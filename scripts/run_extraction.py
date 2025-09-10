# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import argparse
from typing import Any, Dict, List
from collections import defaultdict

from openai import OpenAI
from neo4j import GraphDatabase, Driver

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import settings

EXTRACTION_PROMPT_TEMPLATE = """
You are a data extraction expert. Your task is to extract entities and relationships from the text and format them as a single, valid JSON object.

**Instructions:**
1.  **Identify Entities:** Extract all relevant entities (e.g., people, organizations, locations, concepts, games). Each entity must have a `name` and a `type`.
2.  **Define Relationships:** Define relationships between entities. Each relationship must have a `source` (entity name), a `target` (entity name), and a `type`.
3.  **Relationship Type Format:** The relationship `type` MUST be an uppercase, snake-cased verb phrase (e.g., `PLAYED_TOGETHER`, `THINKS_IS_COOL`).
4.  **Output Format:** Your final output MUST be only a single, raw JSON object. Do not include any explanations, notes, or markdown fences (like ```json) before or after the JSON.
5.  **Compact JSON:** The JSON output MUST be a compact, single-line string without any newlines or formatting whitespace.

**Example:**
Text: 'Elon Musk, the founder of SpaceX, is also involved with Neuralink.'
JSON Output: {{"entities": [{{"name": "Elon Musk", "type": "PERSON"}}, {{"name": "SpaceX", "type": "ORGANIZATION"}}, {{"name": "Neuralink", "type": "ORGANIZATION"}}], "relationships": [{{"source": "Elon Musk", "target": "SpaceX", "type": "IS_FOUNDER_OF"}}, {{"source": "Elon Musk", "target": "Neuralink", "type": "IS_INVOLVED_WITH"}}]}}

**Text to Process:**
Text: '{text_input}'
JSON Output:
""".strip()


# ------------------------
# Clients
# ------------------------
def get_llm_client() -> OpenAI:
    return OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url
    )


def get_neo4j_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password)
    )


# ------------------------
# Robust fence stripping & JSON slicing
# ------------------------
def _strip_code_fence(s: str) -> str:
    """Remove ```...``` fences (with optional language tag),
    supporting both real newline and literal '\\n' after language tag."""
    fence = "```"
    s = s.strip()
    if s.startswith(fence):
        end = s.find(fence, 3)
        if end != -1:
            inner = s[3:end]
            inner = re.sub(r'^\s*[A-Za-z0-9_+\-]+\s*(?:\n|\\n)', '', inner)
            return inner.strip()
    if fence in s:
        i = s.find(fence)
        j = s.find(fence, i + 3)
        if j != -1:
            inner = s[i + 3:j]
            inner = re.sub(r'^\s*[A-Za-z0-9_+\-]+\s*(?:\n|\\n)', '', inner)
            return inner.strip()
    return s


def _slice_first_balanced_json(s: str) -> str:
    """Slice the first balanced JSON object text, respecting strings/escapes."""
    start = s.find('{')
    if start == -1:
        raise ValueError("No JSON object start '{' found.")

    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start:idx + 1]
    raise ValueError("Unbalanced JSON braces.")


def _pre_unescape_for_slice(s: str) -> str:
    """Minimal safe unescape to turn {\"...\"} 风格为 {"..."}，便于平衡扫描。"""
    # 顺序很重要：先去掉 \"，再收敛双反斜杠，最后把 \n、\t 变成真实控制符
    s = s.replace('\\"', '"')
    s = s.replace('\\\\', '\\')
    s = s.replace('\\n', '\n').replace('\\t', '\t')
    return s


# ------------------------
# Parsing strategies
# ------------------------
def _try_parse_json(candidate: str) -> Dict[str, Any]:
    # Strategy 1: direct loads
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip cosmetic escapes then loads
    cleaned = candidate.replace('\\n', '').replace('\\t', '')
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: try minimal de-escape then loads
    try:
        deq = _pre_unescape_for_slice(candidate)
        return json.loads(deq)
    except json.JSONDecodeError:
        pass

    # Strategy 4: conservative remove backslash-before-quote
    try:
        dequoted = re.sub(r'\\(?=")', '', candidate)
        return json.loads(dequoted)
    except json.JSONDecodeError as e:
        preview = candidate[:1000]
        raise ValueError(
            f"Failed to parse JSON after multiple attempts. Last error: {e}\nCandidate Preview: {preview}"
        )


def _call_llm_for_json(client: OpenAI, prompt: str) -> str:
    """Prefer strict JSON mode if backend supports it; fallback to plain text."""
    try:
        resp = client.chat.completions.create(
            model=settings.llm.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    except Exception:
        resp = client.chat_completions.create(  # some gateways alias; fallback to standard if needed
            model=settings.llm.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        ) if hasattr(client, "chat_completions") else client.chat.completions.create(
            model=settings.llm.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


# ------------------------
# Extraction & normalization
# ------------------------
def extract_graph_from_text(client: OpenAI, text: str) -> Dict[str, Any]:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(text_input=text)

    print("--- Calling LLM for extraction ---")
    content = _call_llm_for_json(client, prompt)

    print("--- LLM Response ---")
    print(f"Raw LLM Response:\n{content}\n")

    # 1) Strip code fences
    inner = _strip_code_fence(content)

    # 2) Slice first JSON; if失败，先做轻量去转义再试
    try:
        candidate = _slice_first_balanced_json(inner)
    except ValueError:
        print("[parser] First pass failed (likely escaped quotes). Retrying with de-escape...")
        inner2 = _pre_unescape_for_slice(inner)
        candidate = _slice_first_balanced_json(inner2)

    # 3) Multi-strategy parse
    data = _try_parse_json(candidate)

    # 4) Shape validation & normalization
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object.")

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    if not isinstance(entities, list):
        raise ValueError("'entities' must be a list.")
    if not isinstance(relationships, list):
        raise ValueError("'relationships' must be a list.")

    norm_entities: List[Dict[str, Any]] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        etype = str(e.get("type", "ENTITY")).strip()
        if not name:
            continue
        norm_entities.append({"name": name, "type": etype or "ENTITY"})

    norm_rels: List[Dict[str, str]] = []
    for r in relationships:
        if not isinstance(r, dict):
            continue
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        rtype = str(r.get("type", "")).strip()
        if not (src and tgt and rtype):
            continue
        if "`" in rtype:
            continue
        norm_rels.append({"source": src, "target": tgt, "type": rtype})

    return {"entities": norm_entities, "relationships": norm_rels}


# ------------------------
# Neo4j ingestion (safe labels)
# ------------------------
_label_cache: Dict[str, str] = {}


def _safe_label(label: str) -> str:
    if label in _label_cache:
        return _label_cache[label]
    cleaned = re.sub(r'[^A-Za-z0-9_]', '_', label or '')
    if not cleaned or not re.match(r'[A-Za-z_]', cleaned[0]):
        cleaned = f'X_{cleaned}' if cleaned else 'X'
    _label_cache[label] = cleaned
    return cleaned


def ingest_graph_to_neo4j(driver: Driver, graph_data: Dict[str, Any]):
    print("\n--- Ingesting data into Neo4j ---")
    entities = graph_data.get('entities', [])
    relationships = graph_data.get('relationships', [])

    # Group entities by label for batching
    entities_by_label = defaultdict(list)
    for entity in entities:
        label = _safe_label(entity.get('type', 'ENTITY'))
        entities_by_label[label].append(entity)

    with driver.session() as session:
        # Ingest entities in batches by label
        for label, entity_list in entities_by_label.items():
            # Using f-string for the label is safe here because `_safe_label` sanitizes it
            query = f"UNWIND $entities AS entity MERGE (n:`{label}` {{name: entity.name}})"
            session.run(query, entities=entity_list)

        # Ingest relationships in a single batch using APOC
        if relationships:
            # We need to make sure we don't have backticks in the relationship type for APOC
            valid_rels = []
            for rel in relationships:
                rel_type = rel.get('type')
                if rel_type and '`' not in rel_type:
                    # APOC expects the type as a string, no need to pre-format
                    valid_rels.append(rel)
                else:
                    print(f"Skipping relationship with invalid type: {rel_type}")

            if valid_rels:
                # Use apoc.create.relationship for dynamic relationship types
                query = """
                UNWIND $rels AS rel
                MATCH (a {name: rel.source}), (b {name: rel.target})
                CALL apoc.create.relationship(a, rel.type, {}, b) YIELD rel as r
                RETURN count(r)
                """
                session.run(query, rels=valid_rels)

    print(f"Ingested {len(entities)} entities and {len(relationships)} relationships.")


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract a knowledge graph from text and ingest it into Neo4j.")
    parser.add_argument("text_input", type=str, nargs='?', default=None, help="The input text to process. If not provided, reads from stdin.")
    args = parser.parse_args()

    if args.text_input:
        text_to_process = args.text_input
    elif not sys.stdin.isatty():
        text_to_process = sys.stdin.read()
    else:
        print("Error: No text provided. Please provide text as an argument or pipe it via stdin.", file=sys.stderr)
        sys.exit(1)


    llm_client = get_llm_client()
    neo4j_driver = get_neo4j_driver()

    try:
        extracted_data = extract_graph_from_text(llm_client, text_to_process)
        print("--- Parsed JSON ---")
        print(json.dumps(extracted_data, ensure_ascii=False, indent=2))
        ingest_graph_to_neo4j(neo4j_driver, extracted_data)

        print("\n--- Verification ---")
        with neo4j_driver.session() as session:
            result = session.run("MATCH (a)-[r]->(b) RETURN a.name AS a, type(r) AS r, b.name AS b LIMIT 10")
            for record in result:
                print(f"{record['a']} -[{record['r']}]-> {record['b']}")
    finally:
        neo4j_driver.close()

    print("\nScript finished successfully.")


if __name__ == "__main__":
    main()
