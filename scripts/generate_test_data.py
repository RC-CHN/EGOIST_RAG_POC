# -*- coding: utf-8 -*-
import os
import sys
import argparse
from openai import OpenAI

# Add the project root to the Python path for settings import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import settings

GENERATE_PROMPT_TEMPLATE = """
You are a creative writer. Your task is to generate a natural and detailed conversation on the topic of "{topic}".

**Instructions:**
1.  The conversation should be engaging and involve multiple speakers.
2.  The total length of the dialogue should be approximately {length} characters.
3.  Clearly label who is speaking, for example: "Alice:", "Bob:", etc.
4.  The output MUST be only the raw text of the conversation. Do not include any titles, summaries, or explanations before or after the dialogue.
5.  The conversation should be rich with entities (like names, places, games, technologies) and relationships between them to create good test data for extraction.
""".strip()


def get_llm_client() -> OpenAI:
    """Initializes and returns the OpenAI client using project settings."""
    return OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url
    )


def generate_dialogue(client: OpenAI, topic: str, length: int) -> str:
    """Generates dialogue by calling the LLM."""
    prompt = GENERATE_PROMPT_TEMPLATE.format(topic=topic, length=length)
    
    print("--- Calling LLM to generate test data... ---", file=sys.stderr)
    
    response = client.chat.completions.create(
        model=settings.llm.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7, # A bit of creativity is good for test data
    )
    
    print("--- LLM response received. ---", file=sys.stderr)
    return response.choices[0].message.content or ""


def main():
    """Main function to generate test data."""
    parser = argparse.ArgumentParser(
        description="Generate conversational test data using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="A group of friends discussing their favorite video games, recent sci-fi movies, and planning a weekend trip.",
        help="The topic for the conversation."
    )
    parser.add_argument(
        "--length",
        type=int,
        default=2500,
        help="The approximate number of characters for the generated dialogue."
    )
    args = parser.parse_args()

    llm_client = get_llm_client()
    
    try:
        dialogue = generate_dialogue(llm_client, args.topic, args.length)
        # Print the final dialogue to stdout, so it can be piped
        print(dialogue)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()