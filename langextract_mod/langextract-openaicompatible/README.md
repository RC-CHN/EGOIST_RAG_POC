        # LangExtract OpenAICompatible Provider

A provider plugin for LangExtract that supports OpenAICompatible models.

## Installation

```bash
pip install -e .
```

## Supported Model IDs

- `openaicompatible*`: Models matching pattern ^openaicompatible

## Environment Variables

- `OPENAICOMPATIBLE_API_KEY`: API key for authentication

## Usage

```python
import langextract as lx

result = lx.extract(
    text="Your document here",
    model_id="openaicompatible-model",
    prompt_description="Extract entities",
    examples=[...]
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

## License

Apache License 2.0
