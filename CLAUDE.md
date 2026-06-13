# ZAMEEN

@context.md

## Python environment
- Always run Python via `venv/bin/python` (or `source venv/bin/activate` first).
- Never use bare `python3` — resolves to system Python 3.9 (LibreSSL), missing our deps.
- Don't run HuggingFace embeddings / similarity-search tests unless I ask.