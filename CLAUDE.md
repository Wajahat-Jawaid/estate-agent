# ZAMEEN

@context.md

## Python environment
- Always run Python via `venv/bin/python` (or `source venv/bin/activate` first).
- Never use bare `python3` — resolves to system Python 3.9 (LibreSSL), missing our deps.
- Don't run HuggingFace embeddings / similarity-search tests unless I ask.

# Real Estate AI Agent Rules

## Product Goal
This is not a property browsing chatbot. It is an AI real estate sales assistant.

## Core Conversation Chain
qualification → guidance → shortlist → comparison → objection handling → contact capture → visit booking → agent handoff → follow-up

## Hard Rules
- Never show properties too early.
- Ask only one clear question at a time.
- Do not produce long paragraphs.
- Do not invent property details not present in JSON.
- Do not mention areas unrelated to the user's budget/context.
