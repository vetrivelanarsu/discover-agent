# Discover Agent

Personal learning project: a multi agent system that suggests headlines, tags,
and internal links for news articles. Built to learn AI engineering end to end.

## Stack
- Python 3.12, managed with uv. Run scripts with `uv run <file>`.
- Postgres with pgvector on Supabase. Connection string in DATABASE_URL in .env.
- Gemini via google-genai. Key in GEMINI_API_KEY in .env.
- psycopg 3 for database access. pydantic for schemas.

## Rules
- Never hardcode secrets. Always load from .env with python-dotenv.
- Never print or log DATABASE_URL or GEMINI_API_KEY.
- Do not add LangChain, LlamaIndex, CrewAI, or any agent framework.
  Retrieval, agent loops, and the graph executor are written by hand on purpose.
- Schema changes go in numbered files under sql/ (001_, 002_, ...).
- Keep scripts simple and readable. Prefer plain functions over classes
  unless there is real state to hold.
- Do not add new dependencies without saying why.

## Current schema
articles(id, url unique, title, section, author, published_at, body, fetched_at,
         embedding vector(768), embedded_at)
Schema files live in sql/, applied in order.

## Existing scripts
- ingest.py: pulls BBC Business RSS, cleans bodies with clean_body, stores articles.
- clean_existing.py: reapplies clean_body to stored articles.
- check_db.py: connection check.