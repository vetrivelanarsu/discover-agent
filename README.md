# Discover Agent

Discover Agent is a personal learning project for building a multi-agent news
workflow from first principles. The finished system will suggest headlines,
tags, and relevant internal links for news articles.

The project intentionally does not use an agent framework. Retrieval, agent
loops, and graph execution will be implemented directly in Python.

## Current status

The project currently includes:

- A Postgres `articles` table with pgvector enabled
- A BBC Business RSS ingestion script
- Article text and author extraction with Trafilatura
- Idempotent inserts, allowing ingestion to be run repeatedly

## Stack

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Postgres with pgvector on Supabase
- Gemini through `google-genai`
- psycopg 3 for database access
- Pydantic for structured schemas

## Setup

Install the project dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```dotenv
DATABASE_URL=your_supabase_postgres_connection_string
GEMINI_API_KEY=your_gemini_api_key
```

The `.env` file is ignored by Git. Do not commit credentials.

## Database

Apply the numbered SQL migrations in order. The initial migration is:

```text
sql/001_create_articles.sql
```

It creates the `articles` table:

```text
articles(id, url, title, section, author, published_at, body, fetched_at)
```

Check the database connection with:

```bash
uv run check_db.py
```

## Ingest articles

Run the BBC Business ingestion script:

```bash
uv run ingest.py
```

The script reads the BBC Business RSS feed, downloads each article, extracts
clean body text, and inserts new articles into Postgres. It strips URL query
strings, ignores unsupported media pages and bodies shorter than 500
characters, and skips URLs already stored in the database.

Each article is reported as `OK`, `SKIPPED`, `TOO SHORT`, or `FAILED`, followed
by a summary for the run. Individual article failures do not stop the remaining
feed from being processed.

## Project structure

```text
.
├── ingest.py                 # Fetch and store BBC Business articles
├── check_db.py               # Verify the database connection
├── sql/                      # Numbered database migrations
└── src/discover_agent/       # Application package
```
