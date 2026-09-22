import os
import sys

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


QUERY = """
    select id, title, 1 - (embedding <=> %s) as similarity
    from articles
    where id != %s
      and embedding is not null
    order by embedding <=> %s
    limit %s
"""


def main() -> None:
    load_dotenv()

    if len(sys.argv) not in (2, 3):
        print("Usage: uv run similar.py <article_id> [limit]")
        return

    article_id = int(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)

        row = conn.execute(
            "select title, embedding from articles where id = %s",
            (article_id,),
        ).fetchone()

        if row is None:
            print(f"No article with id {article_id}")
            return

        title, embedding = row

        if embedding is None:
            print(f"Article {article_id} has no embedding yet. Run embed.py first.")
            return

        print(f"\n{title}")
        print("-" * len(title))

        results = conn.execute(
            QUERY, (embedding, article_id, embedding, limit)
        ).fetchall()

        for other_id, other_title, similarity in results:
            print(f"{similarity:.3f}  [{other_id}]  {other_title}")

        print()


if __name__ == "__main__":
    main()