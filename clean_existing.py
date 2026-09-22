import os

import psycopg
from dotenv import load_dotenv

from ingest import clean_body


def main() -> None:
    load_dotenv()

    changed = 0
    unchanged = 0

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        rows = conn.execute("SELECT id, title, body FROM articles").fetchall()

        for article_id, title, body in rows:
            if body is None:
                unchanged += 1
                continue

            cleaned = clean_body(body)
            if cleaned == body:
                unchanged += 1
                continue

            removed = len(body) - len(cleaned)
            conn.execute(
                "UPDATE articles SET body = %s WHERE id = %s",
                (cleaned, article_id),
            )
            changed += 1
            print(f"CLEANED: {title[:70]} (removed {removed} characters)")

        conn.commit()

    print(f"\nChanged {changed}, unchanged {unchanged}")


if __name__ == "__main__":
    main()