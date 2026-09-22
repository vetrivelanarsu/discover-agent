import math
import os
import time

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pgvector.psycopg import register_vector
from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential


MODEL = "gemini-embedding-001"
BATCH_SIZE = 20
BATCH_SLEEP_SECONDS = 2
OUTPUT_DIMENSIONALITY = 768


def build_embedding_text(title: str, body: str | None) -> str:
    return f"{title}\n\n{(body or '')[:1500]}"


def normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise ValueError("Gemini returned a zero-length embedding")
    return [value / magnitude for value in vector]


def report_retry(retry_state: RetryCallState) -> None:
    batch_number = retry_state.args[2]
    print(
        f"Batch {batch_number}: attempt {retry_state.attempt_number} failed; retrying",
        flush=True,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=report_retry,
    reraise=True,
)
def embed_batch(
    client: genai.Client, texts: list[str], batch_number: int
) -> list[list[float]]:
    response = client.models.embed_content(
        model=MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=OUTPUT_DIMENSIONALITY,
        ),
    )

    if response.embeddings is None or len(response.embeddings) != len(texts):
        raise ValueError("Gemini returned an unexpected number of embeddings")

    vectors = []
    for embedding in response.embeddings:
        if embedding.values is None:
            raise ValueError("Gemini returned an embedding without values")
        if len(embedding.values) != OUTPUT_DIMENSIONALITY:
            raise ValueError("Gemini returned an embedding with the wrong dimensions")
        vectors.append(normalize(embedding.values))
    return vectors


def main() -> None:
    load_dotenv()

    embedded_count = 0
    failed_count = 0

    with (
        psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn,
        genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client,
    ):
        register_vector(conn)
        articles = conn.execute(
            """
            SELECT id, title, body
            FROM articles
            WHERE embedding IS NULL
            ORDER BY id
            """
        ).fetchall()

        total_articles = len(articles)
        total_batches = math.ceil(total_articles / BATCH_SIZE)

        for batch_index, start in enumerate(
            range(0, total_articles, BATCH_SIZE), start=1
        ):
            batch = articles[start : start + BATCH_SIZE]
            texts = [build_embedding_text(title, body) for _, title, body in batch]

            try:
                vectors = embed_batch(client, texts, batch_index)
                with conn.transaction():
                    with conn.cursor() as cursor:
                        cursor.executemany(
                            """
                            UPDATE articles
                            SET embedding = %s, embedded_at = now()
                            WHERE id = %s
                            """,
                            [
                                (vector, article_id)
                                for (article_id, _, _), vector in zip(
                                    batch, vectors, strict=True
                                )
                            ],
                        )

                embedded_count += len(batch)
                print(
                    f"Batch {batch_index}/{total_batches}: embedded {len(batch)} "
                    f"articles ({embedded_count}/{total_articles})",
                    flush=True,
                )
            except Exception as exc:
                failed_count += len(batch)
                print(
                    f"Batch {batch_index}/{total_batches}: FAILED "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )

            if batch_index < total_batches:
                time.sleep(BATCH_SLEEP_SECONDS)

    print(f"Embedded {embedded_count} articles; failed {failed_count}.")


if __name__ == "__main__":
    main()
