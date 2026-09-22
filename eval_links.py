import argparse
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg import register_vector

from rerank import MODEL, rerank_article, retrieve_candidates


CASES_PATH = Path(__file__).parent / "eval" / "link_cases.json"
METRICS = ("should_link", "acceptable", "should_reject", "missed")


def score(chosen_ids: list[int], case: dict[str, list[int]]) -> dict[str, int]:
    chosen = set(chosen_ids)
    should_link = set(case["should_link"])
    return {
        "should_link": len(chosen & should_link),
        "acceptable": len(chosen & set(case["acceptable"])),
        "should_reject": len(chosen & set(case["should_reject"])),
        "missed": len(should_link - chosen),
    }


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    separator = "-+-".join("-" * width for width in widths)

    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print(separator)
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate internal link retrieval")
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Gemini model to use (default: {MODEL})",
    )
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    totals: dict[tuple[str, str, str], dict[str, int]] = {}
    rows = []

    with (
        psycopg.connect(os.environ["DATABASE_URL"]) as conn,
        genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client,
    ):
        register_vector(conn)

        for article_id_text, case in cases.items():
            article_id = int(article_id_text)
            _, _, candidates = retrieve_candidates(conn, article_id)
            baseline_ids = [candidate.id for candidate in candidates[:3]]
            reranked = rerank_article(
                conn,
                client,
                article_id,
                model=args.model,
            )
            reranked_ids = [link.id for link in reranked.chosen]
            actual_model = reranked.model or "none"
            if (
                reranked.model is not None
                and reranked.model != reranked.requested_model
            ):
                actual_model += " (FALLBACK)"

            for method, requested_model, result_model, chosen_ids in (
                ("baseline", "n/a", "similarity-only", baseline_ids),
                (
                    "reranker",
                    reranked.requested_model,
                    actual_model,
                    reranked_ids,
                ),
            ):
                result = score(chosen_ids, case)
                total_key = (method, requested_model, result_model)
                if total_key not in totals:
                    totals[total_key] = {metric: 0 for metric in METRICS}
                for metric in METRICS:
                    totals[total_key][metric] += result[metric]
                rows.append(
                    [
                        article_id_text,
                        method,
                        requested_model,
                        result_model,
                        ", ".join(map(str, chosen_ids)) or "none",
                        *(str(result[metric]) for metric in METRICS),
                    ]
                )

    print("\nPer-case results")
    print_table(
        [
            "Case",
            "Method",
            "Requested model",
            "Actual model",
            "Chosen ids",
            "Should",
            "Acceptable",
            "Reject",
            "Missed",
        ],
        rows,
    )

    total_rows = [
        [
            method,
            requested_model,
            result_model,
            *(str(results[metric]) for metric in METRICS),
        ]
        for (method, requested_model, result_model), results in totals.items()
    ]
    print("\nTotals")
    print_table(
        [
            "Method",
            "Requested model",
            "Actual model",
            "Should",
            "Acceptable",
            "Reject",
            "Missed",
        ],
        total_rows,
    )


if __name__ == "__main__":
    main()
