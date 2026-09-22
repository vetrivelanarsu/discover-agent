import argparse
import json
import os
from collections import Counter
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg import register_vector

from rerank import MODEL, rerank_article, retrieve_candidates


CASES_PATH = Path(__file__).parent / "eval" / "link_cases.json"
METRICS = ("should_link", "acceptable", "should_reject", "missed")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


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


def add_metrics(total: dict[str, int], result: dict[str, int]) -> None:
    for metric in METRICS:
        total[metric] += result[metric]


def format_frequencies(counts: Counter, total: int) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{value}: {count}/{total}"
        for value, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
    )


def actual_model_label(requested_model: str, actual_model: str | None) -> str:
    if actual_model is None:
        return "none"
    if actual_model != requested_model:
        return f"{actual_model} (FALLBACK)"
    return actual_model


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate internal link retrieval")
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Gemini model to use (default: {MODEL})",
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=1,
        help="Number of times to rerank each case (default: 1)",
    )
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    totals = {
        "baseline": {metric: 0 for metric in METRICS},
        "reranker": {metric: 0 for metric in METRICS},
    }
    total_models = {
        "baseline": Counter(),
        "reranker": Counter(),
    }
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
            baseline_result = score(baseline_ids, case)
            baseline_metrics = {
                metric: baseline_result[metric] * args.runs for metric in METRICS
            }
            add_metrics(totals["baseline"], baseline_metrics)
            baseline_frequency = Counter(
                {candidate_id: args.runs for candidate_id in baseline_ids}
            )
            baseline_models = Counter({"similarity-only": args.runs})
            total_models["baseline"].update(baseline_models)

            reranker_metrics = {metric: 0 for metric in METRICS}
            reranker_frequency = Counter()
            reranker_models = Counter()

            for run_number in range(1, args.runs + 1):
                if args.runs > 1:
                    print(
                        f"Case {article_id_text}: reranker run "
                        f"{run_number}/{args.runs}",
                        flush=True,
                    )
                reranked = rerank_article(
                    conn,
                    client,
                    article_id,
                    model=args.model,
                )
                reranked_ids = [link.id for link in reranked.chosen]
                reranker_frequency.update(reranked_ids)
                model_label = actual_model_label(
                    reranked.requested_model,
                    reranked.model,
                )
                reranker_models[model_label] += 1
                add_metrics(reranker_metrics, score(reranked_ids, case))

            add_metrics(totals["reranker"], reranker_metrics)
            total_models["reranker"].update(reranker_models)

            for method, requested_model, models, frequency, metrics in (
                (
                    "baseline",
                    "n/a",
                    baseline_models,
                    baseline_frequency,
                    baseline_metrics,
                ),
                (
                    "reranker",
                    args.model,
                    reranker_models,
                    reranker_frequency,
                    reranker_metrics,
                ),
            ):
                rows.append(
                    [
                        article_id_text,
                        method,
                        requested_model,
                        format_frequencies(models, args.runs),
                        format_frequencies(frequency, args.runs),
                        *(
                            f"{metrics[metric] / args.runs:.2f}"
                            for metric in METRICS
                        ),
                    ]
                )

    print("\nPer-case results")
    print_table(
        [
            "Case",
            "Method",
            "Requested model",
            "Actual model frequency",
            "Chosen id frequency",
            "Avg should",
            "Avg acceptable",
            "Avg reject",
            "Avg missed",
        ],
        rows,
    )

    total_evaluations = len(cases) * args.runs
    total_rows = [
        [
            method,
            "n/a" if method == "baseline" else args.model,
            format_frequencies(total_models[method], total_evaluations),
            *(f"{totals[method][metric] / args.runs:.2f}" for metric in METRICS),
        ]
        for method in ("baseline", "reranker")
    ]
    print("\nAverage totals per run")
    print_table(
        [
            "Method",
            "Requested model",
            "Actual model frequency",
            "Avg should",
            "Avg acceptable",
            "Avg reject",
            "Avg missed",
        ],
        total_rows,
    )

    if len(total_models["reranker"]) > 1:
        print(
            "\nWarning: reranker averages include results from different actual "
            "models, which are not directly comparable."
        )


if __name__ == "__main__":
    main()
