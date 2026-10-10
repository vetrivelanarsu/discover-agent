import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg import register_vector

from rerank import MODEL, SuggestedLink, rerank_article, retrieve_candidates, select_links


CASES_PATH = Path(__file__).parent / "eval" / "link_cases.json"
RUNS_DIR = Path(__file__).parent / "eval" / "runs"
METRICS = ("should_link", "acceptable", "should_reject", "missed")
NO_LINK_METRICS = ("expect_none_pass", "expect_none_fail")
ALL_METRICS = METRICS + NO_LINK_METRICS


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def score(chosen_ids: list[int], case: dict[str, Any]) -> dict[str, int]:
    chosen = set(chosen_ids)
    should_link = set(case.get("should_link", []))
    expect_none = bool(case.get("expect_none"))
    return {
        "should_link": len(chosen & should_link),
        "acceptable": len(chosen & set(case.get("acceptable", []))),
        "should_reject": len(chosen & set(case.get("should_reject", []))),
        "missed": len(should_link - chosen),
        "expect_none_pass": int(expect_none and not chosen),
        "expect_none_fail": int(expect_none and bool(chosen)),
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
    for metric in ALL_METRICS:
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


def safe_model_name(model: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in model)


def saved_candidate(link: SuggestedLink, selected_ids: set[int]) -> dict[str, Any]:
    return {
        "id": link.id,
        "title": link.title,
        "label": link.classification,
        "usefulness": link.usefulness,
        "reason": link.reason,
        "similarity": link.similarity,
        "selected": link.id in selected_ids,
    }


def candidate_from_saved(candidate: dict[str, Any]) -> SuggestedLink:
    return SuggestedLink(
        id=candidate["id"],
        title=candidate.get("title", f"Article {candidate['id']}"),
        similarity=candidate["similarity"],
        classification=candidate["label"],
        usefulness=candidate["usefulness"],
        reason=candidate["reason"],
    )


def selected_ids_from_saved(run: dict[str, Any]) -> list[int]:
    candidates = [candidate_from_saved(candidate) for candidate in run["candidates"]]
    return [candidate.id for candidate in select_links(candidates)]


def save_run_file(run_data: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{safe_model_name(run_data['model'])}-{timestamp}.json"
    path.write_text(json.dumps(run_data, indent=2), encoding="utf-8")
    return path


def collect_live_run_data(model: str, runs: int) -> tuple[dict[str, Any], Path]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    run_data = {
        "version": 1,
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "cases": {},
    }

    with (
        psycopg.connect(os.environ["DATABASE_URL"]) as conn,
        genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client,
    ):
        register_vector(conn)

        for article_id_text, case in cases.items():
            article_id = int(article_id_text)
            _, _, candidates = retrieve_candidates(conn, article_id)
            case_data = {
                "case": case,
                "baseline_ids": [candidate.id for candidate in candidates[:3]],
                "runs": [],
            }

            for run_number in range(1, runs + 1):
                if runs > 1:
                    print(
                        f"Case {article_id_text}: reranker run "
                        f"{run_number}/{runs}",
                        flush=True,
                    )
                reranked = rerank_article(conn, client, article_id, model=model)
                selected_ids = {link.id for link in reranked.chosen}
                case_data["runs"].append(
                    {
                        "run_number": run_number,
                        "requested_model": reranked.requested_model,
                        "actual_model": reranked.model,
                        "candidates": [
                            saved_candidate(candidate, selected_ids)
                            for candidate in reranked.candidates
                        ],
                    }
                )

            run_data["cases"][article_id_text] = case_data

    return run_data, save_run_file(run_data)


def load_replay_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_run_data(run_data: dict[str, Any]) -> None:
    cases = run_data["cases"]
    model = run_data["model"]
    rows = []
    stability_rows = []
    totals = {
        "baseline": {metric: 0 for metric in ALL_METRICS},
        "reranker": {metric: 0 for metric in ALL_METRICS},
    }
    total_models = {
        "baseline": Counter(),
        "reranker": Counter(),
    }
    total_evaluations = 0

    for article_id_text, case_data in cases.items():
        case = case_data["case"]
        reranker_runs = case_data["runs"]
        runs = len(reranker_runs)
        total_evaluations += runs

        baseline_ids = case_data["baseline_ids"]
        baseline_result = score(baseline_ids, case)
        baseline_metrics = {
            metric: baseline_result[metric] * runs for metric in ALL_METRICS
        }
        add_metrics(totals["baseline"], baseline_metrics)
        baseline_frequency = Counter({candidate_id: runs for candidate_id in baseline_ids})
        baseline_models = Counter({"similarity-only": runs})
        total_models["baseline"].update(baseline_models)

        reranker_metrics = {metric: 0 for metric in ALL_METRICS}
        reranker_frequency = Counter()
        reranker_models = Counter()

        for run in reranker_runs:
            reranked_ids = selected_ids_from_saved(run)
            reranker_frequency.update(reranked_ids)
            model_label = actual_model_label(
                run.get("requested_model", model),
                run.get("actual_model"),
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
                model,
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
                    format_frequencies(models, runs),
                    format_frequencies(frequency, runs),
                    *(f"{metrics[metric] / runs:.2f}" for metric in METRICS),
                ]
            )

        watched_ids = (
            case.get("should_link", [])
            + case.get("acceptable", [])
            + case.get("should_reject", [])
        )
        for candidate_id in dict.fromkeys(watched_ids):
            per_run = []
            appeared = False
            for run in reranker_runs:
                candidates_by_id = {
                    candidate["id"]: candidate for candidate in run["candidates"]
                }
                candidate = candidates_by_id.get(candidate_id)
                if candidate is None:
                    per_run.append("-")
                    continue
                appeared = True
                per_run.append(f"{candidate['label']} {candidate['usefulness']}")
            if appeared:
                stability_rows.append(
                    [article_id_text, str(candidate_id), ", ".join(per_run)]
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

    total_rows = [
        [
            method,
            "n/a" if method == "baseline" else model,
            format_frequencies(total_models[method], total_evaluations),
            *(f"{totals[method][metric] / run_data['runs']:.2f}" for metric in METRICS),
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

    no_link_rows = [
        [
            method,
            f"{totals[method]['expect_none_pass'] / run_data['runs']:.2f}",
            f"{totals[method]['expect_none_fail'] / run_data['runs']:.2f}",
        ]
        for method in ("baseline", "reranker")
    ]
    print("\nNo-link expectation totals per run")
    print_table(["Method", "Avg expect-none pass", "Avg expect-none fail"], no_link_rows)

    print("\nStability by case")
    print_table(["Case", "Candidate", "Label/usefulness by run"], stability_rows)

    if len(total_models["reranker"]) > 1:
        print(
            "\nWarning: reranker averages include results from different actual "
            "models, which are not directly comparable."
        )


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
    parser.add_argument(
        "--replay",
        type=Path,
        help="Replay a saved eval run without calling Gemini",
    )
    args = parser.parse_args()

    if args.replay:
        evaluate_run_data(load_replay_file(args.replay))
        return

    run_data, saved_path = collect_live_run_data(args.model, args.runs)
    print(f"\nSaved run file: {saved_path}")
    evaluate_run_data(run_data)


if __name__ == "__main__":
    main()
