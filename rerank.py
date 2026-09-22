import json
import os
import sys
from pathlib import Path
from typing import Literal

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.7-flash"
PROMPT_PATH = Path(__file__).parent / "prompts" / "rerank_v1.txt"
CANDIDATE_LIMIT = 15
MAX_LINKS = 3
MAX_MODEL_ATTEMPTS = 5

CANDIDATE_QUERY = """
    select id, title, body, 1 - (embedding <=> %s) as similarity
    from articles
    where id != %s
      and embedding is not null
    order by embedding <=> %s
    limit %s
"""

LinkType = Literal["same_story", "related", "explainer", "not_useful"]


class CandidateAssessment(BaseModel):
    id: int
    classification: LinkType
    reason: str = Field(min_length=1, description="A concise one-sentence reason")


class RerankResponse(BaseModel):
    assessments: list[CandidateAssessment]
    chosen_ids: list[int] = Field(max_length=MAX_LINKS)


class RetrievedCandidate(BaseModel):
    id: int
    title: str
    body: str
    similarity: float


class SuggestedLink(BaseModel):
    id: int
    title: str
    similarity: float
    classification: LinkType
    reason: str


class RerankResult(BaseModel):
    model: str | None
    candidates: list[SuggestedLink]
    chosen: list[SuggestedLink]


def retrieve_candidates(
    conn: psycopg.Connection, article_id: int
) -> tuple[str, str, list[RetrievedCandidate]]:
    source = conn.execute(
        "select title, body, embedding from articles where id = %s",
        (article_id,),
    ).fetchone()
    if source is None:
        raise ValueError(f"No article with id {article_id}")

    title, body, embedding = source
    if embedding is None:
        raise ValueError(f"Article {article_id} has no embedding. Run embed.py first.")

    rows = conn.execute(
        CANDIDATE_QUERY,
        (embedding, article_id, embedding, CANDIDATE_LIMIT),
    ).fetchall()
    candidates = [
        RetrievedCandidate(
            id=candidate_id,
            title=candidate_title,
            body=candidate_body or "",
            similarity=float(similarity),
        )
        for candidate_id, candidate_title, candidate_body, similarity in rows
    ]
    return title, body or "", candidates


def build_prompt(
    source_title: str, source_body: str, candidates: list[RetrievedCandidate]
) -> str:
    article_data = {
        "source": {
            "title": source_title,
            "body_excerpt": source_body[:1500],
        },
        "candidates": [
            {
                "id": candidate.id,
                "title": candidate.title,
                "body_excerpt": candidate.body[:400],
            }
            for candidate in candidates
        ],
    }
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        article_data=json.dumps(article_data, ensure_ascii=False, indent=2)
    )


def validate_response(
    response: RerankResponse, candidates: list[RetrievedCandidate]
) -> None:
    candidate_ids = {candidate.id for candidate in candidates}
    assessment_ids = [assessment.id for assessment in response.assessments]

    if len(assessment_ids) != len(set(assessment_ids)):
        raise ValueError("Gemini returned a candidate assessment more than once")
    if not set(assessment_ids).issubset(candidate_ids):
        raise ValueError("Gemini assessed an id outside the retrieved candidates")
    if set(assessment_ids) != candidate_ids:
        raise ValueError("Gemini did not assess every retrieved candidate")

    if len(response.chosen_ids) != len(set(response.chosen_ids)):
        raise ValueError("Gemini chose the same candidate more than once")
    if not set(response.chosen_ids).issubset(candidate_ids):
        raise ValueError("Gemini chose an id outside the retrieved candidates")

    assessments = {assessment.id: assessment for assessment in response.assessments}
    if any(
        assessments[candidate_id].classification == "not_useful"
        for candidate_id in response.chosen_ids
    ):
        raise ValueError("Gemini chose a candidate classified as not_useful")


def is_temporary_error(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and (
        (exc.code, exc.status)
        in {
            (503, "UNAVAILABLE"),
            (429, "RESOURCE_EXHAUSTED"),
        }
    )


def log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait_seconds = (
        retry_state.next_action.sleep if retry_state.next_action is not None else 0
    )
    model = retry_state.args[1]
    print(
        f"Retry {retry_state.attempt_number + 1}/{MAX_MODEL_ATTEMPTS} for "
        f"{model} after {exc.code} {exc.status}; "
        f"waiting {wait_seconds:.1f}s",
        flush=True,
    )


def generate_response(
    client: genai.Client, model: str, prompt: str
) -> types.GenerateContentResponse:
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=RerankResponse,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )


@retry(
    retry=retry_if_exception(is_temporary_error),
    stop=stop_after_attempt(MAX_MODEL_ATTEMPTS),
    wait=wait_random_exponential(multiplier=1, max=30),
    before_sleep=log_retry,
    reraise=True,
)
def generate_response_with_retry(
    client: genai.Client, model: str, prompt: str
) -> types.GenerateContentResponse:
    return generate_response(client, model, prompt)


def rerank_article(
    conn: psycopg.Connection, client: genai.Client, article_id: int
) -> RerankResult:
    source_title, source_body, candidates = retrieve_candidates(conn, article_id)
    if not candidates:
        return RerankResult(model=None, candidates=[], chosen=[])

    prompt = build_prompt(source_title, source_body, candidates)
    try:
        response = generate_response_with_retry(client, MODEL, prompt)
        result_model = MODEL
    except errors.APIError as exc:
        if (exc.code, exc.status) != (503, "UNAVAILABLE"):
            raise
        print(
            f"{MODEL} remained unavailable after {MAX_MODEL_ATTEMPTS} attempts; "
            f"trying fallback {FALLBACK_MODEL} once",
            flush=True,
        )
        response = generate_response(client, FALLBACK_MODEL, prompt)
        result_model = FALLBACK_MODEL

    if isinstance(response.parsed, RerankResponse):
        parsed = response.parsed
    elif response.text:
        parsed = RerankResponse.model_validate_json(response.text)
    else:
        raise ValueError("Gemini returned no reranking response")

    validate_response(parsed, candidates)

    assessments = {assessment.id: assessment for assessment in parsed.assessments}
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    classified = [
        SuggestedLink(
            id=candidate.id,
            title=candidate.title,
            similarity=candidate.similarity,
            classification=assessments[candidate.id].classification,
            reason=assessments[candidate.id].reason,
        )
        for candidate in candidates
    ]
    chosen = [
        SuggestedLink(
            id=candidate_id,
            title=candidate_by_id[candidate_id].title,
            similarity=candidate_by_id[candidate_id].similarity,
            classification=assessments[candidate_id].classification,
            reason=assessments[candidate_id].reason,
        )
        for candidate_id in parsed.chosen_ids
    ]
    return RerankResult(model=result_model, candidates=classified, chosen=chosen)


def suggest_links(
    conn: psycopg.Connection, client: genai.Client, article_id: int
) -> list[SuggestedLink]:
    return rerank_article(conn, client, article_id).chosen


def main() -> None:
    load_dotenv()

    if len(sys.argv) != 2:
        print("Usage: uv run rerank.py <article_id>")
        return

    try:
        article_id = int(sys.argv[1])
    except ValueError:
        print("article_id must be an integer")
        return

    with (
        psycopg.connect(os.environ["DATABASE_URL"]) as conn,
        genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client,
    ):
        register_vector(conn)
        try:
            result = rerank_article(conn, client, article_id)
        except ValueError as exc:
            print(exc)
            return

    model_label = result.model or "none (no candidates)"
    fallback_label = " (FALLBACK)" if result.model == FALLBACK_MODEL else ""
    print(f"\nModel: {model_label}{fallback_label}")

    print("\nCandidate classifications:")
    for candidate in result.candidates:
        print(
            f"{candidate.similarity:.3f}  [{candidate.id}] {candidate.title} - "
            f"{candidate.classification}: {candidate.reason}"
        )

    print("\nChosen links:")
    if not result.chosen:
        print("None")
    for index, link in enumerate(result.chosen, start=1):
        print(
            f"{index}. [{link.id}] {link.title} "
            f"({link.classification}) - {link.reason}"
        )


if __name__ == "__main__":
    main()
