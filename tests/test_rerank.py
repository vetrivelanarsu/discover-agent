import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from eval_links import main as eval_main
from eval_links import score
from rerank import (
    CandidateAssessment,
    LinkType,
    RerankResult,
    SuggestedLink,
    select_links,
    suggest_links,
)


def link(
    article_id: int,
    similarity: float,
    classification: LinkType,
    usefulness: int,
) -> SuggestedLink:
    return SuggestedLink(
        id=article_id,
        title=f"Article {article_id}",
        similarity=similarity,
        classification=classification,
        usefulness=usefulness,
        reason="Test reason.",
    )


class SelectLinksTests(unittest.TestCase):
    def test_picks_two_same_story_then_best_other_type(self) -> None:
        classified = [
            link(1, 0.50, "same_story", 5),
            link(2, 0.80, "same_story", 4),
            link(3, 0.90, "same_story", 4),
            link(4, 0.70, "related", 5),
            link(5, 0.80, "explainer", 5),
            link(6, 0.99, "not_useful", 5),
        ]

        self.assertEqual(
            [candidate.id for candidate in select_links(classified)],
            [1, 3, 5],
        )

    def test_returns_empty_when_every_candidate_is_not_useful(self) -> None:
        classified = [
            link(1, 0.95, "not_useful", 5),
            link(2, 0.90, "not_useful", 4),
        ]

        self.assertEqual(select_links(classified), [])

    def test_returns_empty_when_all_candidates_are_below_their_floor(self) -> None:
        classified = [
            link(1, 0.99, "same_story", 2),
            link(2, 0.98, "related", 2),
            link(3, 0.97, "explainer", 2),
        ]

        self.assertEqual(select_links(classified), [])

    def test_related_and_same_story_usefulness_three_are_included(
        self,
    ) -> None:
        classified = [
            link(1, 0.80, "same_story", 3),
            link(2, 0.70, "related", 3),
        ]

        self.assertEqual(
            [candidate.id for candidate in select_links(classified)],
            [1, 2],
        )

    def test_returns_one_link_only_when_one_candidate_qualifies(self) -> None:
        classified = [
            link(1, 0.80, "same_story", 4),
            link(2, 0.70, "related", 2),
        ]

        self.assertEqual(
            [candidate.id for candidate in select_links(classified)],
            [1],
        )

    def test_returns_exactly_two_same_story_links_without_padding(self) -> None:
        classified = [
            link(1, 0.50, "same_story", 5),
            link(2, 0.80, "same_story", 4),
            link(3, 0.90, "same_story", 2),
        ]

        self.assertEqual(
            [candidate.id for candidate in select_links(classified)],
            [1, 2],
        )

    def test_only_one_related_or_explainer_is_picked(self) -> None:
        classified = [
            link(1, 0.80, "related", 4),
            link(2, 0.90, "explainer", 4),
        ]

        self.assertEqual(
            [candidate.id for candidate in select_links(classified)],
            [2],
        )

    def test_usefulness_must_be_between_one_and_five(self) -> None:
        for usefulness in (0, 6):
            with self.subTest(usefulness=usefulness):
                with self.assertRaises(ValidationError):
                    CandidateAssessment(
                        id=1,
                        classification="related",
                        usefulness=usefulness,
                        reason="Test reason.",
                    )

    def test_suggest_links_prints_when_no_links_are_selected(self) -> None:
        result = RerankResult(
            requested_model="test-model",
            model="test-model",
            candidates=[],
            chosen=[],
        )

        with (
            patch("rerank.rerank_article", return_value=result),
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            self.assertEqual(suggest_links(None, None, 1, model="test-model"), [])

        self.assertIn("No suitable links found", output.getvalue())


class EvalScoreTests(unittest.TestCase):
    def test_expect_none_passes_only_when_no_links_are_returned(self) -> None:
        case = {
            "expect_none": True,
            "should_link": [],
            "acceptable": [],
            "should_reject": [],
        }

        self.assertEqual(score([], case)["expect_none_pass"], 1)
        self.assertEqual(score([], case)["expect_none_fail"], 0)
        self.assertEqual(score([1], case)["expect_none_pass"], 0)
        self.assertEqual(score([1], case)["expect_none_fail"], 1)


class EvalReplayTests(unittest.TestCase):
    def test_replay_uses_saved_file_without_live_clients(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "eval_replay_run.json"

        with (
            patch("sys.argv", ["eval_links.py", "--replay", str(fixture)]),
            patch("eval_links.genai.Client", side_effect=AssertionError),
            patch("eval_links.psycopg.connect", side_effect=AssertionError),
            patch("sys.stdout", new_callable=StringIO) as output,
        ):
            eval_main()

        report = output.getvalue()
        self.assertIn("Per-case results", report)
        self.assertIn("Average totals per run", report)
        self.assertIn("Stability by case", report)
        self.assertIn("same_story 3, same_story 3", report)


if __name__ == "__main__":
    unittest.main()
