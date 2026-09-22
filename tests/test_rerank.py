import unittest

from pydantic import ValidationError

from rerank import CandidateAssessment, LinkType, SuggestedLink, select_links


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

    def test_never_picks_low_usefulness_or_not_useful(self) -> None:
        classified = [
            link(1, 0.99, "same_story", 2),
            link(2, 0.98, "related", 2),
            link(3, 0.97, "not_useful", 5),
            link(4, 0.70, "explainer", 3),
        ]

        self.assertEqual(
            [candidate.id for candidate in select_links(classified)],
            [4],
        )

    def test_all_not_useful(self) -> None:
        classified = [
            link(1, 0.95, "not_useful", 5),
            link(2, 0.90, "not_useful", 4),
        ]

        self.assertEqual(select_links(classified), [])

    def test_fewer_than_three_useful_candidates(self) -> None:
        classified = [
            link(1, 0.80, "same_story", 4),
            link(2, 0.70, "related", 3),
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


if __name__ == "__main__":
    unittest.main()
