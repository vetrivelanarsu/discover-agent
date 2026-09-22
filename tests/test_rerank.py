import unittest

from rerank import LinkType, SuggestedLink, select_links


def link(
    article_id: int,
    similarity: float,
    classification: LinkType,
) -> SuggestedLink:
    return SuggestedLink(
        id=article_id,
        title=f"Article {article_id}",
        similarity=similarity,
        classification=classification,
        reason="Test reason.",
    )


class SelectLinksTests(unittest.TestCase):
    def test_normal_mix(self) -> None:
        classified = [
            link(1, 0.70, "same_story"),
            link(2, 0.90, "same_story"),
            link(3, 0.80, "related"),
            link(4, 0.60, "explainer"),
            link(5, 0.99, "not_useful"),
        ]

        self.assertEqual(
            [candidate.id for candidate in select_links(classified)],
            [2, 3, 4],
        )

    def test_no_related_candidates(self) -> None:
        classified = [
            link(1, 0.90, "same_story"),
            link(2, 0.85, "same_story"),
            link(3, 0.70, "explainer"),
            link(4, 0.99, "not_useful"),
        ]

        self.assertEqual(
            [candidate.id for candidate in select_links(classified)],
            [1, 2, 3],
        )

    def test_all_not_useful(self) -> None:
        classified = [
            link(1, 0.95, "not_useful"),
            link(2, 0.90, "not_useful"),
        ]

        self.assertEqual(select_links(classified), [])

    def test_fewer_than_three_useful_candidates(self) -> None:
        classified = [
            link(1, 0.80, "related"),
            link(2, 0.99, "not_useful"),
        ]

        self.assertEqual(
            [candidate.id for candidate in select_links(classified)],
            [1],
        )


if __name__ == "__main__":
    unittest.main()
