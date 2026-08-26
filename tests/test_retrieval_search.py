from byzantine.retrieval.search import build_metadata_filter, render_evidence


def test_build_metadata_filter_uses_overlap_for_dates() -> None:
    query_filter = build_metadata_filter(
        people=["Basil II"], places=["Constantinople"], date_start=1000, date_end=1010
    )

    assert query_filter is not None
    assert len(query_filter.must) == 4
    assert query_filter.must[0].key == "metadata.people"
    assert query_filter.must[2].range.gte == 1000
    assert query_filter.must[3].range.lte == 1010


def test_render_evidence_keeps_citations_and_truncates_text() -> None:
    rendered = render_evidence(
        {
            "query": "Who was Basil II?",
            "hits": [
                {
                    "score": 0.75,
                    "section_path": ["PART II", "EMPIRE"],
                    "page_start": 100,
                    "page_end": 101,
                    "metadata": {
                        "people": ["Basil II"],
                        "places": ["Constantinople"],
                        "topics": ["warfare"],
                        "date_start": 976,
                        "date_end": 1025,
                    },
                    "text": "x" * 30,
                }
            ],
        },
        text_limit=10,
    )

    assert "pp. 100-101" in rendered
    assert "Basil II" in rendered
    assert "xxxxxxxxxx…" in rendered
