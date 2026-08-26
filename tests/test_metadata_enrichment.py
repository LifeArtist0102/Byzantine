from byzantine.metadata.enrichment import enrich_chunk, extract_date_range

SEED = {
    "people": [{"canonical": "Basil II", "aliases": ["Basil II"]}],
    "places": [{"canonical": "Constantinople", "aliases": ["Constantinople"]}],
    "topics": {"warfare": ["war", "army"], "government": ["emperor"]},
}


def test_extracts_short_year_range() -> None:
    assert extract_date_range("Basil ruled in 976-1025.")[:2] == (976, 1025)


def test_excludes_modern_bibliographic_years_from_historical_range() -> None:
    assert extract_date_range("See Smith 1997 and the year 1025.")[:2] == (1025, 1025)


def test_enrichment_keeps_only_trusted_aliases_as_people() -> None:
    enriched = enrich_chunk({"text": "Basil II was emperor in Constantinople in 976-1025."}, SEED)
    assert enriched["metadata"]["people"] == ["Basil II"]
    assert enriched["metadata"]["places"] == ["Constantinople"]
    assert enriched["metadata"]["topics"] == ["government"]
