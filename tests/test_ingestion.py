from byzantine.ingestion.pipeline import _safe_book_id, _word_count


def test_safe_book_id() -> None:
    assert _safe_book_id("The Oxford Handbook!") == "the_oxford_handbook"


def test_word_count_handles_punctuation() -> None:
    assert _word_count("Basil II's reign: 976-1025.") == 4
