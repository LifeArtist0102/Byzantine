from byzantine.chunking.semantic import (
    PageLocator,
    build_semantic_chunks,
    parse_markdown_paragraphs,
)


def test_parser_excludes_references_and_retains_section_path() -> None:
    markdown = """## PART I\n## A chapter\n\nUseful historical prose.\n\n## References\n\nExcluded entry."""
    paragraphs = parse_markdown_paragraphs(markdown)
    assert len(paragraphs) == 1
    assert paragraphs[0].section_path == ["PART I", "A chapter"]


def test_parser_retains_title_when_an_ocr_split_author_heading_follows() -> None:
    markdown = """## PART I\n## Byzantine History\n## JOH N SMIT H\n\nUseful historical prose."""
    paragraphs = parse_markdown_paragraphs(markdown)
    assert paragraphs[0].section_path == ["PART I", "Byzantine History"]


def test_parser_excludes_everything_after_an_ocr_split_index_heading() -> None:
    markdown = (
        """## PART I\n## A chapter\n\nUseful historical prose.\n\n## INDE X\n\nAbbasid 265."""
    )
    paragraphs = parse_markdown_paragraphs(markdown)
    assert len(paragraphs) == 1


def test_chunks_have_page_provenance_and_neighbours() -> None:
    paragraphs = parse_markdown_paragraphs(
        "## PART I\n## A chapter\n\n" + "Byzantium had institutions. " * 60
    )
    locator = PageLocator([{"pdf_page": 101, "text": "Byzantium had institutions. " * 60}])
    chunks = build_semantic_chunks(
        paragraphs,
        book_id="test",
        page_locator=locator,
        target_characters=500,
        min_characters=100,
        max_characters=800,
        overlap_characters=50,
    )
    assert chunks[0].page_start == 101
    assert chunks[0].next_chunk_id is not None
