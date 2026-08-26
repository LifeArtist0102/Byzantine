"""Streamlit entry point for the local Byzantine history research Agent MVP."""

from __future__ import annotations

import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

from byzantine.citations import format_chicago_note, format_gbt7714
from byzantine.generation.deepseek import generate_grounded_answer, load_deepseek_api_key
from byzantine.models.document import BibliographicMetadata
from byzantine.models.evidence import Evidence
from byzantine.paths import ensure_app_data_dir
from byzantine.research.services import (
    audit_draft,
    classify_difference,
    counter_queries,
    parallel_reading,
)
from byzantine.retrieval.hybrid import hybrid_search, keyword_evidence
from byzantine.storage.database import LibraryDatabase
from byzantine.workflows.process_document import process_document


def _database() -> LibraryDatabase:
    root = ensure_app_data_dir()
    database = LibraryDatabase(root / "library.db")
    database.initialize()
    return database


def _evidence_result(question: str, evidence: list[Evidence]) -> dict[str, Any]:
    return {"query": question, "hits": [{"section_path": item.section_path, "page_start": item.pdf_page_start, "page_end": item.pdf_page_end, "text": item.text, "title": item.title, "author": item.author, "edition": item.edition, "collection_type": item.collection_type, "source_regions": [region.model_dump() for region in item.source_regions]} for item in evidence]}


def _search(database: LibraryDatabase, query: str, *, collection_ids: list[str] | tuple[str, ...] = (), top_k: int = 5) -> list[Evidence]:
    """Use dense retrieval when locally configured, with FTS-only graceful fallback."""
    try:
        from byzantine.indexing.library_index import search_evidence
        return hybrid_search(query, database=database, vector_search=lambda text, **kwargs: search_evidence(text, qdrant_path=str(ensure_app_data_dir() / "qdrant"), **kwargs), collection_ids=collection_ids, top_k=top_k)
    except (RuntimeError, ValueError):
        return hybrid_search(query, database=database, collection_ids=collection_ids, top_k=top_k)


def _evidence_cards(st: Any, evidence: list[Evidence]) -> None:
    for index, item in enumerate(evidence, start=1):
        pages = f"PDF p. {item.pdf_page_start}" if item.pdf_page_start == item.pdf_page_end else f"PDF pp. {item.pdf_page_start}-{item.pdf_page_end}"
        with st.expander(f"[S{index}] {item.title} | {pages}"):
            st.caption(f"{item.collection_type} · {item.author or '作者未填写'} · {item.edition or '版本未填写'} · {item.source_type}")
            st.write(item.text)
            st.code(format_gbt7714(item), language=None)
            st.code(format_chicago_note(item), language=None)
            st.caption(f"原文件：{item.source_file}；原文区域：{', '.join(region.region_id for region in item.source_regions)}")


def _ask_page(st: Any, database: LibraryDatabase) -> None:
    st.header("Agent 问答")
    collections = [item["collection_id"] for item in database.collections()]
    selected = st.multiselect("检索资料库", collections, default=collections)
    question = st.text_area("历史问题", placeholder="例如：巴西尔二世如何巩固皇权？")
    seek_counter = st.checkbox("主动寻找反面或限制性证据")
    if st.button("检索并回答", type="primary"):
        evidence = _search(database, question, collection_ids=selected, top_k=6)
        if not evidence:
            st.warning("当前所选资料证据不足，无法回答该问题。")
            return
        if not load_deepseek_api_key():
            st.warning("未配置 DeepSeek API Key；以下展示可核查证据。")
            _evidence_cards(st, evidence)
            return
        try:
            generated = generate_grounded_answer(question, _evidence_result(question, evidence), api_key=load_deepseek_api_key(), base_url="https://api.deepseek.com", model="deepseek-chat", max_output_tokens=1200, temperature=0.1, max_evidence_characters=2200)
            st.subheader("回答")
            st.write(generated["answer"])
        except Exception as exc:  # noqa: BLE001 - UI boundary must show API errors to the researcher.
            st.error(f"生成回答失败：{exc}")
        _evidence_cards(st, evidence)
        if seek_counter:
            st.subheader("反面／限制性检索")
            for query in counter_queries(question)[1:]:
                st.write(query)
                _evidence_cards(st, keyword_evidence(database, query, collection_ids=selected, limit=3))


def _library_page(st: Any, database: LibraryDatabase) -> None:
    st.header("资料库")
    documents = database.list_documents()
    st.dataframe([{"文献": item.title, "作者": item.author, "资料库": item.collection_id, "类型": item.source_type, "页数": item.page_count, "状态": item.status, "错误": item.error_message} for item in documents], use_container_width=True)
    st.caption("删除文献会同时清除该文献的 SQLite/FTS 记录；Qdrant 向量删除将在向量索引可用时同步执行。")


def _upload_page(st: Any, database: LibraryDatabase) -> None:
    st.header("上传资料")
    uploaded = st.file_uploader("选择 PDF、TXT、Markdown、JPG、JPEG 或 PNG", type=["pdf", "txt", "md", "markdown", "jpg", "jpeg", "png"])
    collection = st.selectbox("资料库", [item["collection_id"] for item in database.collections()], index=1)
    title = st.text_input("书名／文献名称")
    author = st.text_input("作者")
    translator = st.text_input("译者")
    edition = st.text_input("版本")
    publisher = st.text_input("出版社")
    year = st.number_input("出版年份（未知请填 0）", min_value=0, max_value=3000, value=0)
    language = st.selectbox("语言", ["English", "Chinese", "Greek", "Latin", "other"])
    source_type = st.selectbox("资料类型", ["primary_source", "translation", "secondary_study", "reference_work"])
    if st.button("确认书目信息并处理", type="primary"):
        if not uploaded or not title.strip():
            st.error("请先选择文件并填写文献名称。")
            return
        staging = ensure_app_data_dir() / ".uploads"
        staging.mkdir(exist_ok=True)
        source = staging / uploaded.name
        source.write_bytes(uploaded.getvalue())
        metadata = BibliographicMetadata(title=title.strip(), author=author.strip() or None, translator=translator.strip() or None, edition=edition.strip() or None, publisher=publisher.strip() or None, publication_year=int(year) or None, language=language, source_type=source_type)
        try:
            with st.spinner("正在提取文本、保留原文坐标并建立全文索引……"):
                document = process_document(source, collection_id=collection, metadata=metadata, database=database, seed_path=Path("config/entity_seed.yaml"))
            st.success(f"处理完成：{document.title}（{document.status}）")
        except Exception as exc:  # noqa: BLE001 - preserve user-visible import failure status.
            st.error(f"处理失败：{exc}")


def _topics_page(st: Any, database: LibraryDatabase) -> None:
    st.header("研究专题")
    with st.form("new-topic"):
        title = st.text_input("专题名称")
        question = st.text_input("研究问题")
        if st.form_submit_button("创建专题") and title:
            database.create_topic(title, question)
            st.success("专题已创建。")
    st.dataframe(database.list_topics(), use_container_width=True)


def _reader_page(st: Any, database: LibraryDatabase) -> None:
    st.header("证据阅读器")
    query = st.text_input("用关键词定位证据", placeholder="人名、地名或短语")
    evidence = keyword_evidence(database, query, limit=20) if query.strip() else []
    if not evidence:
        st.info("输入关键词后选择证据；高亮始终指向支持结论的原文区域。")
        return
    item = st.selectbox("证据", evidence, format_func=lambda value: f"{value.title} | {value.evidence_id}")
    st.caption(f"{item.title} · {item.author or '作者未填写'} · {item.edition or '版本未填写'}")
    source = Path(item.source_file)
    try:
        if source.suffix.lower() == ".pdf" and item.source_regions:
            import fitz

            region = next((region for region in item.source_regions if region.page), None)
            if region and region.page:
                with fitz.open(source) as pdf:
                    page = pdf[region.page - 1]
                    for candidate in item.source_regions:
                        if candidate.page == region.page and candidate.bbox:
                            page.draw_rect(fitz.Rect(candidate.bbox), color=(1, 0.85, 0), fill=(1, 0.9, 0), fill_opacity=0.35, overlay=True)
                    st.image(page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).tobytes("png"), caption=f"PDF 第 {region.page} 页；黄色区域为本条 Evidence 的原文坐标")
        elif source.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            st.image(str(source), caption="原始图片；OCR 区域信息见下方")
        else:
            st.text_area("原文段落", item.text, height=280, disabled=True)
    except Exception as exc:  # noqa: BLE001 - the evidence text remains available on render failure.
        st.warning(f"无法渲染原文件，但证据文本仍可用：{exc}")
    st.subheader("证据文字")
    st.write(item.text)
    st.write("GB/T 7714：", format_gbt7714(item))
    st.write("Chicago Notes：", format_chicago_note(item))


def _comparison_page(st: Any, database: LibraryDatabase) -> None:
    st.header("史料平行对读")
    question = st.text_input("比较问题", key="comparison-question")
    dimensions = st.multiselect("比较维度", ["事件描述", "关键措辞", "原因解释", "责任归属", "人物评价", "作者立场", "时间记载"], default=["事件描述", "原因解释"])
    if st.button("开始对读"):
        evidence = _search(database, question, top_k=12)
        if len({item.document_id for item in evidence}) < 2:
            st.warning("请先导入至少两本文献，并使用能命中两者的比较问题。")
            return
        comparison = parallel_reading(question, evidence, dimensions)
        database.save_comparison(comparison)
        st.success("对读结果已保存；每个单元格均保留它自己的证据编号。")
        st.dataframe(comparison["comparison_cells"], use_container_width=True)
        _evidence_cards(st, evidence)


def _claims_page(st: Any, database: LibraryDatabase) -> None:
    st.header("主张—证据账本")
    claim = st.text_area("历史主张")
    relation = st.selectbox("将初始检索结果标为", ["context", "support", "oppose", "qualify"], format_func=lambda value: {"context": "背景", "support": "支持", "oppose": "反对", "qualify": "限制／修正"}[value])
    if st.button("建立主张") and claim.strip():
        claim_id = database.create_claim(claim.strip())
        evidence = _search(database, claim, top_k=5)
        for item in evidence:
            database.link_claim_evidence(claim_id, item, relation)
        st.success("主张与证据关系已保存；研究者可在后续账本中继续补充或修正。")
        _evidence_cards(st, evidence)


def _audit_page(st: Any, database: LibraryDatabase) -> None:
    st.header("论文证据审计")
    draft = st.text_area("粘贴论文草稿（原文不会被自动改写）", height=240)
    if st.button("审计草稿") and draft.strip():
        results = audit_draft(draft, lambda item: _search(database, item, top_k=3))
        database.save_audit(title="论文证据审计", original_text=draft, sentence_results=results)
        st.success("审计记录已保存，原稿没有被改写。")
        st.dataframe(results, use_container_width=True)


def _contradiction_page(st: Any, database: LibraryDatabase) -> None:
    st.header("矛盾与反证")
    claim = st.text_input("要检验的问题或主张", key="counter-claim")
    if st.button("主动寻找反证") and claim:
        evidence = []
        for query in counter_queries(claim):
            evidence.extend(_search(database, query, top_k=3))
        unique = list({item.chunk_id: item for item in evidence}.values())
        for left, right in pairwise(unique):
            classification = classify_difference(left, right)
            database.save_contradiction(subject=claim, description="主动反证检索发现的待核查差异。", classification=classification, evidence_side_a=left, evidence_side_b=right)
            st.write({"分类": classification, "A": left.evidence_id, "B": right.evidence_id})
        _evidence_cards(st, unique)


def _profile_page(st: Any, database: LibraryDatabase) -> None:
    st.header("史料批判卡")
    documents = database.list_documents()
    if not documents:
        st.info("请先导入文献。")
        return
    document = st.selectbox("选择文献", documents, format_func=lambda item: item.title)
    existing = database.source_profile(document.document_id)
    profile = (existing or {}).get("profile", {})
    fields = {"author_identity": "作者身份", "composition_date": "写作时间", "events_described_date": "所述事件时间", "genre": "文献体裁", "intended_audience": "预期读者", "political_affiliation": "政治关联／立场", "relationship_to_subject": "与叙述对象关系", "edition_notes": "版本说明", "translation_notes": "译本说明", "known_limitations": "使用局限"}
    profile_data = {key: st.text_input(label, value=str(profile.get(key, ""))) for key, label in fields.items()}
    status = st.selectbox("确认状态", ["unreviewed", "agent_suggested", "user_confirmed"], index=["unreviewed", "agent_suggested", "user_confirmed"].index((existing or {}).get("review_status", "unreviewed")))
    if st.button("保存批判卡"):
        database.save_source_profile(document.document_id, {**profile_data, "provenance": "user_entered"}, status)
        st.success("史料批判卡已保存。未确认信息不会被呈现为既定事实。")


def _settings_page(st: Any) -> None:
    st.header("设置")
    st.write(f"用户数据目录：`{ensure_app_data_dir()}`")
    st.write(f"DeepSeek API Key：{'已配置' if bool(load_deepseek_api_key()) else '未配置'}（不会显示密钥）")
    st.info("DeepSeek 是在线模型：只有本次检索出的相关片段会被发送给它。SQLite FTS5 的默认中文分词能力有限，中文问题建议同时使用明确的人名、地名或英文术语。")


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="Byzantine Research Agent", layout="wide")
    database = _database()
    pages = {"Agent 问答": _ask_page, "资料库": _library_page, "上传资料": _upload_page, "证据阅读器": _reader_page, "研究专题": _topics_page, "史料平行对读": _comparison_page, "主张—证据账本": _claims_page, "论文证据审计": _audit_page, "矛盾与反证": _contradiction_page, "史料批判卡": _profile_page, "设置": lambda st_, db: _settings_page(st_)}
    choice = st.sidebar.radio("页面", list(pages))
    pages[choice](st, database)


def main() -> None:
    """Launch Streamlit via the installed console command."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())], check=False)


if __name__ == "__main__":
    render()
