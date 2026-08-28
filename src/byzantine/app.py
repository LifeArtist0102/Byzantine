"""Polished Streamlit workspace for Byzantine source-based research."""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from itertools import pairwise
from pathlib import Path
from typing import Any

from byzantine.citations import format_chicago_note, format_gbt7714
from byzantine.generation.deepseek import (
    generate_grounded_answer,
    load_deepseek_api_key,
    summarize_research_chat,
)
from byzantine.models.document import BibliographicMetadata
from byzantine.models.evidence import Evidence
from byzantine.paths import ensure_app_data_dir
from byzantine.research.services import classify_difference, counter_queries, parallel_reading
from byzantine.retrieval.hybrid import hybrid_search
from byzantine.storage.database import LibraryDatabase
from byzantine.workflows.delete_document import delete_document_from_library
from byzantine.workflows.process_document import process_document, reprocess_document

COLLECTION_LABELS = {"starter": "基础资料库", "personal": "个人资料库"}


def _database() -> LibraryDatabase:
    database = LibraryDatabase(ensure_app_data_dir() / "library.db")
    database.initialize()
    return database


def _inject_style(st: Any) -> None:
    st.markdown(
        """<style>
        .stApp { background: radial-gradient(circle at 15% -5%, #283b55 0, #111822 34%, #0b1017 78%); color: #eaf0f7; }
        [data-testid="stSidebar"] { background: #0c131d; border-right: 1px solid #263445; }
        .hero { padding: 1.5rem 1.7rem; border: 1px solid #2e455f; border-radius: 20px; background: linear-gradient(135deg, rgba(39,61,84,.9), rgba(17,26,37,.88)); margin-bottom: 1.25rem; }
        .hero h1 { margin: 0; font-size: 1.7rem; letter-spacing: -.03em; color: #f5f8fc; }
        .hero p { margin: .55rem 0 0; color: #b7c5d6; }
        .eyebrow { color:#8ec5ff; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; }
        .card { border:1px solid #2c3b4c; border-radius:16px; padding:1rem 1.1rem; background:rgba(18,27,38,.86); margin:.55rem 0; }
        .tag { display:inline-block; padding:.18rem .52rem; border-radius:999px; margin:.15rem .28rem .15rem 0; background:#203a55; color:#a8d8ff; font-size:.78rem; }
        .source-line { color:#9db0c5; font-size:.84rem; }
        [data-testid="stChatMessage"] { border: 1px solid #29394a; border-radius: 14px; padding: .55rem .9rem; margin-bottom: .65rem; background: rgba(15,23,33,.75); }
        [data-testid="stProgress"] > div > div > div { background: linear-gradient(90deg, #4da3ff, #79d7c5); }
        </style>""",
        unsafe_allow_html=True,
    )


def _hero(st: Any, eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def _scope_picker(
    st: Any,
    database: LibraryDatabase,
    *,
    key: str,
    default_collection_ids: list[str] | None = None,
    default_document_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Two-level source scope: library category first, individual books second."""
    collections = database.collections()
    by_type = {item["collection_type"]: item["collection_id"] for item in collections}
    default_types = [
        collection_type
        for collection_type, collection_id in by_type.items()
        if default_collection_ids is None or collection_id in default_collection_ids
    ]
    chosen_types = st.multiselect(
        "第一层：资料库类型",
        options=["personal", "starter"],
        default=default_types or ["personal", "starter"],
        format_func=lambda value: COLLECTION_LABELS[value],
        key=f"{key}-collection-types",
        help="先选择个人资料库、基础资料库，下一层只展示其中的文献。",
    )
    collection_ids = [by_type[item] for item in chosen_types if item in by_type]
    documents = database.list_documents(collection_ids=collection_ids)
    document_by_id = {document.document_id: document for document in documents}
    defaults = [
        item for item in (default_document_ids or list(document_by_id)) if item in document_by_id
    ]
    document_ids = st.multiselect(
        "第二层：具体文献（默认全选）",
        options=list(document_by_id),
        default=defaults,
        format_func=lambda document_id: (
            f"{document_by_id[document_id].title} · {COLLECTION_LABELS.get(document_by_id[document_id].collection_id, document_by_id[document_id].collection_id)}"
        ),
        key=f"{key}-documents",
        help="本次检索只会使用这里选中的文献；取消选择即可排除某本书。",
    )
    return collection_ids, document_ids


def _search(
    database: LibraryDatabase,
    query: str,
    *,
    collection_ids: list[str] | tuple[str, ...] = (),
    document_ids: list[str] | tuple[str, ...] = (),
    top_k: int = 6,
) -> list[Evidence]:
    try:
        from byzantine.indexing.library_index import search_evidence

        return hybrid_search(
            query,
            database=database,
            vector_search=lambda text, **kwargs: search_evidence(
                text,
                qdrant_path=str(ensure_app_data_dir() / "qdrant"),
                **kwargs,
            ),
            collection_ids=collection_ids,
            document_ids=document_ids,
            top_k=top_k,
        )
    except (RuntimeError, ValueError):
        return hybrid_search(
            query,
            database=database,
            collection_ids=collection_ids,
            document_ids=document_ids,
            top_k=top_k,
        )


def _retrieval_result(question: str, evidence: list[Evidence]) -> dict[str, Any]:
    return {
        "query": question,
        "hits": [
            {
                "section_path": item.section_path,
                "page_start": item.pdf_page_start,
                "page_end": item.pdf_page_end,
                "text": item.text,
                "title": item.title,
                "author": item.author,
                "edition": item.edition,
                "collection_type": item.collection_type,
                "source_regions": [region.model_dump() for region in item.source_regions],
            }
            for item in evidence
        ],
    }


def _evidence_from_snapshot(snapshot: list[dict[str, Any]]) -> list[Evidence]:
    evidence: list[Evidence] = []
    for item in snapshot:
        try:
            evidence.append(Evidence.model_validate(item))
        except ValueError:
            continue
    return evidence


def _evidence_cards(st: Any, evidence: list[Evidence]) -> None:
    for index, item in enumerate(evidence, start=1):
        pages = (
            f"PDF p. {item.pdf_page_start}"
            if item.pdf_page_start == item.pdf_page_end
            else f"PDF pp. {item.pdf_page_start}-{item.pdf_page_end}"
        )
        with st.expander(f"[S{index}] {item.title} · {pages}"):
            st.markdown(
                f'<div class="source-line">{COLLECTION_LABELS.get(item.collection_type, item.collection_type)} · '
                f"{item.author or '作者未填写'} · {item.edition or '版本未填写'}</div>",
                unsafe_allow_html=True,
            )
            st.write(item.text)
            left, right = st.columns(2)
            left.caption(format_gbt7714(item))
            right.caption(format_chicago_note(item))


def _chat_context(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{item['role']}: {item['content'][:900]}" for item in messages[-8:])


def _agent_page(st: Any, database: LibraryDatabase) -> None:
    _hero(
        st,
        "Research dialogue",
        "Agent 问答",
        "在选定资料库与具体文献范围内，进行可追溯、带上下文的历史讨论。",
    )
    conversations = database.list_conversations()
    if "active_conversation" not in st.session_state:
        st.session_state.active_conversation = None
    left, right = st.columns([3.2, 1], gap="large")
    with right:
        st.markdown("#### 对话工作台")
        if st.button("＋ 新建对话", use_container_width=True):
            st.session_state.active_conversation = None
            st.rerun()
        if conversations:
            current_ids = [item["conversation_id"] for item in conversations]
            active = st.session_state.active_conversation
            index = current_ids.index(active) if active in current_ids else 0
            selected = st.selectbox(
                "历史对话",
                current_ids,
                index=index,
                format_func=lambda value: next(
                    item["title"] for item in conversations if item["conversation_id"] == value
                ),
            )
            if selected != active:
                st.session_state.active_conversation = selected
                st.rerun()
        topics = database.list_topics()
        topic_options = [""] + [item["topic_id"] for item in topics]
        active_data = (
            database.get_conversation(st.session_state.active_conversation)
            if st.session_state.active_conversation
            else None
        )
        topic_index = (
            topic_options.index(active_data["topic_id"])
            if active_data and active_data["topic_id"] in topic_options
            else 0
        )
        selected_topic = st.selectbox(
            "归属研究专题",
            topic_options,
            index=topic_index,
            format_func=lambda value: (
                "暂不归档"
                if not value
                else next(item["title"] for item in topics if item["topic_id"] == value)
            ),
            help="可先聊天，确认研究价值后再归档到专题。",
        )
        if active_data and selected_topic != (active_data["topic_id"] or ""):
            database.update_conversation(
                active_data["conversation_id"], topic_id=selected_topic or None
            )
            active_data = database.get_conversation(active_data["conversation_id"])
        st.caption("每次回答只使用当前两层筛选中的文献；历史聊天内容仅用于理解你的追问意图。")

    with left:
        default_collections = active_data["collection_ids"] if active_data else None
        default_documents = active_data["document_ids"] if active_data else None
        collection_ids, document_ids = _scope_picker(
            st,
            database,
            key=f"chat-{active_data['conversation_id'] if active_data else 'new'}",
            default_collection_ids=default_collections,
            default_document_ids=default_documents,
        )
        st.caption(f"本次范围：{len(collection_ids)} 个资料库 · {len(document_ids)} 本文献")
        messages = (
            database.conversation_messages(active_data["conversation_id"]) if active_data else []
        )
        if not messages:
            st.info("开始一个研究问题。系统会保留对话上下文，但每轮史实仍只根据本轮选定文献检索。")
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["labels"]:
                    st.caption(" · ".join(message["labels"]))
                evidence = _evidence_from_snapshot(message["evidence_snapshot"])
                if evidence:
                    _evidence_cards(st, evidence)
        prompt = st.chat_input("围绕所选文献继续提问…")
        if prompt:
            if not document_ids:
                st.warning("请至少选择一本具体文献后再提问。")
                return
            if not active_data:
                conversation_id = database.create_conversation(
                    title=prompt[:36] + ("…" if len(prompt) > 36 else ""),
                    collection_ids=collection_ids,
                    document_ids=document_ids,
                    topic_id=selected_topic or None,
                )
                st.session_state.active_conversation = conversation_id
                active_data = database.get_conversation(conversation_id)
                messages = []
            else:
                conversation_id = active_data["conversation_id"]
                database.update_conversation(
                    conversation_id,
                    topic_id=selected_topic or None,
                    collection_ids=collection_ids,
                    document_ids=document_ids,
                )
            database.add_chat_message(conversation_id, role="user", content=prompt)
            with st.chat_message("assistant"):
                with st.status("正在检索选定文献并核对出处…", expanded=True) as status:
                    evidence = _search(
                        database,
                        prompt,
                        collection_ids=collection_ids,
                        document_ids=document_ids,
                        top_k=6,
                    )
                    if not evidence:
                        answer = "当前所选资料证据不足，无法回答该问题。"
                        status.update(label="未找到充分证据", state="complete")
                    elif not load_deepseek_api_key():
                        answer = (
                            "已检索到相关证据，但尚未配置 DeepSeek API Key，因此不能生成综合回答。"
                        )
                        status.update(label="已完成本地证据检索", state="complete")
                    else:
                        status.write("正在基于本轮证据生成回答…")
                        result = generate_grounded_answer(
                            prompt,
                            _retrieval_result(prompt, evidence),
                            api_key=load_deepseek_api_key(),
                            base_url="https://api.deepseek.com",
                            model="deepseek-chat",
                            max_output_tokens=1200,
                            temperature=0.1,
                            max_evidence_characters=2200,
                            conversation_context=_chat_context(messages),
                        )
                        answer = result["answer"]
                        status.update(label="回答与出处已生成", state="complete")
                st.markdown(answer)
                if evidence:
                    _evidence_cards(st, evidence)
            database.add_chat_message(
                conversation_id, role="assistant", content=answer, evidence=evidence
            )
            st.rerun()

        if (
            active_data
            and selected_topic
            and messages
            and st.button("将当前对话归档为专题研究卡", type="primary")
        ):
            evidence = []
            for message in messages:
                evidence.extend(_evidence_from_snapshot(message["evidence_snapshot"]))
            unique = list({item.chunk_id: item for item in evidence}.values())
            try:
                with st.spinner("正在由模型整理专题标题、标签、摘要与出处…"):
                    summary = summarize_research_chat(
                        [{"role": item["role"], "content": item["content"]} for item in messages],
                        _retrieval_result("专题聊天归纳", unique),
                        api_key=load_deepseek_api_key(),
                        base_url="https://api.deepseek.com",
                        model="deepseek-chat",
                        max_evidence_characters=1500,
                    )
                database.save_topic_chat_summary(
                    topic_id=selected_topic,
                    conversation_id=active_data["conversation_id"],
                    title=summary["title"],
                    tags=summary["tags"],
                    summary=summary["summary"],
                    evidence=unique,
                )
                st.success("已归档到研究专题。")
            except Exception as exc:  # noqa: BLE001
                st.error(f"归档失败：{exc}")


def _upload_page(st: Any, database: LibraryDatabase) -> None:
    _hero(
        st,
        "Library intake",
        "批量上传资料",
        "支持批量导入；系统会显示每份文献的阶段、总体进度和动态预计完成时间。",
    )
    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "txt", "md", "markdown", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="可以一次选择多本文献。批量时将以文件名作为默认标题。",
    )
    collection_id = st.selectbox(
        "导入到资料库",
        ["personal", "starter"],
        format_func=lambda value: COLLECTION_LABELS[value],
    )
    left, right = st.columns(2)
    author = left.text_input("作者（可批量共用）")
    language = right.selectbox("语言", ["English", "Chinese", "Greek", "Latin", "other"])
    source_type = left.selectbox(
        "资料类型", ["secondary_study", "primary_source", "translation", "reference_work"]
    )
    edition = right.text_input("版本（可批量共用）")
    publisher = left.text_input("出版社（可批量共用）")
    year = right.number_input("出版年份（未知填 0）", min_value=0, max_value=3000, value=0)
    if st.button("开始批量处理", type="primary", disabled=not uploaded_files):
        overall = st.progress(0, text="准备导入…")
        report = st.status("批量导入进行中", expanded=True)
        started = time.monotonic()
        completed: list[str] = []
        failures: list[str] = []
        total = len(uploaded_files)
        staging = ensure_app_data_dir() / ".uploads"
        staging.mkdir(exist_ok=True)
        for index, uploaded in enumerate(uploaded_files):
            local = staging / f"{uuid.uuid4().hex}_{uploaded.name}"
            local.write_bytes(uploaded.getvalue())
            title = Path(uploaded.name).stem.replace("_", " ")

            def update(
                stage: str, fraction: float, *, index: int = index, title: str = title
            ) -> None:
                whole = (index + max(0.0, min(fraction, 1.0))) / total
                elapsed = time.monotonic() - started
                eta = elapsed / max(index + fraction, 0.1) * (total - index - fraction)
                overall.progress(
                    whole,
                    text=f"{index + 1}/{total} · {title} · {stage} · 预计剩余 {max(0, int(eta))} 秒",
                )

            try:
                report.write(f"正在处理：{uploaded.name}")
                document = process_document(
                    local,
                    collection_id=collection_id,
                    metadata=BibliographicMetadata(
                        title=title,
                        author=author.strip() or None,
                        edition=edition.strip() or None,
                        publisher=publisher.strip() or None,
                        publication_year=int(year) or None,
                        language=language,
                        source_type=source_type,
                    ),
                    database=database,
                    seed_path=Path("config/entity_seed.yaml"),
                    progress=update,
                )
                completed.append(document.title)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{uploaded.name}：{exc}")
            overall.progress((index + 1) / total, text=f"已完成 {index + 1}/{total} 份文献")
        report.update(label="批量导入完成", state="complete", expanded=bool(failures))
        if completed:
            st.success(f"成功处理 {len(completed)} 份：{'、'.join(completed)}")
        if failures:
            st.error("\n".join(failures))


def _library_page(st: Any, database: LibraryDatabase) -> None:
    _hero(
        st,
        "Source control",
        "资料库",
        "管理文献、处理状态与本地副本。删除将清除该文献所有派生记录，但不影响你外部保存的原始文件。",
    )
    documents = database.list_documents()
    metrics = st.columns(3)
    metrics[0].metric("文献总数", len(documents))
    metrics[1].metric("已就绪", sum(item.status == "ready" for item in documents))
    metrics[2].metric("处理中", sum(item.status != "ready" for item in documents))
    st.dataframe(
        [
            {
                "文献": item.title,
                "资料库": COLLECTION_LABELS.get(item.collection_id, item.collection_id),
                "作者": item.author,
                "页数": item.page_count,
                "状态": item.status,
                "错误": item.error_message,
            }
            for item in documents
        ],
        use_container_width=True,
        hide_index=True,
    )
    retryable = [
        item
        for item in documents
        if item.status in {"failed", "indexing", "extracting", "enriching"}
    ]
    if retryable:
        st.subheader("恢复未完成处理")
        target = st.selectbox(
            "选择文献", retryable, format_func=lambda item: f"{item.title}（{item.status}）"
        )
        if st.button("重新处理并显示进度"):
            bar = st.progress(0, text="准备重新处理…")
            try:
                with st.status("正在恢复文献", expanded=True) as status:
                    document = reprocess_document(
                        target.document_id,
                        database=database,
                        seed_path=Path("config/entity_seed.yaml"),
                        progress=lambda stage, fraction: bar.progress(fraction, text=stage),
                    )
                    status.update(label=f"已完成：{document.title}", state="complete")
                st.success("重新处理完成。")
            except Exception as exc:  # noqa: BLE001
                st.error(f"重新处理失败：{exc}")
    st.divider()
    st.subheader("删除文献")
    st.warning("不可恢复：会删除向量、全文、专题关联证据、聊天归档以及应用保存的文件副本。")
    if documents:
        target = st.selectbox(
            "选择要删除的文献",
            documents,
            format_func=lambda item: item.title,
            key="delete-document",
        )
        confirmed = st.checkbox(
            f"我确认永久删除《{target.title}》的本地记录", key="delete-confirmation"
        )
        if st.button("永久删除所选文献", disabled=not confirmed):
            try:
                delete_document_from_library(target.document_id, database=database)
            except Exception as exc:  # noqa: BLE001
                st.error(f"删除失败，未完成删除：{exc}")
            else:
                st.success("文献与其派生记录已删除。")
                st.rerun()


def _topics_page(st: Any, database: LibraryDatabase) -> None:
    _hero(
        st,
        "Research map",
        "研究专题",
        "把多轮问答整理成带标签、摘要和出处的研究卡，而不是散落的聊天记录。",
    )
    create, browse = st.tabs(["创建专题", "专题档案"])
    with create, st.form("new-topic", clear_on_submit=True):
        title = st.text_input("专题名称", placeholder="例如：十一世纪军事区的演变")
        question = st.text_input("核心研究问题")
        description = st.text_area("研究说明（可选）")
        if st.form_submit_button("创建研究专题", type="primary"):
            if not title.strip():
                st.error("请填写专题名称。")
            else:
                database.create_topic(title.strip(), question.strip(), description.strip())
                st.success("专题已创建；可回到 Agent 问答，将聊天归档到此专题。")
    with browse:
        topics = database.list_topics()
        if not topics:
            st.info("还没有专题。请先创建一个研究问题。")
            return
        topic_id = st.selectbox(
            "打开专题",
            [item["topic_id"] for item in topics],
            format_func=lambda value: next(
                item["title"] for item in topics if item["topic_id"] == value
            ),
        )
        topic = database.get_topic(topic_id)
        summaries = database.topic_chat_summaries(topic_id)
        st.markdown(f"### {topic['title']}")
        st.caption(topic["research_question"] or "尚未设置核心问题")
        if topic["description"]:
            st.write(topic["description"])
        st.metric("已归档聊天", len(summaries))
        if not summaries:
            st.info("从 Agent 问答选择该专题后，点击“将当前对话归档为专题研究卡”。")
            return
        for summary in summaries:
            tags = "".join(f'<span class="tag">{tag}</span>' for tag in summary["tags"])
            st.markdown(
                f'<div class="card"><strong>{summary["title"]}</strong><br>{tags}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("查看模型整理的研究卡与原对话"):
                st.markdown(summary["summary"])
                evidence = _evidence_from_snapshot(summary["evidence_snapshot"])
                if evidence:
                    _evidence_cards(st, evidence)
                st.caption("原始聊天记录")
                for message in database.conversation_messages(summary["conversation_id"]):
                    with st.chat_message(message["role"]):
                        st.write(message["content"])


def _comparison_page(st: Any, database: LibraryDatabase) -> None:
    _hero(
        st,
        "Comparative reading",
        "史料平行对读",
        "在明确选择的文献之间保留差异，而不是生成混合式总结；每次对照都会保存。",
    )
    compose, history = st.tabs(["新建对读", "历史记录"])
    with compose:
        collection_ids, document_ids = _scope_picker(st, database, key="comparison")
        question = st.text_input("比较问题", placeholder="不同文献如何解释第四次十字军东征的转向？")
        dimensions = st.multiselect(
            "比较维度",
            ["事件描述", "关键措辞", "原因解释", "责任归属", "人物评价", "作者立场", "时间记载"],
            default=["事件描述", "原因解释", "共同点", "差异点"],
        )
        if st.button("运行平行对读", type="primary"):
            if len(document_ids) < 2:
                st.warning("请在第二层至少选择两本文献。")
                return
            with st.status("正在按所选文献检索并分列证据…", expanded=True) as status:
                evidence = _search(
                    database,
                    question,
                    collection_ids=collection_ids,
                    document_ids=document_ids,
                    top_k=16,
                )
                if len({item.document_id for item in evidence}) < 2:
                    status.update(label="证据覆盖不足", state="error")
                    st.warning("当前问题没有同时命中至少两本文献；请调整问题或选择范围。")
                    return
                comparison = parallel_reading(question, evidence, dimensions)
                database.save_comparison(comparison)
                status.update(label="对读已保存", state="complete")
            st.dataframe(comparison["comparison_cells"], use_container_width=True, hide_index=True)
            _evidence_cards(st, evidence)
    with history:
        comparisons = database.list_comparisons()
        if not comparisons:
            st.info("暂无已保存的对读记录。")
        else:
            record = st.selectbox(
                "已保存对读", comparisons, format_func=lambda item: item["question"]
            )
            st.caption(f"比较维度：{'、'.join(record['dimensions'])}")
            st.dataframe(record["comparison_cells"], use_container_width=True, hide_index=True)


def _contradiction_page(st: Any, database: LibraryDatabase) -> None:
    _hero(
        st,
        "Counter-evidence",
        "矛盾与反证",
        "不只寻找支持材料：在选定文献中主动检索反面、限制性或不同解释的证据。",
    )
    collection_ids, document_ids = _scope_picker(st, database, key="counter")
    claim = st.text_input("需要检验的问题或主张")
    if st.button("寻找反证与差异", type="primary"):
        if not document_ids:
            st.warning("请先选择具体文献。")
            return
        with st.status("正在执行支持、反面与不同解释三类检索…", expanded=True) as status:
            evidence: list[Evidence] = []
            for query in counter_queries(claim):
                evidence.extend(
                    _search(
                        database,
                        query,
                        collection_ids=collection_ids,
                        document_ids=document_ids,
                        top_k=4,
                    )
                )
            unique = list({item.chunk_id: item for item in evidence}.values())
            for left, right in pairwise(unique):
                database.save_contradiction(
                    subject=claim,
                    description="主动反证检索发现的待核查差异。",
                    classification=classify_difference(left, right),
                    evidence_side_a=left,
                    evidence_side_b=right,
                )
            status.update(label="已保存待核查差异", state="complete")
        _evidence_cards(st, unique)


def _settings_page(st: Any) -> None:
    _hero(
        st,
        "Local-first",
        "设置",
        "所有文献、索引、聊天和专题均保存在本机；只有选择生成或归档时才会调用 DeepSeek。",
    )
    st.code(str(ensure_app_data_dir()), language=None)
    st.write(
        f"DeepSeek API Key：{'已配置' if load_deepseek_api_key() else '未配置'}（不会显示密钥）"
    )
    st.info(
        "提示：SQLite FTS5 的中文分词能力有限。清晰的人名、地名、年代与英文术语可改善关键词检索；BGE-M3 会补充语义检索。"
    )


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="Byzantine Research Studio", page_icon="🏛️", layout="wide")
    _inject_style(st)
    database = _database()
    st.sidebar.markdown("## 🏛️ Byzantine")
    st.sidebar.caption("History Research Studio")
    pages = {
        "Agent 问答": _agent_page,
        "上传资料": _upload_page,
        "资料库": _library_page,
        "研究专题": _topics_page,
        "史料平行对读": _comparison_page,
        "矛盾与反证": _contradiction_page,
        "设置": lambda st_, db: _settings_page(st_),
    }
    choice = st.sidebar.radio("研究工作流", list(pages))
    st.sidebar.divider()
    st.sidebar.caption("所有结论必须回到被选中的文献证据。")
    pages[choice](st, database)


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())], check=False
    )


if __name__ == "__main__":
    render()
