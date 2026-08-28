"""Historia: a local, evidence-grounded Byzantine history research studio."""

from __future__ import annotations

import html
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from byzantine.citations import format_chicago_note, format_gbt7714
from byzantine.generation.deepseek import (
    generate_grounded_answer,
    load_deepseek_api_key,
    summarize_research_chat,
)
from byzantine.models.document import BibliographicMetadata, DocumentRecord
from byzantine.models.evidence import Evidence
from byzantine.paths import ensure_app_data_dir
from byzantine.research.services import classify_difference, parallel_reading
from byzantine.retrieval.hybrid import hybrid_search
from byzantine.storage.database import LibraryDatabase
from byzantine.workflows.delete_document import delete_document_from_library
from byzantine.workflows.process_document import process_document, reprocess_document

RESEARCH_PAGES = ["Agent 问答", "研究专题", "史料平行对读", "矛盾与反证"]
COLLECTION_LABELS = {"starter": "公共资料库", "personal": "个人资料库"}
STATUS_LABELS = {
    "ready": "已就绪",
    "uploaded": "等待处理",
    "extracting": "提取中",
    "ocr_processing": "OCR 处理中",
    "enriching": "整理中",
    "indexing": "索引中",
    "failed": "处理失败",
}


def _database() -> LibraryDatabase:
    database = LibraryDatabase(ensure_app_data_dir() / "library.db")
    database.initialize()
    return database


def _inject_design_system(st: Any, *, first_visit: bool) -> None:
    entrance = ""
    if first_visit:
        entrance = """
        @media (prefers-reduced-motion: no-preference) {
          [data-testid="stSidebar"] { animation: sidebar-in .48s cubic-bezier(.16,1,.3,1) both; }
          [data-testid="stMainBlockContainer"] { animation: workspace-in .62s .05s cubic-bezier(.16,1,.3,1) both; }
        }
        @keyframes sidebar-in { from { opacity:0; transform:translateX(-14px); } to { opacity:1; transform:none; } }
        @keyframes workspace-in { from { opacity:0; transform:translateY(12px) scale(.995); } to { opacity:1; transform:none; } }
        """
    st.markdown(
        f"""<style>
        :root {{
          --bg:#f8f8fa; --panel:#ffffff; --sidebar:#fbfaf9; --ink:#111827; --muted:#697386;
          --line:#e3e6eb; --line-strong:#cfd5de; --blue:#0868e8; --blue-dark:#0458c7;
          --blue-soft:#edf4ff; --green:#15935b; --green-soft:#eaf7f0; --amber:#b87209;
          --amber-soft:#fff5e7; --shadow:0 10px 28px rgba(26,38,58,.065);
        }}
        html {{ background:var(--bg); }}
        .stApp {{ background:var(--bg); color:var(--ink); font-family:"Segoe UI","PingFang SC","Microsoft YaHei UI",sans-serif; }}
        #MainMenu, footer, [data-testid="stHeader"] {{ visibility:hidden; }}
        [data-testid="stAppViewContainer"] > .main {{ background:var(--bg); }}
        [data-testid="stMainBlockContainer"] {{ max-width:1440px; padding:1.9rem 2.35rem 2.8rem; }}
        [data-testid="stSidebar"] {{ width:294px!important; min-width:294px!important; background:var(--sidebar); border-right:1px solid var(--line); }}
        [data-testid="stSidebar"] > div:first-child {{ width:294px!important; padding:1.8rem 1.35rem 1.35rem; }}
        [data-testid="stSidebar"] hr {{ border-color:var(--line); margin:.85rem 0; }}
        .historia-brand {{ padding:.1rem .15rem 1.25rem; }}
        .historia-wordmark {{ font-family:Georgia,"Times New Roman",serif; color:#15171b; font-size:2rem; letter-spacing:-.045em; line-height:1; }}
        .historia-subtitle {{ color:var(--muted); font-size:.78rem; margin-top:.4rem; }}
        .sidebar-label {{ color:#344054; font-size:.78rem; font-weight:650; margin:.72rem 0 .35rem; }}
        .sidebar-note {{ color:var(--muted); font-size:.72rem; line-height:1.65; margin:.35rem 0 .85rem; }}
        .sidebar-section {{ color:#7b8494; font-size:.7rem; letter-spacing:.04em; margin:1rem 0 .35rem; }}
        .app-topbar {{ min-height:48px; border-bottom:1px solid var(--line); margin-bottom:1.35rem; }}
        .trace-badge {{ display:inline-flex; align-items:center; gap:.42rem; padding:.52rem .78rem; border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.78); color:#202834; font-size:.77rem; }}
        .st-key-research_topbar [data-testid="stHorizontalBlock"] {{ align-items:flex-start; }}
        .st-key-research_topbar .app-topbar {{ min-height:14px; margin-bottom:1rem; }}
        .st-key-research_topbar [data-testid="stSelectbox"] {{ max-width:210px; }}
        .st-key-research_topbar [data-baseweb="select"]>div {{ min-height:46px; background:rgba(255,255,255,.68)!important; border-color:rgba(188,196,208,.72)!important; box-shadow:inset 0 1px 0 rgba(255,255,255,.9),0 9px 24px rgba(39,52,73,.045); backdrop-filter:blur(18px) saturate(1.18); }}
        .page-intro {{ margin:0 0 1.25rem; }}
        .page-intro h1 {{ margin:0; color:var(--ink); font-size:1.72rem; line-height:1.2; letter-spacing:-.035em; }}
        .page-intro p {{ margin:.42rem 0 0; color:var(--muted); font-size:.88rem; line-height:1.6; max-width:70ch; }}
        .scope-bar {{ display:flex; align-items:center; gap:.65rem; min-height:54px; padding:.7rem 1rem; border:1px solid var(--line); border-radius:11px; background:var(--panel); box-shadow:0 2px 8px rgba(26,38,58,.025); margin:.6rem 0 1.15rem; }}
        .scope-name {{ color:#2c3440; font-size:.8rem; font-weight:650; }}
        .scope-chip {{ color:#333b47; font-size:.72rem; background:#f1f2f4; border-radius:7px; padding:.28rem .55rem; }}
        .scope-count {{ color:var(--muted); font-size:.75rem; }}
        .agent-mode-marker {{ display:none; }}
        [data-testid="stAppViewContainer"]:has(.agent-mode-marker)>.main {{ background:radial-gradient(circle at 52% 12%,rgba(221,232,249,.38),transparent 34%),linear-gradient(180deg,#fafbfd 0%,#f6f7f9 100%); }}
        [data-testid="stMainBlockContainer"]:has(.agent-mode-marker) {{ max-width:1040px; padding:1rem 2rem 2rem; }}
        .hero-welcome {{ min-height:clamp(245px,36vh,330px); display:flex; align-items:center; justify-content:center; text-align:center; }}
        .hero-inner {{ width:min(620px,100%); }}
        .hero-orb {{ width:54px; height:54px; margin:0 auto 1.2rem; border-radius:50%; background:radial-gradient(circle at 30% 24%,#e3efff 0,#7bb0ff 34%,#0b68e8 74%); box-shadow:inset 0 2px 3px rgba(255,255,255,.72),0 12px 30px rgba(20,107,230,.20); }}
        .hero-title {{ color:#101a31; font-size:2.35rem; font-weight:750; letter-spacing:-.055em; }}
        .hero-copy {{ color:var(--muted); font-size:.92rem; margin:.55rem auto 0; }}
        .prompt-chips {{ display:flex; justify-content:center; flex-wrap:wrap; gap:.5rem; margin-top:1.25rem; }}
        .prompt-chip {{ border:1px solid rgba(207,213,222,.72); border-radius:999px; background:rgba(255,255,255,.52); color:#3b4350; padding:.42rem .9rem; font-size:.76rem; box-shadow:inset 0 1px 0 rgba(255,255,255,.88); backdrop-filter:blur(12px); }}
        .st-key-agent_scope_glass {{ max-width:720px; margin:.2rem auto 1.25rem; padding:.52rem .62rem .62rem; border:1px solid rgba(255,255,255,.82); border-radius:20px; background:rgba(255,255,255,.52); box-shadow:inset 0 1px 0 rgba(255,255,255,.96),0 20px 48px rgba(35,48,70,.075); backdrop-filter:blur(24px) saturate(1.22); }}
        .st-key-agent_scope_glass [data-testid="stHorizontalBlock"] {{ gap:.55rem; }}
        .st-key-agent_scope_glass [data-testid="stPopover"] button {{ min-height:42px; border-color:rgba(194,202,214,.68); border-radius:13px; background:rgba(255,255,255,.58); box-shadow:inset 0 1px 0 rgba(255,255,255,.92); }}
        .st-key-agent_scope_glass .scope-bar {{ min-height:42px; margin:0; padding:.45rem .65rem; border:0; border-radius:13px; background:rgba(239,242,246,.58); box-shadow:none; }}
        .card {{ border:1px solid var(--line); border-radius:12px; background:var(--panel); box-shadow:0 2px 8px rgba(26,38,58,.025); padding:1rem 1.15rem; }}
        .card-title {{ color:var(--ink); font-size:1rem; font-weight:720; }}
        .card-copy {{ color:var(--muted); font-size:.78rem; line-height:1.58; margin-top:.35rem; }}
        .tag {{ display:inline-block; border:1px solid var(--line); border-radius:7px; padding:.23rem .55rem; color:#455063; background:#f8f8fa; font-size:.71rem; margin:.45rem .3rem 0 0; }}
        .status-good {{ color:var(--green); background:var(--green-soft); border-radius:7px; padding:.22rem .5rem; font-size:.71rem; }}
        .status-warn {{ color:var(--amber); background:var(--amber-soft); border-radius:7px; padding:.22rem .5rem; font-size:.71rem; }}
        .document-card {{ min-height:235px; padding:1.35rem; border:1px solid var(--line); border-radius:13px; background:var(--panel); box-shadow:var(--shadow); }}
        .doc-label {{ display:inline-block; background:var(--blue-soft); color:var(--blue); border-radius:7px; padding:.28rem .58rem; font-size:.75rem; }}
        .doc-icon {{ width:55px; height:68px; border:2px solid #7a8495; border-radius:5px; margin:1.45rem 0 .8rem; position:relative; }}
        .doc-icon:after {{ content:""; position:absolute; left:9px; right:9px; bottom:11px; height:2px; background:#7a8495; box-shadow:0 -9px 0 #a2a9b4; }}
        .doc-title {{ font-size:1.08rem; font-weight:720; color:var(--ink); }}
        .doc-meta {{ color:var(--muted); font-size:.78rem; margin-top:.38rem; }}
        .evidence-card {{ min-height:150px; border:1px solid var(--line); border-radius:10px; padding:.9rem; background:var(--panel); }}
        .evidence-text {{ color:#26303d; font-size:.83rem; line-height:1.65; }}
        .evidence-source {{ color:var(--muted); font-size:.72rem; margin-top:.6rem; }}
        .topic-card {{ border:1px solid var(--line); border-radius:12px; background:var(--panel); padding:1.05rem 1.15rem; margin-bottom:.75rem; box-shadow:0 2px 8px rgba(26,38,58,.025); transition:transform .22s cubic-bezier(.16,1,.3,1),box-shadow .22s ease; }}
        .topic-card:hover {{ transform:translateY(-2px); box-shadow:var(--shadow); }}
        .topic-title {{ color:var(--ink); font-size:1rem; font-weight:720; }}
        .topic-question {{ color:#465061; font-size:.78rem; margin-top:.32rem; }}
        .topic-meta {{ color:var(--muted); font-size:.7rem; margin-top:.7rem; }}
        .empty-state {{ text-align:center; border:1px dashed var(--line-strong); border-radius:12px; background:rgba(255,255,255,.55); padding:2.2rem 1rem; }}
        .empty-title {{ color:var(--ink); font-weight:700; }} .empty-copy {{ color:var(--muted); font-size:.78rem; margin-top:.35rem; }}
        .privacy-row {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; border:1px solid var(--line); background:var(--panel); border-radius:11px; padding:.9rem 1.05rem; }}
        .privacy-title {{ color:var(--ink); font-size:.88rem; font-weight:700; }} .privacy-copy {{ color:var(--muted); font-size:.75rem; margin-top:.25rem; }}
        .source-line {{ color:var(--muted); font-size:.73rem; margin-bottom:.55rem; }}
        [data-testid="stButton"]>button,[data-testid="stFormSubmitButton"]>button {{ border:1px solid var(--line-strong); border-radius:9px; background:var(--panel); color:#17202d; font-weight:600; transition:transform .16s cubic-bezier(.16,1,.3,1),box-shadow .18s ease,border-color .18s ease,background .18s ease; }}
        [data-testid="stButton"]>button:hover,[data-testid="stFormSubmitButton"]>button:hover {{ border-color:#aab3c1; box-shadow:0 7px 18px rgba(26,38,58,.08); transform:translateY(-1px); }}
        [data-testid="stButton"]>button:active,[data-testid="stFormSubmitButton"]>button:active {{ transform:translateY(1px) scale(.985); }}
        [data-testid="stButton"]>button[kind="primary"],[data-testid="stFormSubmitButton"]>button[kind="primary"] {{ background:var(--blue); border-color:var(--blue); color:#fff; }}
        [data-testid="stButton"]>button[kind="primary"]:hover,[data-testid="stFormSubmitButton"]>button[kind="primary"]:hover {{ background:var(--blue-dark); border-color:var(--blue-dark); }}
        [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stNumberInput"] input,[data-baseweb="select"]>div {{ background:var(--panel)!important; border-color:var(--line-strong)!important; color:var(--ink)!important; border-radius:9px!important; }}
        [data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus {{ border-color:var(--blue)!important; box-shadow:0 0 0 3px rgba(8,104,232,.10)!important; }}
        [data-testid="stBottom"] {{ background:transparent!important; }}
        [data-testid="stBottom"]>div {{ background:linear-gradient(180deg,rgba(247,248,250,0),rgba(247,248,250,.84) 34%,rgba(247,248,250,.96) 100%)!important; }}
        [data-testid="stChatInput"] {{ max-width:720px; min-height:66px; margin:0 auto 1rem; border:1px solid rgba(255,255,255,.9); border-radius:22px; background:rgba(255,255,255,.68); box-shadow:inset 0 1px 0 rgba(255,255,255,.98),0 18px 48px rgba(31,43,64,.13); backdrop-filter:blur(26px) saturate(1.25); }}
        [data-testid="stChatInput"] textarea {{ background:transparent!important; }}
        [data-testid="stChatInput"] button {{ border-radius:999px!important; transition:transform .18s cubic-bezier(.16,1,.3,1),background .18s ease!important; }}
        [data-testid="stChatInput"] button:active {{ transform:scale(.92); }}
        [data-testid="stMainBlockContainer"]:has(.agent-mode-marker) [data-testid="stChatMessage"] {{ max-width:720px; margin:.4rem auto; border:0; background:transparent; padding:1rem .15rem; }}
        [data-testid="stMainBlockContainer"]:has(.agent-mode-marker) [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ width:fit-content; max-width:min(78%,620px); margin-left:auto; margin-right:calc((100% - min(720px,100%))/2); padding:.72rem .95rem; border:1px solid rgba(217,222,229,.72); border-radius:20px 20px 6px 20px; background:rgba(235,237,241,.78); box-shadow:inset 0 1px 0 rgba(255,255,255,.82); backdrop-filter:blur(16px); }}
        [data-testid="stChatMessageAvatarUser"] {{ display:none!important; }}
        [data-testid="stChatMessageAvatarAssistant"] {{ width:30px!important; height:30px!important; border-radius:50%!important; background:radial-gradient(circle at 30% 24%,#e3efff 0,#7bb0ff 34%,#0b68e8 74%)!important; color:transparent!important; box-shadow:inset 0 1px 2px rgba(255,255,255,.7),0 6px 16px rgba(20,107,230,.16); }}
        [class*="st-key-archive-chat-"] {{ max-width:720px; margin:.65rem auto; }}
        [data-testid="stExpander"] {{ border:1px solid var(--line)!important; border-radius:10px!important; background:var(--panel)!important; }}
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{ border:1px solid var(--line); border-radius:10px; background:var(--panel); gap:0; overflow:hidden; width:fit-content; }}
        [data-testid="stTabs"] button {{ min-width:175px; border-right:1px solid var(--line); border-radius:0; padding:.65rem 1rem; }}
        [data-testid="stTabs"] button[aria-selected="true"] {{ color:var(--blue); background:var(--blue-soft); }}
        [data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
        [data-testid="stFileUploaderDropzone"] {{ min-height:150px; border:1px dashed #9ba6b7; border-radius:10px; background:#fcfcfd; }}
        [data-testid="stProgress"]>div>div {{ background:#e7effa; }} [data-testid="stProgress"]>div>div>div {{ background:var(--blue); }}
        [data-baseweb="popover"] {{ animation:popover-in .18s cubic-bezier(.16,1,.3,1) both; }}
        [data-baseweb="popover"]>div {{ border-color:rgba(255,255,255,.86)!important; background:rgba(255,255,255,.82)!important; box-shadow:inset 0 1px 0 rgba(255,255,255,.96),0 18px 44px rgba(31,43,64,.12)!important; backdrop-filter:blur(24px) saturate(1.2); }}
        @keyframes popover-in {{ from {{ opacity:0; transform:translateY(-5px) scale(.985); }} to {{ opacity:1; transform:none; }} }}
        .st-key-topic_drawer {{ animation:drawer-in .34s cubic-bezier(.16,1,.3,1) both; }}
        @keyframes drawer-in {{ from {{ opacity:0; transform:translateX(22px); }} to {{ opacity:1; transform:none; }} }}
        {entrance}
        @media(max-width:900px) {{
          [data-testid="stSidebar"] {{ width:260px!important; min-width:260px!important; }}
          [data-testid="stSidebar"]>div:first-child {{ width:260px!important; }}
          [data-testid="stMainBlockContainer"] {{ padding:1.15rem 1rem 2.4rem; }}
          [data-testid="stMainBlockContainer"]:has(.agent-mode-marker) {{ padding-left:1rem; padding-right:1rem; }}
          .st-key-research_topbar [data-testid="stSelectbox"] {{ max-width:180px; }}
          .st-key-agent_scope_glass {{ max-width:100%; border-radius:17px; }}
          .st-key-agent_scope_glass [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; }}
          .hero-welcome {{ min-height:300px; }} .hero-title {{ font-size:1.75rem; }}
          [data-testid="stTabs"] button {{ min-width:auto; }}
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def _topbar(st: Any, current_page: str, *, badge: str = "可追溯引用") -> str:
    with st.container(key="research_topbar"):
        left, right = st.columns([0.45, 1.55])
        with left:
            selected = st.selectbox(
                "研究功能",
                RESEARCH_PAGES,
                index=RESEARCH_PAGES.index(current_page),
                label_visibility="collapsed",
                key=f"research_page_selector_{current_page}",
            )
        with right:
            st.markdown(
                f'<div style="text-align:right"><span class="trace-badge">{html.escape(badge)}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="app-topbar"></div>', unsafe_allow_html=True)
    return selected


def _page_intro(st: Any, title: str, description: str) -> None:
    st.markdown(
        f'<section class="page-intro"><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></section>',
        unsafe_allow_html=True,
    )


def _empty_state(st: Any, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="empty-state"><div class="empty-title">{html.escape(title)}</div>'
        f'<div class="empty-copy">{html.escape(copy)}</div></div>',
        unsafe_allow_html=True,
    )


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
                text, qdrant_path=str(ensure_app_data_dir() / "qdrant"), **kwargs
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
            f"PDF 第 {item.pdf_page_start} 页"
            if item.pdf_page_start == item.pdf_page_end
            else f"PDF 第 {item.pdf_page_start}–{item.pdf_page_end} 页"
        )
        with st.expander(f"[S{index}] {item.title} · {pages}"):
            st.markdown(
                f'<div class="source-line">{html.escape(item.author or "作者未填写")} · '
                f"{html.escape(item.edition or '版本未填写')}</div>",
                unsafe_allow_html=True,
            )
            st.write(item.text)
            left, right = st.columns(2)
            left.caption(format_gbt7714(item))
            right.caption(format_chicago_note(item))


def _scope_picker(
    st: Any,
    database: LibraryDatabase,
    *,
    key: str,
    default_collection_ids: list[str] | None = None,
    default_document_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    collections = database.collections()
    by_type = {item["collection_type"]: item["collection_id"] for item in collections}
    default_types = [
        kind
        for kind, collection_id in by_type.items()
        if default_collection_ids is None or collection_id in default_collection_ids
    ]
    left, right, status = st.columns([1, 1, 1.25])
    with left, st.popover("选择资料库", width="stretch"):
        chosen_types = st.multiselect(
            "资料库",
            ["personal", "starter"],
            default=default_types or ["personal", "starter"],
            format_func=lambda value: COLLECTION_LABELS[value],
            key=f"{key}-collections",
        )
    collection_ids = [by_type[item] for item in chosen_types if item in by_type]
    documents = database.list_documents(collection_ids)
    ready_documents = [item for item in documents if item.status == "ready"]
    by_id = {item.document_id: item for item in ready_documents}
    defaults = [item for item in (default_document_ids or list(by_id)) if item in by_id]
    with right, st.popover("选择文献", width="stretch"):
        document_ids = st.multiselect(
            "可引用文献",
            list(by_id),
            default=defaults,
            format_func=lambda value: by_id[value].title,
            key=f"{key}-documents",
        )
    with status:
        st.markdown(
            f'<div class="scope-bar" style="margin:0"><span class="scope-count">'
            f"{len(collection_ids)} 个资料库 · {len(document_ids)} 篇文献</span></div>",
            unsafe_allow_html=True,
        )
    return collection_ids, document_ids


def _scope_summary(st: Any, collection_ids: list[str], document_ids: list[str]) -> None:
    chips = "".join(
        f'<span class="scope-chip">{COLLECTION_LABELS.get(item, item)}</span>'
        for item in collection_ids
    )
    st.markdown(
        f'<div class="scope-bar"><span class="scope-name">当前研究范围</span>{chips}'
        f'<span class="scope-count">{len(document_ids)} 篇引用文献</span></div>',
        unsafe_allow_html=True,
    )


def _chat_context(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{item['role']}: {item['content'][:900]}" for item in messages[-8:])


def _sidebar(st: Any, database: LibraryDatabase) -> None:
    st.markdown(
        '<div class="historia-brand"><div class="historia-wordmark">Historia</div>'
        '<div class="historia-subtitle">史料研究助手</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("＋  新建研究对话", width="stretch", key="new_research_chat"):
        st.session_state.active_conversation = None
        st.session_state.page = "Agent 问答"
        st.rerun()

    active_id = st.session_state.get("active_conversation")
    try:
        active = database.get_conversation(active_id) if active_id else None
    except KeyError:
        st.session_state.active_conversation = None
        active_id = None
        active = None
    topics = database.list_topics()
    topic_options = [""] + [item["topic_id"] for item in topics]
    st.markdown('<div class="sidebar-label">将当前对话归纳到研究专题</div>', unsafe_allow_html=True)
    selected_topic = st.selectbox(
        "专题归档",
        topic_options,
        index=(
            topic_options.index(active["topic_id"])
            if active and active["topic_id"] in topic_options
            else 0
        ),
        format_func=lambda value: (
            "暂不归档"
            if not value
            else next(item["title"] for item in topics if item["topic_id"] == value)
        ),
        label_visibility="collapsed",
        key="sidebar_topic",
    )
    if active and selected_topic != (active["topic_id"] or ""):
        database.update_conversation(active_id, topic_id=selected_topic or None)
    st.markdown(
        '<div class="sidebar-note">回答中的每一条史实都必须回到当前选定文献。删除文献后，相关对话会同步移除。</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section">研究对话</div>', unsafe_allow_html=True)
    conversations = database.list_conversations()
    if conversations:
        for conversation in conversations[:7]:
            if st.button(
                conversation["title"],
                width="stretch",
                key=f"open-chat-{conversation['conversation_id']}",
            ):
                st.session_state.active_conversation = conversation["conversation_id"]
                st.session_state.page = "Agent 问答"
                st.rerun()
    else:
        st.caption("暂无研究对话")
    st.divider()
    if st.button("设置", width="stretch", key="settings_nav"):
        st.session_state.page = "设置"
        st.rerun()


def _agent_page(st: Any, database: LibraryDatabase) -> None:
    st.markdown('<div class="agent-mode-marker"></div>', unsafe_allow_html=True)
    active_id = st.session_state.get("active_conversation")
    try:
        active = database.get_conversation(active_id) if active_id else None
    except KeyError:
        st.session_state.active_conversation = None
        active_id = None
        active = None
    messages = database.conversation_messages(active_id) if active_id else []
    default_collections = active["collection_ids"] if active else None
    default_documents = active["document_ids"] if active else None

    if not messages:
        st.markdown(
            '<section class="hero-welcome"><div class="hero-inner"><div class="hero-orb"></div>'
            '<div class="hero-title">从史料中开始提问</div>'
            '<div class="hero-copy">回答会保存为可追溯的研究对话，并附带对应出处。</div>'
            '<div class="prompt-chips"><span class="prompt-chip">人物与年代</span>'
            '<span class="prompt-chip">因果链条</span><span class="prompt-chip">制度演变</span></div></div></section>',
            unsafe_allow_html=True,
        )
    with st.container(key="agent_scope_glass"):
        collection_ids, document_ids = _scope_picker(
            st,
            database,
            key=f"chat-{active_id or 'new'}",
            default_collection_ids=default_collections,
            default_document_ids=default_documents,
        )
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            evidence = _evidence_from_snapshot(message["evidence_snapshot"])
            if evidence:
                _evidence_cards(st, evidence)

    selected_topic = st.session_state.get("sidebar_topic")
    if (
        active
        and selected_topic
        and messages
        and st.button(
            "将当前对话整理为专题研究卡",
            width="stretch",
            key=f"archive-chat-{active_id}",
        )
    ):
        evidence_by_key: dict[tuple[str, str], Evidence] = {}
        for message in messages:
            for item in _evidence_from_snapshot(message["evidence_snapshot"]):
                evidence_by_key[(item.document_id, item.node_id)] = item
        evidence = list(evidence_by_key.values())[:10]
        if not evidence:
            st.warning("当前对话还没有可归档的文献证据。")
        elif not load_deepseek_api_key():
            st.warning("请先在设置中配置 DeepSeek API Key。")
        else:
            with st.status("正在整理对话并核对引用", expanded=True) as status:
                summary = summarize_research_chat(
                    messages,
                    _retrieval_result("研究对话归档", evidence),
                    api_key=load_deepseek_api_key(),
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat",
                    max_evidence_characters=1800,
                )
                database.save_topic_chat_summary(
                    topic_id=selected_topic,
                    conversation_id=active_id,
                    title=summary["title"],
                    tags=summary["tags"],
                    summary=summary["summary"],
                    evidence=evidence,
                )
                status.update(label="专题研究卡已更新", state="complete")
            st.success("已归档到所选研究专题。")

    prompt = st.chat_input("向史料提出问题…")
    if not prompt:
        return
    if not document_ids:
        st.warning("请先选择至少一篇可引用文献。")
        return
    if active is None:
        conversation_id = database.create_conversation(
            title=prompt[:36] + ("…" if len(prompt) > 36 else ""),
            collection_ids=collection_ids,
            document_ids=document_ids,
            topic_id=st.session_state.get("sidebar_topic") or None,
        )
        st.session_state.active_conversation = conversation_id
        messages = []
    else:
        conversation_id = active["conversation_id"]
        database.update_conversation(
            conversation_id,
            collection_ids=collection_ids,
            document_ids=document_ids,
            topic_id=st.session_state.get("sidebar_topic") or None,
        )
    database.add_chat_message(conversation_id, role="user", content=prompt)
    with st.chat_message("assistant"):
        with st.status("正在检索史料并核对出处", expanded=True) as status:
            evidence = _search(
                database,
                prompt,
                collection_ids=collection_ids,
                document_ids=document_ids,
                top_k=6,
            )
            if not evidence:
                answer = "当前所选资料的证据不足，无法回答该问题。"
                status.update(label="未找到充分证据", state="complete")
            elif not load_deepseek_api_key():
                answer = "已检索到相关证据，但尚未配置 DeepSeek API Key，无法生成综合回答。"
                status.update(label="证据检索已完成", state="complete")
            else:
                status.write("正在根据本轮证据生成回答")
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
    database.add_chat_message(conversation_id, role="assistant", content=answer, evidence=evidence)
    st.rerun()


def _topic_metrics(database: LibraryDatabase, topic_id: str) -> tuple[int, int, list[str]]:
    conversations = [item for item in database.list_conversations() if item["topic_id"] == topic_id]
    summaries = database.topic_chat_summaries(topic_id)
    document_ids: set[str] = set()
    tags: list[str] = []
    for summary in summaries:
        for evidence in summary["evidence_snapshot"]:
            if evidence.get("document_id"):
                document_ids.add(str(evidence["document_id"]))
        for tag in summary["tags"]:
            if tag not in tags:
                tags.append(tag)
    return len(conversations), len(document_ids), tags[:4]


def _topic_detail(st: Any, database: LibraryDatabase, topic_id: str) -> None:
    topic = database.get_topic(topic_id)
    if st.button("返回专题库", key="back-topic-list"):
        st.session_state.selected_topic = None
        st.rerun()
    _page_intro(st, topic["title"], topic["research_question"] or "尚未设置核心研究问题")
    if topic["description"]:
        st.write(topic["description"])
    summaries = database.topic_chat_summaries(topic_id)
    if not summaries:
        _empty_state(st, "该专题还没有研究卡", "在 Agent 问答中归档一段已完成的研究对话。")
        return
    for summary in summaries:
        tags = "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in summary["tags"])
        st.markdown(
            f'<div class="topic-card"><div class="topic-title">{html.escape(summary["title"])}</div>'
            f'{tags}<div class="topic-meta">由研究对话整理，展开后可核查证据出处。</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("查看研究卡、出处与原始对话"):
            st.markdown(summary["summary"])
            evidence = _evidence_from_snapshot(summary["evidence_snapshot"])
            if evidence:
                _evidence_cards(st, evidence)
            st.caption("原始研究对话")
            for message in database.conversation_messages(summary["conversation_id"]):
                with st.chat_message(message["role"]):
                    st.write(message["content"])


def _topics_page(st: Any, database: LibraryDatabase) -> None:
    selected_topic = st.session_state.get("selected_topic")
    if selected_topic:
        _topic_detail(st, database, selected_topic)
        return
    header, action = st.columns([4, 1])
    with header:
        _page_intro(st, "研究专题", "把经过验证的研究对话整理为可持续推进的专题档案。")
    with action:
        if st.button("＋ 新建专题", type="primary", width="stretch"):
            st.session_state.topic_drawer_open = True
            st.rerun()

    drawer_open = st.session_state.get("topic_drawer_open", False)
    main, drawer = st.columns([2.2, 1]) if drawer_open else (st.container(), None)
    with main:
        search, sort = st.columns([3, 1])
        query = search.text_input(
            "搜索研究专题", placeholder="搜索研究专题", label_visibility="collapsed"
        )
        sort.selectbox("排序", ["最近更新", "最早创建"], label_visibility="collapsed")
        topics = database.list_topics()
        if query.strip():
            topics = [item for item in topics if query.lower() in item["title"].lower()]
        if not topics:
            _empty_state(st, "还没有研究专题", "点击“新建专题”建立第一个研究档案。")
        for topic in topics:
            conversation_count, document_count, tags = _topic_metrics(database, topic["topic_id"])
            tag_html = "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in tags)
            st.markdown(
                f'<div class="topic-card"><div class="topic-title">{html.escape(topic["title"])}</div>'
                f'<div class="topic-question">{html.escape(topic["research_question"] or "尚未设置核心研究问题")}</div>'
                f'{tag_html}<div class="topic-meta">{conversation_count} 条对话 · {document_count} 篇文献 · {html.escape(topic["updated_at"][:10])}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("进入专题", key=f"enter-topic-{topic['topic_id']}"):
                st.session_state.selected_topic = topic["topic_id"]
                st.rerun()
    if drawer is not None:
        with drawer, st.container(border=True, key="topic_drawer"):
            close_col, _ = st.columns([1, 4])
            if close_col.button("关闭", key="close-topic-drawer"):
                st.session_state.topic_drawer_open = False
                st.rerun()
            st.subheader("新建研究专题")
            st.caption("将研究问题整理成可持续推进的专题档案。")
            with st.form("create-topic", clear_on_submit=True):
                title = st.text_input("专题名称", placeholder="例如：十一世纪军事区的演变")
                question = st.text_input("核心研究问题", placeholder="输入你希望持续追踪的问题")
                description = st.text_area("研究说明（可选）", height=180)
                if st.form_submit_button("创建专题", type="primary", width="stretch"):
                    if not title.strip():
                        st.error("请填写专题名称。")
                    else:
                        database.create_topic(title.strip(), question.strip(), description.strip())
                        st.session_state.topic_drawer_open = False
                        st.rerun()


def _document_card(st: Any, document: DocumentRecord, label: str) -> None:
    st.markdown(
        f'<div class="document-card"><span class="doc-label">{html.escape(label)}</span>'
        '<div class="doc-icon"></div>'
        f'<div class="doc-title">《{html.escape(document.title)}》</div>'
        f'<div class="doc-meta">{html.escape(document.author or "作者未填写")} · '
        f"{html.escape(document.language)} · {COLLECTION_LABELS.get(document.collection_id, document.collection_id)}</div></div>",
        unsafe_allow_html=True,
    )


def _comparison_page(st: Any, database: LibraryDatabase) -> None:
    _page_intro(st, "史料平行对读", "将不同文献对同一问题的记载并列呈现，保留出处与差异。")
    mode = st.segmented_control(
        "对读视图", ["新建对读", "历史对读"], default="新建对读", label_visibility="collapsed"
    )
    if mode == "历史对读":
        comparisons = database.list_comparisons()
        if not comparisons:
            _empty_state(st, "暂无历史对读", "完成一次平行对读后，记录会保存在这里。")
            return
        record = st.selectbox("已保存对读", comparisons, format_func=lambda item: item["question"])
        st.caption(f"比较维度：{'、'.join(record['dimensions'])}")
        st.dataframe(record["comparison_cells"], width="stretch", hide_index=True)
        return

    collections = database.collections()
    by_type = {item["collection_type"]: item["collection_id"] for item in collections}
    selected_types = st.pills(
        "当前研究范围",
        ["personal", "starter"],
        default=["personal", "starter"],
        selection_mode="multi",
        format_func=lambda value: COLLECTION_LABELS[value],
    )
    collection_ids = [by_type[item] for item in selected_types or []]
    documents = [item for item in database.list_documents(collection_ids) if item.status == "ready"]
    _scope_summary(st, collection_ids, [item.document_id for item in documents])
    if len(documents) < 2:
        _empty_state(st, "至少需要两篇可用文献", "请先到设置中的批量导入添加资料。")
        return
    left, right = st.columns(2, gap="large")
    with left:
        document_a = st.selectbox(
            "文献 A", documents, format_func=lambda item: item.title, key="compare-a"
        )
        _document_card(st, document_a, "文献 A")
    with right:
        options_b = [item for item in documents if item.document_id != document_a.document_id]
        document_b = st.selectbox(
            "文献 B", options_b, format_func=lambda item: item.title, key="compare-b"
        )
        _document_card(st, document_b, "文献 B")
    question = st.text_input("比较问题", placeholder="不同文献如何解释第四次十字军东征的转向？")
    dimensions = st.pills(
        "比较维度",
        ["事件描述", "原因解释", "责任归属", "作者立场", "共同点", "差异点"],
        default=["原因解释", "作者立场", "差异点"],
        selection_mode="multi",
    )
    if st.button("开始对读", type="primary", width="stretch"):
        if not question.strip():
            st.warning("请先填写比较问题。")
            return
        selected_ids = [document_a.document_id, document_b.document_id]
        with st.status("正在按文献分别检索证据", expanded=True) as status:
            evidence = _search(
                database,
                question,
                collection_ids=collection_ids,
                document_ids=selected_ids,
                top_k=16,
            )
            if len({item.document_id for item in evidence}) < 2:
                status.update(label="证据覆盖不足", state="error")
                st.warning("当前问题没有同时命中两篇文献，请调整问题。")
                return
            comparison = parallel_reading(question, evidence, dimensions or [])
            database.save_comparison(comparison)
            status.update(label="对读结果已保存", state="complete")
        st.dataframe(comparison["comparison_cells"], width="stretch", hide_index=True)
        _evidence_cards(st, evidence)


def _evidence_column(st: Any, title: str, evidence: list[Evidence], tone: str) -> None:
    st.markdown(
        f'<div class="card-title">{html.escape(title)}</div><div class="card-copy">{len(evidence)} 条证据</div>',
        unsafe_allow_html=True,
    )
    for item in evidence[:4]:
        label_class = "status-good" if tone == "support" else "status-warn"
        st.markdown(
            f'<div class="evidence-card"><div class="evidence-text">{html.escape(item.text[:240])}</div>'
            f'<div class="evidence-source">{html.escape(item.title)} · 第 {item.pdf_page_start or "—"} 页</div>'
            f'<div style="margin-top:.55rem"><span class="{label_class}">{html.escape(title)}</span></div></div>',
            unsafe_allow_html=True,
        )


def _contradiction_page(st: Any, database: LibraryDatabase) -> None:
    _page_intro(
        st, "矛盾与反证", "主动寻找反面证据、限制条件与替代解释，只提示证据关系，不代替研究者判断。"
    )
    mode = st.segmented_control(
        "审计视图", ["证据审计", "历史检验"], default="证据审计", label_visibility="collapsed"
    )
    if mode == "历史检验":
        rows = database.list_contradictions()
        if not rows:
            _empty_state(st, "暂无历史检验", "完成一次矛盾与反证检索后，结果会保存在这里。")
            return
        st.dataframe(
            [
                {
                    "主张": row["subject"],
                    "差异类型": row["classification"],
                    "状态": row["review_status"],
                    "创建时间": row["created_at"][:16],
                }
                for row in rows
            ],
            width="stretch",
            hide_index=True,
        )
        return

    collection_ids, document_ids = _scope_picker(st, database, key="contradiction")
    _scope_summary(st, collection_ids, document_ids)
    claim = st.text_input("待校验主张", placeholder="输入需要检验的历史主张")
    if st.button("重新检验", type="primary"):
        if not document_ids or not claim.strip():
            st.warning("请先选择文献并填写待校验主张。")
            return
        with st.status("正在检索支持、反面与限制性证据", expanded=True) as status:
            groups = {
                "support": _search(
                    database,
                    claim,
                    collection_ids=collection_ids,
                    document_ids=document_ids,
                    top_k=4,
                ),
                "oppose": _search(
                    database,
                    f"evidence against or contradicting: {claim}",
                    collection_ids=collection_ids,
                    document_ids=document_ids,
                    top_k=4,
                ),
                "qualify": _search(
                    database,
                    f"limitations and alternative explanations for: {claim}",
                    collection_ids=collection_ids,
                    document_ids=document_ids,
                    top_k=4,
                ),
            }
            for support in groups["support"]:
                for opposing in groups["oppose"][:2]:
                    database.save_contradiction(
                        subject=claim,
                        description="主动反证检索发现的待核查差异。",
                        classification=classify_difference(support, opposing),
                        evidence_side_a=support,
                        evidence_side_b=opposing,
                    )
            st.session_state.contradiction_result = {
                key: [item.model_dump(mode="json") for item in value]
                for key, value in groups.items()
            }
            st.session_state.contradiction_claim = claim
            status.update(label="证据关系已整理", state="complete")
    saved = st.session_state.get("contradiction_result")
    if not saved:
        return
    groups = {key: _evidence_from_snapshot(value) for key, value in saved.items()}
    st.markdown(
        f'<div class="card"><span class="status-good">待校验主张</span>'
        f'<div class="card-title" style="margin-top:.65rem">{html.escape(st.session_state.contradiction_claim)}</div>'
        f'<div class="card-copy">已检索 {len(document_ids)} 篇文献 · 找到 {sum(len(value) for value in groups.values())} 条相关证据</div></div>',
        unsafe_allow_html=True,
    )
    support_col, oppose_col, qualify_col = st.columns(3)
    with support_col:
        _evidence_column(st, "支持该主张", groups["support"], "support")
    with oppose_col:
        _evidence_column(st, "反面证据", groups["oppose"], "oppose")
    with qualify_col:
        _evidence_column(st, "限制与替代解释", groups["qualify"], "qualify")
    st.markdown(
        '<div class="privacy-row" style="margin-top:1rem"><div><div class="privacy-title">当前判断边界</div>'
        '<div class="privacy-copy">现有材料可支持部分解释，但不足以将其视为唯一原因。结论仍需研究者核查。</div></div></div>',
        unsafe_allow_html=True,
    )


def _save_api_key(api_key: str) -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    kept = [line for line in existing if not line.startswith("DEEPSEEK_API_KEY=")]
    kept.append(f"DEEPSEEK_API_KEY={api_key.strip()}")
    env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.environ["DEEPSEEK_API_KEY"] = api_key.strip()


def _test_api_connection() -> str:
    api_key = load_deepseek_api_key()
    if not api_key:
        raise RuntimeError("尚未配置 DeepSeek API Key。")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    models = client.models.list()
    count = len(list(models.data))
    return f"连接成功，服务返回 {count} 个可用模型。"


def _copy_path(path: Path) -> None:
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", "Set-Clipboard", "-Value", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _system_settings(st: Any, database: LibraryDatabase) -> None:
    root = ensure_app_data_dir()
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.subheader("本地数据")
        st.caption("文献、索引、研究对话与专题默认保存在本地。")
        st.text_input("数据目录", str(root), disabled=True)
        copy_col, open_col = st.columns(2)
        if copy_col.button("复制路径", width="stretch"):
            try:
                _copy_path(root)
                st.success("路径已复制。")
            except Exception as exc:  # noqa: BLE001
                st.error(f"复制失败：{exc}")
        if open_col.button("打开文件夹", width="stretch"):
            try:
                os.startfile(root)  # type: ignore[attr-defined]
            except OSError as exc:
                st.error(f"无法打开目录：{exc}")
    with right, st.container(border=True):
        st.subheader("模型服务")
        configured = bool(load_deepseek_api_key())
        st.markdown(
            f'<div class="privacy-row"><span>回答模型</span><strong>DeepSeek</strong>'
            f'<span class="{"status-good" if configured else "status-warn"}">'
            f"{'已配置' if configured else '未配置'}</span></div>",
            unsafe_allow_html=True,
        )
        test_col, edit_col = st.columns(2)
        if test_col.button("测试连接", width="stretch"):
            try:
                with st.spinner("正在连接 DeepSeek"):
                    st.success(_test_api_connection())
            except Exception as exc:  # noqa: BLE001
                st.error(f"连接失败：{exc}")
        if edit_col.button("修改配置", width="stretch"):
            st.session_state.edit_api_key = not st.session_state.get("edit_api_key", False)
        if st.session_state.get("edit_api_key"):
            with st.form("api-key-form"):
                api_key = st.text_input("DeepSeek API Key", type="password")
                if st.form_submit_button("保存配置", type="primary"):
                    if not api_key.strip():
                        st.error("API Key 不能为空。")
                    else:
                        _save_api_key(api_key)
                        st.session_state.edit_api_key = False
                        st.success("配置已保存到本地 .env。")

    with st.container(border=True):
        st.subheader("检索与向量索引")
        st.caption("SQLite FTS5 提供关键词检索；BGE-M3 与 Qdrant 提供语义召回。")
        check_col, rebuild_col = st.columns([1, 1])
        if check_col.button("检查索引", width="stretch"):
            try:
                from byzantine.indexing.library_index import index_status

                report = index_status(qdrant_path=str(root / "qdrant"))
                if report["healthy"]:
                    st.success(f"{report['message']}，共 {report['points']} 个向量点。")
                else:
                    st.warning(report["message"])
            except Exception as exc:  # noqa: BLE001
                st.error(f"检查失败：{exc}")
        if rebuild_col.button("重建索引", width="stretch"):
            all_evidence = [
                evidence
                for document in database.list_documents()
                for evidence in database.document_evidence(document.document_id)
            ]
            if not all_evidence:
                st.warning("资料库没有可用于重建索引的证据。")
            else:
                from byzantine.indexing.library_index import rebuild_index

                progress = st.progress(0, text="准备重建索引")
                try:
                    rebuild_index(
                        all_evidence,
                        qdrant_path=str(root / "qdrant"),
                        progress=lambda completed, total: progress.progress(
                            completed / max(total, 1),
                            text=f"正在重建索引 {completed}/{total}",
                        ),
                    )
                    st.success("索引重建完成。")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"索引重建失败：{exc}")
    st.markdown(
        '<div class="privacy-row"><div><div class="privacy-title">隐私说明</div>'
        '<div class="privacy-copy">只有生成回答、专题摘要或证据分析时，才会发送当前检索到的必要内容。完整文献与研究专题不会自动上传。</div>'
        '</div><span class="status-good">数据保存在本地</span></div>',
        unsafe_allow_html=True,
    )


def _batch_import(st: Any, database: LibraryDatabase) -> None:
    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "docx", "txt", "md", "markdown", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="支持 PDF、DOCX、TXT、Markdown 与图片；单个文件建议不超过 100 MB。",
    )
    if uploaded_files:
        st.dataframe(
            [
                {
                    "文件名": item.name,
                    "大小": f"{len(item.getvalue()) / 1024 / 1024:.1f} MB",
                    "状态": "等待处理",
                }
                for item in uploaded_files
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"已选择 {len(uploaded_files)} 个文件 · 共 {sum(len(item.getvalue()) for item in uploaded_files) / 1024 / 1024:.1f} MB"
        )
    left, right = st.columns([1, 1.55], gap="large")
    with left, st.container(border=True):
        st.subheader("导入设置")
        collection_id = st.selectbox(
            "导入资料库",
            ["personal", "starter"],
            format_func=lambda value: COLLECTION_LABELS[value],
        )
        language = st.selectbox("语言", ["Chinese", "English", "Greek", "Latin", "other"])
        source_type = st.selectbox(
            "资料类型", ["secondary_study", "primary_source", "translation", "reference_work"]
        )
    with right, st.container(border=True):
        st.subheader("共享书目信息")
        st.caption("以下信息将应用于本批次全部文献，可留空。")
        a, b = st.columns(2)
        author = a.text_input("作者")
        publisher = b.text_input("出版社")
        edition = a.text_input("版本")
        year = b.number_input("出版年份", min_value=0, max_value=3000, value=0)
    if st.button("开始批量处理", type="primary", disabled=not uploaded_files, width="stretch"):
        overall = st.progress(0, text="准备导入")
        report = st.status("批量导入进行中", expanded=True)
        started = time.monotonic()
        total = len(uploaded_files)
        completed: list[str] = []
        failures: list[str] = []
        staging = ensure_app_data_dir() / ".uploads"
        staging.mkdir(exist_ok=True)
        for index, uploaded in enumerate(uploaded_files):
            safe_name = Path(uploaded.name).name
            local = staging / f"{uuid.uuid4().hex}_{safe_name}"
            local.write_bytes(uploaded.getvalue())
            title = Path(safe_name).stem.replace("_", " ")

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
                report.write(f"正在处理：{safe_name}")
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
                failures.append(f"{safe_name}：{exc}")
            finally:
                local.unlink(missing_ok=True)
            overall.progress((index + 1) / total, text=f"已完成 {index + 1}/{total} 份文献")
        report.update(label="批量导入完成", state="complete", expanded=bool(failures))
        if completed:
            st.success(f"成功处理 {len(completed)} 份：{'、'.join(completed)}")
        if failures:
            st.error("\n".join(failures))


def _library_management(st: Any, database: LibraryDatabase) -> None:
    documents = database.list_documents()
    metrics = st.columns(4)
    metrics[0].metric("文献总数", len(documents))
    metrics[1].metric("已就绪", sum(item.status == "ready" for item in documents))
    metrics[2].metric("处理中", sum(item.status not in {"ready", "failed"} for item in documents))
    metrics[3].metric("处理失败", sum(item.status == "failed" for item in documents))
    search_col, collection_col, status_col = st.columns([2, 1, 1])
    query = search_col.text_input("搜索文献", placeholder="搜索文献", label_visibility="collapsed")
    collection_filter = collection_col.selectbox(
        "资料库",
        ["all", "personal", "starter"],
        format_func=lambda value: "全部资料库" if value == "all" else COLLECTION_LABELS[value],
        label_visibility="collapsed",
    )
    status_filter = status_col.selectbox(
        "状态",
        ["all", "ready", "processing", "failed"],
        format_func=lambda value: {
            "all": "全部状态",
            "ready": "已就绪",
            "processing": "处理中",
            "failed": "处理失败",
        }[value],
        label_visibility="collapsed",
    )
    filtered = documents
    if query.strip():
        filtered = [item for item in filtered if query.lower() in item.title.lower()]
    if collection_filter != "all":
        filtered = [item for item in filtered if item.collection_id == collection_filter]
    if status_filter == "ready":
        filtered = [item for item in filtered if item.status == "ready"]
    elif status_filter == "failed":
        filtered = [item for item in filtered if item.status == "failed"]
    elif status_filter == "processing":
        filtered = [item for item in filtered if item.status not in {"ready", "failed"}]
    st.dataframe(
        [
            {
                "文献名称": item.title,
                "资料库": COLLECTION_LABELS.get(item.collection_id, item.collection_id),
                "类型与语言": f"{item.source_type} · {item.language}",
                "索引状态": STATUS_LABELS.get(item.status, item.status),
                "更新时间": item.updated_at[:10],
            }
            for item in filtered
        ],
        width="stretch",
        hide_index=True,
    )
    if not filtered:
        _empty_state(st, "没有匹配文献", "请调整搜索词或筛选条件。")
        return
    st.subheader("文献操作")
    target = st.selectbox("选择文献", filtered, format_func=lambda item: item.title)
    retry_col, delete_col = st.columns(2)
    if retry_col.button("重新处理", width="stretch"):
        bar = st.progress(0, text="准备重新处理")
        try:
            document = reprocess_document(
                target.document_id,
                database=database,
                seed_path=Path("config/entity_seed.yaml"),
                progress=lambda stage, fraction: bar.progress(fraction, text=stage),
            )
            st.success(f"《{document.title}》重新处理完成。")
        except Exception as exc:  # noqa: BLE001
            st.error(f"重新处理失败：{exc}")
    with delete_col.popover("删除文献", width="stretch"):
        st.warning("删除会同步清理索引、相关研究对话和专题引用，且不可恢复。")
        confirmed = st.checkbox(f"确认删除《{target.title}》")
        if st.button("永久删除", disabled=not confirmed, type="primary"):
            try:
                delete_document_from_library(target.document_id, database=database)
                st.success("文献与派生记录已删除。")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"删除失败：{exc}")
    st.markdown(
        '<div class="privacy-row" style="margin-top:1rem"><div><div class="privacy-title">删除规则</div>'
        '<div class="privacy-copy">删除文献将同步清理索引，并移除依赖该文献的研究对话与专题引用。</div></div></div>',
        unsafe_allow_html=True,
    )


def _settings_page(st: Any, database: LibraryDatabase) -> None:
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.caption("设置")
    with top_right:
        if st.button("返回研究工作区", width="stretch"):
            st.session_state.page = "Agent 问答"
            st.rerun()
    _page_intro(st, "设置", "管理本地数据、模型服务、资料导入与文献索引。")
    system_tab, import_tab, library_tab = st.tabs(["系统与数据", "批量导入", "资料库管理"])
    with system_tab:
        _system_settings(st, database)
    with import_tab:
        _batch_import(st, database)
    with library_tab:
        _library_management(st, database)


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="Historia", page_icon="H", layout="wide")
    first_visit = not st.session_state.get("_historia_entered", False)
    st.session_state._historia_entered = True
    st.session_state.setdefault("page", "Agent 问答")
    st.session_state.setdefault("active_conversation", None)
    st.session_state.setdefault("topic_drawer_open", False)
    st.session_state.setdefault("selected_topic", None)
    _inject_design_system(st, first_visit=first_visit)
    database = _database()
    with st.sidebar:
        _sidebar(st, database)

    current_page = st.session_state.page
    if current_page != "设置":
        selected_page = _topbar(
            st,
            current_page,
            badge={
                "Agent 问答": "可追溯引用",
                "研究专题": "专题档案",
                "史料平行对读": "保留差异",
                "矛盾与反证": "证据审计",
            }[current_page],
        )
        if selected_page != current_page:
            st.session_state.page = selected_page
            st.rerun()
    pages = {
        "Agent 问答": _agent_page,
        "研究专题": _topics_page,
        "史料平行对读": _comparison_page,
        "矛盾与反证": _contradiction_page,
        "设置": _settings_page,
    }
    pages[current_page](st, database)


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())], check=False
    )


if __name__ == "__main__":
    render()
