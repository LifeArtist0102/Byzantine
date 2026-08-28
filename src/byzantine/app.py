"""Product-grade Streamlit workspace for Byzantine evidence-based research."""

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
from byzantine.models.document import BibliographicMetadata, DocumentRecord
from byzantine.models.evidence import Evidence
from byzantine.paths import ensure_app_data_dir
from byzantine.research.services import classify_difference, counter_queries, parallel_reading
from byzantine.retrieval.hybrid import hybrid_search
from byzantine.storage.database import LibraryDatabase
from byzantine.workflows.delete_document import delete_document_from_library
from byzantine.workflows.process_document import process_document, reprocess_document

COLLECTION_LABELS = {"starter": "公共资料库", "personal": "个人资料库"}
STATUS_LABELS = {
    "ready": "已就绪",
    "uploaded": "待处理",
    "extracting": "正在提取",
    "enriching": "正在整理",
    "indexing": "正在索引",
    "failed": "处理失败",
}


def _database() -> LibraryDatabase:
    database = LibraryDatabase(ensure_app_data_dir() / "library.db")
    database.initialize()
    return database


def _inject_style(st: Any) -> None:
    """Install the single-palette, editorial product system for the whole app."""
    st.markdown(
        """<style>
        :root {
            --paper: #f5f4ee;
            --surface: #fcfcf8;
            --surface-muted: #f0f1ea;
            --ink: #17231f;
            --muted: #627068;
            --line: #d9ddd4;
            --line-strong: #bdc7bc;
            --forest: #116149;
            --forest-dark: #0a4635;
            --forest-soft: #e1eee6;
            --warning: #9a5b18;
            --danger: #9b3932;
            --shadow: 0 18px 48px rgba(28, 46, 37, .07);
        }
        .stApp {
            background: var(--paper);
            color: var(--ink);
            font-family: "Aptos", "Segoe UI", "Microsoft YaHei UI", sans-serif;
        }
        #MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; }
        [data-testid="stAppViewContainer"] > .main { background: var(--paper); }
        [data-testid="stMainBlockContainer"] {
            max-width: 1440px;
            padding: 2rem 3.25rem 3.5rem;
        }
        [data-testid="stSidebar"] {
            background: #17251f;
            border-right: 1px solid rgba(239, 244, 237, .12);
        }
        [data-testid="stSidebar"] * { color: #edf2ec; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #aebdb3; }
        [data-testid="stSidebar"] hr { border-color: rgba(239, 244, 237, .14); }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 8px;
            margin: .12rem 0;
            padding: .34rem .42rem;
            transition: background .18s ease, transform .18s ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(239, 244, 237, .09);
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: #2c5141;
            box-shadow: inset 2px 0 0 #9cc7ad;
        }
        .brand-lockup { padding: .9rem .25rem 1.35rem; }
        .brand-mark {
            display: inline-flex; align-items: center; justify-content: center;
            width: 31px; height: 31px; border: 1px solid #a9c6b3; border-radius: 6px;
            color: #f5f7f2; font-family: "Arial Narrow", sans-serif; font-size: 1rem;
            font-weight: 700; letter-spacing: -.06em; margin-bottom: .72rem;
        }
        .brand-name { color: #f4f6f1; font-size: .95rem; font-weight: 700; letter-spacing: .08em; }
        .brand-subtitle { color: #aebdb3; font-size: .73rem; letter-spacing: .04em; margin-top: .25rem; }
        .page-head {
            display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2rem;
            align-items: end; border-bottom: 1px solid var(--line-strong);
            padding: .2rem 0 1.2rem; margin-bottom: 1.5rem;
        }
        .page-kicker, .section-kicker {
            color: var(--forest); font-size: .72rem; font-weight: 750; letter-spacing: .13em;
            text-transform: uppercase; margin-bottom: .55rem;
        }
        .page-head h1 { color: var(--ink); font-size: clamp(1.8rem, 3vw, 2.85rem); letter-spacing: -.055em; line-height: 1; margin: 0; font-weight: 720; }
        .page-head p { color: var(--muted); margin: .65rem 0 0; font-size: .96rem; line-height: 1.65; max-width: 67ch; }
        .page-meta { color: var(--muted); text-align: right; font-family: Consolas, "Cascadia Mono", monospace; font-size: .72rem; line-height: 1.6; white-space: nowrap; }
        .workflow-note {
            border-left: 2px solid var(--forest); background: #e9f0ea; color: #426151;
            padding: .64rem .82rem; margin: 0 0 1.25rem; font-size: .85rem; line-height: 1.55;
        }
        .scope-shell {
            border: 1px solid var(--line); background: var(--surface); padding: 1.1rem 1.2rem 1.15rem;
            border-radius: 10px; box-shadow: 0 8px 25px rgba(28, 46, 37, .035); margin-bottom: 1rem;
        }
        .scope-title { color: var(--ink); font-size: .92rem; font-weight: 720; letter-spacing: -.01em; }
        .scope-caption { color: var(--muted); font-size: .79rem; margin-top: .24rem; line-height: 1.5; }
        .scope-step { color: var(--forest); font-family: Consolas, "Cascadia Mono", monospace; font-size: .69rem; letter-spacing: .1em; }
        .metric-strip { display: flex; align-items: stretch; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); margin: 0 0 1.35rem; }
        .metric-unit { padding: .72rem 1.25rem .72rem 0; margin-right: 1.25rem; border-right: 1px solid var(--line); min-width: 112px; }
        .metric-unit:last-child { border-right: 0; }
        .metric-number { color: var(--ink); font-size: 1.35rem; font-weight: 720; letter-spacing: -.05em; line-height: 1; }
        .metric-label { color: var(--muted); font-size: .74rem; margin-top: .32rem; }
        .rail { border-left: 1px solid var(--line); padding-left: 1.25rem; }
        .rail-title { color: var(--ink); font-size: .86rem; font-weight: 720; margin: 0 0 .6rem; }
        .rail-copy { color: var(--muted); font-size: .79rem; line-height: 1.55; margin: .55rem 0 0; }
        .record {
            border-top: 1px solid var(--line); padding: 1rem 0; transition: background .2s ease;
        }
        .record:last-child { border-bottom: 1px solid var(--line); }
        .record:hover { background: rgba(225, 238, 230, .32); }
        .record-title { color: var(--ink); font-size: 1rem; font-weight: 720; letter-spacing: -.015em; }
        .record-meta { color: var(--muted); font-size: .78rem; margin-top: .35rem; line-height: 1.45; }
        .tag { display: inline-block; color: var(--forest-dark); background: var(--forest-soft); border: 1px solid #c8ddce; border-radius: 999px; padding: .14rem .48rem; font-size: .72rem; margin: .2rem .28rem .1rem 0; }
        .empty-state { border: 1px dashed var(--line-strong); border-radius: 10px; padding: 2.1rem 1.4rem; background: rgba(252, 252, 248, .55); }
        .empty-eyebrow { color: var(--forest); font-family: Consolas, "Cascadia Mono", monospace; font-size: .72rem; letter-spacing: .1em; }
        .empty-title { color: var(--ink); font-size: 1.08rem; font-weight: 720; margin: .55rem 0 .32rem; }
        .empty-copy { color: var(--muted); font-size: .86rem; max-width: 58ch; line-height: 1.6; }
        .source-line { color: var(--muted); font-size: .79rem; margin: .1rem 0 .65rem; }
        [data-testid="stChatMessage"] { border: 0; border-top: 1px solid var(--line); border-radius: 0; padding: 1.1rem .15rem; background: transparent; }
        [data-testid="stChatMessage"]:last-of-type { border-bottom: 1px solid var(--line); }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color: var(--ink); line-height: 1.75; }
        [data-testid="stChatMessageAvatarUser"] { background: #d5e6da !important; color: #174c39 !important; }
        [data-testid="stChatMessageAvatarAssistant"] { background: #1d3128 !important; color: #f2f5ee !important; }
        [data-testid="stChatInput"] { border: 1px solid var(--line-strong); box-shadow: 0 12px 28px rgba(28, 46, 37, .08); background: var(--surface); }
        [data-testid="stChatInput"]:focus-within { border-color: var(--forest); box-shadow: 0 0 0 3px rgba(17, 97, 73, .12); }
        [data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 8px !important; background: var(--surface) !important; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.3rem; border-bottom: 1px solid var(--line); }
        [data-testid="stTabs"] button { color: var(--muted); background: transparent !important; border-radius: 0; padding: .55rem 0; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: var(--forest-dark); border-bottom: 2px solid var(--forest) !important; }
        [data-testid="stMetric"] { border: 0; background: transparent; padding: 0; }
        [data-testid="stMetricLabel"] { color: var(--muted); font-size: .76rem; }
        [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.45rem; letter-spacing: -.045em; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
        [data-testid="stProgress"] > div > div { background: #dce7df; }
        [data-testid="stProgress"] > div > div > div { background: var(--forest); animation: progress-breathe 1.7s ease-in-out infinite; }
        @keyframes progress-breathe { 50% { filter: brightness(1.16); } }
        .skeleton { height: 11px; border-radius: 4px; margin: .65rem 0; background: linear-gradient(90deg, #e8ebe4 25%, #f7f8f4 37%, #e8ebe4 63%); background-size: 400% 100%; animation: shimmer 1.35s ease infinite; }
        .skeleton.short { width: 48%; } .skeleton.medium { width: 76%; }
        @keyframes shimmer { 0% { background-position: 100% 0; } 100% { background-position: -100% 0; } }
        [data-testid="stButton"] > button, [data-testid="stFormSubmitButton"] > button {
            border: 1px solid var(--forest); background: var(--forest); color: #fff; border-radius: 7px;
            font-weight: 650; transition: transform .18s cubic-bezier(.16, 1, .3, 1), background .18s ease, box-shadow .18s ease;
        }
        [data-testid="stButton"] > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background: var(--forest-dark); box-shadow: 0 8px 18px rgba(17, 97, 73, .17); transform: translateY(-1px); }
        [data-testid="stButton"] > button:active, [data-testid="stFormSubmitButton"] > button:active { transform: translateY(1px) scale(.985); }
        [data-testid="stButton"] > button[kind="secondary"] { background: transparent; color: var(--forest-dark); border-color: var(--line-strong); box-shadow: none; }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input, [data-baseweb="select"] > div {
            border-color: var(--line-strong) !important; border-radius: 7px !important; background: var(--surface) !important; color: var(--ink) !important;
        }
        [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus { border-color: var(--forest) !important; box-shadow: 0 0 0 3px rgba(17, 97, 73, .1) !important; }
        label, [data-testid="stWidgetLabel"] p { color: #34483e !important; font-size: .82rem !important; font-weight: 650 !important; }
        [data-testid="stAlert"] { border-radius: 8px; }
        @media (max-width: 860px) {
            [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 2.5rem; }
            .page-head { display: block; }
            .page-meta { text-align: left; margin-top: .8rem; }
            .metric-strip { overflow-x: auto; }
            .rail { border-left: 0; border-top: 1px solid var(--line); padding: 1rem 0 0; margin-top: 1.25rem; }
        }
        </style>""",
        unsafe_allow_html=True,
    )


def _inject_reference_layout(st: Any, *, play_entrance: bool) -> None:
    """Apply the framed, calm chat-workspace direction from the visual brief."""
    entrance = ""
    if play_entrance:
        entrance = """
        @media (prefers-reduced-motion: no-preference) {
            [data-testid="stSidebar"] { animation: sidebar-enter .55s cubic-bezier(.16, 1, .3, 1) both; }
            [data-testid="stAppViewContainer"] > .main { animation: workspace-enter .7s .08s cubic-bezier(.16, 1, .3, 1) both; }
        }
        @keyframes sidebar-enter { from { opacity: 0; transform: translateX(-18px); } to { opacity: 1; transform: translateX(0); } }
        @keyframes workspace-enter { from { opacity: 0; transform: translateY(16px) scale(.992); } to { opacity: 1; transform: translateY(0) scale(1); } }
        """
    st.markdown(
        f"""<style>
        :root {{ --canvas: #151716; --window: #fbfbf9; --panel: #ffffff; --soft: #f5f6f3; --ink: #142019; --muted: #728078; --line: #e5e8e3; --forest: #116149; --forest-dark: #0b4031; --forest-soft: #e4f0e8; }}
        .stApp, [data-testid="stAppViewContainer"] {{ background: var(--canvas) !important; }}
        [data-testid="stAppViewContainer"] > .main {{
            background: var(--window) !important; margin: 28px 28px 28px 0; min-height: calc(100vh - 56px);
            border-radius: 0 24px 24px 0; box-shadow: 18px 22px 56px rgba(0, 0, 0, .18); overflow: hidden;
        }}
        [data-testid="stMainBlockContainer"] {{ max-width: 1280px; padding: 1.65rem 2.65rem 3rem; }}
        [data-testid="stSidebar"] {{
            background: var(--window) !important; color: var(--ink) !important; margin: 28px 0 28px 28px;
            min-height: calc(100vh - 56px); border-radius: 24px 0 0 24px; border: 0; border-right: 1px solid var(--line);
            box-shadow: -10px 22px 56px rgba(0, 0, 0, .08); overflow: hidden;
        }}
        [data-testid="stSidebar"] * {{ color: var(--ink) !important; }}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p, [data-testid="stSidebar"] .sidebar-session-meta {{ color: var(--muted) !important; }}
        [data-testid="stSidebar"] hr {{ border-color: var(--line); }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{ border-radius: 8px; padding: .4rem .52rem; margin: .08rem 0; }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background: #f0f4f1; transform: translateX(2px); }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{ background: var(--forest-soft); box-shadow: inset 2px 0 0 var(--forest); }}
        .brand-lockup {{ padding: .55rem .1rem .8rem; }}
        .brand-mark {{ background: #18271f; border: 0; border-radius: 9px; color: #fff !important; width: 30px; height: 30px; margin-bottom: .58rem; }}
        .brand-name {{ color: var(--ink) !important; font-size: .87rem; letter-spacing: .105em; }}
        .brand-subtitle {{ color: var(--muted) !important; font-size: .64rem; }}
        .page-head {{ margin-bottom: 1.1rem; padding: .15rem 0 .95rem; border-color: var(--line); }}
        .page-head h1 {{ font-size: clamp(1.45rem, 2.35vw, 2.25rem); font-weight: 720; }}
        .page-head p {{ font-size: .87rem; line-height: 1.55; }}
        .page-kicker, .section-kicker {{ font-size: .65rem; letter-spacing: .14em; }}
        .page-meta {{ font-size: .66rem; color: #849087; }}
        .scope-shell {{ border-color: var(--line); border-radius: 12px; box-shadow: none; background: var(--panel); padding: 1rem 1.1rem; }}
        .metric-strip {{ margin-bottom: 1rem; border-color: var(--line); }}
        .metric-unit {{ min-width: 92px; padding: .6rem 1rem .6rem 0; margin-right: 1rem; border-color: var(--line); }}
        .metric-number {{ font-size: 1.1rem; }}
        .rail {{ border-color: var(--line); }}
        .chat-welcome {{ min-height: 245px; display: flex; align-items: center; justify-content: center; text-align: center; padding: 1.2rem 0 .2rem; }}
        .chat-welcome-inner {{ max-width: 550px; }}
        .welcome-orb {{ width: 40px; height: 40px; margin: 0 auto 1rem; border-radius: 50%; background: radial-gradient(circle at 30% 25%, #b7dbc5 0, #3f9472 35%, #0c5b43 76%); box-shadow: inset 0 1px 2px rgba(255,255,255,.55); }}
        .welcome-kicker {{ color: var(--forest); font-family: Consolas, "Cascadia Mono", monospace; font-size: .68rem; letter-spacing: .12em; }}
        .welcome-title {{ color: var(--ink); font-size: 1.8rem; letter-spacing: -.055em; font-weight: 730; margin: .6rem 0 .45rem; }}
        .welcome-copy {{ color: var(--muted); font-size: .88rem; line-height: 1.62; }}
        .prompt-hints {{ display: flex; justify-content: center; flex-wrap: wrap; gap: .45rem; margin-top: 1rem; }}
        .prompt-hint {{ background: var(--panel); border: 1px solid var(--line); border-radius: 999px; color: #526158; padding: .34rem .62rem; font-size: .72rem; }}
        [data-testid="stChatInput"] {{ max-width: 760px; margin: .55rem auto 0; border-radius: 14px; background: var(--panel); box-shadow: 0 13px 32px rgba(24, 39, 31, .09); }}
        [data-testid="stChatInput"] textarea {{ min-height: 48px; }}
        [data-testid="stButton"] > button, [data-testid="stFormSubmitButton"] > button {{ border-radius: 999px; font-size: .82rem; padding: .38rem .85rem; }}
        .sidebar-session-heading {{ color: var(--ink); font-size: .72rem; font-weight: 720; letter-spacing: .1em; margin: .65rem 0 .35rem; }}
        .sidebar-session-note {{ color: var(--muted); font-size: .72rem; line-height: 1.5; margin-top: .55rem; }}
        {entrance}
        @media (max-width: 860px) {{
            [data-testid="stAppViewContainer"] > .main {{ margin: 0; min-height: 100dvh; border-radius: 0; box-shadow: none; }}
            [data-testid="stSidebar"] {{ margin: 0; min-height: 100dvh; border-radius: 0; box-shadow: none; }}
            [data-testid="stMainBlockContainer"] {{ padding: 1.15rem 1rem 2.5rem; }}
            .chat-welcome {{ min-height: 190px; }} .welcome-title {{ font-size: 1.45rem; }}
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def _chat_welcome(st: Any) -> None:
    st.markdown(
        """<section class="chat-welcome"><div class="chat-welcome-inner"><div class="welcome-orb"></div>
        <div class="welcome-kicker">BYZANTINE RESEARCH AGENT</div><div class="welcome-title">从史料中开始提问</div>
        <div class="welcome-copy">选择资料范围后，提出一个明确的问题。回答会保存为可追溯的研究对话，并附带对应出处。</div>
        <div class="prompt-hints"><span class="prompt-hint">人物与年代</span><span class="prompt-hint">因果链条</span><span class="prompt-hint">制度演变</span></div></div></section>""",
        unsafe_allow_html=True,
    )


def _page_head(st: Any, eyebrow: str, title: str, description: str, meta: str) -> None:
    st.markdown(
        f"""<section class="page-head"><div><div class="page-kicker">{eyebrow}</div>
        <h1>{title}</h1><p>{description}</p></div><div class="page-meta">{meta}</div></section>""",
        unsafe_allow_html=True,
    )


def _workflow_note(st: Any, text: str) -> None:
    st.markdown(f'<div class="workflow-note">{text}</div>', unsafe_allow_html=True)


def _empty_state(st: Any, eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f"""<div class="empty-state"><div class="empty-eyebrow">{eyebrow}</div>
        <div class="empty-title">{title}</div><div class="empty-copy">{copy}</div></div>""",
        unsafe_allow_html=True,
    )


def _metric_strip(st: Any, items: list[tuple[str, str]]) -> None:
    body = "".join(
        f'<div class="metric-unit"><div class="metric-number">{value}</div><div class="metric-label">{label}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-strip">{body}</div>', unsafe_allow_html=True)


def _scope_picker(
    st: Any,
    database: LibraryDatabase,
    *,
    key: str,
    default_collection_ids: list[str] | None = None,
    default_document_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Two layers of scope: collection kind, then specific documents."""
    collections = database.collections()
    by_type = {item["collection_type"]: item["collection_id"] for item in collections}
    default_types = [
        collection_type
        for collection_type, collection_id in by_type.items()
        if default_collection_ids is None or collection_id in default_collection_ids
    ]
    st.markdown(
        '<div class="scope-shell"><div class="scope-title">本次研究范围</div>'
        '<div class="scope-caption">先确定资料库，再确定本轮可引用的具体文献。系统不会越过此范围检索。</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="scope-step">01 / LIBRARY</div>', unsafe_allow_html=True)
        chosen_types = st.multiselect(
            "选择资料库",
            options=["personal", "starter"],
            default=default_types or ["personal", "starter"],
            format_func=lambda value: COLLECTION_LABELS[value],
            key=f"{key}-collection-types",
            help="个人资料库是你上传的资料；公共资料库是项目内置的资料。",
        )
    collection_ids = [by_type[item] for item in chosen_types if item in by_type]
    documents = database.list_documents(collection_ids=collection_ids)
    document_by_id = {document.document_id: document for document in documents}
    defaults = [
        item for item in (default_document_ids or list(document_by_id)) if item in document_by_id
    ]
    with right:
        st.markdown('<div class="scope-step">02 / DOCUMENTS</div>', unsafe_allow_html=True)
        document_ids = st.multiselect(
            "选择文献（默认全选）",
            options=list(document_by_id),
            default=defaults,
            format_func=lambda document_id: (
                f"{document_by_id[document_id].title} · "
                f"{COLLECTION_LABELS.get(document_by_id[document_id].collection_id, document_by_id[document_id].collection_id)}"
            ),
            key=f"{key}-documents",
            help="取消勾选即可排除某本书；空选择不会进行检索。",
        )
    st.markdown("</div>", unsafe_allow_html=True)
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
            f"PDF p. {item.pdf_page_start}"
            if item.pdf_page_start == item.pdf_page_end
            else f"PDF pp. {item.pdf_page_start}-{item.pdf_page_end}"
        )
        with st.expander(f"[S{index}]  {item.title}  ·  {pages}"):
            st.markdown(
                f'<div class="source-line">{COLLECTION_LABELS.get(item.collection_type, item.collection_type)}'
                f" · {item.author or '作者未填写'} · {item.edition or '版本未填写'}</div>",
                unsafe_allow_html=True,
            )
            st.write(item.text)
            left, right = st.columns(2)
            left.caption(format_gbt7714(item))
            right.caption(format_chicago_note(item))


def _chat_context(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{item['role']}: {item['content'][:900]}" for item in messages[-8:])


def _agent_page(st: Any, database: LibraryDatabase) -> None:
    _page_head(
        st,
        "Research dialogue",
        "Agent 问答",
        "在明确的文献边界内进行带出处的多轮研究对话；上下文用于理解追问，史实只来自本轮选定资料。",
        "LOCAL-FIRST\nEVIDENCE-BOUND",
    )
    conversations = database.list_conversations()
    if "active_conversation" not in st.session_state:
        st.session_state.active_conversation = None

    with st.sidebar:
        st.markdown(
            '<aside class="rail"><div class="rail-title">对话工作台</div>', unsafe_allow_html=True
        )
        if st.button("新建对话", use_container_width=True):
            st.session_state.active_conversation = None
            st.rerun()
        if conversations:
            ids = [item["conversation_id"] for item in conversations]
            active = st.session_state.active_conversation
            selected = st.selectbox(
                "历史对话",
                ids,
                index=ids.index(active) if active in ids else 0,
                format_func=lambda value: next(
                    item["title"] for item in conversations if item["conversation_id"] == value
                ),
            )
            if selected != active:
                st.session_state.active_conversation = selected
                st.rerun()
        active_data = (
            database.get_conversation(st.session_state.active_conversation)
            if st.session_state.active_conversation
            else None
        )
        topics = database.list_topics()
        topic_options = [""] + [item["topic_id"] for item in topics]
        selected_topic = st.selectbox(
            "归属研究专题",
            topic_options,
            index=(
                topic_options.index(active_data["topic_id"])
                if active_data and active_data["topic_id"] in topic_options
                else 0
            ),
            format_func=lambda value: (
                "暂不归档"
                if not value
                else next(item["title"] for item in topics if item["topic_id"] == value)
            ),
            help="先完成有价值的讨论，再归档为专题研究卡。",
        )
        if active_data and selected_topic != (active_data["topic_id"] or ""):
            database.update_conversation(
                active_data["conversation_id"], topic_id=selected_topic or None
            )
            active_data = database.get_conversation(active_data["conversation_id"])
        st.markdown(
            '<p class="rail-copy">回答中的每一条史实都必须回到当前选定文献。删除文献后，相关对话会同步移除。</p></aside>',
            unsafe_allow_html=True,
        )

    with st.container():
        default_collections = active_data["collection_ids"] if active_data else None
        default_documents = active_data["document_ids"] if active_data else None
        collection_ids, document_ids = _scope_picker(
            st,
            database,
            key=f"chat-{active_data['conversation_id'] if active_data else 'new'}",
            default_collection_ids=default_collections,
            default_document_ids=default_documents,
        )
        _metric_strip(
            st,
            [("已选资料库", str(len(collection_ids))), ("可检索文献", str(len(document_ids)))],
        )
        messages = (
            database.conversation_messages(active_data["conversation_id"]) if active_data else []
        )
        if not messages:
            _chat_welcome(st)
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["labels"]:
                    st.caption(" · ".join(message["labels"]))
                saved_evidence = _evidence_from_snapshot(message["evidence_snapshot"])
                if saved_evidence:
                    _evidence_cards(st, saved_evidence)

        prompt = st.chat_input("围绕当前选定的文献继续提问")
        if prompt:
            if not document_ids:
                st.warning("请在第二层至少选择一本具体文献后再提问。")
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
                st.markdown(
                    '<div class="skeleton medium"></div><div class="skeleton"></div>',
                    unsafe_allow_html=True,
                )
                with st.status("正在检索选定文献并核对出处", expanded=True) as status:
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
                        status.update(label="本地证据检索已完成", state="complete")
                    else:
                        status.write("正在依据本轮证据生成回答")
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

        if active_data and selected_topic and messages:
            st.divider()
            if st.button("将当前对话归档为专题研究卡", type="primary"):
                evidence = []
                for message in messages:
                    evidence.extend(_evidence_from_snapshot(message["evidence_snapshot"]))
                unique = list({item.chunk_id: item for item in evidence}.values())
                try:
                    with st.spinner("正在整理标题、标签、摘要与出处"):
                        summary = summarize_research_chat(
                            [
                                {"role": item["role"], "content": item["content"]}
                                for item in messages
                            ],
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
    _page_head(
        st,
        "Library intake",
        "批量上传资料",
        "导入后依次完成文本提取、语义整理与向量索引。进度和预计剩余时间会随实际处理速度更新。",
        "BATCH INGEST\nLOCAL PROCESSING",
    )
    _workflow_note(
        st,
        "上传文件会先复制到本地资料库。只有文本、元数据和向量写入成功后，文献才会标记为“已就绪”。",
    )
    uploaded_files = st.file_uploader(
        "选择一个或多个文件",
        type=["pdf", "txt", "md", "markdown", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="支持一次选择多本书或多份资料。批量处理时会以文件名作为默认书名。",
    )
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown('<div class="section-kicker">DESTINATION</div>', unsafe_allow_html=True)
        collection_id = st.selectbox(
            "导入资料库",
            ["personal", "starter"],
            format_func=lambda value: COLLECTION_LABELS[value],
        )
        language = st.selectbox("语言", ["English", "Chinese", "Greek", "Latin", "other"])
        source_type = st.selectbox(
            "资料类型", ["secondary_study", "primary_source", "translation", "reference_work"]
        )
    with right:
        st.markdown('<div class="section-kicker">SHARED BIBLIOGRAPHY</div>', unsafe_allow_html=True)
        author = st.text_input("作者（可批量共用）")
        publisher = st.text_input("出版社（可批量共用）")
        edition = st.text_input("版本（可批量共用）")
        year = st.number_input("出版年份（未知填 0）", min_value=0, max_value=3000, value=0)
    if uploaded_files:
        _metric_strip(
            st,
            [
                ("待处理文件", str(len(uploaded_files))),
                ("目标资料库", COLLECTION_LABELS[collection_id]),
            ],
        )
    if st.button("开始批量处理", type="primary", disabled=not uploaded_files):
        overall = st.progress(0, text="准备导入")
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
                    text=(
                        f"{index + 1}/{total} · {title} · {stage} · 预计剩余 {max(0, int(eta))} 秒"
                    ),
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


def _document_record(st: Any, document: DocumentRecord) -> None:
    status = STATUS_LABELS.get(document.status, document.status)
    st.markdown('<div class="record">', unsafe_allow_html=True)
    left, right = st.columns([4, 1])
    with left:
        st.markdown(f'<div class="record-title">{document.title}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="record-meta">{COLLECTION_LABELS.get(document.collection_id, document.collection_id)}'
            f" · {document.author or '作者未填写'} · {document.page_count or '—'} 页 · {status}</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.caption(document.updated_at[:10])
    st.markdown("</div>", unsafe_allow_html=True)


def _library_page(st: Any, database: LibraryDatabase) -> None:
    _page_head(
        st,
        "Source control",
        "资料库",
        "查看资料库中的文献、处理状态与本地副本。删除会同步清理索引、聊天和专题中的相关记录。",
        "SOURCE OF TRUTH\nLOCAL ARCHIVE",
    )
    documents = database.list_documents()
    _metric_strip(
        st,
        [
            ("文献总数", str(len(documents))),
            ("已就绪", str(sum(item.status == "ready" for item in documents))),
            ("待处理", str(sum(item.status != "ready" for item in documents))),
        ],
    )
    if not documents:
        _empty_state(
            st,
            "EMPTY LIBRARY",
            "资料库仍为空",
            "前往“批量上传资料”导入第一本文献；其后才能选择文献并开始问答。",
        )
        return
    st.markdown('<div class="section-kicker">LIBRARY CATALOGUE</div>', unsafe_allow_html=True)
    for document in documents:
        _document_record(st, document)

    retryable = [
        item
        for item in documents
        if item.status in {"failed", "indexing", "extracting", "enriching"}
    ]
    if retryable:
        st.divider()
        st.markdown('<div class="section-kicker">RECOVERY</div>', unsafe_allow_html=True)
        target = st.selectbox("选择需要恢复的文献", retryable, format_func=lambda item: item.title)
        if st.button("重新处理并显示进度"):
            bar = st.progress(0, text="准备重新处理")
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
    st.markdown('<div class="section-kicker">DESTRUCTIVE ACTION</div>', unsafe_allow_html=True)
    st.warning(
        "永久删除会移除向量、全文、存储副本，以及只依赖该文献的聊天与专题摘要；外部原始文件不会受影响。"
    )
    target = st.selectbox(
        "选择要删除的文献", documents, format_func=lambda item: item.title, key="delete-document"
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
    _page_head(
        st,
        "Research map",
        "研究专题",
        "把已验证的多轮问答组织成可继续研究的专题档案：包含标签、模型摘要、原始对话与证据出处。",
        "SYNTHESIS\nWITH PROVENANCE",
    )
    create, browse = st.tabs(["创建专题", "专题档案"])
    with create, st.form("new-topic", clear_on_submit=True):
        st.markdown('<div class="section-kicker">NEW RESEARCH THREAD</div>', unsafe_allow_html=True)
        title = st.text_input("专题名称", placeholder="例如：十一世纪军事区的演变")
        question = st.text_input("核心研究问题")
        description = st.text_area("研究说明（可选）", help="说明研究边界、假设或准备比较的材料。")
        if st.form_submit_button("创建研究专题", type="primary"):
            if not title.strip():
                st.error("请填写专题名称。")
            else:
                database.create_topic(title.strip(), question.strip(), description.strip())
                st.success("专题已创建。回到 Agent 问答后，可将对话归档到这里。")
    with browse:
        topics = database.list_topics()
        if not topics:
            _empty_state(
                st,
                "NO TOPICS",
                "还没有研究专题",
                "先创建一个研究问题，再把 Agent 问答中的有价值对话归档为带出处的研究卡。",
            )
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
        _metric_strip(st, [("已归档研究卡", str(len(summaries))), ("专题状态", "进行中")])
        st.markdown(f'<div class="record-title">{topic["title"]}</div>', unsafe_allow_html=True)
        st.caption(topic["research_question"] or "尚未设置核心研究问题")
        if topic["description"]:
            st.write(topic["description"])
        if not summaries:
            _empty_state(
                st,
                "AWAITING SYNTHESIS",
                "尚未归档对话",
                "在 Agent 问答中选择这个专题，然后点击“将当前对话归档为专题研究卡”。",
            )
            return
        for summary in summaries:
            tags = "".join(f'<span class="tag">{tag}</span>' for tag in summary["tags"])
            st.markdown(
                f'<div class="record"><div class="record-title">{summary["title"]}</div>{tags}'
                f'<div class="record-meta">由一段已保存对话整理；展开后可审查摘要、出处与原始上下文。</div></div>',
                unsafe_allow_html=True,
            )
            with st.expander("查看研究卡、证据与原始对话"):
                st.markdown(summary["summary"])
                evidence = _evidence_from_snapshot(summary["evidence_snapshot"])
                if evidence:
                    _evidence_cards(st, evidence)
                st.caption("原始聊天记录")
                for message in database.conversation_messages(summary["conversation_id"]):
                    with st.chat_message(message["role"]):
                        st.write(message["content"])


def _comparison_page(st: Any, database: LibraryDatabase) -> None:
    _page_head(
        st,
        "Comparative reading",
        "史料平行对读",
        "并列保存不同文献在同一问题上的证据与表述，不将不同材料混为一段没有出处的概括。",
        "COMPARE\nPRESERVE DIFFERENCE",
    )
    compose, history = st.tabs(["新建对读", "历史记录"])
    with compose:
        _workflow_note(
            st, "请在第二层至少选择两本文献。系统只比较当前选定范围中的命中文本，并保存本次对读。"
        )
        collection_ids, document_ids = _scope_picker(st, database, key="comparison")
        question = st.text_input("比较问题", placeholder="不同文献如何解释第四次十字军东征的转向？")
        dimensions = st.multiselect(
            "比较维度",
            [
                "事件描述",
                "关键词",
                "原因解释",
                "责任归属",
                "人物评价",
                "作者立场",
                "时间记录",
                "共同点",
                "差异点",
            ],
            default=["事件描述", "原因解释", "共同点", "差异点"],
        )
        if st.button("运行平行对读", type="primary"):
            if len(document_ids) < 2:
                st.warning("请在第二层至少选择两本文献。")
                return
            if not question.strip():
                st.warning("请先填写比较问题。")
                return
            with st.status("正在按选定文献检索并分列证据", expanded=True) as status:
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
            _empty_state(
                st,
                "NO COMPARISONS",
                "还没有保存的对读",
                "选择至少两本文献，填写问题和维度后运行平行对读。",
            )
        else:
            record = st.selectbox(
                "已保存对读", comparisons, format_func=lambda item: item["question"]
            )
            st.caption(f"比较维度：{'、'.join(record['dimensions'])}")
            st.dataframe(record["comparison_cells"], use_container_width=True, hide_index=True)


def _contradiction_page(st: Any, database: LibraryDatabase) -> None:
    _page_head(
        st,
        "Counter-evidence",
        "矛盾与反证",
        "主动检索支持、限制与替代解释，帮助你区分直接冲突、翻译差异和不同史学视角。",
        "CHALLENGE\nTHE FIRST ANSWER",
    )
    _workflow_note(
        st, "此功能不判定真伪；它只保存需要研究者核查的证据差异，并始终受当前文献范围约束。"
    )
    collection_ids, document_ids = _scope_picker(st, database, key="counter")
    claim = st.text_input(
        "需要检验的问题或主张", placeholder="例如：某项制度变化的主要原因是什么？"
    )
    if st.button("寻找反证与差异", type="primary"):
        if not document_ids:
            st.warning("请先选择具体文献。")
            return
        if not claim.strip():
            st.warning("请先填写需要检验的问题或主张。")
            return
        with st.status("正在执行支持、反面与替代解释三类检索", expanded=True) as status:
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
            status.update(label="待核查差异已保存", state="complete")
        if unique:
            _evidence_cards(st, unique)
        else:
            _empty_state(
                st,
                "NO COUNTER-EVIDENCE",
                "没有检索到可比较的证据",
                "可以调整问题表述，或扩大第二层中选择的文献范围。",
            )


def _settings_page(st: Any) -> None:
    _page_head(
        st,
        "Local-first",
        "设置",
        "文献、索引、聊天与专题均保存于本地。仅在生成回答或归档专题时，才会向 DeepSeek 发送当前检索到的证据。",
        "PRIVACY\nLOCAL DATA",
    )
    _metric_strip(
        st,
        [
            ("数据位置", "本地"),
            ("DeepSeek", "已配置" if load_deepseek_api_key() else "未配置"),
        ],
    )
    st.markdown('<div class="section-kicker">DATA DIRECTORY</div>', unsafe_allow_html=True)
    st.code(str(ensure_app_data_dir()), language=None)
    st.info(
        "提示：SQLite FTS5 对中文分词有限；清晰的人名、地名、年份和英文术语可提升检索效果，BGE-M3 会补充语义召回。"
    )


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="Byzantine Research Studio", page_icon="B", layout="wide")
    play_entrance = not st.session_state.get("_entry_motion_seen", False)
    st.session_state._entry_motion_seen = True
    _inject_style(st)
    _inject_reference_layout(st, play_entrance=play_entrance)
    database = _database()
    st.sidebar.markdown(
        '<div class="brand-lockup"><div class="brand-mark">B</div><div class="brand-name">BYZANTINE</div>'
        '<div class="brand-subtitle">EVIDENCE RESEARCH STUDIO</div></div>',
        unsafe_allow_html=True,
    )
    pages = {
        "Agent 问答": _agent_page,
        "批量上传资料": _upload_page,
        "资料库": _library_page,
        "研究专题": _topics_page,
        "史料平行对读": _comparison_page,
        "矛盾与反证": _contradiction_page,
        "设置": lambda st_, db: _settings_page(st_),
    }
    choice = st.sidebar.radio("研究工作流", list(pages), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("所有结论必须回到已选择文献中的可追溯证据。")
    pages[choice](st, database)


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())], check=False
    )


if __name__ == "__main__":
    render()
