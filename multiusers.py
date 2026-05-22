"""멀티유저·멀티세션 RAG 챗봇 — app_users 테이블 로그인, Supabase 세션·벡터 저장."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"


def _resolve_log_dir() -> Path:
    """로컬은 repo/logs, Streamlit Cloud 등 읽기 전용 환경은 temp로 폴백."""
    for candidate in (LOG_DIR, Path(tempfile.gettempdir()) / "mpb_ai_edu" / "logs"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return Path(tempfile.gettempdir())

MODEL_NAME = "gpt-4o-mini"
VECTOR_BATCH_SIZE = 10
RETRIEVER_K = 10
USER_TABLE = "app_users"


def _get_secret(key: str) -> str:
    """Streamlit secrets 우선, 없으면 환경 변수."""
    try:
        value = st.secrets.get(key)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(key, "").strip()


def _apply_config() -> None:
    load_dotenv(dotenv_path=ENV_PATH, override=False)


def _setup_logging() -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log_path = _resolve_log_dir() / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.WARNING)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("multiusers")


logger = _setup_logging()

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _missing_keys() -> list[str]:
    missing: list[str] = []
    if not _get_secret("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not _get_secret("SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    if not _get_secret("SUPABASE_ANON_KEY"):
        missing.append("SUPABASE_ANON_KEY")
    return missing


def _get_supabase() -> Client | None:
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _get_llm() -> ChatOpenAI:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return ChatOpenAI(model=MODEL_NAME, temperature=0.7, api_key=api_key)


def _get_embeddings() -> OpenAIEmbeddings:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAIEmbeddings(api_key=api_key)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def _user_id() -> str | None:
    user = st.session_state.get("current_user")
    if not user:
        return None
    return str(user.get("id") or "")


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "current_user": None,
        "auth_tab": "login",
        "chat_history": [],
        "conversation_memory": [],
        "current_session_id": None,
        "processed_names": [],
        "session_list": [],
        "selected_session_id": None,
        "rag_enabled": True,
        "sidebar_session_pick": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _register_user(sb: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id:
        return False, "아이디를 입력하세요."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."

    existing = (
        sb.table(USER_TABLE)
        .select("id")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False, "이미 사용 중인 아이디입니다."

    try:
        sb.table(USER_TABLE).insert(
            {"login_id": login_id, "password_hash": _hash_password(password)}
        ).execute()
        return True, "회원가입이 완료되었습니다. 로그인해 주세요."
    except Exception as exc:  # noqa: BLE001
        logger.warning("Signup failed: %s", exc)
        err = str(exc)
        if "PGRST205" in err or "Could not find the table" in err:
            return (
                False,
                "회원 테이블(`app_users`)이 Supabase에 없습니다. "
                "SQL Editor에서 `multi-users-migrate.sql`을 실행한 뒤 다시 시도하세요.",
            )
        return False, f"회원가입 실패: {exc}"


def _authenticate_user(sb: Client, login_id: str, password: str) -> tuple[dict[str, str] | None, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return None, "아이디와 비밀번호를 입력하세요."

    res = (
        sb.table(USER_TABLE)
        .select("id, login_id, password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None, "아이디 또는 비밀번호가 올바르지 않습니다."

    row = rows[0]
    if not _verify_password(password, row["password_hash"]):
        return None, "아이디 또는 비밀번호가 올바르지 않습니다."

    return {"id": row["id"], "login_id": row["login_id"]}, ""


def _logout() -> None:
    st.session_state.current_user = None
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.current_session_id = None
    st.session_state.processed_names = []
    st.session_state.selected_session_id = None
    st.session_state.session_list = []
    st.session_state.sidebar_session_pick = "(세션 선택)"


def _fetch_sessions(sb: Client, user_id: str) -> list[dict[str, Any]]:
    res = (
        sb.table("chat_sessions")
        .select("id, title, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return res.data or []


def _session_options(sessions: list[dict[str, Any]]) -> dict[str, str]:
    return {f"{s['title']} ({s['id'][:8]}…)": s["id"] for s in sessions}


def _session_owned(sb: Client, session_id: str, user_id: str) -> bool:
    res = (
        sb.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def _generate_session_title(llm: ChatOpenAI, first_user: str, first_assistant: str) -> str:
    prompt = (
        "다음 첫 질문과 답변을 한 줄로 요약한 세션 제목을 한국어로 작성하세요. "
        "20자 내외, 따옴표·설명 없이 제목만 출력하세요.\n\n"
        f"[질문]\n{first_user[:500]}\n\n[답변]\n{first_assistant[:800]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = str(getattr(out, "content", "") or "").strip().strip('"\'')
        return title[:80] if title else "새 세션"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Title generation failed: %s", exc)
        return first_user[:40] or "새 세션"


def _save_messages(
    sb: Client, session_id: str, user_id: str, messages: list[dict[str, str]]
) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).eq(
        "user_id", user_id
    ).execute()
    rows = [
        {
            "session_id": session_id,
            "user_id": user_id,
            "role": m["role"],
            "content": m["content"],
            "sort_order": idx,
        }
        for idx, m in enumerate(messages)
    ]
    if rows:
        sb.table("chat_messages").insert(rows).execute()
    sb.table("chat_sessions").update({"updated_at": datetime.utcnow().isoformat()}).eq(
        "id", session_id
    ).eq("user_id", user_id).execute()


def _save_session_files(sb: Client, session_id: str, names: list[str]) -> None:
    sb.table("session_files").delete().eq("session_id", session_id).execute()
    if not names:
        return
    sb.table("session_files").insert(
        [{"session_id": session_id, "file_name": n} for n in names]
    ).execute()


def _load_session_into_state(sb: Client, session_id: str, user_id: str) -> bool:
    if not _session_owned(sb, session_id, user_id):
        return False

    msgs = (
        sb.table("chat_messages")
        .select("role, content, sort_order")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("sort_order")
        .execute()
    )
    files = (
        sb.table("session_files")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    st.session_state.current_session_id = session_id
    st.session_state.selected_session_id = session_id
    st.session_state.chat_history = [
        {"role": m["role"], "content": m["content"]} for m in (msgs.data or [])
    ]
    st.session_state.conversation_memory = list(st.session_state.chat_history)
    st.session_state.processed_names = [f["file_name"] for f in (files.data or [])]
    return True


def _auto_save_current(sb: Client, user_id: str, llm: ChatOpenAI | None = None) -> str | None:
    history = st.session_state.chat_history
    if not history and not st.session_state.processed_names:
        return st.session_state.current_session_id

    sid = st.session_state.current_session_id
    if sid and not _session_owned(sb, sid, user_id):
        sid = None
        st.session_state.current_session_id = None

    if not sid:
        title = "새 세션"
        if llm and len(history) >= 2:
            users = [m for m in history if m["role"] == "user"]
            assistants = [m for m in history if m["role"] == "assistant"]
            if users and assistants:
                title = _generate_session_title(
                    llm, users[0]["content"], assistants[0]["content"]
                )
        row = (
            sb.table("chat_sessions")
            .insert({"title": title, "user_id": user_id})
            .execute()
        )
        sid = row.data[0]["id"]
        st.session_state.current_session_id = sid

    if history:
        _save_messages(sb, sid, user_id, history)
    _save_session_files(sb, sid, st.session_state.processed_names)
    return sid


def _insert_new_session_copy(sb: Client, user_id: str, llm: ChatOpenAI) -> str | None:
    history = st.session_state.chat_history
    if not history:
        st.sidebar.warning("저장할 대화가 없습니다.")
        return None

    users = [m for m in history if m["role"] == "user"]
    assistants = [m for m in history if m["role"] == "assistant"]
    title = "새 세션"
    if users and assistants:
        title = _generate_session_title(llm, users[0]["content"], assistants[0]["content"])

    new_row = (
        sb.table("chat_sessions")
        .insert({"title": title, "user_id": user_id})
        .execute()
    )
    new_id = new_row.data[0]["id"]
    _save_messages(sb, new_id, user_id, history)
    _save_session_files(sb, new_id, st.session_state.processed_names)

    src_id = st.session_state.current_session_id
    if src_id and _session_owned(sb, src_id, user_id):
        _copy_vectors(sb, src_id, new_id)
    return new_id


def _copy_vectors(sb: Client, source_session_id: str, target_session_id: str) -> None:
    res = (
        sb.table("vector_documents")
        .select("file_name, content, embedding, metadata")
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return
    for i in range(0, len(rows), VECTOR_BATCH_SIZE):
        batch = rows[i : i + VECTOR_BATCH_SIZE]
        inserts = [
            {
                "session_id": target_session_id,
                "file_name": r["file_name"],
                "content": r["content"],
                "embedding": r["embedding"],
                "metadata": r.get("metadata") or {},
            }
            for r in batch
        ]
        sb.table("vector_documents").insert(inserts).execute()


def _delete_session(sb: Client, session_id: str, user_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()


def _store_vectors_for_file(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    file_name: str,
    splits: list[Document],
) -> int:
    count = 0
    for i in range(0, len(splits), VECTOR_BATCH_SIZE):
        batch = splits[i : i + VECTOR_BATCH_SIZE]
        texts = [d.page_content for d in batch]
        vectors = embeddings.embed_documents(texts)
        rows = [
            {
                "session_id": session_id,
                "file_name": file_name,
                "content": doc.page_content,
                "embedding": emb,
                "metadata": doc.metadata or {},
            }
            for doc, emb in zip(batch, vectors)
        ]
        sb.table("vector_documents").insert(rows).execute()
        count += len(rows)
    return count


def _process_pdf_uploads(
    sb: Client,
    uploaded_files: list[Any],
    session_id: str,
    user_id: str,
) -> tuple[list[str], int]:
    if not _session_owned(sb, session_id, user_id):
        raise ValueError("현재 세션에 접근할 수 없습니다.")

    embeddings = _get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed: list[str] = []
    total_chunks = 0

    for uf in uploaded_files:
        file_name = Path(uf.name).name
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            docs = PyPDFLoader(tmp_path).load()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not docs:
            continue

        for d in docs:
            d.metadata["file_name"] = file_name
            d.metadata["source"] = file_name

        splits = splitter.split_documents(docs)
        if not splits:
            continue

        existing = (
            sb.table("vector_documents")
            .select("id")
            .eq("session_id", session_id)
            .eq("file_name", file_name)
            .limit(1)
            .execute()
        )
        if not existing.data:
            total_chunks += _store_vectors_for_file(
                sb, embeddings, session_id, file_name, splits
            )
        processed.append(file_name)

    return processed, total_chunks


def _retrieve_documents_rpc(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    query: str,
    session_id: str,
    k: int = RETRIEVER_K,
) -> list[Document]:
    query_emb = embeddings.embed_query(query)
    try:
        res = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_emb,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in res.data or []:
            docs.append(
                Document(
                    page_content=row.get("content", ""),
                    metadata={
                        "file_name": row.get("file_name", ""),
                        "session_id": row.get("session_id", ""),
                        "similarity": row.get("similarity"),
                    },
                )
            )
        return docs
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC retrieval failed, fallback: %s", exc)
        return _retrieve_documents_fallback(sb, embeddings, query, session_id, k)


def _retrieve_documents_fallback(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    query: str,
    session_id: str,
    k: int,
) -> list[Document]:
    try:
        vs = SupabaseVectorStore(
            client=sb,
            embedding=embeddings,
            table_name="vector_documents",
            query_name="match_vector_documents",
        )
        retriever = vs.as_retriever(search_kwargs={"k": k * 3})
        docs = retriever.invoke(query)
        filtered = [
            d
            for d in docs
            if (d.metadata or {}).get("session_id") == session_id
            or str((d.metadata or {}).get("session_id", "")) == session_id
        ]
        return filtered[:k] if filtered else docs[:k]
    except Exception as exc2:  # noqa: BLE001
        logger.warning("Fallback retriever failed: %s", exc2)
        res = (
            sb.table("vector_documents")
            .select("content, file_name, metadata")
            .eq("session_id", session_id)
            .limit(k)
            .execute()
        )
        return [
            Document(page_content=r["content"], metadata={"file_name": r["file_name"]})
            for r in (res.data or [])
        ]


def _list_vector_file_names(sb: Client, session_id: str | None, user_id: str) -> list[str]:
    if not session_id or not _session_owned(sb, session_id, user_id):
        return []
    res = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    return sorted({r["file_name"] for r in (res.data or []) if r.get("file_name")})


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if m["role"] == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str,
    context: str,
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = remove_separators(str(getattr(out, "content", "") or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def _clear_screen() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.current_session_id = None
    st.session_state.processed_names = []
    st.session_state.selected_session_id = None


def _render_styles() -> None:
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    _render_styles()
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">기획예산처</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def _render_auth_screen(sb: Client) -> None:
    _render_header()
    st.markdown("### 로그인 / 회원가입")
    st.caption("Supabase Auth 없이 `app_users` 테이블로 계정을 관리합니다.")

    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    with tab_login:
        login_id = st.text_input("아이디", key="login_id_input")
        password = st.text_input("비밀번호", type="password", key="login_pw_input")
        if st.button("로그인", type="primary", key="btn_login"):
            user, err = _authenticate_user(sb, login_id, password)
            if user:
                st.session_state.current_user = user
                st.session_state.session_list = _fetch_sessions(sb, user["id"])
                st.success(f"{user['login_id']}님, 환영합니다.")
                st.rerun()
            else:
                st.error(err)

    with tab_signup:
        new_id = st.text_input("새 아이디", key="signup_id_input")
        new_pw = st.text_input("비밀번호", type="password", key="signup_pw_input")
        new_pw2 = st.text_input("비밀번호 확인", type="password", key="signup_pw2_input")
        if st.button("회원가입", key="btn_signup"):
            if new_pw != new_pw2:
                st.error("비밀번호 확인이 일치하지 않습니다.")
            else:
                ok, msg = _register_user(sb, new_id, new_pw)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


def _on_session_select_change() -> None:
    pick = st.session_state.get("sidebar_session_pick")
    if not pick or pick == "(세션 선택)":
        return
    sb = _get_supabase()
    uid = _user_id()
    if not sb or not uid:
        return
    options = st.session_state.get("_session_id_map", {})
    sid = options.get(pick)
    if sid:
        if not _load_session_into_state(sb, sid, uid):
            st.sidebar.error("해당 세션에 접근할 수 없습니다.")


def _render_sidebar(sb: Client, user_id: str, login_id: str) -> None:
    try:
        st.session_state.session_list = _fetch_sessions(sb, user_id)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"세션 목록을 불러올 수 없습니다. "
            f"`multi-users-ref.sql`을 Supabase SQL Editor에서 실행했는지 확인하세요.\n\n`{exc}`"
        )
        st.stop()

    id_map = _session_options(st.session_state.session_list)
    st.session_state._session_id_map = id_map
    labels = ["(세션 선택)"] + list(id_map.keys())

    with st.sidebar:
        st.markdown(f"**로그인:** `{login_id}`")
        if st.button("로그아웃"):
            _logout()
            st.rerun()

        st.markdown("---")
        st.markdown("**LLM 모델**")
        st.radio("LLM 모델 선택", (MODEL_NAME,), index=0, disabled=True)

        st.markdown("---")
        st.markdown("**세션 관리**")

        current_label = None
        if st.session_state.current_session_id:
            for lbl, sid in id_map.items():
                if sid == st.session_state.current_session_id:
                    current_label = lbl
                    break

        default_idx = 0
        if current_label and current_label in labels[1:]:
            default_idx = labels.index(current_label)

        st.selectbox(
            "저장된 세션",
            labels,
            index=default_idx,
            key="sidebar_session_pick",
            on_change=_on_session_select_change,
        )

        llm_for_save: ChatOpenAI | None = None
        try:
            llm_for_save = _get_llm()
        except ValueError:
            pass

        col1, col2 = st.columns(2)
        with col1:
            if st.button("세션저장"):
                if llm_for_save is None:
                    st.warning("OPENAI_API_KEY가 필요합니다.")
                else:
                    new_id = _insert_new_session_copy(sb, user_id, llm_for_save)
                    if new_id:
                        st.session_state.current_session_id = new_id
                        st.session_state.session_list = _fetch_sessions(sb, user_id)
                        st.success("새 세션이 저장되었습니다.")
                        st.rerun()

            if st.button("세션로드"):
                pick = st.session_state.get("sidebar_session_pick")
                if not pick or pick == "(세션 선택)":
                    st.warning("세션을 선택하세요.")
                else:
                    sid = id_map.get(pick)
                    if sid and _load_session_into_state(sb, sid, user_id):
                        st.success("세션을 불러왔습니다.")
                        st.rerun()
                    else:
                        st.error("세션을 불러올 수 없습니다.")

        with col2:
            if st.button("세션삭제"):
                sid = st.session_state.current_session_id
                if not sid:
                    pick = st.session_state.get("sidebar_session_pick")
                    if pick and pick != "(세션 선택)":
                        sid = id_map.get(pick)
                if not sid:
                    st.warning("삭제할 세션을 선택하세요.")
                else:
                    _delete_session(sb, sid, user_id)
                    if st.session_state.current_session_id == sid:
                        _clear_screen()
                    st.session_state.session_list = _fetch_sessions(sb, user_id)
                    st.success("세션이 삭제되었습니다.")
                    st.rerun()

            if st.button("화면초기화"):
                _clear_screen()
                st.session_state.sidebar_session_pick = "(세션 선택)"
                st.rerun()

        if st.button("vectordb"):
            names = _list_vector_file_names(
                sb, st.session_state.current_session_id, user_id
            )
            if not names:
                st.info("현재 세션에 저장된 벡터 파일이 없습니다.")
            else:
                st.markdown("**Vector DB 파일 목록**")
                for n in names:
                    st.text(f"- {n}")

        st.markdown("---")
        rag_on = st.radio("RAG (PDF 검색)", ("RAG 사용", "사용 안 함"), index=0)
        st.session_state.rag_enabled = rag_on == "RAG 사용"

        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기"):
            if not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    if not st.session_state.current_session_id:
                        row = (
                            sb.table("chat_sessions")
                            .insert({"title": "새 세션", "user_id": user_id})
                            .execute()
                        )
                        st.session_state.current_session_id = row.data[0]["id"]
                    sid = st.session_state.current_session_id
                    names, n_chunks = _process_pdf_uploads(
                        sb, list(uploads), sid, user_id
                    )
                    merged = list(
                        dict.fromkeys(st.session_state.processed_names + names)
                    )
                    st.session_state.processed_names = merged
                    _save_session_files(sb, sid, merged)
                    if llm_for_save:
                        _auto_save_current(sb, user_id, llm_for_save)
                    st.success(f"PDF 처리 완료 ({n_chunks}개 청크 저장).")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        sid = st.session_state.current_session_id
        vec_count = len(_list_vector_file_names(sb, sid, user_id)) if sid else 0
        st.text(
            f"모델: {MODEL_NAME}\n"
            f"사용자: {login_id}\n"
            f"현재 세션 ID: {sid or '(없음)'}\n"
            f"처리된 PDF: {len(st.session_state.processed_names)}\n"
            f"벡터 DB 파일 수: {vec_count}\n"
            f"대화 메시지 수: {len(st.session_state.chat_history)}"
        )


def main() -> None:
    st.set_page_config(page_title="기획예산처 RAG 챗봇", page_icon="📚", layout="wide")
    _apply_config()
    _init_session_state()

    missing = _missing_keys()
    if missing:
        st.error(
            "다음 키가 설정되지 않았습니다: "
            + ", ".join(missing)
            + "\n\n로컬: `.env` · Streamlit Cloud: `st.secrets`"
            + f"\n\n`.env` 경로: `{ENV_PATH}`"
        )
        return

    sb = _get_supabase()
    if sb is None:
        st.error("Supabase 클라이언트를 초기화할 수 없습니다.")
        return

    user = st.session_state.current_user
    if not user:
        _render_auth_screen(sb)
        return

    user_id = str(user["id"])
    login_id = str(user.get("login_id", ""))

    _render_header()
    _render_sidebar(sb, user_id, login_id)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    if not st.session_state.current_session_id:
        row = (
            sb.table("chat_sessions")
            .insert({"title": "새 세션", "user_id": user_id})
            .execute()
        )
        st.session_state.current_session_id = row.data[0]["id"]

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            llm = _get_llm()
            use_rag = st.session_state.rag_enabled
            sid = st.session_state.current_session_id

            if use_rag and sid:
                vec_files = _list_vector_file_names(sb, sid, user_id)
                if not vec_files:
                    full_answer = (
                        "# 안내\n\n"
                        "RAG를 사용하려면 PDF를 업로드한 뒤 **파일 처리하기**를 눌러 주세요."
                    )
                    placeholder.markdown(remove_separators(full_answer))
                else:
                    embeddings = _get_embeddings()
                    mem_txt = _format_memory_block(
                        st.session_state.conversation_memory[:-1]
                    )
                    docs = _retrieve_documents_rpc(sb, embeddings, user_input, sid)
                    context = "\n\n".join(d.page_content for d in docs) or "(관련 문서 없음)"
                    messages = _build_rag_messages(user_input, context, mem_txt)
                    acc = ""
                    for chunk in llm.stream(messages):
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            acc += piece
                            placeholder.markdown(remove_separators(acc) + "▌")
                    full_answer = remove_separators(acc)
                    placeholder.markdown(full_answer)
                    follow = _generate_followup_section(llm, user_input, full_answer)
                    if follow:
                        full_answer += follow
                        placeholder.markdown(remove_separators(full_answer))
            else:
                mem_txt = _format_memory_block(
                    st.session_state.conversation_memory[:-1]
                )
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                msgs = [SystemMessage(content=sys), HumanMessage(content=user_input)]
                acc = ""
                for chunk in llm.stream(msgs):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        acc += piece
                        placeholder.markdown(remove_separators(acc) + "▌")
                full_answer = remove_separators(acc)
                placeholder.markdown(full_answer)

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청 처리 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

    st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
    st.session_state.conversation_memory.append(
        {"role": "assistant", "content": full_answer}
    )

    try:
        llm_auto = _get_llm()
        _auto_save_current(sb, user_id, llm_auto)
        st.session_state.session_list = _fetch_sessions(sb, user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("자동 저장 실패: %s", exc)


if __name__ == "__main__":
    main()
