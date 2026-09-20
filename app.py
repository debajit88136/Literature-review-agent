import logging
import os
import re

import streamlit as st

from lit_review.demo_guard import DemoGuard
from lit_review.llm import GeminiClient, LLMError
from lit_review.pipeline import DEFAULT_PAPERS, MAX_PAPERS, MIN_PAPERS, PipelineError, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_DEMO_RUNS_PER_DAY = 10
DEFAULT_DEMO_MAX_PAPERS = 6

st.set_page_config(page_title="Autonomous Literature Review Agent", page_icon="📚", layout="wide")


def read_setting(name: str) -> str:
    try:
        value = st.secrets[name]
    except Exception:
        value = os.environ.get(name)
    return str(value).strip() if value else ""


def read_int_setting(name: str, default: int) -> int:
    try:
        value = int(read_setting(name))
    except ValueError:
        return default
    return value if value > 0 else default


@st.cache_resource
def get_guard(max_runs_per_day: int) -> DemoGuard:
    return DemoGuard(max_runs_per_day)


def file_slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:50] or "review"


def check_api_key(llm: GeminiClient) -> None:
    try:
        llm.complete("You are a health check.", 'Reply with exactly this JSON: {"ok": true}')
    except LLMError as e:
        raise PipelineError(f"Could not use the Gemini API: {e}") from e


def run_review(topic: str, num_papers: int, api_key: str, model: str):
    bar = st.progress(0.0, text="Starting...")

    def on_progress(fraction: float, message: str) -> None:
        bar.progress(fraction, text=message)

    try:
        llm = GeminiClient(api_key, model)
        check_api_key(llm)
        return run_pipeline(topic, llm, num_papers=num_papers, model_name=model, progress=on_progress)
    except (ValueError, PipelineError) as e:
        st.error(str(e))
    except Exception:
        logger.exception("Unexpected failure while running the review")
        st.error("Something unexpected went wrong. Please try again in a minute.")
    finally:
        bar.empty()
    return None


server_key = read_setting("GEMINI_API_KEY")
model = read_setting("GEMINI_MODEL") or DEFAULT_MODEL
demo_max_papers = max(MIN_PAPERS + 1, read_int_setting("DEMO_MAX_PAPERS", DEFAULT_DEMO_MAX_PAPERS))
guard = get_guard(read_int_setting("DEMO_MAX_RUNS_PER_DAY", DEFAULT_DEMO_RUNS_PER_DAY))

st.title("📚 Autonomous Literature Review Agent")
st.caption(
    "Enter a research topic. The agent searches arXiv, reads the papers, groups them into themes, "
    "finds recurring open problems and writes a cited literature review."
)

with st.sidebar:
    st.header("Settings")
    key_label = "Your Gemini API key (optional)" if server_key else "Gemini API key"
    user_key = st.text_input(key_label, type="password").strip()
    demo_mode = bool(server_key) and not user_key
    max_papers = min(MAX_PAPERS, demo_max_papers) if demo_mode else MAX_PAPERS
    num_papers = st.slider("Number of papers", MIN_PAPERS, max_papers, min(DEFAULT_PAPERS, max_papers))
    if demo_mode:
        st.info(
            f"Demo mode: up to {max_papers} papers, {guard.runs_left_today} demo runs left today "
            "(shared by all visitors). Paste your own free Gemini API key above to remove these limits."
        )
    elif user_key:
        st.success("Using your own key: no demo limits.")
    else:
        st.caption("Get a free key at aistudio.google.com/apikey. It is used for this session only and is not saved by this app.")
    st.caption(f"Model: {model}")

topic = st.text_input("Research topic", placeholder="e.g. LLM hallucination detection", max_chars=200)
run_clicked = st.button("Run", type="primary")
st.caption("A review takes roughly 20 seconds per paper on the free tier.")

if run_clicked:
    api_key = user_key or server_key
    if not topic.strip():
        st.warning("Please enter a research topic.")
    elif not api_key:
        st.warning("Please enter your Gemini API key in the sidebar.")
    else:
        st.session_state.pop("result", None)
        started, reason = guard.try_start() if demo_mode else (True, "")
        if not started:
            st.warning(reason)
        else:
            try:
                outcome = run_review(topic.strip(), min(num_papers, max_papers), api_key, model)
            finally:
                if demo_mode:
                    guard.finish()
            if outcome is not None:
                st.session_state["result"] = outcome

result = st.session_state.get("result")
if result is not None:
    analysed = len(result.notes) - result.num_failed
    cols = st.columns(4)
    cols[0].metric("Candidates searched", result.num_candidates)
    cols[1].metric("Papers analysed", analysed)
    cols[2].metric("Read from PDF", result.num_full_text)
    cols[3].metric("Themes", len(result.themes))
    if result.num_failed:
        st.warning(f"{result.num_failed} paper(s) could not be analysed and are listed separately in the review.")

    st.download_button(
        "⬇️ Download Markdown",
        data=result.markdown,
        file_name=f"literature_review_{file_slug(result.topic)}.md",
        mime="text/markdown",
        type="primary",
    )
    tab_review, tab_source = st.tabs(["Review", "Markdown source"])
    with tab_review:
        st.markdown(result.markdown)
    with tab_source:
        st.code(result.markdown, language="markdown")

with st.expander("How this works and what to keep in mind"):
    st.markdown(
        """
1. **Search:** finds candidate papers on arXiv.
2. **Select:** an LLM picks the papers that best fit a general review, from titles and abstracts.
3. **Read:** downloads each PDF and extracts its text. If a PDF fails, it falls back to the abstract.
4. **Extract:** an LLM writes structured notes for each paper (method, findings, datasets, limitations).
5. **Synthesise:** groups papers into themes and finds problems raised by at least two papers.
6. **Write:** assembles the Markdown review. Citations come from arXiv metadata, never from the LLM.

**Limitations:** summaries are written by an LLM and can contain errors, so verify important claims
against the papers. Long papers are trimmed to their start and end. Results can differ between runs.
"""
    )
