import streamlit as st
import asyncio
import nest_asyncio
nest_asyncio.apply()

import os
import time
import httpx
import logging
import operator
from typing import Any, Dict, List, Literal, Optional, Annotated, TypedDict
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CLIPBOARD HELPER  (from sample — works for arbitrarily large outputs)
# ─────────────────────────────────────────────────────────────────────────────
def copy_button(text: str, label: str = "📋 Copy to Clipboard"):
    """Full text written into a hidden <textarea> at render time.
    JS reads from the DOM node — not a JS string literal — so there is
    no browser template-literal size cap or Streamlit iframe serialization
    truncation regardless of output length."""
    import streamlit.components.v1 as components
    import html as html_mod
    encoded = html_mod.escape(text, quote=True)
    btn_id = f"copy-btn-{abs(hash(text)) % 1000000}"
    ta_id  = f"copy-ta-{abs(hash(text)) % 1000000}"
    components.html(
        f"""
        <textarea id="{ta_id}"
            style="position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;"
        >{encoded}</textarea>
        <button id="{btn_id}"
            style="background:#1f77b4;color:white;border:none;padding:10px 20px;
                   border-radius:6px;font-size:14px;cursor:pointer;width:100%;margin-top:4px;">
            {label}
        </button>
        <script>
        (function() {{
            var btn = document.getElementById('{btn_id}');
            var ta  = document.getElementById('{ta_id}');
            btn.addEventListener('click', function() {{
                var txt = ta.value;
                function markDone() {{
                    btn.innerText = '✅ Copied!';
                    btn.style.background = '#2d6a2d';
                    setTimeout(function() {{
                        btn.innerText = '{label}';
                        btn.style.background = '#1f77b4';
                    }}, 2000);
                }}
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(txt).then(markDone).catch(function() {{
                        ta.style.cssText = 'position:static;width:100%;height:2px;';
                        ta.select();
                        document.execCommand('copy');
                        ta.style.cssText = 'position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;';
                        markDone();
                    }});
                }} else {{
                    ta.style.cssText = 'position:static;width:100%;height:2px;';
                    ta.select();
                    document.execCommand('copy');
                    ta.style.cssText = 'position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;';
                    markDone();
                }}
            }});
        }})();
        </script>
        """,
        height=60,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adversarial Search Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]        { background:#111 !important; color:#e0e0e0; }
[data-testid="stSidebar"]                 { background:#1a1a1a !important; border-right:1px solid #2d2d2d; }
[data-testid="stSidebar"] label           { color:#aaa !important; font-size:0.82rem; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stTextArea textarea {
    background:#252525 !important; color:#e0e0e0 !important;
    border:1px solid #3a3a3a !important; border-radius:6px;
}
[data-testid="stSidebar"] .stButton button {
    background:#e63b2e !important; color:#fff !important;
    border:none; border-radius:6px; font-weight:600; width:100%;
}
[data-testid="stSidebar"] .stButton button:hover { background:#c0302a !important; }
.main-header { font-size:2rem; font-weight:700; color:#f0f0f0; margin-bottom:0.2rem; }
.sub-header  { color:#888; font-size:0.9rem; margin-bottom:1.5rem; }
.result-box  {
    background:#1e1e1e; border:1px solid #2d2d2d; border-radius:8px;
    padding:1.2rem 1.4rem; margin-top:0.6rem;
}
.result-box h4 { color:#e63b2e; margin:0 0 0.5rem; font-size:0.85rem; letter-spacing:.06em; text-transform:uppercase; }
.result-box p  { color:#d0d0d0; line-height:1.65; margin:0; white-space:pre-wrap; }
.badge { display:inline-block; padding:2px 10px; border-radius:4px; font-size:0.75rem; font-weight:600; margin-bottom:0.8rem; }
.badge-search  { background:#1d3a2f; color:#4caf8a; }
.badge-browse  { background:#1e2e3a; color:#5aafdf; }
.badge-unknown { background:#2d2d1e; color:#c9b84a; }
.error-box { background:#2a1a1a; border:1px solid #5c2020; border-radius:8px; padding:1rem 1.2rem; margin-top:0.6rem; color:#e07070; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    st.markdown("### 🔑 API Keys")
    nvidia_key      = st.text_input("NVIDIA API Key",      type="password", placeholder="nvapi-…")
    serper_key      = st.text_input("Serper API Key",      type="password", placeholder="…")
    browserless_key = st.text_input("Browserless API Key", type="password", placeholder="…")
    keys_applied    = st.button("✅  Apply API Keys")

    st.divider()
    st.markdown("### 🤖 Model")
    _nvidia_models = {
        "GPT OSS 120B (Default)":    "openai/gpt-oss-120b",
        "Kimi K2.5 (32k output)":    "moonshotai/kimi-k2.5",
        "MiniMax M2.5 (32k output)": "minimaxai/minimax-m2.5",
    }
    selected_model_label = st.selectbox(
        "Select Model",
        options=list(_nvidia_models.keys()),
        index=0,
        help="All NVIDIA NIM models support up to 32,000 output tokens.",
    )
    model_name = _nvidia_models[selected_model_label]
    st.caption("⚡ 32,000 output token limit · Hosted on NVIDIA infrastructure")
    if st.button("✅ Apply Model", type="primary", use_container_width=True):
        st.success(f"✅ Model set to: {selected_model_label}")
    max_tokens  = st.slider("Max Tokens",   512, 8192, 4096, step=256)
    temperature = st.slider("Temperature",  0.0,  2.0,  1.0, step=0.1)

    st.divider()
    st.markdown("### ⚙️ Pipeline")
    max_search_results = st.slider("Max Search Results", 3, 20, 10)
    browser_top_n      = st.slider("Browser Top-N URLs", 1, 10,  3)

# Apply keys to env
if keys_applied or (nvidia_key and serper_key and browserless_key):
    if nvidia_key:      os.environ["NVIDIA_API_KEY"]      = nvidia_key
    if serper_key:      os.environ["SERPER_API_KEY"]      = serper_key
    if browserless_key: os.environ["BROWSERLESS_API_KEY"] = browserless_key

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS  (Cell 4 — exact)
# ─────────────────────────────────────────────────────────────────────────────
class SearchResult(BaseModel):
    title:   str   = Field(..., min_length=1)
    url:     str   = Field(...)
    snippet: str   = Field(..., max_length=500)
    rank:    int   = Field(..., ge=1)

class WebPageContent(BaseModel):
    url:        str   = Field(...)
    html:       str   = Field(...)
    text:       str   = Field(...)
    fetched_at: float = Field(..., description="epoch seconds")

def merge_dicts(left: dict, right: dict) -> dict:
    result = dict(left) if left else {}
    if right:
        result.update(right)
    return result

class GraphState(TypedDict):
    user_query:       str
    intent:           str
    search_results:   list
    page_contents:    list
    aggregated_text:  str
    strategy_outputs: Annotated[dict, merge_dicts]
    final_answer:     str
    errors:           Annotated[list, operator.add]

# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS  (Cell 5 — exact)
# ─────────────────────────────────────────────────────────────────────────────
class SerperAuthError(Exception): pass
class SerperRateLimit(Exception): pass
class SerperTimeout(Exception): pass
class SerperNetworkError(Exception): pass
class BrowserlessAuthError(Exception): pass
class BrowserlessRateLimit(Exception): pass
class BrowserlessTimeout(Exception): pass
class BrowserlessNetworkError(Exception): pass
class ReducerError(Exception): pass
class ClassificationError(Exception): pass
class LLMError(Exception): pass

# ─────────────────────────────────────────────────────────────────────────────
# SERPER CLIENT  (Cell 6 — exact)
# ─────────────────────────────────────────────────────────────────────────────
async def serper_search(query: str, *, max_results: int = 10) -> List[SearchResult]:
    if not query.strip():
        raise ValueError("query must be non-empty")

    url     = "https://google.serper.dev/search"
    headers = {"X-API-KEY": os.environ.get("SERPER_API_KEY", ""), "Content-Type": "application/json"}
    payload = {"q": query, "num": max_results}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code == 401: raise SerperAuthError(f"Serper auth failed: {resp.text}")
        if resp.status_code == 429: raise SerperRateLimit("Serper rate limit hit")
        resp.raise_for_status()

        data    = resp.json()
        results = []
        for i, item in enumerate(data.get("organic", [])[:max_results], start=1):
            try:
                results.append(SearchResult(
                    title=item.get("title", "No title"),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", "")[:500],
                    rank=i,
                ))
            except Exception as e:
                log.warning(f"Skipping malformed search result: {e}")
        return results

    except httpx.TimeoutException:
        raise SerperTimeout("Serper request timed out")
    except (SerperAuthError, SerperRateLimit):
        raise
    except Exception as e:
        raise SerperNetworkError(f"Serper network error: {e}") from e

# ─────────────────────────────────────────────────────────────────────────────
# BROWSERLESS CLIENT  (Cell 7 — exact)
# ─────────────────────────────────────────────────────────────────────────────
async def browserless_fetch(url: str, *, timeout: int = 30) -> WebPageContent:
    endpoint = f"https://chrome.browserless.io/content?token={os.environ.get('BROWSERLESS_API_KEY', '')}"
    payload  = {"url": url}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json=payload)
        if resp.status_code == 401: raise BrowserlessAuthError(f"Browserless auth failed for {url}")
        if resp.status_code == 429: raise BrowserlessRateLimit(f"Browserless rate limit for {url}")
        resp.raise_for_status()

        html = resp.text
        if len(html.encode()) > 5 * 1024 * 1024:
            html = html[:5 * 1024 * 1024]
            log.warning(f"HTML truncated at 5MB for {url}")

        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        return WebPageContent(url=url, html=html, text=text, fetched_at=time.time())

    except httpx.TimeoutException:
        raise BrowserlessTimeout(f"Browserless timed out for {url}")
    except (BrowserlessAuthError, BrowserlessRateLimit):
        raise
    except Exception as e:
        raise BrowserlessNetworkError(f"Browserless network error for {url}: {e}") from e

# ─────────────────────────────────────────────────────────────────────────────
# REDUCERS  (Cell 8 — exact)
# ─────────────────────────────────────────────────────────────────────────────
def reducer_max_confidence(prev: List[SearchResult], incoming: List[SearchResult]) -> List[SearchResult]:
    try:
        combined: Dict[str, SearchResult] = {}
        for r in prev + incoming:
            key = str(r.url)
            if key not in combined or r.rank < combined[key].rank:
                combined[key] = r
        sorted_results = sorted(combined.values(), key=lambda x: x.rank)
        return [
            SearchResult(title=r.title, url=r.url, snippet=r.snippet, rank=i+1)
            for i, r in enumerate(sorted_results)
        ]
    except Exception as e:
        raise ReducerError(f"max_confidence merge failed: {e}") from e

def reducer_append_unique(prev: List[WebPageContent], incoming: List[WebPageContent]) -> List[WebPageContent]:
    try:
        seen   = {str(p.url) for p in prev}
        merged = list(prev)
        for item in incoming:
            if str(item.url) not in seen:
                merged.append(item)
                seen.add(str(item.url))
        return merged
    except Exception as e:
        raise ReducerError(f"append_unique merge failed: {e}") from e

def reducer_merge(
    prev:     Dict[str, Any],
    incoming: Dict[str, Any],
    key:      str,
    strategy: Literal["max_confidence", "append_unique", "overwrite"],
) -> Dict[str, Any]:
    result = dict(prev)
    try:
        if strategy == "max_confidence":
            result[key] = reducer_max_confidence(prev.get(key, []), incoming.get(key, []))
        elif strategy == "append_unique":
            result[key] = reducer_append_unique(prev.get(key, []), incoming.get(key, []))
        elif strategy == "overwrite":
            result[key] = incoming.get(key, prev.get(key))
        else:
            raise ReducerError(f"Unknown strategy: {strategy}")
    except ReducerError:
        raise
    except Exception as e:
        raise ReducerError(f"reducer_merge failed on key={key}: {e}") from e
    return result

# ─────────────────────────────────────────────────────────────────────────────
# NVIDIA LLM  (Cell 9 — exact, uses sidebar settings)
# ─────────────────────────────────────────────────────────────────────────────
def call_nvidia_llm(prompt: str, max_tokens: int = 4096,
                    _model: str = "openai/gpt-oss-120b", _temperature: float = 1.0) -> str:
    from openai import OpenAI
    nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ.get("NVIDIA_API_KEY", ""),
    )
    completion = nvidia_client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=_temperature,
        top_p=1,
        max_tokens=max_tokens,
        stream=True,
    )

    full_response      = []
    reasoning_response = []

    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_response.append(reasoning)

        content = delta.content
        if content is not None:
            full_response.append(content)

    content_final   = "".join(full_response).strip()
    reasoning_final = "".join(reasoning_response).strip()

    log.info(f"[llm] content='{repr(content_final)}' | reasoning_len={len(reasoning_final)}")
    return content_final if content_final else reasoning_final

# ─────────────────────────────────────────────────────────────────────────────
# NODES  (Cells 10-15 — exact)
# ─────────────────────────────────────────────────────────────────────────────
def classifier_node(state: dict) -> dict:
    query  = state["user_query"]
    prompt = f"""Classify this query into one category: search, browse, or unknown.

User query: {query}

Reply with ONE word only: search, browse, or unknown."""

    for attempt in range(3):
        try:
            raw = call_nvidia_llm(
                prompt, max_tokens=500, _model=model_name, _temperature=temperature
            ).strip().lower()
            log.info(f"[ClassifierNode] raw='{raw}'")
            intent_raw = "unknown"
            for word in reversed(raw.split()):
                cleaned = word.strip(".,!?\"':-\n")
                if cleaned in ("search", "browse", "unknown"):
                    intent_raw = cleaned
                    break
            log.info(f"[ClassifierNode] intent={intent_raw}")
            return {"intent": intent_raw}
        except Exception as e:
            log.warning(f"[ClassifierNode] attempt {attempt+1} failed: {e}")
            if attempt == 2:
                return {"intent": "unknown", "errors": state.get("errors", []) + [str(e)]}


def search_node(state: dict) -> dict:
    query  = state["user_query"]
    errors = list(state.get("errors", []))

    async def _run():
        for attempt in range(4):
            try:
                results = await serper_search(query, max_results=max_search_results)
                log.info(f"[SearchNode] Got {len(results)} results")
                return {"search_results": [r.model_dump() for r in results]}
            except SerperAuthError as e:
                log.error(f"[SearchNode] Auth error: {e}")
                errors.append(str(e))
                return {"search_results": [], "errors": errors}
            except SerperRateLimit as e:
                wait = 2 ** attempt
                log.warning(f"[SearchNode] Rate limit, waiting {wait}s (attempt {attempt+1})")
                await asyncio.sleep(wait)
                if attempt == 3:
                    errors.append(str(e))
                    return {"search_results": [], "errors": errors}
            except SerperTimeout as e:
                log.warning(f"[SearchNode] Timeout: {e} — returning empty results")
                errors.append(str(e))
                return {"search_results": [], "errors": errors}
            except SerperNetworkError as e:
                log.error(f"[SearchNode] Network error: {e}")
                errors.append(str(e))
                return {"search_results": [], "errors": errors}

    return asyncio.get_event_loop().run_until_complete(_run())


BROWSER_SEMAPHORE = asyncio.Semaphore(10)

def browser_node(state: dict) -> dict:
    search_results = state.get("search_results", [])
    errors         = list(state.get("errors", []))

    if not search_results:
        log.warning("[BrowserNode] No search results to browse")
        return {"page_contents": [], "errors": errors}

    results_parsed = [
        SearchResult(**r) if isinstance(r, dict) else r
        for r in search_results
    ]
    top_urls = [str(r.url) for r in sorted(results_parsed, key=lambda x: x.rank)[:browser_top_n]]

    async def fetch_with_guard(url: str):
        async with BROWSER_SEMAPHORE:
            for attempt in range(3):
                try:
                    content = await browserless_fetch(url)
                    return content
                except BrowserlessAuthError as e:
                    log.error(f"[BrowserNode] Auth error for {url}: {e}")
                    errors.append(str(e))
                    return None
                except BrowserlessRateLimit:
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                except BrowserlessTimeout as e:
                    log.warning(f"[BrowserNode] Timeout dropping {url}: {e}")
                    errors.append(str(e))
                    return None
                except MemoryError:
                    log.error(f"[BrowserNode] OOM fetching {url}")
                    errors.append(f"OOM for {url}")
                    return None
                except BrowserlessNetworkError as e:
                    log.error(f"[BrowserNode] Network error {url}: {e}")
                    errors.append(str(e))
                    return None
            return None

    async def _run():
        fetched = await asyncio.gather(*[fetch_with_guard(u) for u in top_urls])
        pages   = [f.model_dump() for f in fetched if f is not None]
        log.info(f"[BrowserNode] Fetched {len(pages)}/{len(top_urls)} pages")
        return {"page_contents": pages, "errors": errors}

    return asyncio.get_event_loop().run_until_complete(_run())


def reducer_node_search(state: dict) -> dict:
    try:
        prev   = [SearchResult(**r) if isinstance(r, dict) else r for r in state.get("search_results", [])]
        merged = reducer_max_confidence(prev, [])
        log.info(f"[ReducerNode:search] Merged → {len(merged)} results")
        return {"search_results": [r.model_dump() for r in merged]}
    except ReducerError as e:
        log.error(f"[ReducerNode:search] ReducerError — keeping prev state: {e}")
        state.get("errors", []).append(str(e))
        return {}


def reducer_node_browse(state: dict) -> dict:
    try:
        prev   = [WebPageContent(**p) if isinstance(p, dict) else p for p in state.get("page_contents", [])]
        merged = reducer_append_unique(prev, [])
        log.info(f"[ReducerNode:browse] Merged → {len(merged)} pages")
        return {"page_contents": [p.model_dump() for p in merged]}
    except ReducerError as e:
        log.error(f"[ReducerNode:browse] ReducerError — keeping prev state: {e}")
        state.get("errors", []).append(str(e))
        return {}


def aggregator_node(state: dict) -> dict:
    try:
        pages          = state.get("page_contents", [])
        search_results = state.get("search_results", [])
        texts = []
        for p in pages:
            text = p.get("text", "") if isinstance(p, dict) else p.text
            if text.strip():
                texts.append(text.strip())

        if not texts and search_results:
            log.info("[AggregatorNode] No page content — falling back to snippets")
            for r in search_results:
                snippet = r.get("snippet", "") if isinstance(r, dict) else r.snippet
                texts.append(snippet)

        aggregated = "\n\n".join(texts)
        aggregated = aggregated[:20000]
        log.info(f"[AggregatorNode] aggregated_text length={len(aggregated)}")
        return {"aggregated_text": aggregated}
    except Exception as e:
        log.error(f"[AggregatorNode] AggregationError: {e}")
        return {"aggregated_text": "", "errors": state.get("errors", []) + [str(e)]}


def answer_node(state: dict) -> dict:
    aggregated_text = state.get("aggregated_text", "")
    intent          = state.get("intent", "unknown")
    errors          = list(state.get("errors", []))

    def build_prompt(text: str, intent: str) -> str:
        context = text if text.strip() else "No context available."
        return f"""You are a helpful assistant. Based on the context below, answer the user's query.
Intent: {intent}
Context:
{context}

Provide a clear, concise answer in 3-5 sentences."""

    for attempt in range(2):
        try:
            prompt = build_prompt(aggregated_text, intent)
            answer = call_nvidia_llm(
                prompt, max_tokens=max_tokens, _model=model_name, _temperature=temperature
            ).strip()
            if not answer:
                raise LLMError("Empty response")
            log.info(f"[AnswerNode] Generated answer ({len(answer)} chars)")
            return {"final_answer": answer}
        except Exception as e:
            log.warning(f"[AnswerNode] attempt {attempt+1} failed: {e}")
            errors.append(str(e))
            if attempt == 1:
                return {"final_answer": "Unable to generate answer.", "errors": errors}

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER  (Cell 16 — build_adversarial_graph, exact logic)
# ─────────────────────────────────────────────────────────────────────────────
def build_adversarial_graph():
    from langgraph.graph import StateGraph, END

    builder = StateGraph(GraphState)

    builder.add_node("classifier",     classifier_node)
    builder.add_node("search",         search_node)
    builder.add_node("browser",        browser_node)
    builder.add_node("reducer_search", reducer_node_search)
    builder.add_node("reducer_browse", reducer_node_browse)
    builder.add_node("aggregator",     aggregator_node)
    builder.add_node("answer",         answer_node)

    builder.set_entry_point("classifier")

    def route_intent(state: dict) -> str:
        intent = state.get("intent", "unknown")
        if intent == "browse":
            return "search_then_browse"
        return "search"

    builder.add_conditional_edges("classifier", route_intent, {
        "search":            "search",
        "search_then_browse": "search",
    })

    def route_after_search(state: dict) -> str:
        return "browser" if state.get("intent") == "browse" else "reducer_search"

    builder.add_conditional_edges("search", route_after_search, {
        "browser":        "browser",
        "reducer_search": "reducer_search",
    })

    builder.add_edge("browser",        "reducer_browse")
    builder.add_edge("reducer_browse", "aggregator")
    builder.add_edge("reducer_search", "aggregator")
    builder.add_edge("aggregator",     "answer")
    builder.add_edge("answer",         END)

    return builder.compile()

# ─────────────────────────────────────────────────────────────────────────────
# RUN QUERY  (Cell 17 — exact)
# ─────────────────────────────────────────────────────────────────────────────
def run_query(user_query: str) -> dict:
    initial_state: GraphState = {
        "user_query":       user_query,
        "intent":           "unknown",
        "search_results":   [],
        "page_contents":    [],
        "aggregated_text":  "",
        "strategy_outputs": {},
        "final_answer":     "",
        "errors":           [],
    }
    log.info(f"[run_query] Starting ADVERSARIAL pipeline for: '{user_query}'")
    graph = build_adversarial_graph()
    result = graph.invoke(initial_state)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 Adversarial Search Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">NVIDIA NIM · LangGraph · Serper · Browserless</div>', unsafe_allow_html=True)

query = st.text_area(
    "Enter your query",
    placeholder="e.g. latest dspy optimizations 2026",
    height=100,
    label_visibility="collapsed",
)

run_btn = st.button("🚀  Run Pipeline", type="primary", use_container_width=True)

keys_missing = not (
    os.environ.get("NVIDIA_API_KEY") and
    os.environ.get("SERPER_API_KEY") and
    os.environ.get("BROWSERLESS_API_KEY")
)

if run_btn:
    if not query.strip():
        st.warning("Please enter a query.")
    elif keys_missing:
        st.error("Please enter all three API keys in the sidebar and click **✅ Apply API Keys**.")
    else:
        with st.spinner("Running adversarial pipeline…"):
            try:
                result = run_query(query.strip())
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                result = None

        if result:
            intent    = result.get("intent", "unknown")
            badge_cls = {"search": "badge-search", "browse": "badge-browse"}.get(intent, "badge-unknown")
            st.markdown(
                f'<span class="badge {badge_cls}">Intent: {intent.upper()}</span>',
                unsafe_allow_html=True,
            )

            # Strategy outputs
            strategy_outputs = result.get("strategy_outputs", {})
            if strategy_outputs:
                st.markdown("#### Strategy Outputs")
                for strategy, answer in strategy_outputs.items():
                    st.markdown(
                        f'<div class="result-box"><h4>{strategy}</h4><p>{answer}</p></div>',
                        unsafe_allow_html=True,
                    )
                    copy_button(str(answer), f"📋 Copy {strategy}")

            # Adversarial critique (if embedded in aggregated_text — Cell 17 logic)
            agg = result.get("aggregated_text", "")
            if "CRITIQUE:" in agg:
                critique_part = agg.split("FINAL ANSWER:")[0]
                with st.expander("🧠 Adversarial Judge's Critique"):
                    st.text(critique_part)

            # Final answer
            final_answer = result.get("final_answer", "")
            st.markdown(
                f'<div class="result-box"><h4>Final Synthesized Answer</h4><p>{final_answer}</p></div>',
                unsafe_allow_html=True,
            )
            if final_answer:
                copy_button(final_answer, "📋 Copy Final Answer")

            # Search results
            search_results = result.get("search_results", [])
            if search_results:
                with st.expander(f"📄 Search Results ({len(search_results)})"):
                    for r in search_results:
                        r = r if isinstance(r, dict) else r.model_dump()
                        st.markdown(f"**{r['rank']}.** [{r['title']}]({r['url']})")
                        st.caption(r["snippet"])

            # Errors
            errors = result.get("errors", [])
            if errors:
                with st.expander(f"⚠️ Errors ({len(errors)})"):
                    for err in errors:
                        st.markdown(
                            f'<div class="error-box">{err}</div>',
                            unsafe_allow_html=True,
                        )
