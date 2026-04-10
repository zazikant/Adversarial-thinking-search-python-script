import streamlit as st
import os
import asyncio
import httpx
import logging
import time
import json
from typing import List, Optional, Any
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import threading
import nest_asyncio

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="RAG Query System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CLIPBOARD HELPER ======================
def copy_button(text: str, label: str = "📋 Copy to Clipboard"):
    """Clipboard copy that works for arbitrarily large outputs."""
    import streamlit.components.v1 as components
    import html as html_mod
    encoded = html_mod.escape(text, quote=True)
    btn_id = f"copy-btn-{abs(hash(text)) % 1000000}"
    ta_id = f"copy-ta-{abs(hash(text)) % 1000000}"
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

# ====================== SCHEMAS ======================
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    rank: int

class WebPageContent(BaseModel):
    url: str
    html: str
    text: str
    fetched_at: float

# ====================== EXCEPTIONS ======================
class SerperAuthError(Exception): pass
class SerperRateLimit(Exception): pass
class SerperTimeout(Exception): pass
class SerperNetworkError(Exception): pass

class BrowserlessAuthError(Exception): pass
class BrowserlessRateLimit(Exception): pass
class BrowserlessTimeout(Exception): pass
class BrowserlessNetworkError(Exception): pass

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# ====================== SIDEBAR - API KEYS ======================
with st.sidebar:
    st.header("⚙️ API Configuration")
    
    st.markdown("### 🔑 API Keys")
    
    # Initialize session state
    if "keys_saved" not in st.session_state:
        st.session_state["keys_saved"] = False
    
    # NVIDIA Key
    nvidia_key = st.text_input(
        "NVIDIA API Key",
        type="password",
        value=st.session_state.get("saved_nvidia_key", ""),
        help="NVIDIA NIM API key for LLM calls"
    )
    
    # Serper Key
    serper_key = st.text_input(
        "Serper API Key", 
        type="password",
        value=st.session_state.get("saved_serper_key", ""),
        help="Serper.dev API key for Google search"
    )
    
    # Browserless Key
    browserless_key = st.text_input(
        "Browserless API Key",
        type="password",
        value=st.session_state.get("saved_browserless_key", ""),
        help="Browserless.io API key for headless Chrome"
    )
    
    # Apply Button
    if st.button("✅ Apply API Keys", type="primary", use_container_width=True):
        st.session_state["saved_nvidia_key"] = nvidia_key
        st.session_state["saved_serper_key"] = serper_key
        st.session_state["saved_browserless_key"] = browserless_key
        st.session_state["keys_saved"] = True
        st.success("✅ Keys applied!")
        st.rerun()
    
    st.divider()
    
    # Show status
    if st.session_state.get("keys_saved"):
        st.success("🔑 Keys configured")
    else:
        st.warning("⚠️ Keys not applied")
    
    st.divider()
    
    # Model Selection
    st.markdown("### 🤖 LLM Model")
    model_options = {
        "deepseek-ai/deepseek-v3.1": "deepseek-ai/deepseek-v3.1",
        "google/gemma-2-9b-it": "google/gemma-2-9b-it",
        "meta/llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
        "mistralai/mistral-small-24b-instruct": "mistralai/mistral-small-24b-instruct",
        "Qwen/Qwen2.5-7b-instruct": "qwen/qwen2.5-7b-instruct",
        "Custom Model": "custom"
    }
    selected_model = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0,
        help="Select the LLM model to use"
    )
    
    if selected_model == "Custom Model":
        model_name = st.text_input("Enter Custom Model Name", value="deepseek-ai/deepseek-v3.1")
    else:
        model_name = model_options[selected_model]
    
    st.divider()
    
    # Search Settings
    st.markdown("### 🔍 Search Settings")
    max_results = st.slider("Max Search Results", 5, 20, 10)
    max_pages = st.slider("Max Pages to Fetch", 3, 10, 5)
    
    st.divider()
    st.caption("💡 All keys are stored in session only. Not saved to disk.")

# ====================== GET SAVED KEYS ======================
nvidia_key = st.session_state.get("saved_nvidia_key", "")
serper_key = st.session_state.get("saved_serper_key", "")
browserless_key = st.session_state.get("saved_browserless_key", "")

# ====================== UTILITY FUNCTIONS ======================
def llm(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """Call NVIDIA LLM API."""
    if not nvidia_key:
        return "ERROR: No NVIDIA API key configured"
    
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {nvidia_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-120b",  # Use full path
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        if response.status_code == 404:
            return f"ERROR: Model not found. Available models: openai/gpt-oss-120b, deepseek-ai/deepseek-v3.1, etc."
        response.raise_for_status()
        data = response.json()
        
        # Handle reasoning models that may return content in reasoning field
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning") or ""
        return content
    except Exception as e:
        log.error(f"[llm] Error: {e}")
        return f"ERROR: {str(e)}"

async def serper_search_async(query: str, max_results: int = 10) -> List[SearchResult]:
    """Async Serper search."""
    if not serper_key:
        return []
    
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": max_results, "gl": "us", "hl": "en", "autocorrect": True}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        results = []
        for i, item in enumerate(data.get("organic", [])[:max_results], start=1):
            results.append(SearchResult(
                title=item.get("title", "No title"),
                url=item.get("link", ""),
                snippet=item.get("snippet", "")[:500],
                rank=i
            ))
        return results
    except Exception as e:
        log.error(f"[serper_search] Error: {e}")
        return []

async def browserless_fetch_async(url: str, timeout: int = 60) -> Optional[WebPageContent]:
    """Async Browserless fetch."""
    if not browserless_key:
        return None
    
    endpoint = f"https://chrome.browserless.io/content?token={browserless_key}"
    payload = {"url": url}
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json=payload)
        
        if resp.status_code == 401:
            return None
        if resp.status_code == 429:
            return None
        if resp.status_code != 200:
            return None
        
        html = resp.text
        if len(html.encode()) > 5 * 1024 * 1024:
            html = html[:5 * 1024 * 1024]
        
        import re
        html_clean = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html_clean = re.sub(r'<style[^>]*>.*?</style>', '', html_clean, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', html_clean)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return WebPageContent(url=url, html=html, text=text, fetched_at=time.time())
    except Exception as e:
        log.error(f"[browserless_fetch] Error for {url}: {e}")
        return None

async def search_node_async(query: str, max_results: int = 10) -> List[SearchResult]:
    """Simplify query for better results."""
    simplified_query = query
    if len(query) > 100:
        keywords = query.replace("?", "").replace(".", "").replace(",", " ").split()
        important = [w for w in keywords if any(k in w.lower() for k in ['dspy', 'mipro', 'gepa', 'optimizer', 'rag', 'instruction', 'drift', 'gpt', 'llama', 'bayesian', 'genetic', 'pareto', 'supabase', 'nextjs', 'search', 'database'])]
        if len(important) >= 3:
            simplified_query = " ".join(important[:10])
    
    return await serper_search_async(simplified_query, max_results)

async def browser_node_async(urls: List[str], max_pages: int = 5) -> List[WebPageContent]:
    """Fetch multiple URLs in parallel."""
    semaphore = asyncio.Semaphore(5)
    
    async def fetch_with_limit(url: str):
        async with semaphore:
            for attempt in range(3):
                result = await browserless_fetch_async(url)
                if result:
                    return result
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            return None
    
    tasks = [fetch_with_limit(url) for url in urls[:max_pages]]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

def classify_intent(query: str) -> str:
    """Classify query intent."""
    prompt = f"""Classify this query as 'search' or 'browse'. 
    - 'search' if it's a question needing information lookup
    - 'browse' if it requires visiting specific URLs
    
    Query: {query}
    
    Respond with just one word: search or browse"""
    
    result = llm(prompt, max_tokens=50).strip().lower()
    if "search" in result:
        return "search"
    elif "browse" in result:
        return "browse"
    return "unknown"

def strategy_quick(search_results: List[SearchResult]) -> str:
    """Quick strategy using snippets only."""
    snippets = [r.snippet for r in search_results[:5]]
    context = "\n".join(snippets)
    
    prompt = f"""Answer this query using ONLY the snippets below. Be concise.

Query: {user_query}

Snippets:
{context}

Provide a clear, direct answer."""
    
    return llm(prompt, max_tokens=1000)

def strategy_deep(page_contents: List[WebPageContent], query: str) -> str:
    """Deep strategy using full page content."""
    if not page_contents:
        return "No pages fetched"
    
    full_text = []
    for p in page_contents:
        text = p.text[:5000] if len(p.text) > 5000 else p.text
        full_text.append(text)
    
    context = "\n\n---PAGE BREAK---\n\n".join(full_text)[:15000]
    
    prompt = f"""Answer this query using the detailed content from multiple web pages.
Be thorough and cite specific information.

Query: {query}

Content:
{context}

Provide a detailed, well-structured answer."""
    
    return llm(prompt, max_tokens=2000)

def strategy_hybrid(search_results: List[SearchResult], page_contents: List[WebPageContent], query: str) -> str:
    """Hybrid strategy combining snippets and pages."""
    snippets = [r.snippet for r in search_results[:5]]
    snippet_context = "\n".join(f"{i+1}. {s}" for i, s in enumerate(snippets))
    
    page_samples = []
    for p in page_contents[:5]:
        text = p.text[:2000] if len(p.text) > 2000 else p.text
        page_samples.append(text)
    page_context = "\n\n".join(page_samples)
    
    prompt = f"""Combine quick facts (snippets) with detailed context (pages) to answer.
Be balanced - not too brief but not overly verbose.

Query: {query}

SNIPPETS:
{snippet_context}

DETAILED CONTENT:
{page_context}

Provide a balanced, informative answer."""
    
    return llm(prompt, max_tokens=1500)

def adversarial_judge(quick: str, deep: str, hybrid: str, query: str) -> str:
    """Synthesize best answer with critique."""
    prompt = f"""You are an adversarial judge evaluating three different answer strategies.
Critique each one honestly, then synthesize the BEST possible answer.

Query: {query}

STRATEGY OUTPUTS:
---
[QUICK - from snippets only]:
{quick}

[DEEP - from full pages]:
{deep}

[HYBRID - combined]:
{hybrid}
---

Your response MUST follow this format:
CRITIQUE:
- Quick: [honest critique]
- Deep: [honest critique]  
- Hybrid: [honest critique]

FINAL ANSWER:
[Synthesize the best elements from all strategies into a superior answer]"""
    
    return llm(prompt, max_tokens=3000)

# ====================== MAIN APP ======================
st.title("🔍 RAG Query System")
st.markdown("Multi-strategy adversarial RAG pipeline with search, browse, and AI synthesis")

# Query Input
user_query = st.text_area(
    "Enter your query:",
    height=100,
    placeholder="What is DSPy GEPA optimizer in 2026?"
)

# Run Button
if st.button("🚀 Run Query", type="primary", use_container_width=True):
    if not user_query.strip():
        st.error("Please enter a query")
    elif not nvidia_key or not serper_key or not browserless_key:
        st.error("Please configure all API keys in the sidebar")
    else:
        with st.spinner("Running RAG pipeline..."):
            try:
                # Step 1: Classify
                st.info("📊 Classifying intent...")
                intent = classify_intent(user_query)
                st.success(f"Intent: {intent}")
                
                # Step 2: Search
                st.info("🔍 Searching...")
                search_results = asyncio.run(search_node_async(user_query, max_results))
                st.success(f"Found {len(search_results)} results")
                
                # Step 3: Browse (if needed)
                page_contents = []
                if intent == "search" or intent == "unknown":
                    st.info("🌐 Fetching pages...")
                    urls = [r.url for r in search_results[:max_pages]]
                    page_contents = asyncio.run(browser_node_async(urls, max_pages))
                    st.success(f"Fetched {len(page_contents)} pages")
                
                # Step 4: Run strategies
                st.info("🧠 Generating answers...")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### Quick (Snippets)")
                    quick_answer = strategy_quick(search_results)
                    st.text_area("", quick_answer, height=200, key="quick")
                
                with col2:
                    st.markdown("### Deep (Full Pages)")
                    deep_answer = strategy_deep(page_contents, user_query)
                    st.text_area("", deep_answer, height=200, key="deep")
                
                with col3:
                    st.markdown("### Hybrid (Combined)")
                    hybrid_answer = strategy_hybrid(search_results, page_contents, user_query)
                    st.text_area("", hybrid_answer, height=200, key="hybrid")
                
                # Step 5: Adversarial Judge
                st.info("⚖️ Synthesizing final answer...")
                final_answer = adversarial_judge(quick_answer, deep_answer, hybrid_answer, user_query)
                
                st.divider()
                st.markdown("### 🎯 FINAL SYNTHESIZED ANSWER")
                st.text_area("", final_answer, height=400, key="final")
                
                # Copy button
                copy_button(final_answer, "📋 Copy Final Answer")
                
                # Full output
                with st.expander("📄 View Full Output (with critique)"):
                    st.code(final_answer, language="markdown")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")

# ====================== FOOTER ======================
st.divider()
st.caption("🔧 Multi-Strategy RAG System | Powered by NVIDIA NIM + Serper + Browserless")