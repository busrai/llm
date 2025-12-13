# -*- coding: utf-8 -*-
"""
RAG + META-RAG Code Review Pipeline (Qdrant + Ollama)
- Index code segments to Qdrant (COLL_CODE)
- Index labeled review comments to Qdrant (COLL_COMMENTS)
- **NEW**: Index repository summaries (files/classes) to Qdrant (COLL_SUMMARIES)
- Evaluate LLM models with/without RAG, k in {1,3,5} for past comments
- Per-subcategory metrics: BLEU, ROUGE-1/ROUGE-L, METEOR, TF-IDF semantic, Actionability, Grounding
- LLM-as-a-Judge (e.g., Llama3-70B) for semantic correctness & usefulness
- Wide-context prompt: multiple files/classes; include actual code snippets

Usage examples:
  python rag_meta_rag_pipeline.py index-code --repo /path/to/repo
  python rag_meta_rag_pipeline.py index-comments --json labeled_comments.json
  python rag_meta_rag_pipeline.py index-summaries --repo /path/to/repo --summarizer ollama --summary_model llama3:8b-instruct
  python rag_meta_rag_pipeline.py eval --eval_json eval_data.json --comment_ks 1,3,5 --code_k 5 --meta_rag --rag --no_rag

Notes:
- No pandas used. Pure dict/list processing.
- METEOR may require NLTK corpora; if unavailable, the METEOR score is set to -1.0.
- Judge model configurable via env LLM_JUDGE_MODEL (default: llama3:70b-instruct).
"""
from __future__ import annotations
import os
import re
import ast
import json
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

# Metrics
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
try:
    from nltk.translate.meteor_score import single_meteor_score
except Exception:
    single_meteor_score = None

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.models import SearchRequest

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_meta_rag")

# Config
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLL_CODE = os.getenv("COLL_CODE", "code_segments")
COLL_COMMENTS = os.getenv("COLL_COMMENTS", "review_comments")
COLL_SUMMARIES = os.getenv("COLL_SUMMARIES", "repo_summaries")
REPO_ID = os.getenv("REPO_ID", "minddb")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "sentence")  # voyage|ollama|sentence
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
SENTENCE_MODEL = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")
STRICT_JSON = True
LLM_MODELS = [m.strip() for m in os.getenv("LLM_MODELS", "codellama:13b,codestral:22b,qwen3-coder:30b,mistral:7b-instruct,deepseek-coder-33b,phi3:latest").split(",") if m.strip()]
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "mistral:7b-instruct")

TOPK_CODE_DEFAULT = int(os.getenv("TOPK_CODE", "3"))
TOPK_COMMENTS_DEFAULT = int(os.getenv("TOPK_COMMENTS", "3"))
TOPK_SUMMARIES_DEFAULT = int(os.getenv("TOPK_SUMMARIES", "3"))

# ---------------- Embeddings -----------------
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class Embedder:
    def __init__(self, provider: str = EMBED_PROVIDER):
        self.provider = provider
        self._session = None
        self._st_model = None
        if provider == "sentence":
            if SentenceTransformer is None:
                raise RuntimeError("SentenceTransformer not available; install sentence-transformers")
            self._st_model = SentenceTransformer(SENTENCE_MODEL)
        elif provider in ("ollama", "voyage"):
            import requests
            self._session = requests.Session()
        else:
            raise ValueError(f"Unknown embed provider: {provider}")

    @property
    def dim(self) -> int:
        if self.provider == "voyage":
            return 1536
        if self.provider == "ollama":
            return 768
        if self.provider == "sentence":
            return 384
        return 768

    def embed_texts(self, texts: List[str], kind: str = "text") -> List[List[float]]:
        if self.provider == "sentence":
            return self._st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).tolist()
        if self.provider == "ollama":
            return [self._embed_ollama(t) for t in texts]
        if self.provider == "voyage":
            return self._embed_voyage(texts, kind)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _embed_ollama(self, text: str) -> List[float]:
        import requests
        payload = {"model": OLLAMA_EMBED_MODEL, "input": text}
        resp = self._session.post(f"{OLLAMA_URL}/api/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding") or (data.get("embeddings", [None])[0])
        return embedding

    def _embed_voyage(self, texts: List[str], kind: str) -> List[List[float]]:
        import requests
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {VOYAGE_API_KEY}", "Content-Type": "application/json"}
        model = "voyage-code-2" if kind == "code" else "voyage-large-2"
        resp = self._session.post(url, json={"model": model, "input": texts}, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]

# ---------------- Qdrant helpers -----------------
from types import SimpleNamespace
import requests

def get_qdrant() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_collection(client: QdrantClient, name: str, dim: int):
    cols = [c.name for c in client.get_collections().collections]
    if name not in cols:
        client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        logger.info(f"Created collection: {name}, dim={dim}")

# Compatibility search (SDK/HTTP/REST)

def safe_search(client: QdrantClient, collection_name: str, query_vector: List[float], limit: int,
                query_filter: Optional[Filter] = None, with_payload: bool = True, with_vectors: bool = False):
    def _normalize(result_list):
        norm = []
        for item in result_list or []:
            if hasattr(item, "payload") and hasattr(item, "id"):
                norm.append(item)
                continue
            if isinstance(item, dict):
                norm.append(SimpleNamespace(id=item.get("id"), payload=item.get("payload") or {},
                                            score=item.get("score"), vector=item.get("vector")))
                continue
            norm.append(SimpleNamespace(id=getattr(item, "id", None), payload=getattr(item, "payload", {}) or {},
                                        score=getattr(item, "score", None), vector=getattr(item, "vector", None)))
        return norm
    # SDK path
    if hasattr(client, "search"):
        res = client.search(collection_name=collection_name, query_vector=query_vector, limit=limit,
                            query_filter=query_filter, with_payload=with_payload, with_vectors=with_vectors)
        return _normalize(res)
    # HTTP/OpenAPI
    req = SearchRequest(vector=query_vector, limit=limit, filter=query_filter, with_payload=with_payload)
    if hasattr(client, "http") and hasattr(client.http, "points_api"):
        pa = client.http.points_api
        if hasattr(pa, "search_points"):
            resp = pa.search_points(collection_name=collection_name, search_request=req)
            items = getattr(resp, "result", resp)
            return _normalize(items)
        if hasattr(pa, "search"):
            resp = pa.search(collection_name=collection_name, search_request=req)
            items = getattr(resp, "result", resp)
            return _normalize(items)
    if hasattr(client, "openapi_client") and hasattr(client.openapi_client, "points_api"):
        pa = client.openapi_client.points_api
        if hasattr(pa, "search_points"):
            resp = pa.search_points(collection_name=collection_name, search_request=req)
            items = getattr(resp, "result", resp)
            return _normalize(items)
        if hasattr(pa, "search"):
            resp = pa.search(collection_name=collection_name, search_request=req)
            items = getattr(resp, "result", resp)
            return _normalize(items)
    # REST fallback
    base = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    url = f"{base}/collections/{collection_name}/points/search"
    body = {"vector": query_vector, "limit": limit, "with_payload": with_payload}
    if query_filter is not None:
        body["filter"] = json.loads(query_filter.model_dump_json())
    r = requests.post(url, json=body)
    r.raise_for_status()
    data = r.json()
    items = data.get("result", [])
    return _normalize(items)

# ---------------- Utils -----------------

def extract_python_segments(src: str) -> List[Dict[str, Any]]:
    segments = []
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                segments.append({"kind": "function", "symbol": node.name,
                                 "line_start": node.lineno, "line_end": getattr(node, "end_lineno", node.lineno)})
            elif isinstance(node, ast.ClassDef):
                segments.append({"kind": "class", "symbol": node.name,
                                 "line_start": node.lineno, "line_end": getattr(node, "end_lineno", node.lineno)})
    except Exception as e:
        logger.warning(f"AST parse error: {e}")
    if not segments:
        segments.append({"kind": "file", "symbol": "module", "line_start": 1, "line_end": len(src.splitlines())})
    return segments

def clip_code_by_lines(code: str, start: int, end: int) -> str:
    lines = code.splitlines()
    start = max(1, start)
    end = min(len(lines), end)
    return "\n".join(lines[start-1:end])

def extract_identifiers(code: str) -> List[str]:
    idents = set()
    try:
        tree = ast.parse(code or "")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                idents.add(node.name)
            elif isinstance(node, ast.ClassDef):
                idents.add(node.name)
            elif isinstance(node, ast.Name):
                idents.add(node.id)
            elif isinstance(node, ast.Attribute):
                idents.add(node.attr)
    except Exception:
        idents.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", code or ""))
    return [i for i in idents if len(i) >= 3 and not i.isdigit()]

# Load snippet from payload

def _load_code_snippet_from_payload(payload: Dict[str, Any], max_chars: int = 1600) -> str:
    path = payload.get("path")
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        start = int(payload.get("line_start", 1))
        end = int(payload.get("line_end", len(text.splitlines())))
        snippet = clip_code_by_lines(text, start, end)
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "\n# ... (truncated) ..."
        return snippet
    except Exception:
        return ""

# ---------------- Indexers -----------------

def index_code_repo(repo_root: Path, glob: str = "**/*.py"):
    client = get_qdrant()
    embedder = Embedder(EMBED_PROVIDER)
    dim = embedder.dim
    ensure_collection(client, COLL_CODE, dim)
    points_batch: List[PointStruct] = []
    files = list(repo_root.rglob(glob))
    logger.info(f"Files to index: {len(files)}")
    for py in files:
        try:
            code = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        segments = extract_python_segments(code)
        for seg in segments:
            seg_code = clip_code_by_lines(code, seg["line_start"], seg["line_end"])
            vec = embedder.embed_texts([seg_code], kind="code")[0]
            pid = str(uuid.uuid4())
            payload = {
                "repo_id": REPO_ID,
                "path": str(py),
                "lang": "py",
                "kind": seg["kind"],
                "symbol": seg["symbol"],
                "line_start": seg["line_start"],
                "line_end": seg["line_end"],
                "segment_id": pid,
            }
            points_batch.append(PointStruct(id=pid, vector=vec, payload=payload))
            if len(points_batch) >= 256:
                client.upsert(collection_name=COLL_CODE, points=points_batch)
                points_batch = []
    if points_batch:
        client.upsert(collection_name=COLL_CODE, points=points_batch)
    logger.info("Code indexing completed.")

# Index labeled comments

def search_segment_ids_for_anchor(client: QdrantClient, path: str, line_number: int, limit: int = 5) -> List[str]:
    flt = Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=REPO_ID)),
                       FieldCondition(key="path", match=MatchValue(value=path))])
    page = client.scroll(collection_name=COLL_CODE, scroll_filter=flt, limit=512, with_payload=True, with_vectors=False)
    seg_ids: List[str] = []
    for point in page[0]:
        p = point.payload
        if p.get("line_start", 0) <= line_number <= p.get("line_end", 0):
            seg_ids.append(p.get("segment_id") or point.id)
    return seg_ids[:limit]


def index_labeled_comments(json_path: Path):
    client = get_qdrant()
    embedder = Embedder(EMBED_PROVIDER)
    dim = embedder.dim
    ensure_collection(client, COLL_COMMENTS, dim)
    data: List[Dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))
    logger.info(f"Comment records: {len(data)}")
    points: List[PointStruct] = []
    for it in data:
        comment_text = it.get("comment") or it.get("comment_text") or ""
        vec = embedder.embed_texts([comment_text], kind="text")[0]
        path = it.get("file_path") or it.get("path")
        line = int(it.get("line_number") or 1)
        seg_ids = search_segment_ids_for_anchor(client, path, line, limit=5) if path else []
        payload = {
            "repo_id": REPO_ID,
            "pr_id": it.get("pr_number"),
            "comment_text": comment_text,
            "author": it.get("author"),
            "created_at": it.get("comment_created_at"),
            "taxonomy": {
                "category": (it.get("category") or "").lower(),
                "subcategory": (it.get("subcategory") or "").lower(),
            },
            "anchors": [{"path": path, "line_number": line}],
            "anchor_segment_ids": seg_ids,
            "file_path": path,
        }
        pid = str(uuid.uuid4())
        points.append(PointStruct(id=pid, vector=vec, payload=payload))
        if len(points) >= 256:
            client.upsert(collection_name=COLL_COMMENTS, points=points)
            points = []
    if points:
        client.upsert(collection_name=COLL_COMMENTS, points=points)
    logger.info("Comment indexing completed.")

# ---------------- META-RAG: Summaries -----------------

# Simple summarizer: heuristics + optional LLM via Ollama

def _simple_summarize_code(code: str, path: str, symbol: str, kind: str, max_chars: int = 800) -> str:
    docstring = ""
    try:
        tree = ast.parse(code)
        docstring = ast.get_docstring(tree) or ""
    except Exception:
        docstring = ""
    ids = extract_identifiers(code)
    top_ids = ", ".join(sorted(ids)[:20])
    summary = (
        f"Path: {path}\nKind: {kind}\nSymbol: {symbol}\nDoc: {docstring}\n"
        f"Identifiers: {top_ids}\n"
    )
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n..."
    return summary

async def _ollama_summarize(text: str, model: str = "llama3:8b-instruct") -> str:
    prompt = (
        "Summarize the following repository unit (file/class/function)."
        " Include purpose, key APIs, dependencies, and common pitfalls."
        " Return a concise paragraph (<= 120 words).\n\n" + text
    )
    return await run_ollama(prompt, model, options="temperature=0,num_ctx=4096")


def index_repo_summaries(repo_root: Path, glob: str = "**/*.py", summarizer: str = "heuristic",
                          summary_model: str = "llama3:8b-instruct"):
    """
    Create COLL_SUMMARIES: per file/class/function summaries to enable Meta-RAG stage-1 retrieval.
    summarizer: 'heuristic' | 'ollama'
    """
    client = get_qdrant()
    embedder = Embedder(EMBED_PROVIDER)
    dim = embedder.dim
    ensure_collection(client, COLL_SUMMARIES, dim)
    files = list(repo_root.rglob(glob))
    points: List[PointStruct] = []
    for py in files:
        try:
            full_code = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        segments = extract_python_segments(full_code)
        for seg in segments:
            seg_code = clip_code_by_lines(full_code, seg["line_start"], seg["line_end"])
            if summarizer == "ollama":
                try:
                    summary = asyncio.get_event_loop().run_until_complete(_ollama_summarize(seg_code, summary_model))
                except Exception:
                    summary = _simple_summarize_code(seg_code, str(py), seg.get("symbol"), seg.get("kind"))
            else:
                summary = _simple_summarize_code(seg_code, str(py), seg.get("symbol"), seg.get("kind"))
            vec = embedder.embed_texts([summary], kind="text")[0]
            sid = str(uuid.uuid4())
            payload = {
                "repo_id": REPO_ID,
                "path": str(py),
                "kind": seg["kind"],
                "symbol": seg["symbol"],
                "line_start": seg["line_start"],
                "line_end": seg["line_end"],
                "summary": summary,
                "summary_id": sid,
            }
            points.append(PointStruct(id=sid, vector=vec, payload=payload))
            if len(points) >= 256:
                client.upsert(collection_name=COLL_SUMMARIES, points=points)
                points = []
    if points:
        client.upsert(collection_name=COLL_SUMMARIES, points=points)
    logger.info("Repo summaries indexing completed.")

# ---------------- Retrieval -----------------

def retrieve_summaries_for_diff(diff_text: str, topk: int = TOPK_SUMMARIES_DEFAULT):
    client = get_qdrant()
    embedder = Embedder(EMBED_PROVIDER)
    vec = embedder.embed_texts([diff_text], kind="text")[0]
    flt = Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=REPO_ID))])
    res = safe_search(client=client, collection_name=COLL_SUMMARIES, query_vector=vec, limit=topk,
                      query_filter=flt, with_payload=True, with_vectors=False)
    return res


def retrieve_code_for_diff(diff_text: str, topk: int = TOPK_CODE_DEFAULT):
    client = get_qdrant()
    embedder = Embedder(EMBED_PROVIDER)
    vec = embedder.embed_texts([diff_text], kind="code")[0]
    flt = Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=REPO_ID))])
    res = safe_search(client=client, collection_name=COLL_CODE, query_vector=vec, limit=topk,
                      query_filter=flt, with_payload=True, with_vectors=False)
    return res


def retrieve_code_via_summaries(diff_text: str, topk_summaries: int, per_summary_code_k: int = 2) -> List[Any]:
    """Meta-RAG stage: first retrieve summaries, then expand to related code segments."""
    client = get_qdrant()
    embedder = Embedder(EMBED_PROVIDER)
    sum_hits = retrieve_summaries_for_diff(diff_text, topk=topk_summaries)
    code_hits: List[Any] = []
    for sh in sum_hits:
        p = sh.payload if hasattr(sh, "payload") else sh.get("payload", {})
        path = p.get("path")
        if not path:
            continue
        flt = Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=REPO_ID)),
                           FieldCondition(key="path", match=MatchValue(value=path))])
        query_vec = embedder.embed_texts([p.get("summary", "")], kind="text")[0]
        part = safe_search(client=client, collection_name=COLL_CODE, query_vector=query_vec, limit=per_summary_code_k,
                           query_filter=flt, with_payload=True, with_vectors=False)
        code_hits.extend(part)
    uniq = {}
    for h in code_hits:
        hid = getattr(h, "id", None) or (h.get("id") if isinstance(h, dict) else None) or uuid.uuid4().hex
        uniq[hid] = h
    return list(uniq.values())


def retrieve_comments_by_segments(segment_ids: List[str], topk: int = TOPK_COMMENTS_DEFAULT):
    client = get_qdrant()
    embedder = Embedder(EMBED_PROVIDER)
    if not segment_ids:
        return []
    hits = []
    for sid in segment_ids:
        flt = Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=REPO_ID)),
                           FieldCondition(key="anchor_segment_ids", match=MatchValue(value=sid))])
        vec = embedder.embed_texts([sid], kind="text")[0]
        part = safe_search(client=client, collection_name=COLL_COMMENTS, query_vector=vec,
                           limit=max(3, topk // max(1, len(segment_ids))), query_filter=flt,
                           with_payload=True, with_vectors=False)
        hits.extend(getattr(part, "result", part))
    unique = {}
    for h in hits:
        hid = getattr(h, "id", None) or (h.get("id") if isinstance(h, dict) else None) or id(h)
        unique[hid] = h
    return list(unique.values())[:topk]

# ---------------- Prompt & LLM -----------------

def build_prompt(diff_text: str, code_hits, comment_hits, style_guide: str = "", summary_hits=None) -> str:
    code_block_lines = []
    for h in (code_hits or []):
        p = h.payload if hasattr(h, "payload") else (h.get("payload", {}) if isinstance(h, dict) else {})
        header = f"- {p.get('path')}:{p.get('line_start')}-{p.get('line_end')} {p.get('symbol')} ({p.get('kind')})"
        snippet = _load_code_snippet_from_payload(p)
        if snippet:
            code_block_lines.append(header + "\n```python\n" + snippet + "\n```")
        else:
            code_block_lines.append(header)

    summaries_lines = []
    for sh in (summary_hits or []):
        sp = sh.payload if hasattr(sh, "payload") else (sh.get("payload", {}) if isinstance(sh, dict) else {})
        summaries_lines.append(f"- {sp.get('path')} {sp.get('symbol')} ({sp.get('kind')})\n{sp.get('summary','')}\n")

    def _payload_of(x):
        return x.payload if hasattr(x, "payload") else (x.get("payload", {}) if isinstance(x, dict) else {})

    comments_block = "\n".join([f"- {_payload_of(h).get('comment_text','')}" for h in (comment_hits or [])])

    if STRICT_JSON:
        system = (
            "You are a senior code reviewer. Use ONLY the DIFF, SUMMARIES, CODE CONTEXT, and PAST COMMENTS.\n"
            "Return STRICT JSON (one object) with keys: issue_type, review_action, severity, scope, subcategory, "
            "finding, recommendation, evidence (list of {path,line_start,line_end,reason}).\n"
        )
    else:
        system = "You are a senior code reviewer.\n"

    prompt = (
        f"{system}"
        f"### DIFF\n{diff_text}\n\n"
        f"### SUMMARIES (Meta-RAG)\n{('\n'.join(summaries_lines) if summaries_lines else '(no summaries)')}\n\n"
        f"### CODE CONTEXT\n{('\n'.join(code_block_lines) if code_block_lines else '(no code context)')}\n\n"
        f"### PAST COMMENTS\n{(comments_block if comments_block else '(no past comments)')}\n\n"
        f"### TEAM POLICY\n{style_guide}\n"
    )
    return prompt


def _parse_options(ollama_options: str) -> Dict[str, Any]:
    opts = {}
    for kv in (o.strip() for o in ollama_options.split(",") if o.strip()):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                opts[k.strip()] = ast.literal_eval(v.strip())
            except Exception:
                opts[k.strip()] = v.strip()
    return opts

async def run_ollama(prompt: str, model_name: str, options: str = "temperature=0,num_ctx=4096") -> str:
    import requests
    payload = {"model": model_name, "prompt": prompt, "stream": False, "options": _parse_options(options)}
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def parse_output(output: str) -> Dict[str, Any]:
    if STRICT_JSON:
        try:
            j = json.loads(output.strip().split("\n")[-1])
            return j
        except Exception:
            return {"finding": output.strip(), "subcategory": "unknown"}
    return {"finding": output.strip(), "subcategory": "unknown"}

# ---------------- Metrics -----------------

def compute_text_metrics(refs: List[str], preds: List[str]) -> Dict[str, float]:
    bleu_scores = [
        sentence_bleu([r.split()], p.split(), smoothing_function=SmoothingFunction().method1) for r, p in zip(refs, preds)
    ]
    avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    avg_r1 = 0.0
    avg_rL = 0.0
    if True:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        r1, rL = [], []
        for r, p in zip(refs, preds):
            s = scorer.score(r, p)
            r1.append(s["rouge1"].fmeasure)
            rL.append(s["rougeL"].fmeasure)
        avg_r1 = float(np.mean(r1)) if r1 else 0.0
        avg_rL = float(np.mean(rL)) if rL else 0.0
    return {"BLEU": avg_bleu, "ROUGE-1 F1": avg_r1, "ROUGE-L F1": avg_rL}


def compute_meteor(refs: List[str], preds: List[str]) -> Optional[float]:
    if single_meteor_score is None:
        return None
    scores = []
    for r, p in zip(refs, preds):
        try:
            scores.append(single_meteor_score(r or "", p or ""))
        except Exception:
            return None
    return float(np.mean(scores)) if scores else 0.0


def compute_semantic_tfidf(refs: List[str], preds: List[str]) -> float:
    sims = []
    for r, p in zip(refs, preds):
        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform([r or "", p or ""])
        sim = cosine_similarity(vec[0], vec[1])[0][0]
        sims.append(float(sim))
    return float(np.mean(sims)) if sims else 0.0

_ACTION_PATTERNS = [
    r"\buse\b", r"\bmove\b", r"\breplace\b", r"\bextract\b", r"\brename\b",
    r"\bremove\b", r"\brefactor\b", r"\badd\b", r"\bvalidate\b", r"\bsanitize\b",
    r"\bparameteri[sz]e\b", r"\bavoid\b", r"\bprefer\b", r"\bmake\b.*\boptional\b",
    r"\bwrite\b.*\btest\b", r"\bcreate\b.*\bfixture\b", r"\bconsider\b", r"\balign\b", r"\benforce\b",
]

def compute_actionability(preds: List[str]) -> float:
    c = 0
    for t in preds:
        text = (t or "").lower()
        if any(re.search(p, text) for p in _ACTION_PATTERNS):
            c += 1
    return c / len(preds) if preds else 0.0


def compute_grounding(codes: List[str], preds: List[str]) -> float:
    total_ids, total_hits = 0, 0
    for code, comment in zip(codes, preds):
        ids = extract_identifiers(code)
        total_ids += len(ids)
        text = (comment or "").lower()
        total_hits += sum(1 for i in ids if i.lower() in text)
    return (total_hits / total_ids) if total_ids else 0.0


def composite_score(f1, sem, act, grd, w=(0.3, 0.4, 0.2, 0.1)) -> float:
    return w[0]*f1 + w[1]*sem + w[2]*act + w[3]*grd


def retrieval_recall_at_k(gold_comment: str, retrieved_comment_texts: List[str], k: int) -> float:
    if not retrieved_comment_texts:
        return 0.0
    vec = TfidfVectorizer(min_df=1, ngram_range=(1,2)).fit_transform([gold_comment] + retrieved_comment_texts[:k])
    sims = cosine_similarity(vec[0], vec[1:])[0]
    return 1.0 if (sims.max() if len(sims) else 0.0) >= 0.5 else 0.0

# ---------------- LLM-as-a-Judge -----------------

def _build_judge_prompt(code_context: str, gold: str, pred: str) -> str:
    return (
        "You are an expert code review judge. Assess the GENERATED review vs GOLD.\n"
        "Score 0-5 for each: semantic_correctness (is the finding technically correct wrt code), "
        "usefulness (is it actionable and helpful for developers).\n"
        "Return STRICT JSON: {\"semantic_correctness\": int, \"usefulness\": int}.\n\n"
        f"### CODE CONTEXT\n{code_context}\n\n"
        f"### GOLD REVIEW\n{gold}\n\n"
        f"### GENERATED REVIEW\n{pred}\n"
    )

async def llm_as_judge(code_hits, gold_comment: str, pred_comment: str) -> Dict[str, float]:
    ctx_parts = []
    for h in (code_hits or []):
        p = h.payload if hasattr(h, "payload") else (h.get("payload", {}) if isinstance(h, dict) else {})
        header = f"{p.get('path')}:{p.get('line_start')}-{p.get('line_end')} {p.get('symbol')} ({p.get('kind')})"
        snippet = _load_code_snippet_from_payload(p)
        ctx_parts.append(header + "\n" + snippet)
    code_ctx = "\n\n".join(ctx_parts)[:4000]

    j_prompt = _build_judge_prompt(code_ctx, gold_comment, pred_comment)
    try:
        resp = await run_ollama(j_prompt, LLM_JUDGE_MODEL, options="temperature=0,num_ctx=4096")
        j = parse_output(resp)
        sem = float(j.get("semantic_correctness", j.get("Judge_Semantic", 0)))
        use = float(j.get("usefulness", j.get("Judge_Useful", 0)))
        return {"Judge_Semantic": sem/5.0 if sem > 1 else sem, "Judge_Useful": use/5.0 if use > 1 else use}
    except Exception:
        return {"Judge_Semantic": 0.0, "Judge_Useful": 0.0}

# ---------------- Evaluation -----------------
import csv
from collections import defaultdict

def take_topk(hits, k: int):
    return list(hits)[:k] if hits else []

async def eval_models(
    eval_items: List[Dict[str, Any]],
    llm_models: List[str],
    comment_ks: List[int] = [1,3,5],
    code_k: int = TOPK_CODE_DEFAULT,
    meta_summ_k: int = TOPK_SUMMARIES_DEFAULT,
    do_rag: bool = True,
    do_no_rag: bool = True,
    use_meta_rag: bool = True,
):
    os.makedirs("out_eval", exist_ok=True)
    all_runs_summary = []

    for model in llm_models:
        logger.info(f"Model evaluating: {model}")
        combos = []
        if do_no_rag:
            combos.append(("no_rag", 0))
        if do_rag:
            for k in comment_ks:
                combos.append(("rag", k))

        for mode, k_comments in combos:
            logger.info(f"==> Mode={mode}  topK_comments={k_comments}  metaRAG={use_meta_rag}")

            per_example_rows = []
            by_subcat_refs = defaultdict(list)
            by_subcat_preds = defaultdict(list)
            by_subcat_codes = defaultdict(list)
            by_subcat_judge = defaultdict(lambda: {"sem":[], "use":[]})
            retrieval_recalls = []

            for it in eval_items:
                diff = it.get("code") or ""
                gold_comment = it.get("comment") or ""
                gold_subcat = (it.get("subcategory") or "").lower()
                file_path = it.get("file_path") or ""
                line_number = int(it.get("line_number") or 1)

                # Retrieval
                if mode == "rag":
                    summary_hits = retrieve_summaries_for_diff(diff, topk=meta_summ_k) if use_meta_rag else []
                    if use_meta_rag:
                        code_hits = retrieve_code_via_summaries(diff_text=diff, topk_summaries=meta_summ_k, per_summary_code_k=max(1, code_k//2))
                        code_hits_direct = retrieve_code_for_diff(diff, topk=max(1, code_k - len(code_hits)))
                        code_hits = (code_hits or []) + (code_hits_direct or [])
                    else:
                        code_hits = retrieve_code_for_diff(diff, topk=code_k)
                    seg_ids = [ (h.payload.get("segment_id") if hasattr(h,"payload") else (h.get("payload",{}).get("segment_id"))) or getattr(h,"id",None) for h in (code_hits or []) ]
                    comment_hits_all = retrieve_comments_by_segments(seg_ids, topk=max(k_comments, 10))
                    comment_hits = take_topk(comment_hits_all, k_comments)
                    rec = retrieval_recall_at_k(gold_comment, [ (h.payload.get("comment_text","") if hasattr(h,"payload") else h.get("payload",{}).get("comment_text","")) for h in comment_hits_all ], k=10)
                    retrieval_recalls.append(rec)
                else:
                    summary_hits = []
                    code_hits = []
                    comment_hits = []

                # Prompt & LLM
                prompt = build_prompt(diff, code_hits, comment_hits, style_guide="", summary_hits=summary_hits)
                out = await run_ollama(prompt, model)
                parsed = parse_output(out)
                pred_comment = parsed.get("finding", "") or ""
                pred_subcat = (parsed.get("subcategory") or "").lower()

                judge_scores = await llm_as_judge(code_hits, gold_comment, pred_comment) if mode == "rag" else {"Judge_Semantic": 0.0, "Judge_Useful": 0.0}

                row = {
                    "model": model,
                    "mode": mode,
                    "k_comments": k_comments,
                    "true_comment": gold_comment,
                    "pred_comment": pred_comment,
                    "true_subcategory": gold_subcat,
                    "pred_subcategory": pred_subcat,
                    "file_path": file_path,
                    "judge_sem": judge_scores.get("Judge_Semantic",0.0),
                    "judge_use": judge_scores.get("Judge_Useful",0.0),
                }
                per_example_rows.append(row)

                by_subcat_refs[gold_subcat].append(gold_comment)
                by_subcat_preds[gold_subcat].append(pred_comment)
                by_subcat_codes[gold_subcat].append(diff)
                by_subcat_judge[gold_subcat]["sem"].append(row["judge_sem"])
                by_subcat_judge[gold_subcat]["use"].append(row["judge_use"])

            # Subcategory metrics
            subcat_metrics = []
            for sc in sorted(by_subcat_refs.keys()):
                refs = by_subcat_refs[sc]
                preds = by_subcat_preds[sc]
                codes = by_subcat_codes[sc]
                true_labels = [sc]*len(refs)
                pred_labels = []
                for r in per_example_rows:
                    if r["true_subcategory"] == sc:
                        pred_labels.append(r["pred_subcategory"])
                labels = list({*true_labels, *pred_labels})
                prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=labels, average="macro", zero_division=0)
                textm = compute_text_metrics(refs, preds)
                meteor = compute_meteor(refs, preds)
                sem = compute_semantic_tfidf(refs, preds)
                act = compute_actionability(preds)
                grd = compute_grounding(codes, preds)
                comp = composite_score(f1, sem, act, grd)
                j_sem = float(np.mean(by_subcat_judge[sc]["sem"])) if by_subcat_judge[sc]["sem"] else 0.0
                j_use = float(np.mean(by_subcat_judge[sc]["use"])) if by_subcat_judge[sc]["use"] else 0.0
                m = {
                    "model": model,
                    "mode": mode,
                    "k_comments": k_comments,
                    "subcategory": sc,
                    "Macro-F1": f1,
                    "Precision": prec,
                    "Recall": rec,
                    "BLEU": textm.get("BLEU", 0.0),
                    "ROUGE-1 F1": textm.get("ROUGE-1 F1", 0.0),
                    "ROUGE-L F1": textm.get("ROUGE-L F1", 0.0),
                    "METEOR": meteor if meteor is not None else -1.0,
                    "SemanticCosine": sem,
                    "ActionabilityRate": act,
                    "Grounding@Identifiers": grd,
                    "CompositeScore": comp,
                    "Judge_Semantic": j_sem,
                    "Judge_Useful": j_use,
                    "Retrieval@10": float(np.mean(retrieval_recalls)) if retrieval_recalls else 0.0
                }
                subcat_metrics.append(m)

            # Outputs
            tag = f"{model.replace(':','_')}.{mode}.k{k_comments}.meta{'1' if use_meta_rag else '0'}"
            with open(f"out_eval/preds_{tag}.jsonl", "w", encoding="utf-8") as f:
                for r in per_example_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(f"out_eval/metrics_subcat_{tag}.json", "w", encoding="utf-8") as f:
                json.dump(subcat_metrics, f, indent=2, ensure_ascii=False)
            all_runs_summary.extend(subcat_metrics)
            logger.info(f"Completed -> out_eval/metrics_subcat_{tag}.json")

    csv_path = "out_eval/summary_all_runs.csv"
    fieldnames = list(all_runs_summary[0].keys()) if all_runs_summary else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_runs_summary:
            w.writerow(row)
    logger.info(f"Global summary -> {csv_path}")

# ---------------- CLI -----------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="RAG + META-RAG code review pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_idx_code = sub.add_parser("index-code", help="Index code segments to Qdrant")
    ap_idx_code.add_argument("--repo", required=True, help="Repository root directory")

    ap_idx_comments = sub.add_parser("index-comments", help="Index labeled past review comments to Qdrant")
    ap_idx_comments.add_argument("--json", required=True, help="Labeled comments JSON file (array)")

    ap_idx_summ = sub.add_parser("index-summaries", help="Index repository summaries (Meta-RAG stage)")
    ap_idx_summ.add_argument("--repo", required=True, help="Repository root directory")
    ap_idx_summ.add_argument("--summarizer", default="heuristic", choices=["heuristic", "ollama"], help="Summary generator")
    ap_idx_summ.add_argument("--summary_model", default="llama3:8b-instruct", help="Ollama model for summarization")

    ap_eval = sub.add_parser("eval", help="Evaluate models with/without RAG (and Meta-RAG)")
    ap_eval.add_argument("--eval_json", required=True, help="Evaluation items JSON")
    ap_eval.add_argument("--comment_ks", default="1,3,5", help="TopK list for past comments, e.g., 1,3,5")
    ap_eval.add_argument("--code_k", type=int, default=TOPK_CODE_DEFAULT, help="TopK code segments for context")
    ap_eval.add_argument("--meta_summ_k", type=int, default=TOPK_SUMMARIES_DEFAULT, help="TopK summaries for Meta-RAG stage")
    ap_eval.add_argument("--meta_rag", action="store_true", help="Enable Meta-RAG (summaries + expansion)")
    ap_eval.add_argument("--rag", action="store_true", help="Enable RAG mode")
    ap_eval.add_argument("--no_rag", action="store_true", help="Evaluate No-RAG mode as well")

    args = ap.parse_args()
    if args.cmd == "index-code":
        index_code_repo(Path(args.repo))
    elif args.cmd == "index-comments":
        index_labeled_comments(Path(args.json))
    elif args.cmd == "index-summaries":
        index_repo_summaries(Path(args.repo), summarizer=args.summarizer, summary_model=args.summary_model)
    elif args.cmd == "eval":
        eval_items = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
        comment_ks = [int(x) for x in args.comment_ks.split(",") if x.strip()]
        # If neither flag provided, run both
        do_rag = args.rag or not (args.rag or args.no_rag)
        do_no_rag = args.no_rag or not (args.rag or args.no_rag)
        asyncio.run(eval_models(eval_items, LLM_MODELS, comment_ks=comment_ks, code_k=args.code_k,
                                meta_summ_k=args.meta_summ_k, do_rag=do_rag, do_no_rag=do_no_rag, use_meta_rag=args.meta_rag))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
