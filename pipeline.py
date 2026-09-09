"""
pipeline.py — chunking, embedding, storage, and all three query modes.

VectorStore  : ChromaDB + local sentence-transformers
chunk_doc    : token-aware splitting with overlap
answer       : RAG Q&A with inline citations
summarize    : structured document summary
search       : pure semantic search, no generation
"""
from __future__ import annotations
import re
import numpy as np
import tiktoken
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import anthropic

from config import cfg

# ── Tokenizer ─────────────────────────────────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")

def _tlen(text: str) -> int:
    return len(_enc.encode(text))

def _hard_split(text: str, max_tok: int) -> list[str]:
    tokens = _enc.encode(text)
    return [_enc.decode(tokens[i:i+max_tok]) for i in range(0, len(tokens), max_tok)]


# ── Chunker ───────────────────────────────────────────────────────────────

def chunk_doc(doc: dict, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap) -> list[dict]:
    paras = [p.strip() for p in re.split(r"\n{2,}", doc["text"]) if p.strip()]
    windows: list[str] = []
    buf, buf_tok = [], 0

    for para in paras:
        pt = _tlen(para)
        if pt > chunk_size:
            if buf:
                windows.append(" ".join(buf)); buf, buf_tok = [], 0
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                st = _tlen(sent)
                if st > chunk_size:
                    windows.extend(_hard_split(sent, chunk_size))
                elif buf_tok + st > chunk_size:
                    windows.append(" ".join(buf)); buf, buf_tok = [sent], st
                else:
                    buf.append(sent); buf_tok += st
        elif buf_tok + pt > chunk_size:
            windows.append(" ".join(buf)); buf, buf_tok = [para], pt
        else:
            buf.append(para); buf_tok += pt

    if buf:
        windows.append(" ".join(buf))

    chunks = []
    for i, w in enumerate(windows):
        if i > 0 and overlap:
            prev_tok = _enc.encode(windows[i-1])
            w = _enc.decode(prev_tok[-overlap:]) + " " + w
        safe = doc["source_ref"].replace("/","_").replace(":","_")[:60]
        chunks.append(dict(
            id=f"{safe}__c{i}",
            text=w.strip(),
            chunk_index=i,
            total_chunks=len(windows),
            doc_title=doc["title"],
            source_ref=doc["source_ref"],
        ))
    return chunks


# ── Vector Store ──────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self._emb = SentenceTransformer(cfg.embedding_model)
        self._client = chromadb.PersistentClient(
            path=str(cfg.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            cfg.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embs  = self._emb.encode(texts, show_progress_bar=False).tolist()
        meta  = [{k: str(v) for k, v in c.items() if k != "text"} for c in chunks]
        self._col.upsert(ids=[c["id"] for c in chunks], documents=texts,
                         embeddings=embs, metadatas=meta)
        return len(chunks)

    def query(self, q: str, top_k=cfg.top_k, doc_title: str | None = None) -> list[dict]:
        where = {"doc_title": {"$eq": doc_title}} if doc_title else None
        qemb  = self._emb.encode([q]).tolist()
        fetch = min(top_k * 3, max(self._col.count(), 1))
        res   = self._col.query(query_embeddings=qemb, n_results=fetch,
                                where=where, include=["documents","metadatas","distances"])
        hits  = [{"text": d, "distance": dist, **m}
                 for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
        return self._mmr(qemb[0], hits, top_k)

    def _mmr(self, qvec, candidates, k, lam=cfg.top_k and 0.6) -> list[dict]:
        if not candidates: return []
        q  = np.array(qvec)
        vs = self._emb.encode([c["text"] for c in candidates], show_progress_bar=False)
        qn = q / (np.linalg.norm(q) + 1e-9)
        vn = vs / (np.linalg.norm(vs, axis=1, keepdims=True) + 1e-9)
        sim = vn @ qn
        sel, rem = [], list(range(len(candidates)))
        for _ in range(min(k, len(candidates))):
            if not sel:
                idx = rem[int(np.argmax(sim[rem]))]
            else:
                sv = vn[sel]
                scores = [0.6*sim[i] - 0.4*float(np.max(vn[i] @ sv.T)) for i in rem]
                idx = rem[int(np.argmax(scores))]
            sel.append(idx); rem.remove(idx)
        return [candidates[i] for i in sel]

    def count(self) -> int:
        return self._col.count()

    def sources(self) -> list[dict]:
        if not self._col.count(): return []
        metas = self._col.get(include=["metadatas"])["metadatas"]
        seen  = {}
        for m in metas:
            r = m.get("source_ref","")
            if r not in seen:
                seen[r] = {"title": m.get("doc_title"), "source_ref": r}
        return list(seen.values())

    def delete(self, source_ref: str) -> int:
        ids = self._col.get(where={"source_ref":{"$eq":source_ref}})["ids"]
        if ids: self._col.delete(ids=ids)
        return len(ids)


# ── Generation helpers ────────────────────────────────────────────────────

_claude = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

def _context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] {c['doc_title']}\n{c['text']}" for i, c in enumerate(chunks, 1)
    )

def _call(system: str, user: str, max_tokens: int) -> str:
    r = _claude.messages.create(model=cfg.claude_model, max_tokens=max_tokens,
                                system=system, messages=[{"role":"user","content":user}])
    return r.content[0].text


# ── Q&A ───────────────────────────────────────────────────────────────────

_QA_SYS = """Answer using ONLY the numbered context chunks below.
Cite inline as [1], [2], etc. If the context is insufficient, say so.
End with: Sources used: [n, n, ...]"""

def answer(query: str, store: VectorStore, doc_title: str | None = None) -> dict:
    chunks = store.query(query, doc_title=doc_title)
    if not chunks:
        return {"answer": "No relevant content found.", "chunks": [], "query": query}
    text = _call(_QA_SYS, f"Context:\n\n{_context(chunks)}\n\nQuestion: {query}", 1000)
    return {"answer": text, "chunks": chunks, "query": query}


# ── Summarize ─────────────────────────────────────────────────────────────

_SUM_SYS = """Produce a structured summary:
## Overview (2–3 sentences)
## Key Points (5–8 specific bullets)
## Gaps or Caveats
Stay grounded in the provided text."""

def summarize(store: VectorStore, doc_title: str | None = None) -> dict:
    all_chunks: list[dict] = []
    seen: set[str] = set()
    for q in ["overview introduction", "findings results", "details methodology"]:
        for c in store.query(q, top_k=8, doc_title=doc_title):
            cid = c["id"]
            if cid not in seen:
                seen.add(cid); all_chunks.append(c)
    all_chunks = all_chunks[:18]
    if not all_chunks:
        return {"summary": "Nothing to summarize.", "chunks_used": 0}
    scope = doc_title or "all documents"
    text = _call(_SUM_SYS, f"Summarize content from: {scope}\n\n{_context(all_chunks)}", 800)
    return {"summary": text, "chunks_used": len(all_chunks)}


# ── Search ────────────────────────────────────────────────────────────────

def search(query: str, store: VectorStore, top_k=cfg.top_k,
           doc_title: str | None = None) -> list[dict]:
    chunks = store.query(query, top_k=top_k, doc_title=doc_title)
    for c in chunks:
        c["score"] = round(1 - c.get("distance", 0), 4)
    return chunks