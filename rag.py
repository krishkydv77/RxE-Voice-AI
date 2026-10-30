import argparse
import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from logger import logger, log_metric, Timer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data_basic"))
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", BASE_DIR / "vectordb"))
MANIFEST_PATH = VECTOR_DIR / "manifest.json"
COLLECTION = "healthcare_knowledge_base"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "3"))
ADD_BATCH = 64

SUPPORTED_EXT = {".md", ".markdown", ".pdf"}

WRITE_LOCK = threading.RLock()

_embeddings = None
_store = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        logger.info("RAG embed model load start | model=%s", EMBED_MODEL)
        with Timer() as t:
            _embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
        logger.info("RAG embed model ready | model=%s load_ms=%.1f", EMBED_MODEL, t.ms)
    return _embeddings


def get_store():
    global _store
    if _store is None:
        with WRITE_LOCK:
            if _store is None:
                from langchain_chroma import Chroma

                VECTOR_DIR.mkdir(parents=True, exist_ok=True)
                logger.info("RAG vector store open | dir=%s collection=%s", VECTOR_DIR, COLLECTION)
                _store = Chroma(
                    collection_name=COLLECTION,
                    embedding_function=get_embeddings(),
                    persist_directory=str(VECTOR_DIR),
                    collection_metadata={"hnsw:space": "cosine"},
                )
                logger.info("RAG vector store ready | existing_chunks=%d", _store._collection.count())
    return _store


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read manifest, creating new: %s", e)
    return {}


def _save_manifest(m: dict) -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "। ", ". ", " ", ""],
    )


def split_markdown_text(text: str, source: str) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)
    chunks = _recursive_splitter().split_documents(sections)
    out: list[Document] = []
    for c in chunks:
        keys = [k for k in ("h2", "h3") if k in c.metadata] or [k for k in ("h1",) if k in c.metadata]
        path = " > ".join(c.metadata[k] for k in keys)
        body = c.page_content.strip()
        if not body:
            continue
        if path and not body.lstrip().startswith("#"):
            body = f"[{path}] {body}"
        out.append(Document(page_content=body,
                            metadata={"source": source, "section": path or source, "type": "md"}))
    return out


def load_and_chunk(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    logger.info("RAG load start | file=%s type=%s size=%.1f KB", path.name, ext, path.stat().st_size / 1024)

    if ext in (".md", ".markdown"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs = split_markdown_text(text, path.name)
        logger.info("RAG load md | file=%s chars=%d chunks=%d", path.name, len(text), len(docs))
        return docs

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(path)).load()
        with_text = [p for p in pages if p.page_content.strip()]
        logger.info("RAG load pdf | file=%s pages=%d pages_with_text=%d", path.name, len(pages), len(with_text))
        if not with_text:
            logger.warning("RAG no text in pdf | file=%s", path.name)
            return []
        chunks = _recursive_splitter().split_documents(with_text)
        out = []
        for c in chunks:
            page_no = int(c.metadata.get("page", 0)) + 1
            out.append(Document(page_content=c.page_content.strip(),
                                metadata={"source": path.name, "section": f"page {page_no}",
                                          "page": page_no, "type": "pdf"}))
        logger.info("RAG chunk pdf | file=%s chunks=%d", path.name, len(out))
        return out

    logger.warning("RAG unsupported file skipped | file=%s", path.name)
    return []


def _add_docs(docs: list[Document], ids: list[str], label: str) -> float:
    store = get_store()
    with Timer() as t:
        for i in range(0, len(docs), ADD_BATCH):
            with Timer() as tb:
                store.add_documents(docs[i:i + ADD_BATCH], ids=ids[i:i + ADD_BATCH])
            logger.info("RAG embed+store batch | %s chunks %d-%d/%d took=%.1f ms",
                        label, i + 1, min(i + ADD_BATCH, len(docs)), len(docs), tb.ms)
    return t.ms


def _delete_source(name: str) -> int:
    store = get_store()
    old = store.get(where={"source": name}).get("ids", [])
    if old:
        store.delete(ids=old)
    return len(old)


def index_file(path: Path, manifest: dict) -> int:
    sha = _file_sha(path)
    with Timer() as t_load:
        docs = load_and_chunk(path)
    removed = _delete_source(path.name)
    if removed:
        logger.info("RAG old chunks removed | file=%s removed=%d", path.name, removed)

    if docs:
        ids = [f"{path.name}:{sha[:8]}:{i}" for i in range(len(docs))]
        embed_ms = _add_docs(docs, ids, path.name)
    else:
        embed_ms = 0.0

    manifest[path.name] = {"sha256": sha, "chunks": len(docs), "type": path.suffix.lower().lstrip("."),
                           "indexed_at": datetime.now().isoformat(timespec="seconds")}
    log_metric("rag_index", file=path.name, chunks=len(docs), load_chunk_ms=t_load.ms,
               embed_store_ms=embed_ms, removed_old=removed)
    return len(docs)


def sync_data_dir(force: bool = False) -> dict:
    with WRITE_LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
        logger.info("RAG sync start | dir=%s files_found=%d force=%s", DATA_DIR, len(files), force)
        if not files:
            logger.warning("RAG data folder empty: %s", DATA_DIR)

        manifest = _load_manifest()
        store = get_store()
        if manifest and store._collection.count() == 0:
            logger.warning("RAG manifest present but vector DB empty - forcing reindex")
            force = True

        summary = {"indexed": [], "skipped": [], "removed": [], "chunks_added": 0}
        with Timer() as t:
            for p in files:
                sha = _file_sha(p)
                if not force and manifest.get(p.name, {}).get("sha256") == sha:
                    logger.info("RAG skip (unchanged) | file=%s chunks=%s", p.name, manifest[p.name].get("chunks"))
                    summary["skipped"].append(p.name)
                    continue
                logger.info("RAG index needed | file=%s reason=%s", p.name,
                            "force" if force else ("new" if p.name not in manifest else "changed"))
                n = index_file(p, manifest)
                summary["indexed"].append(p.name)
                summary["chunks_added"] += n
                _save_manifest(manifest)

            present = {p.name for p in files}
            for name in [n for n in manifest if n not in present]:
                removed = _delete_source(name)
                manifest.pop(name)
                summary["removed"].append(name)
                logger.info("RAG file deleted - chunks purged | file=%s removed=%d", name, removed)
            _save_manifest(manifest)

        summary["total_chunks"] = store._collection.count()
        summary["sync_ms"] = t.ms
        log_metric("rag_sync", **summary)
        logger.info("RAG sync done | indexed=%s skipped=%s removed=%s total_chunks=%d took=%.1f ms",
                    summary["indexed"], summary["skipped"], summary["removed"],
                    summary["total_chunks"], t.ms)
        return summary


def rebuild() -> dict:
    global _store
    with WRITE_LOCK:
        logger.warning("RAG REBUILD: collection + manifest purge")
        try:
            get_store().delete_collection()
        except Exception as e:
            logger.info("RAG rebuild delete_collection skipped (%s)", e)
        _store = None
        MANIFEST_PATH.unlink(missing_ok=True)
        return sync_data_dir(force=True)


def append_markdown_entry(filename: str, entry_md: str, entry_key: str, file_header: str) -> int:
    with WRITE_LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / filename
        store = get_store()
        first_id = f"{filename}:entry-{entry_key}:0"
        if store.get(ids=[first_id]).get("ids"):
            logger.info("RAG entry already saved - skipped | file=%s key=%s", filename, entry_key)
            return 0

        if not path.exists():
            path.write_text(file_header.rstrip() + "\n", encoding="utf-8")
            logger.info("RAG created file | %s", path)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + entry_md.strip() + "\n")
        logger.info("RAG entry written | file=%s key=%s chars=%d", filename, entry_key, len(entry_md))

        docs = split_markdown_text(entry_md, filename)
        ids = [f"{filename}:entry-{entry_key}:{i}" for i in range(len(docs))]
        embed_ms = _add_docs(docs, ids, f"{filename}#{entry_key}") if docs else 0.0

        manifest = _load_manifest()
        prev = manifest.get(filename, {})
        manifest[filename] = {"sha256": _file_sha(path), "chunks": prev.get("chunks", 0) + len(docs),
                              "type": "md", "indexed_at": datetime.now().isoformat(timespec="seconds")}
        _save_manifest(manifest)
        log_metric("rag_entry", file=filename, key=entry_key, chunks=len(docs), embed_store_ms=embed_ms)
        return len(docs)


def search(query: str, k: int = TOP_K) -> list[dict]:
    store = get_store()
    with Timer() as t:
        res = store.similarity_search_with_score(query, k=k)
    hits = []
    for doc, dist in res:
        hits.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "?"),
            "section": doc.metadata.get("section", ""),
            "page": doc.metadata.get("page"),
            "score": round(1 - float(dist), 3),
        })
    logger.info("RAG search | q=%r k=%d hits=%d took=%.1f ms", query[:80], k, len(hits), t.ms)
    log_metric("rag_search", latency_ms=t.ms, k=k, top_score=hits[0]["score"] if hits else None,
               sources=[h["source"] for h in hits])
    return hits


def format_context(hits: list[dict], max_chars: int = 1800) -> str:
    parts, used = [], 0
    for h in hits:
        piece = f"({h['source']} | {h['section']}) {h['text']}"
        if used + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - used)]
        if piece:
            parts.append(piece)
            used += len(piece)
        if used >= max_chars:
            break
    return "\n---\n".join(parts)


def stats() -> dict:
    m = _load_manifest()
    return {"files": m, "total_chunks": get_store()._collection.count(), "data_dir": str(DATA_DIR),
            "embed_model": EMBED_MODEL}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Healthcare data_basic -> vector DB")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--query")
    a = ap.parse_args()
    out = rebuild() if a.rebuild else sync_data_dir()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.query:
        for h in search(a.query):
            print(f"{h['score']:.3f} | {h['source']} | {h['section']}\n   {h['text'][:160]}\n")