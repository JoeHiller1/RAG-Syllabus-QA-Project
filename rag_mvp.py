"""
RAG MVP (analytics-first):
- Ingest .txt/.md and .pdf from a folder
- Chunk text
- Embed chunks (SentenceTransformers)
- Index with FAISS
- Retrieve top-k for a query
- (Optional) generate an answer (stub you can swap with an LLM)
- Log retrieval metrics + latency

Run:
  python rag_mvp.py --docs ./docs --query "What is covered in week 5?" --top_k 5
"""




from __future__ import annotations
import os, re, time, json, argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
import ollama 

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")



try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import torch
except Exception:
    torch = None



# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Chunk:
    doc_id: str
    source_path: str
    chunk_id: int
    text: str
    start_char: int
    end_char: int

@dataclass
class RetrievalResult:
    query: str
    top_k: int
    retrieved: List[Dict]  # each includes score, chunk metadata
    timings_ms: Dict[str, float]

# -----------------------------
# Course extraction and filtering
# -----------------------------
def extract_course_from_query(query: str) -> Optional[str]:
    """
    Extracts course code from query (e.g., "CS 124", "CS124", "ECE 391").
    Returns normalized course code like "CS 124" or None if not found.
    """
    # Pattern: letter(s) followed by optional space and numbers
    # Matches: CS 124, CS124, ECE 391, STAT 107, CS/ECE/STAT/IS etc.
    pattern = r'\b([A-Z]{2,4})\s*(\d{3,4})\b'
    match = re.search(pattern, query.upper())
    if match:
        dept = match.group(1)
        num = match.group(2)
        return f"{dept} {num}"
    return None

def course_matches_doc(course_code: str, doc_id: str) -> bool:
    """
    Check if course code matches a document ID (filename).
    Handles variations like "CS 124" matching "CS124" or "CS 124_".
    """
    # Normalize: remove spaces, convert to upper
    course_normalized = re.sub(r'\s+', '', course_code.upper())
    doc_normalized = re.sub(r'\s+', '', doc_id.upper())
    
    # Check if normalized course code appears in normalized doc_id
    return course_normalized in doc_normalized

# -----------------------------
# Ingestion
# -----------------------------
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def load_documents(docs_dir: str) -> List[Tuple[str, str, str]]:
    """
    Returns: list of (doc_id, source_path, text)
    doc_id can be filename (without extension) or include extension.
    """
    docs = []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            path = os.path.join(root, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in [".txt", ".md"]:
                text = read_text_file(path)
            elif ext == ".pdf":
                text = read_pdf(path)
            else:
                continue
            doc_id = os.path.relpath(path, docs_dir)
            docs.append((doc_id, path, normalize_text(text)))
    return docs

def normalize_text(text: str) -> str:
    # Light normalization—don’t overdo it for MVP
    text = re.sub("\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_syllabus_text(text: str) -> str:
    # Normalize newlines
 
    text = re.sub("\r\n", "\n", text)

    # Remove repeated page headers/footers
    text = re.sub(r"CS\s*\d+\s*[-–]\s*Fall\s*\d{4}", "", text)

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common boilerplate sections
    boilerplate_patterns = [
        r"Academic Integrity[\s\S]*?(?=\n[A-Z][A-Za-z ]+:|\Z)",
        r"Disability Resources[\s\S]*?(?=\n[A-Z][A-Za-z ]+:|\Z)",
        r"COVID[- ]?19[\s\S]*?(?=\n[A-Z][A-Za-z ]+:|\Z)",
    ]

    for pat in boilerplate_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    return text.strip()


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, doc_id, source_path, chunk_size=800, chunk_overlap=150, encoding_name="cl100k_base"):
    """
    Token-based chunking (more accurate for LLMs than character-based).
    encoding_name: 
    - "cl100k_base" for GPT-3.5/4 (default)
    - "p50k_base" for GPT-3
    - "r50k_base" for GPT-2
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size to ensure progress")
    
    # Use tokenizer if available, otherwise fall back to character-based
    if tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except KeyError:
            print(f"Warning: encoding {encoding_name} not found, using cl100k_base")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        # Tokenize the entire text
        tokens = encoding.encode(text)
        n_tokens = len(tokens)
        
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < n_tokens:
            end_idx = min(start_idx + chunk_size, n_tokens)
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text_str = encoding.decode(chunk_tokens).strip()
            
            if chunk_text_str:
                # Calculate approximate character positions for metadata
                # (not exact since token decode may differ slightly)
                approx_start_char = len(encoding.decode(tokens[:start_idx]))
                approx_end_char = len(encoding.decode(tokens[:end_idx]))
                
                chunks.append(Chunk(
                    doc_id=doc_id,
                    source_path=source_path,
                    chunk_id=chunk_id,
                    text=chunk_text_str,
                    start_char=approx_start_char,
                    end_char=approx_end_char,
                ))
                chunk_id += 1
            
            # Move start position forward (with overlap)
            next_start = end_idx - chunk_overlap
            if next_start <= start_idx:
                next_start = end_idx  # Ensure forward progress
            start_idx = next_start
    else:
        # Fallback to character-based chunking if tiktoken not available
        print("Warning: tiktoken not available, falling back to character-based chunking")
        chunks = []
        n = len(text)
        start = 0
        chunk_id = 0
        
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(Chunk(
                    doc_id=doc_id,
                    source_path=source_path,
                    chunk_id=chunk_id,
                    text=chunk,
                    start_char=start,
                    end_char=end,
                ))
                chunk_id += 1
            
            next_start = end - chunk_overlap
            if next_start <= start:
                next_start = end
            start = next_start
    
    return chunks



# -----------------------------
# Embeddings + Index
# -----------------------------
def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Returns float32 embeddings shaped (N, D), L2 normalized for cosine similarity via inner product.
    """
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")
    return embs

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    For normalized embeddings, cosine similarity == inner product.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


# -----------------------------
# Retrieval
# -----------------------------
def retrieve(
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Chunk],
    query: str,
    top_k: int = 5
) -> RetrievalResult:
    timings = {}

    t0 = time.perf_counter()
    q_emb = embed_texts(model, [query], batch_size=1)
    timings["embed_query"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    scores, ids = index.search(q_emb, top_k)
    timings["faiss_search"] = (time.perf_counter() - t1) * 1000

    retrieved = []
    for rank, (idx, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
        if idx < 0:
            continue
        ch = chunks[idx]
        retrieved.append({
            "rank": rank,
            "score": float(score),
            "doc_id": ch.doc_id,
            "source_path": ch.source_path,
            "chunk_id": ch.chunk_id,
            "start_char": ch.start_char,
            "end_char": ch.end_char,
            "text": ch.text,
        })

    return RetrievalResult(query=query, top_k=top_k, retrieved=retrieved, timings_ms=timings)


# -----------------------------
# Generation (Stub)
# -----------------------------
def generate_answer_stub(query: str, contexts: List[Dict]) -> str:
    """
    Replace this later with your LLM call (OpenAI/Claude/local).
    For MVP, this is a "grounded summary" baseline.
    """
    if not contexts:
        return "I couldn't find relevant context in the documents."

    # Naive: just concatenate the top contexts and “answer” by quoting
    # Later: prompt an LLM and include citations.
    bullets = []
    ollama.pull("qwen2.5:3b")

    for c in contexts[:3]:
        snippet = c["text"][:300].replace("\n", " ")
        bullets.append(f"- (doc={c['doc_id']}, chunk={c['chunk_id']}) {snippet}...")
    
   
    response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {"role": "user", "content": query + "\n".join(bullets)}
    ]
    )
    return response["message"]["content"]
  


# -----------------------------
# Streamlit App Functions
# -----------------------------
def load_or_build_index(
    docs_dir: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    encoding_name: str = "cl100k_base",
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[faiss.Index, List[Chunk], SentenceTransformer]:
    """
    Load documents, chunk them, embed, and build FAISS index.
    Returns: (index, chunks, model)
    """
    # 1) Load docs
    docs = load_documents(docs_dir)
    if not docs:
        raise RuntimeError(f"No supported docs found in {docs_dir}. Use .txt/.md/.pdf")
    
    # 2) Clean syllabus text
    docs = [(doc_id, path, clean_syllabus_text(text)) for doc_id, path, text in docs]
    
    # 3) Chunk
    all_chunks: List[Chunk] = []
    for doc_id, path, text in docs:
        all_chunks.extend(
            chunk_text(
                text=text,
                doc_id=doc_id,
                source_path=path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name=encoding_name,
            )
        )
    
    if not all_chunks:
        raise RuntimeError("No chunks created—are your documents empty?")
    
    # 4) Embed + index
    # Force CPU usage explicitly
    model = SentenceTransformer(model_name, device='cpu')
    
    embeddings = embed_texts(model, [c.text for c in all_chunks])
    index = build_faiss_index(embeddings)
    
    return index, all_chunks, model

def answer_query(
    query: str,
    index: faiss.Index,
    chunks: List[Chunk],
    model: SentenceTransformer,
    top_k: int = 5
) -> Tuple[str, List[Dict]]:
    """
    Answer a query using the index and chunks.
    Returns: (answer, retrieved_chunks)
    """
    # Extract course from query and filter chunks if course is mentioned
    course_code = extract_course_from_query(query)
    if course_code:
        filtered_chunks = [ch for ch in chunks if course_matches_doc(course_code, ch.doc_id)]
        if filtered_chunks:
            chunks = filtered_chunks
    
    # Retrieve
    result = retrieve(model, index, chunks, query, top_k=top_k)
    
    # Generate answer
    answer = generate_answer_stub(query, result.retrieved)
    
    return answer, result.retrieved


# -----------------------------
# Logging / Analytics
# -----------------------------
def log_retrieval_to_csv(result: RetrievalResult, out_csv: str = "retrieval_logs.csv") -> None:
    rows = []
    for r in result.retrieved:
        row = {
            "query": result.query,
            "top_k": result.top_k,
            "rank": r["rank"],
            "score": r["score"],
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "source_path": r["source_path"],
            "start_char": r["start_char"],
            "end_char": r["end_char"],
            "embed_query_ms": result.timings_ms.get("embed_query", None),
            "faiss_search_ms": result.timings_ms.get("faiss_search", None),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if os.path.exists(out_csv):
        df.to_csv(out_csv, mode="w", header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=str, required=True, help="Folder containing .txt/.md/.pdf docs")
    ap.add_argument("--query", type=str, required=True, help="User question")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--chunk_size", type=int, default=800)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    ap.add_argument("--encoding", type=str, default="cl100k_base", 
                help="Tokenizer encoding (cl100k_base for GPT-3.5/4, p50k_base for GPT-3)")
    ap.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--log_csv", type=str, default="retrieval_logs.csv")
    args = ap.parse_args()

    # 1) Load docs
    print(f"Loading documents from {args.docs}...")
    docs = load_documents(args.docs)
    print(f"Loaded {len(docs)} documents")
    if not docs:
        raise RuntimeError(f"No supported docs found in {args.docs}. Use .txt/.md/.pdf")

    # 2) Clean syllabus text

    docs = [(doc_id, path, clean_syllabus_text(text)) for doc_id, path, text in docs]
    # 3) Chunk
    print("Starting to chunk documents...")
    all_chunks: List[Chunk] = []
    for i, (doc_id, path, text) in enumerate(docs, 1):
        print(f"Chunking document {i}/{len(docs)}: {doc_id}")
        all_chunks.extend(
            chunk_text(
                text=text,
                doc_id=doc_id,
                source_path=path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                encoding_name=getattr(args, 'encoding', 'cl100k_base'),
            )
        )
    print(f"Total chunks created: {len(all_chunks)}")
    if not all_chunks:
        raise RuntimeError("No chunks created—are your documents empty?")
    
    # Extract course from query and filter chunks if course is mentioned
    course_code = extract_course_from_query(args.query)
    if course_code:
        print(f"Detected course in query: {course_code}")
        original_count = len(all_chunks)
        all_chunks = [ch for ch in all_chunks if course_matches_doc(course_code, ch.doc_id)]
        filtered_count = len(all_chunks)
        print(f"Filtered to {filtered_count} chunks from {original_count} (matching {course_code})")
        if not all_chunks:
            raise RuntimeError(f"No chunks found for course {course_code}. Check if the course name matches any document.")
    else:
        print("No course code detected in query - searching all documents")
    
    print(len(all_chunks))
    # 4) Embed + index
    # Force CPU usage explicitly
    model = SentenceTransformer(args.model_name, device='cpu')
    
    # Check device (CPU/GPU)
    print("\n=== Device Information ===")
    print("Forcing CPU usage (GPU disabled)")
    if torch is not None:
        if torch.cuda.is_available():
            print(f"CUDA available: Yes (but disabled for this run)")
        else:
            print("CUDA available: No (running on CPU)")
    else:
        print("PyTorch not directly available (may be used internally by SentenceTransformer)")
    
    # Try to detect model device
    if hasattr(model, 'device'):
        print(f"SentenceTransformer device: {model.device}")
    elif hasattr(model, '_modules') and len(model._modules) > 0:
        try:
            first_module = list(model._modules.values())[0]
            if hasattr(first_module, 'device'):
                print(f"Model device: {first_module.device}")
            elif hasattr(first_module, 'parameters'):
                device = next(first_module.parameters()).device
                print(f"Model device: {device}")
        except Exception:
            pass
    print()

    t0 = time.perf_counter()
    embeddings = embed_texts(model, [c.text for c in all_chunks])
    build_ms = (time.perf_counter() - t0) * 1000

    index = build_faiss_index(embeddings)

    # 5) Retrieve
    result = retrieve(model, index, all_chunks, args.query, top_k=args.top_k)

    # Add build time into timings (helpful for profiling)
    result.timings_ms["embed_corpus+build_index"] = build_ms

    # 6) Print results
    print("\n=== Retrieval Results ===")
    print(json.dumps({
        "query": result.query,
        "top_k": result.top_k,
        "timings_ms": result.timings_ms,
        "hits": [
            {k: r[k] for k in ["rank", "score", "doc_id", "chunk_id", "start_char", "end_char"]}
            for r in result.retrieved
        ]
    }, indent=2))

    
    print("\n=== Answer (stub) ===")
    print(generate_answer_stub(args.query, result.retrieved))

    # 7) Log
    log_retrieval_to_csv(result, args.log_csv)
    print(f"\nLogged retrieval to: {args.log_csv}")


if __name__ == "__main__":
    main()
