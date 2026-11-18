"""SciFact retrieval API: TF-IDF cosine and BM25."""
from collections import Counter, defaultdict
import json
import math
import re
import zipfile
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DATA_DIR = Path("data/scifact")
ROOT = DATA_DIR / "scifact"
DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def ensure_data():
    if (ROOT / "corpus.jsonl").exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "scifact.zip"
    if not archive.exists():
        response = requests.get(DATA_URL, timeout=180)
        response.raise_for_status()
        archive.write_bytes(response.content)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR)


ensure_data()
with (ROOT / "corpus.jsonl").open(encoding="utf-8") as f:
    documents = {str(d["_id"]): d for d in (json.loads(line) for line in f)}

stemmer = PorterStemmer()
stopwords = set(ENGLISH_STOP_WORDS)


def tokenize(text):
    return [stemmer.stem(w) for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in stopwords]


index = defaultdict(dict)
doc_lengths = {}
doc_frequencies = {}
for doc_id, doc in documents.items():
    frequencies = Counter(tokenize(doc.get("title", "") + " " + doc.get("text", "")))
    doc_frequencies[doc_id] = frequencies
    doc_lengths[doc_id] = sum(frequencies.values())
    for term, frequency in frequencies.items():
        index[term][doc_id] = frequency

N = len(documents)
avgdl = sum(doc_lengths.values()) / max(1, N)
doc_tfidf_norms = {}
for doc_id, frequencies in doc_frequencies.items():
    norm_sq = 0.0
    for term, tf in frequencies.items():
        df = len(index[term])
        idf = math.log((N + 1) / (df + 1)) + 1
        weight = (1 + math.log(tf)) * idf
        norm_sq += weight * weight
    doc_tfidf_norms[doc_id] = math.sqrt(norm_sq)
app = FastAPI(title="SciFact Retrieval API", version="1.0.0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    model: str = "bm25"
    top_k: int = Field(default=10, ge=1, le=100)


def retrieve(query, model, top_k):
    qfreq = Counter(tokenize(query))
    scores = defaultdict(float)
    if model == "bm25":
        for term, qf in qfreq.items():
            postings = index.get(term, {})
            df = len(postings)
            if not df:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                denominator = tf + 1.5 * (0.25 + 0.75 * doc_lengths[doc_id] / (avgdl or 1))
                scores[doc_id] += idf * (tf * 2.5 / denominator) * qf
    else:
        qweights = {t: f * (math.log((N + 1) / (len(index.get(t, {})) + 1)) + 1) for t, f in qfreq.items()}
        qnorm = math.sqrt(sum(v * v for v in qweights.values())) or 1.0
        for term, qweight in qweights.items():
            idf = math.log((N + 1) / (len(index.get(term, {})) + 1)) + 1
            for doc_id, tf in index.get(term, {}).items():
                weight = (1 + math.log(tf)) * idf
                scores[doc_id] += weight * qweight
        for doc_id in list(scores):
            scores[doc_id] /= (doc_tfidf_norms[doc_id] or 1.0) * qnorm
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


@app.get("/health")
def health():
    return {"status": "ok", "corpus": "SciFact", "documents": N}


@app.post("/search")
def search(request: SearchRequest):
    model = request.model.lower()
    if model not in {"bm25", "tfidf"}:
        raise HTTPException(status_code=400, detail="model debe ser bm25 o tfidf")
    results = []
    for rank, (doc_id, score) in enumerate(retrieve(request.query, model, request.top_k), 1):
        doc = documents[doc_id]
        results.append({
            "rank": rank,
            "document_id": doc_id,
            "score": float(score),
            "title": doc.get("title", ""),
            "snippet": doc.get("text", "")[:300],
        })
    return {"model": model, "results": results}
