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
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root():
    page = r'''<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SciFact | Buscador</title>
  <style>
    :root{color-scheme:light;--ink:#172033;--muted:#667085;--line:#dfe5ed;--blue:#2556c7;--bg:#f5f7fb}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
    main{max-width:900px;margin:40px auto;padding:0 20px}.card{background:white;border:1px solid var(--line);border-radius:16px;padding:26px;box-shadow:0 8px 24px #1b2b4a0a}
    h1{margin:0 0 6px;font-size:28px}p{color:var(--muted);margin:0 0 20px}.meta{font-size:14px;color:var(--muted)}
    form{display:grid;grid-template-columns:1fr 180px 110px auto;gap:12px;align-items:end;margin-top:22px}
    label{display:grid;gap:6px;font-size:14px;font-weight:600}input,select,button{font:inherit;border-radius:9px;padding:10px 12px;border:1px solid var(--line);min-height:44px;background:#fff;color:var(--ink)}
    button{background:var(--blue);color:white;border-color:var(--blue);font-weight:650;cursor:pointer}button:disabled{opacity:.65;cursor:wait}
    #status{margin:18px 0 8px;color:var(--muted);min-height:24px}.result{border-top:1px solid var(--line);padding:16px 0}.result h3{font-size:17px;margin:0 0 5px}.result p{margin:6px 0;color:#344054}.tag{font-size:12px;color:var(--muted)}
    .links{display:flex;gap:16px;margin-top:18px;font-size:14px}.links a{color:var(--blue)}
    @media(max-width:700px){main{margin:20px auto}form{grid-template-columns:1fr 1fr}.query{grid-column:1/-1}button{grid-column:1/-1}}
  </style>
</head>
<body>
<main>
  <section class="card">
    <h1>Buscador SciFact</h1>
    <p>Recuperación de artículos científicos por consulta. Prueba BM25 y TF‑IDF sobre el corpus SciFact.</p>
    <div class="meta"><span id="health">Comprobando servicio…</span> · Corpus: __DOC_COUNT__ documentos</div>
    <form id="search-form">
      <label class="query">Consulta<input id="query" name="query" required placeholder="Ej.: Vitamin D prevents cancer"></label>
      <label>Modelo<select id="model" name="model"><option value="bm25">BM25</option><option value="tfidf">TF‑IDF (coseno)</option></select></label>
      <label>Resultados<input id="top-k" name="top_k" type="number" min="1" max="100" value="10" required></label>
      <button id="submit" type="submit">Buscar</button>
    </form>
    <div id="status" role="status" aria-live="polite"></div>
    <section id="results" aria-label="Resultados de búsqueda"></section>
    <div class="links"><a href="/docs">API interactiva</a><a href="/health">Estado del servicio</a></div>
  </section>
</main>
<script>
const form=document.getElementById('search-form'),statusBox=document.getElementById('status'),resultsBox=document.getElementById('results'),submitButton=document.getElementById('submit');
fetch('/health').then(r=>r.json()).then(d=>{document.getElementById('health').textContent=`Servicio activo · ${d.documents} documentos`}).catch(()=>{document.getElementById('health').textContent='No se pudo comprobar el estado del servicio'});
form.addEventListener('submit',async event=>{
  event.preventDefault();resultsBox.replaceChildren();submitButton.disabled=true;statusBox.textContent='Buscando…';
  const payload={query:document.getElementById('query').value.trim(),model:document.getElementById('model').value,top_k:Number(document.getElementById('top-k').value)};
  try{
    const response=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await response.json();if(!response.ok)throw new Error(data.detail||'Error en la búsqueda');
    statusBox.textContent=`${data.results.length} resultados · ${data.model.toUpperCase()}`;
    if(!data.results.length){const empty=document.createElement('p');empty.textContent='No se encontraron documentos para esta consulta.';resultsBox.append(empty);return;}
    for(const item of data.results){
      const article=document.createElement('article');article.className='result';
      const heading=document.createElement('h3');heading.textContent=`${item.rank}. ${item.title||'Documento sin título'}`;
      const meta=document.createElement('div');meta.className='tag';meta.textContent=`ID: ${item.document_id} · Puntuación: ${Number(item.score).toFixed(4)}`;
      const snippet=document.createElement('p');snippet.textContent=item.snippet;
      article.append(heading,meta,snippet);resultsBox.append(article);
    }
  }catch(error){statusBox.textContent=`No se pudo completar la búsqueda: ${error.message}`;}
  finally{submitButton.disabled=false;}
});
</script>
</body>
</html>'''.replace("__DOC_COUNT__", str(N))
    return HTMLResponse(content=page)


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
