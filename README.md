---
title: SciFact Retrieval API
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# SciFact Retrieval API

API de recuperación sobre SciFact con BM25 y TF-IDF/coseno.

- `GET /health`
- `POST /search` con JSON: `{"query":"vitamin D cancer","model":"bm25","top_k":5}`

El corpus oficial se descarga automáticamente al iniciar el contenedor.
