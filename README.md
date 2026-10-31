# Document Search & Q&A

Search PDF documents with BM25 and (optionally) answer questions from them with an LLM, citing file and page.

## How it works
1. PDFs in `docs/` are split into ~150-word chunks (with overlap), each tagged with file and page.
2. A question is matched against the chunks with BM25 (keyword search).
3. `/ask` (optional) sends the top chunks to an LLM that answers only from them and cites sources. It needs an API key.

## Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```
- `GET /health` – chunk count
- `GET /search?q=overfitting` – top matching chunks with page numbers
  - `POST /ask` – LLM answer with sources (needs a free `GEMINI_API_KEY` from Google AI Studio)
- `POST /reload` – re-index after adding PDFs

## Evaluation
Two eval scripts, both testing retrieval only (no LLM needed):
- `eval.py`: 10 easy questions on one 3-page document. Top-1 10/10.
- `eval2.py`: 2 documents, 8 paraphrased questions + 3 unanswerable ones.

| Version | Top-1 | Top-4 | Abstained correctly |
|---|---|---|---|
| BM25 only | 7/8 | 8/8 | 0/3 |
| + stopword removal | 7/8 | 7/8 | 3/3 |
| + simple suffix stemming | 7/8 | 8/8 | 3/3 |

Stopword removal fixed unanswerable questions (common words were matching random chunks).
Stemming fixed a miss caused by "memorizing" vs "memorizes". Note the stemmer is crude (e.g. "class" and "classes" don't match).
The eval sets are small and written by me, so these numbers show the pipeline works, not that it is robust.

## Limitations
- BM25 matches exact words: paraphrased questions can miss.
- Text PDFs only (scanned PDFs need OCR). Arabic PDF text extraction is untested.
- The eval set is tiny and written by me.

## Next steps
- Larger document set and paraphrased / unanswerable questions in the eval.
- Compare BM25 with embeddings and a hybrid.