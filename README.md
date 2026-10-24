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
- `POST /ask` – LLM answer with sources (needs `ANTHROPIC_API_KEY`)
- `POST /reload` – re-index after adding PDFs

## Evaluation
`python eval.py` checks whether the correct page is retrieved for 10 questions on a sample 3-page document: Top-1 10/10, Top-4 10/10.
This is a small, easy test (questions share wording with the text), so it shows the pipeline works, not that it is robust.

## Limitations
- BM25 matches exact words: paraphrased questions can miss.
- Text PDFs only (scanned PDFs need OCR). Arabic PDF text extraction is untested.
- The eval set is tiny and written by me.

## Next steps
- Larger document set and paraphrased / unanswerable questions in the eval.
- Compare BM25 with embeddings and a hybrid.