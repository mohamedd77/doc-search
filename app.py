import os, re, glob
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
DOCS_DIR = "docs"
CHUNK_WORDS = 150
OVERLAP = 30

state = {"chunks": [], "bm25": None, "llm_ready": False}



STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "do", "does", "how", "what", "who", "why", "which", "i",
    "my", "it", "for", "with", "from", "that", "this", "can", "should",
    "be", "by", "as", "at", "than", "not", "when", "use", "used",
}

def normalize(text):
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)
    text = re.sub("[إأآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text.lower()

SUFFIXES = ("ing", "ed", "es", "s")

def stem(word):
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word

def tokenize(text):
    words = re.findall(r"\w+", normalize(text))
    return [stem(w) for w in words if w not in STOPWORDS]


def load_chunks():
    chunks = []
    step = CHUNK_WORDS - OVERLAP
    for path in glob.glob(os.path.join(DOCS_DIR, "*.pdf")):
        reader = PdfReader(path)
        for page_no, page in enumerate(reader.pages, start=1):
            words = (page.extract_text() or "").split()
            for start in range(0, max(len(words), 1), step):
                piece = " ".join(words[start:start + CHUNK_WORDS])
                if piece.strip():
                    chunks.append({
                        "text": piece,
                        "file": os.path.basename(path),
                        "page": page_no,
                    })
    return chunks

def build_index():
    state["chunks"] = load_chunks()
    corpus = [tokenize(c["text"]) for c in state["chunks"]]
    state["bm25"] = BM25Okapi(corpus) if corpus else None



def retrieve(question, k=4):
    if not state["bm25"]:
        return []
    scores = state["bm25"].get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [dict(state["chunks"][i], score=float(scores[i]))
            for i in top if scores[i] > 0]



def ask_llm(question, hits):
    context = "\n\n".join(
        f"[{i+1}] ({h['file']}, p.{h['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )
    prompt = f"""Answer ONLY from the sources below. Cite sources like [1].
If the answer isn't in the sources, say you don't know.
Reply in the language of the question.

<sources>
{context}
</sources>

Question: {question}"""
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# ---------- الـ API ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    state["llm_ready"] = bool(os.getenv("GEMINI_API_KEY"))
    build_index()
    yield

app = FastAPI(lifespan=lifespan)

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

class Source(BaseModel):
    file: str
    page: int
    score: float

class Answer(BaseModel):
    answer: str
    sources: list[Source]

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["chunks"]),
            "llm_ready": state["llm_ready"], "model": MODEL}

@app.post("/reload")
def reload_docs():
    build_index()
    return {"chunks": len(state["chunks"])}

@app.get("/search")
def search(q: str):
    hits = retrieve(q)
    return [{"file": h["file"], "page": h["page"], "score": round(h["score"], 2),
             "text": h["text"][:200]} for h in hits]

@app.post("/ask", response_model=Answer)
def ask(body: Question):
    if not state["llm_ready"]:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set")
    hits = retrieve(body.question)
    if not hits:
        return Answer(answer="مفيش معلومة عن ده في المستندات.", sources=[])
    try:
        answer = ask_llm(body.question, hits)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"LLM service error: {e}")
    return Answer(
        answer=answer,
        sources=[Source(file=h["file"], page=h["page"], score=h["score"]) for h in hits],
    )