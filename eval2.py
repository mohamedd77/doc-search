from app import build_index, retrieve, state

build_index()
print("chunks:", len(state["chunks"]))

ML, ENG = "ml_notes.pdf", "engineering_notes.pdf"

ANSWERABLE = [
    ("Why does my model score great on training data but poorly on new data?", (ML, 1)),
    ("How can I stop a network from memorizing the dataset?",                  (ML, 1)),
    ("Which layer should I replace when reusing a pretrained network?",        (ML, 2)),
    ("What is the difference between a Docker image and a container?",         (ENG, 4)),
    ("How do I combine rows from two tables in SQL?",                          (ENG, 2)),
    ("What does status code 422 mean?",                                        (ENG, 3)),
    ("How do I keep my passwords out of the repository?",                      (ENG, 4)),
    ("Why should I use a virtual environment?",                                (ENG, 1)),
]

UNANSWERABLE = [
    "Who won the 2022 football world cup?",
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
]

top1 = top4 = 0
for q, expected in ANSWERABLE:
    hits = retrieve(q, k=4)
    got = [(h["file"], h["page"]) for h in hits]
    ok1 = bool(got) and got[0] == expected
    ok4 = expected in got
    top1 += ok1
    top4 += ok4
    tag = "PASS" if ok1 else "MISS" if ok4 else "FAIL"
    print(tag, "|", q, "->", got[:2])

abstain = 0
for q in UNANSWERABLE:
    hits = retrieve(q, k=4)
    ok = len(hits) == 0
    abstain += ok
    print("OK  " if ok else "BAD ", "|", q, "->",
          [(h["file"], h["page"], round(h["score"], 2)) for h in hits[:2]])

print(f"\nTop-1: {top1}/{len(ANSWERABLE)}")
print(f"Top-4: {top4}/{len(ANSWERABLE)}")
print(f"Abstained correctly: {abstain}/{len(UNANSWERABLE)}")