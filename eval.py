from app import build_index, retrieve

build_index()

# (السؤال، الصفحات الصح المتوقعة في ml_notes.pdf)
TESTS = [
    ("What is overfitting and how do I fix it?",            {1}),
    ("What does the learning rate control?",                {1}),
    ("What is the difference between precision and recall?", {1}),
    ("How does a convolutional filter work on images?",     {2}),
    ("How does transfer learning work with little data?",   {2}),
    ("What are the steps of the PyTorch training loop?",    {2}),
    ("How does attention use queries keys and values?",     {3}),
    ("What is a hallucination?",                            {3}),
    ("How does RAG reduce hallucinations?",                 {3}),
    ("Why does Arabic text produce more tokens?",           {3}),
]

top1 = topk = 0
for q, expected in TESTS:
    hits = retrieve(q, k=4)
    pages = [h["page"] for h in hits]
    ok1 = bool(pages) and pages[0] in expected
    okk = bool(expected & set(pages))
    top1 += ok1
    topk += okk
    print(("PASS" if ok1 else "MISS" if okk else "FAIL"), "|", q, "->", pages)

n = len(TESTS)
print(f"\nTop-1 accuracy: {top1}/{n}")
print(f"Top-4 recall:   {topk}/{n}")from app import build_index, retrieve

build_index()

# (السؤال، الصفحات الصح المتوقعة في ml_notes.pdf)
TESTS = [
    ("What is overfitting and how do I fix it?",            {1}),
    ("What does the learning rate control?",                {1}),
    ("What is the difference between precision and recall?", {1}),
    ("How does a convolutional filter work on images?",     {2}),
    ("How does transfer learning work with little data?",   {2}),
    ("What are the steps of the PyTorch training loop?",    {2}),
    ("How does attention use queries keys and values?",     {3}),
    ("What is a hallucination?",                            {3}),
    ("How does RAG reduce hallucinations?",                 {3}),
    ("Why does Arabic text produce more tokens?",           {3}),
]

top1 = topk = 0
for q, expected in TESTS:
    hits = retrieve(q, k=4)
    pages = [h["page"] for h in hits]
    ok1 = bool(pages) and pages[0] in expected
    okk = bool(expected & set(pages))
    top1 += ok1
    topk += okk
    print(("PASS" if ok1 else "MISS" if okk else "FAIL"), "|", q, "->", pages)

n = len(TESTS)
print(f"\nTop-1 accuracy: {top1}/{n}")
print(f"Top-4 recall:   {topk}/{n}")