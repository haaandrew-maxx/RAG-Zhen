from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import pandas as pd
load_dotenv()
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy

GROUND_TRUTH_FILE = "EPD-questions-ground_truths.xlsx"
LOGS_FILE = "rag_logs.jsonl"

df = pd.read_excel(GROUND_TRUTH_FILE)

if "question" not in df.columns or "ground_truth" not in df.columns:
    raise ValueError("Excel must have 'question' and 'ground_truth' columns")

ground_truth_map = dict(zip(df["question"], df["ground_truth"]))

rag_answers = {}   # {question: answer}

with open(LOGS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)

        q = obj.get("question")
        ans = None

        if isinstance(obj.get("answer"), dict):
            ans = obj["answer"].get("answer")
        else:
            ans = obj.get("answer")

        if q and ans:
            rag_answers[q] = ans

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = AnswerAccuracy(llm=llm)

# Evaluate

results = []
missing = []

for q, reference in ground_truth_map.items():
    if q not in rag_answers:
        missing.append(q)
        continue

    user_input = q
    response = rag_answers[q]
    print(f"\nPregunta: {user_input}")
    score = scorer.score(user_input = q, response = rag_answers[q], reference = reference)
    print(f" Score: {score.value:.4f}")
    results.append(score.value)


print("\n==============================")
print("RAG Accuracy Evaluation Result")
print("==============================")

print(f"Total questions: {len(ground_truth_map)}")
print(f"Matched logs:    {len(results)}")
print(f"Missing logs:    {len(missing)}")

if results:
    avg_score = sum(results) / len(results)
    print(f"\nFinal accuracy score: {avg_score:.4f}")

print("\nMissing questions (no logs found):")
for q in missing:
    print("  -", q)