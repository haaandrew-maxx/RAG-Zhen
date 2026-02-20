from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import pandas as pd
load_dotenv()
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy

GROUND_TRUTH_FILE = "Groundtruth.csv"
LOGS_FILE = "rag_logs.jsonl"

df = pd.read_csv(GROUND_TRUTH_FILE)

if "query" not in df.columns or "groundtruth" not in df.columns:
    raise ValueError("CSV must have 'query' and 'groundtruth' columns")

ground_truth_map = dict(zip(df["query"], df["groundtruth"]))

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
    print(f"\nQuestion: {user_input}")
    
    # Get score from LLM
    score = scorer.score(user_input = q, response = rag_answers[q], reference = reference)
    
    # Convert to binary score (0 or 1)
    # Threshold is 0.5: if score >= 0.5 then correct (1), otherwise incorrect (0)
    binary_score = 1 if score.value >= 0.5 else 0
    
    print(f" Raw score: {score.value:.4f}")
    print(f" Binary score: {binary_score} ({'✓ Correct' if binary_score == 1 else '✗ Incorrect'})")
    results.append(binary_score)


print("\n==============================")
print("RAG Accuracy Evaluation Result")
print("==============================")

print(f"Total questions:    {len(ground_truth_map)}")
print(f"Matched logs:       {len(results)}")
print(f"Missing logs:       {len(missing)}")

if results:
    correct_count = sum(results)
    total_count = len(results)
    accuracy = correct_count / total_count
    print(f"\nCorrect answers:    {correct_count}")
    print(f"Total evaluated:    {total_count}")
    print(f"Accuracy rate:      {accuracy:.2%} ({accuracy:.4f})")

print("\nMissing questions (no logs found):")
for q in missing:
    print("  -", q)