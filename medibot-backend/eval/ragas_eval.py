import json
import os

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness


LOG_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "eval_log.jsonl"
)

FAITHFULNESS_THRESHOLD = 0.7


def load_log():

    if not os.path.exists(LOG_PATH):
        return []

    records = []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def build_dataset(records):

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for record in records:

        # Skip blocked answers — judge already flagged these
        if record.get("judge_verdict") == "block":
            continue

        # Skip records where judge could not produce a ground truth
        if not record.get("ground_truth"):
            continue

        data["question"].append(record["question"])
        data["answer"].append(record["answer"])
        data["contexts"].append(record["contexts"])
        data["ground_truth"].append(record["ground_truth"])

    return Dataset.from_dict(data)


def run_eval():

    records = load_log()
    dataset = build_dataset(records)

    if len(dataset) == 0:
        print("No records to evaluate.")
        return None

    print(f"Running RAGAS evaluation on {len(dataset)} records...\n")

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness]
    )

    df = results.to_pandas()

    print(df[["question", "faithfulness"]].to_string(index=False))
    print(f"\nMean faithfulness: {df['faithfulness'].mean():.3f}")
    print(f"Pass (>= {FAITHFULNESS_THRESHOLD}): {(df['faithfulness'] >= FAITHFULNESS_THRESHOLD).sum()} / {len(df)}")

    return df


if __name__ == "__main__":
    run_eval()
