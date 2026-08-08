import json
import os
from datetime import datetime


LOG_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "eval_log.jsonl"
)


def log_eval(state):

    contexts = [
        doc.page_content
        for doc in state.get("documents", [])
    ]

    contexts += [
        result["content"]
        for result in state.get("prescription_sources", [])
        if result.get("content")
    ]

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "chat_id": state.get("chat_id", ""),
        "question": state.get("question", ""),
        "answer": state.get("answer", ""),
        "contexts": contexts,
        "ground_truth": state.get("ground_truth", ""),
        "judge_verdict": state.get("judge_verdict", ""),
        "judge_block_reason": state.get("judge_block_reason", ""),
        "route": state.get("route", "")
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
