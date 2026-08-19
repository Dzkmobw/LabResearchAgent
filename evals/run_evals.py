"""Run the minimal LabResearchBench against the LangGraph agent."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agent.graph import graph
from evals.evaluators import evaluate_case


DATASET_PATH = Path(__file__).with_name("dataset.jsonl")


async def run() -> None:
    rows = _load_dataset(DATASET_PATH)
    results: list[dict[str, Any]] = []
    for row in rows:
        output = await graph.ainvoke(row["input"])
        eval_result = evaluate_case(output, row.get("expectations", {}))
        results.append({"case_id": row["case_id"], **eval_result})

    passed = sum(1 for result in results if result["passed"])
    print(json.dumps({"passed": passed, "total": len(results), "results": results}, indent=2))


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    asyncio.run(run())
