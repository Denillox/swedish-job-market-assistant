import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langsmith import Client
from assistant.agent import _extract_text, _invoke_agent

DATASET_NAME = "swedish-job-market-assistant-eval"
EVAL_QUESTIONS_PATH = Path(__file__).resolve().parent / "eval_questions.json"


def load_questions() -> list[dict]:
    with open(EVAL_QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def create_or_get_dataset(client: Client):
    questions = load_questions()
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        return existing[0]

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Evaluation questions for the Swedish job market assistant",
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": q["question"]},
                "outputs": {
                    "expected_tool": q["expected_tool"],
                    "category": q["category"],
                    "notes": q["notes"],
                },
            }
            for q in questions
        ],
    )
    return dataset


def _extract_tool_calls(messages) -> list[str]:
    tool_names = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    tool_names.append(name)
        name = getattr(msg, "name", None)
        if name and msg.__class__.__name__ == "ToolMessage":
            tool_names.append(name)
    return tool_names


def target(inputs: dict) -> dict:
    response = _invoke_agent([("user", inputs["question"])])
    messages = response["messages"]
    return {
        "answer": _extract_text(messages[-1].content),
        "tool_calls": _extract_tool_calls(messages),
    }


def correctness_evaluator(run, example) -> dict:
    expected = example.outputs.get("expected_tool", "")
    expected_tools = [tool.strip() for tool in expected.split("+")]
    actual_tools = run.outputs.get("tool_calls", [])
    score = all(tool in actual_tools for tool in expected_tools)
    return {"key": "tool_correctness", "score": 1 if score else 0}


def main():
    client = Client()
    create_or_get_dataset(client)

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator],
        experiment_prefix="swedish-job-market-assistant",
        max_concurrency=2,
    )

    df = results.to_pandas()
    col = "feedback.tool_correctness"
    if col in df.columns:
        passed = df[col].sum()
        total = len(df)
        print(f"Tool correctness: {passed:.0f}/{total} ({100 * passed / total:.0f}%)")


if __name__ == "__main__":
    main()
