import json
import time
from src.agent import build_agent, extract_text

DOCUMENT_ID_PATH = "eval/test_document_id.json"
DELAY_BETWEEN_QUESTIONS = 5


def load_document_id():
    with open(DOCUMENT_ID_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["document_id"]


def get_tool_calls_used(messages):
    tool_names = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                tool_names.append(call["name"])
    return tool_names


def check_answer(answer, expected_substrings):
    answer_lower = answer.lower()
    return any(sub.lower() in answer_lower for sub in expected_substrings)


def run_eval(questions_path="eval/eval_questions.json"):
    document_id = load_document_id()
    document_label = "test paper"

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    agent, used_images = build_agent([document_id], {document_id: document_label})

    results = []

    for q in questions:
        print(f"\n[{q['id']}] {q['question']}")

        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": q["question"]}]
            })
            final_content = result["messages"][-1].content
            answer = extract_text(final_content)

            tools_used = get_tool_calls_used(result["messages"])
            tool_correct = q["expected_tool"] in tools_used
            answer_correct = check_answer(answer, q["expected_answer_contains"])

            passed = tool_correct and answer_correct

            print(f"  Tools used: {tools_used}")
            print(f"  Expected tool: {q['expected_tool']} -> {'OK' if tool_correct else 'MISS'}")
            print(f"  Answer: {answer[:150]}")
            print(f"  Expected content: {q['expected_answer_contains']} -> {'OK' if answer_correct else 'MISS'}")
            print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

            results.append({
                "id": q["id"],
                "question": q["question"],
                "passed": passed,
                "tool_correct": tool_correct,
                "answer_correct": answer_correct
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": q["id"], "question": q["question"], "passed": False, "error": str(e)})

        time.sleep(DELAY_BETWEEN_QUESTIONS)

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    pass_rate = passed_count / total * 100

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed_count}/{total} passed ({pass_rate:.1f}%)")
    print(f"{'='*50}")

    with open("eval/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "passed": passed_count,
            "pass_rate": pass_rate,
            "results": results
        }, f, indent=2)

    return results, pass_rate


if __name__ == "__main__":
    results, pass_rate = run_eval()

    THRESHOLD = 70.0
    if pass_rate < THRESHOLD:
        print(f"\nFAILED: pass rate {pass_rate:.1f}% is below threshold {THRESHOLD}%")
        exit(1)
    else:
        print(f"\nPASSED: pass rate {pass_rate:.1f}% meets threshold {THRESHOLD}%")
        exit(0)