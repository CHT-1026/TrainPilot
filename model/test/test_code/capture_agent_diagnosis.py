"""用Agent实际默认节点捕获模型输入；脚本模型只用于观测提示词，不评模型能力。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model.test.test_code.prepare_agent_diagnosis import ROOT, OUT, canonical, read_rows, save, sha


def capture(agent_dir, output):
    sys.path.insert(0, str(agent_dir))
    from app.workflows.chat_nodes.recognition import create_recognition_nodes
    from app.workflows.intent_prompt import INTENT_TEMPLATE
    from app.capabilities.retrieval.answer import verify_answer
    assert INTENT_TEMPLATE == (ROOT / "model/dataset/datasets/trainpilot_agent/intent_prompt_v4.txt").read_text(encoding="utf-8")
    records = read_rows(output / "cases.jsonl")
    captured, ablations = [], []
    suffixes = ("confirm-0", "confirm-4", "confirm-5", "confirm-6", "cancel-0", "cancel-4",
                "cancel-5", "cancel-6", "quoted-1", "actual-1", "leave-end", "approval-comment")
    for record in records:
        case = record["case"]
        current = {**record, "inference_messages": case["messages"][:-1], "sampling_enabled": True}
        if case["mode"] == "intent_parse":
            scenario = case["scenario"]
            state = {"user_message": scenario["user_message"], "messages": scenario.get("history", []),
                     "pending_action": scenario.get("pending_action"), "summary": scenario.get("summary", "")}
            llm = Mock()
            llm.invoke.return_value = SimpleNamespace(content=canonical(case["expected"]))
            with patch("app.workflows.chat_nodes.recognition.datetime") as clock:
                clock.now.return_value = datetime.fromisoformat(scenario["current_date"])
                result = create_recognition_nodes(llm)["intent_recognition"](state)
            llm.invoke.assert_called_once()
            assert result["intent"] == case["expected"]["intent"]
            prompt = llm.invoke.call_args.args[0]
            current["inference_messages"] = [{"role": "user", "content": prompt}]
            current["prompt_variant"] = "agent_default_trainpilot"
            stored_context = json.loads(case["messages"][0]["content"].split("\n给定上下文：\n", 1)[1].rsplit("\n用户输入：\n", 1)[0])
            runtime_context = json.loads(prompt.split("\n给定上下文：\n", 1)[1].rsplit("\n用户输入：\n", 1)[0])
            assert runtime_context == stored_context, case["id"]
            for family in ("v6-train-f00-", "v6-val-f10-"):
                if case["id"].startswith(family) and case["id"][len(family):] in suffixes:
                    ablations.append({**record, "id": "ablation/stored/" + case["id"],
                        "inference_messages": case["messages"][:-1], "sampling_enabled": False,
                        "prompt_variant": "stored_same_protocol", "paired_primary_id": record["id"]})
        captured.append(current)
    assert len(captured) == 214 and len(ablations) == 24
    all_cases = captured + ablations
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_cases)
    (output / "runtime_cases.jsonl").write_text(content, encoding="utf-8")
    paths = ("app/workflows/intent_prompt.py", "app/workflows/intent_prompt.txt",
             "app/workflows/chat_nodes/recognition.py", "local_runtime/trainpilot_cli.py",
             "app/capabilities/retrieval/answer.py")
    # 独立核验真实OA入口；结构通过不表示事实正确，不生成新的语义奖励。
    oa = []
    previous = ROOT / "runs/v6-eval-review-20261030/received/v6.jsonl"
    for row in read_rows(previous):
        if row["case"]["mode"] != "oa_rag": continue
        answer, citations, accepted = verify_answer(row["text"], row["case"]["evidence"])
        oa.append({"id": row["case"]["id"], "input": row["text"], "runtime_accepted": accepted,
                   "runtime_answer": answer, "citations": len(citations),
                   "strict_reference_exact": row["reference_exact"]})
    save(output / "runtime_capture.json", {"agent_directory": str(agent_dir),
         "source_sha256": {name: sha(agent_dir / name) for name in paths},
         "canonical_prompt_sha256": sha(agent_dir / "app/workflows/intent_prompt.txt"),
         "primary_cases": len(captured), "business_prompts_captured_from_actual_default_node": 145,
         "stored_vs_runtime_context_mismatches": 0, "stored_prompt_greedy_ablations": len(ablations),
         "runtime_cases_sha256": sha(output / "runtime_cases.jsonl"),
         "expected_completions_both_models": 3900,
         "model_called": False, "database_called": False,
         "note": "捕获节点使用脚本回答，只验证输入构造，不作为模型成绩；RAG保留给定证据输入。"})
    save(output / "oa_runtime_verification.json", {"items": oa,
         "note": "真实verify_answer主要核对引用结构/拒答表达，不能识别所有主体混淆或无依据数值。"})
    print("RUNTIME_CAPTURE_OK", len(all_cases), "cases; 3900 generations; no real model/DB calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, default=ROOT.parent / "TrainPilot-Agent-RAG")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    capture(args.agent_dir, args.output_dir)
