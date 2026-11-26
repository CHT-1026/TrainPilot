"""使用原 Agent 状态逻辑检查人工编写的 V3 预览；不训练或调用模型。

在 Agent 的 Python 环境运行，--agent-project 指向其项目目录。
输出是待审核预览，不是模型评测成绩，也不生成 train/val/test。
"""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PREVIEW = ROOT / "model/dataset/datasets/trainpilot_agent/v3_preview"
FIELDS = {"leave_type", "start_date", "end_date", "reason", "app_id", "decision", "comments"}


class ScriptedIntent:
    def __init__(self):
        self.expected = None
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=json.dumps(self.expected, ensure_ascii=False, separators=(",", ":")))


async def build_preview(agent_project, output):
    sys.path.insert(0, str(agent_project))
    from app.domain.user.models import UserContext
    from local_runtime.chat_cli import create_terminal_graph, run_turn

    source = json.loads((output / "scenarios.json").read_text(encoding="utf-8"))
    fixed_now = datetime.fromisoformat(source["current_date"])

    class PreviewClock:
        @staticmethod
        def now():
            return fixed_now

    old_questions = {row["scenario"]["user_message"] for row in (
        json.loads(line) for line in (ROOT / "model/dataset/datasets/trainpilot_agent/v2/test.jsonl").read_text(encoding="utf-8").splitlines()
    )}
    samples = []
    for trajectory in source["trajectories"]:
        teacher = ScriptedIntent()
        graph = create_terminal_graph(teacher)
        thread = "preview-" + trajectory["id"]
        if trajectory.get("initial_state"):
            await graph.aupdate_state({"configurable": {"thread_id": thread}}, trajectory["initial_state"])
        user = UserContext(user_id="preview_account", name="", role=trajectory.get("role", "employee"))
        for index, turn in enumerate(trajectory["turns"], 1):
            expected = turn["expected"]
            assert set(expected) == {"intent", "extracted_info"}
            assert set(expected["extracted_info"]) <= FIELDS
            assert all(isinstance(value, str) and value for value in expected["extracted_info"].values())
            assert turn["user"] not in old_questions
            teacher.expected = expected
            previous_calls = len(teacher.prompts)
            trace = []
            with patch("app.workflows.chat_nodes.recognition.datetime", PreviewClock):
                state = await run_turn(graph, thread, turn["user"], user, trace.append)
            assert len(teacher.prompts) == previous_calls + 1, "预览只允许脚本化意图识别，不调用其他模型节点"
            prompt = teacher.prompts[-1]
            context = json.loads(prompt.split("\n给定上下文：\n", 1)[1].split("\n用户输入：\n", 1)[0])
            pending = state.get("pending_action")
            checks = {
                "intent": state["intent"] == expected["intent"],
                "route": all(f"[节点] {node}" in trace for node in turn["expected_nodes"]),
                "pending_type": (pending or {}).get("action_type") == turn["expected_pending_type"],
                "response": all(text in state.get("response", "") for text in turn.get("response_contains", [])),
            }
            if "expected_accumulated" in turn:
                checks["accumulated_fields"] = state.get("conversation_context", {}).get("accumulated_extracted_info") == turn["expected_accumulated"]
            if "expected_draft" in turn:
                draft = (pending or {}).get("draft", {})
                checks["draft"] = all(draft.get(key) == value for key, value in turn["expected_draft"].items())
            assert all(checks.values()), (turn["id"], checks)
            samples.append({
                "id": turn["id"], "trajectory_id": trajectory["id"], "turn": index,
                "status": "pending_user_review", "scenario": {**context, "user_message": turn["user"]},
                "expected": expected,
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False, separators=(",", ":"))}],
                "expected_nodes": turn["expected_nodes"], "runtime_checks": checks,
                "observed_response": state.get("response", ""),
                "observed_pending_action": pending,
                "execution_scope": turn.get("execution_scope", "原Agent本地草稿或状态更新；未调用模型、未访问数据库。"),
                "seed_note": trajectory.get("seed_note"),
                "trace": trace,
            })
    source_files = ["app/workflows/intent_protocol.py", "app/workflows/intent_protocol.txt",
                    "app/workflows/chat_nodes/recognition.py", "app/workflows/chat_nodes/leave.py",
                    "app/workflows/chat_nodes/response.py", "app/workflows/chat_nodes/execute.py",
                    "local_runtime/chat_cli.py"]
    artifact = {
        "status": "pending_user_review", "output_policy": "complete_known_fields",
        "note": "标签由人工编写后注入识别节点；检查的是协议和真实状态链路，不是模型准确率。",
        "scenarios_sha256": hashlib.sha256((output / "scenarios.json").read_bytes()).hexdigest(),
        "agent_sources_sha256": {name: hashlib.sha256((agent_project / name).read_bytes()).hexdigest() for name in source_files},
        "trajectory_count": len(source["trajectories"]), "sample_count": len(samples),
        "external_model_calls": 0, "database_writes": 0, "samples": samples,
    }
    (output / "samples.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# TrainPilot V3 多轮样例预览", "", "状态：待用户审核。只完成样例与实际状态链路检查，尚未生成正式数据或训练。", "",
             "已确认沿用 V2 的完整已知字段协议。下面的 JSON 是人工编写的预期输出，不是当前模型的回答。",
             "上下文由脚本化识别结果驱动原 Agent 图生成；初始审批状态为模拟数据，数据库节点明确不可用。", "",
             "## 本次重点", "", "追问补齐、修改后保留字段、确认与审批区分、取消草稿与撤销已提交申请区分、临时查询后返回原任务。", "",
             "## 样例", ""]
    for sample in samples:
        lines += [f"### {sample['id']} · {sample['trajectory_id']} · 第 {sample['turn']} 轮", "",
                  "用户：" + sample["scenario"]["user_message"], "",
                  f"输入包含历史 {len(sample['scenario']['history'])} 条；待确认动作：`{(sample['scenario'].get('pending_action') or {}).get('action_type')}`。", "",
                  "预期模型输出：", "", "```json", json.dumps(sample["expected"], ensure_ascii=False, separators=(",", ":")), "```", "",
                  "预期节点：" + " → ".join(sample["expected_nodes"]), "",
                  "原 Agent 在预设标签下的实际回复：", "", "```text", sample["observed_response"], "```", "",
                  "执行范围：" + sample["execution_scope"], ""]
    lines += ["## 审核后再进行的工作", "", "1. 确认表达、标签和状态边界，尤其是操作另一对象时如何处理旧待确认动作。",
              "2. 扩展真实轨迹和表达变化，按整条轨迹及模板族划分，不将旧评测题写入训练集。",
              "3. 用 Qwen tokenizer 检查完整历史和监督目标是否被截断，再决定长度和批次。",
              "4. 保留 V2；从同一 Full SFT 底座建立新的 LoRA 对照，并重测完整业务流程。", "",
              "详细上下文、模型输入、源代码哈希及状态检查见 [samples.json](samples.json)；人工场景定义见 [scenarios.json](scenarios.json)。", ""]
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"trajectories": artifact["trajectory_count"], "samples": len(samples),
                      "runtime_checks": "passed", "external_model_calls": 0, "database_writes": 0,
                      "status": "pending_user_review"}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PREVIEW)
    args = parser.parse_args()
    if (args.output_dir / "samples.json").exists():
        raise FileExistsError("已有预览结果，请先审核，或使用新的输出目录")
    asyncio.run(build_preview(args.agent_project.resolve(), args.output_dir.resolve()))


if __name__ == "__main__":
    main()
