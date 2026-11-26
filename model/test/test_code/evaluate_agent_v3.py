"""V2/V3业务对照及原Agent闭环测试；结果只存runs，不训练、不访问外部API。"""
import argparse
import asyncio
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from http.server import HTTPServer
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(rows):
    counts = {key: sum(bool(row[key]) for row in rows)
              for key in ("passed", "strict_json", "intent_ok", "fields_exact")}
    categories = {}
    for row in rows:
        bucket = categories.setdefault(row["case"]["category"], {"passed": 0, "total": 0})
        bucket["passed"] += bool(row["passed"])
        bucket["total"] += 1
    return {**counts, "total": len(rows), "exact_rate": counts["passed"] / len(rows),
            "truncated": sum(row["stop_reason"] != "eos" for row in rows),
            "generation_seconds": sum(row["seconds"] for row in rows), "categories": categories}


def model_evaluation(args):
    import datasets
    import torch
    from safetensors.torch import load_file
    from model.dataset.dataset import QwenSFTEncoder
    from model.test.test_code.evaluate_agent_lora import score_business
    from model.test.test_code.evaluate_qwen_sft import generate, sha256, supervised_stats
    from model.test.test_code.qwen_agent_api import load_pair, make_handler

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model_summary.json").exists():
        raise FileExistsError("已有模型评测，请换output-dir")
    torch.set_num_threads(2)
    versions = {name: ROOT / f"runs/trainpilot_agent_lora_{name}/adapter" for name in ("v2", "v3")}
    suites = {}
    for name in ("v2", "v3"):
        folder = ROOT / f"model/dataset/datasets/trainpilot_agent/{name}"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["outputs"].values():
            assert sha256(folder / item["file"]) == item["sha256"]
        suites[name] = [row for row in read_rows(folder / "test.jsonl") if row["mode"] == "intent_parse"]
    save(output / "model_cases.json", suites)  # 生成前冻结题目和标签。
    state = torch.load(ROOT / "runs/trainpilot_agent_lora_v3/checkpoint.pt", map_location="cpu", weights_only=True)
    export = load_file(str(versions["v3"] / "adapter_model.safetensors"))
    assert state["weights"].keys() == export.keys()
    assert all(torch.equal(state["weights"][name], value) for name, value in export.items())
    source = json.loads((ROOT / "runs/trainpilot_agent_lora_v3/run_manifest.json").read_text(encoding="utf-8"))
    assert source["data"]["source_sha256"] == sha256(ROOT / "model/dataset/datasets/trainpilot_agent/v3/train.jsonl")
    artifact = {"progress": state["progress"], "adapter_tensors": len(export),
                "checkpoint_matches_export": True, "training_manifest": source}
    del state, export
    save(output / "artifact_check.json", artifact)
    model, tokenizer, metadata = load_pair(ROOT / "runs/qwen-fullsft/checkpoint.pt", ROOT / "runs/qwen-base-reference", versions["v2"])
    model.load_adapter(str(versions["v3"]), adapter_name="v3", is_trainable=False, autocast_adapter_dtype=False)
    model.eval()
    metadata.update(adapters={name: {"path": str(path), "sha256": sha256(path / "adapter_model.safetensors")} for name, path in versions.items()},
                    control="同一份Full SFT底座、BF16推理、贪心解码，分别启用V2和V3；不合并、不训练。",
                    max_new_tokens=args.max_new_tokens, timestamp=datetime.now().astimezone().isoformat())
    save(output / "model_metadata.json", metadata)
    encoder = QwenSFTEncoder(tokenizer, 1536)
    summaries = {}
    with torch.inference_mode():
        for version in ("v2", "v3"):
            model.set_adapter("default" if version == "v2" else "v3")
            summaries[version] = {}
            with (output / f"{version}_generations.jsonl").open("w", encoding="utf-8") as stream:
                for suite_name, cases in suites.items():
                    rows, loss_sum, correct, targets = [], 0.0, 0, 0
                    for index, case in enumerate(cases, 1):
                        ids = tokenizer.apply_chat_template(case["messages"][:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                        result = generate(model, tokenizer, ids, "cuda", args.max_new_tokens)
                        row = {"variant": version, "suite": suite_name, "case": case, **result, **score_business(case, result["text"])}
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        rows.append(row)
                        encoded = encoder(case)
                        inputs = torch.tensor([encoded["input_ids"]], device="cuda")
                        labels = torch.tensor([encoded["labels"]], device="cuda")
                        logits = model(input_ids=inputs, use_cache=False).logits
                        loss, matched, count = supervised_stats(logits, labels)
                        loss_sum += loss
                        correct += matched
                        targets += count
                        del inputs, labels, logits
                        if index == 1 or index % 15 == 0:
                            print(f"{version} suite={suite_name} {index}/{len(cases)} exact={sum(r['passed'] for r in rows)}", flush=True)
                    summaries[version][suite_name] = {**summarize(rows), "loss": loss_sum / targets,
                                                     "ppl": math.exp(loss_sum / targets), "supervised_tokens": targets,
                                                     "teacher_forced_token_accuracy": correct / targets}
                    save(output / "model_summary.json", summaries)
    metadata["checkpoint_sha256_after"] = sha256(ROOT / "runs/qwen-fullsft/checkpoint.pt")
    assert metadata["checkpoint_sha256_after"] == metadata["checkpoint_sha256"]
    for name, path in versions.items():
        assert sha256(path / "adapter_model.safetensors") == metadata["adapters"][name]["sha256"]
    metadata["weights_unchanged"] = True
    save(output / "model_metadata.json", metadata)
    print("MODEL SUMMARY " + json.dumps(summaries, ensure_ascii=False), flush=True)
    if args.serve:
        def infer(messages, limit, version):
            if version == "sft":
                context = model.disable_adapter()
            else:
                model.set_adapter("default" if version == "v2" else "v3")
                context = nullcontext()
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            with context:
                result = generate(model, tokenizer, ids, "cuda", min(limit, args.max_new_tokens))
            with (output / "agent_requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"variant": version, "messages": messages, **result}, ensure_ascii=False) + "\n")
            return result
        handler = make_handler(lambda messages, limit: infer(messages, limit, "v3"), "trainpilot-v3", {
            "trainpilot-v2": lambda messages, limit: infer(messages, limit, "v2"),
            "trainpilot-sft": lambda messages, limit: infer(messages, limit, "sft"),
            "trainpilot-lora": lambda messages, limit: infer(messages, limit, "v3"),
        })
        print(f"READY http://127.0.0.1:{args.port}/v1", flush=True)
        with HTTPServer(("127.0.0.1", args.port), handler) as server:
            server.serve_forever()


def diagnostics():
    leave = {"leave_type": "事假", "start_date": "2026-11-09", "end_date": "2026-11-10", "reason": "家中维修"}
    pending = {"action_type": "submit_leave", "draft": {**leave, "working_days": 2, "requires_hr": False}}
    seed = {"messages": [{"role": "user", "content": "申请2026年11月9日到10日的事假，原因是家中维修。"},
                         {"role": "assistant", "content": "请核对草稿：事假，2026-11-09至2026-11-10，原因家中维修。回复确认提交或取消放弃。"}],
            "pending_action": pending, "conversation_context": {"accumulated_extracted_info": leave}}

    def turn(user, intent, fields=None, nodes=None, pending_type="submit_leave", draft=None):
        result = {"user": user, "expected": {"intent": intent, "extracted_info": fields or {}},
                  "expected_nodes": nodes or ["submit_leave"], "expected_pending_type": pending_type}
        if draft is not None:
            result["expected_draft"] = draft
        return result

    rows = [{"id": "legacy_four_turn", "kind": "legacy_regression", "turns": [
        turn("我想请事假", "submit_leave", {"leave_type": "事假"}, pending_type=None),
        turn("时间是2026年11月2日到11月3日，原因是个人事务。", "submit_leave",
             {"leave_type": "事假", "start_date": "2026-11-02", "end_date": "2026-11-03", "reason": "个人事务"}),
        turn("确认提交", "confirm_action", nodes=["execute_pending"]),
        turn("算了，取消草稿", "cancel_action", nodes=["clear_pending"], pending_type=None),
    ]}, {"id": "cancel_then_new_natural", "kind": "cancel_reset", "initial_state": deepcopy(seed), "turns": [
        turn("这份先不要了。", "cancel_action", nodes=["clear_pending"], pending_type=None),
        turn("再开一份年假，什么时候休我还没想好。", "submit_leave", {"leave_type": "年假"}, pending_type=None),
    ]}, {"id": "reason_only_update", "kind": "unseen_field_edit", "initial_state": deepcopy(seed), "turns": [
        turn("原因改成“办理证件”，时间和假别不改。", "submit_leave", {**leave, "reason": "办理证件"}, draft={**leave, "reason": "办理证件"})
    ]}, {"id": "not_submit_but_modify", "kind": "negation_and_edit", "initial_state": deepcopy(seed), "turns": [
        turn("先不要提交，把结束时间改到2026年11月12日，其他不变。", "submit_leave", {**leave, "end_date": "2026-11-12"}, draft={**leave, "end_date": "2026-11-12"})
    ]}, {"id": "short_confirmation", "kind": "natural_confirmation", "initial_state": deepcopy(seed), "turns": [
        turn("行，就这样办。", "confirm_action", nodes=["execute_pending"])
    ]}, {"id": "no_pending_confirm", "kind": "missing_object", "turns": [
        turn("确认。", "clarify", nodes=["clarify"], pending_type=None)
    ]}, {"id": "no_pending_cancel", "kind": "missing_object", "turns": [
        turn("取消。", "clarify", nodes=["clarify"], pending_type=None)
    ]}]
    withdrawal = {"messages": [{"role": "user", "content": "我要撤回已提交的leave_2026_A102，请先准备草稿。"},
                                {"role": "assistant", "content": "待确认撤回申请leave_2026_A102，回复确认执行撤回，回复取消放弃这项撤回。"}],
                  "pending_action": {"action_type": "cancel_leave", "draft": {"app_id": "leave_2026_A102"}}}
    rows.extend([
        {"id": "confirm_withdrawal", "kind": "withdrawal_boundary", "initial_state": deepcopy(withdrawal), "turns": [
            turn("确认取消这张申请。", "confirm_action", nodes=["execute_pending"], pending_type="cancel_leave")]},
        {"id": "abandon_withdrawal", "kind": "withdrawal_boundary", "initial_state": deepcopy(withdrawal), "turns": [
            turn("不撤回了，把刚才准备的撤回操作取消掉。", "cancel_action", nodes=["clear_pending"], pending_type=None)]},
        {"id": "other_request_natural", "kind": "object_selection", "initial_state": deepcopy(withdrawal), "turns": [
            turn("刚才那单先放着，我要撤的是已提交的leave_2026_B309，不是leave_2026_A102。", "delete_application",
                 {"app_id": "leave_2026_B309"}, nodes=["delete"], pending_type="cancel_leave")]},
        {"id": "suggestion_then_decision", "kind": "advice_and_action", "role": "manager", "turns": [
            turn("申请leave_2026_C782该不该通过？先给我建议。", "get_suggestion", {"app_id": "leave_2026_C782"}, nodes=["suggestion"], pending_type=None),
            turn("我决定通过leave_2026_C782，意见写“材料齐全”。", "approve",
                 {"app_id": "leave_2026_C782", "decision": "通过", "comments": "材料齐全"}, nodes=["approve"], pending_type=None)]},
    ])
    for index, question in enumerate(("你是谁？", "你叫什么名字？"), 1):
        item = turn(question, "general_chat", nodes=["general_chat"], pending_type=None)
        item["response_contains"] = ["TrainPilot"]
        rows.append({"id": f"identity_in_agent_{index}", "kind": "agent_identity", "turns": [item]})
    for row in rows:
        row.update(split="diagnostic", current_date="2026-11-02", role=row.get("role", "employee"))
    return rows


async def workflow_evaluation(args):
    import logging
    logging.disable(logging.INFO)
    sys.path.insert(0, str(args.agent_project.resolve()))
    from app.domain.user.models import UserContext
    from app.capabilities.policy_engine import validate_leave_draft
    from langchain_openai import ChatOpenAI
    from local_runtime.chat_cli import create_terminal_graph, run_turn
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "workflow_summary.json").exists():
        raise FileExistsError("已有流程评测，请换output-dir")
    heldout = [row for row in read_rows(ROOT / "model/dataset/datasets/trainpilot_agent/v3/trajectories.jsonl") if row["split"] == "test"]
    cases = heldout + diagnostics()
    save(output / "workflow_cases.json", cases)
    summaries = {}
    for version in ("v2", "v3"):
        llm = ChatOpenAI(model=f"trainpilot-{version}", base_url=args.base_url, api_key="local-evaluation", temperature=0, timeout=120, max_retries=0)
        rows = []
        with (output / f"workflow_{version}.jsonl").open("w", encoding="utf-8") as stream:
            for number, case in enumerate(cases, 1):
                graph = create_terminal_graph(llm)
                thread = f"eval-{version}-{case['id']}"
                if case.get("initial_state"):
                    await graph.aupdate_state({"configurable": {"thread_id": thread}}, deepcopy(case["initial_state"]))
                fixed_now = datetime.fromisoformat(case["current_date"])
                class Clock:
                    @staticmethod
                    def now():
                        return fixed_now
                user = UserContext(user_id="evaluation_account", name="", role=case.get("role", "employee"))
                for index, turn in enumerate(case["turns"], 1):
                    trace = []
                    with patch("app.workflows.chat_nodes.recognition.datetime", Clock), patch(
                        "app.workflows.chat_nodes.leave.validate_leave_draft", side_effect=lambda raw: validate_leave_draft(raw, today=fixed_now.date())
                    ):
                        state = await run_turn(graph, thread, turn["user"], user, trace.append)
                    draft = (state.get("pending_action") or {}).get("draft", {})
                    extracted = state.get("extracted_info", {})
                    nodes = [item.removeprefix("[节点] ") for item in trace if item.startswith("[节点] ")]
                    checks = {"intent": state.get("intent") == turn["expected"]["intent"],
                              "fields_exact": extracted == turn["expected"]["extracted_info"],
                              "nodes": all(node in nodes for node in turn["expected_nodes"]),
                              "pending_type": (state.get("pending_action") or {}).get("action_type") == turn["expected_pending_type"],
                              "draft": all(draft.get(key) == value for key, value in turn.get("expected_draft", {}).items())}
                    accumulated = state.get("conversation_context", {}).get("accumulated_extracted_info", {})
                    if "expected_accumulated" in turn:
                        checks["accumulated"] = accumulated == turn["expected_accumulated"]
                    if "response_contains" in turn:
                        checks["response"] = all(text.lower() in (state.get("response") or "").lower()
                                                 for text in turn["response_contains"])
                    row = {"variant": version, "trajectory_id": case["id"], "suite": case["split"],
                           "category": case["kind"], "turn": index, "user": turn["user"], "expected": turn["expected"],
                           "parsed": {"intent": state.get("intent"), "extracted_info": extracted},
                           "response": state.get("response"), "pending": state.get("pending_action"),
                           "accumulated": accumulated, "trace": trace, "checks": checks, "passed": all(checks.values())}
                    stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    stream.flush()
                    rows.append(row)
                print(f"workflow {version} {number}/{len(cases)} {case['id']} passed={all(r['passed'] for r in rows if r['trajectory_id']==case['id'])}", flush=True)
        summaries[version] = {}
        for suite in ("test", "diagnostic"):
            selected = [row for row in rows if row["suite"] == suite]
            ids = {row["trajectory_id"] for row in selected}
            summaries[version][suite] = {
                "turns": len(selected), "turns_exact": sum(row["passed"] for row in selected),
                "intent_correct": sum(row["checks"]["intent"] for row in selected),
                "fields_exact": sum(row["checks"]["fields_exact"] for row in selected),
                "trajectories": len(ids),
                "trajectories_all_passed": sum(all(row["passed"] for row in selected if row["trajectory_id"] == key) for key in ids),
            }
        save(output / "workflow_summary.json", summaries)
    print("WORKFLOW SUMMARY " + json.dumps(summaries, ensure_ascii=False), flush=True)


def comparison(args):
    output = args.output_dir
    models = json.loads((output / "model_summary.json").read_text(encoding="utf-8"))
    workflows = json.loads((output / "workflow_summary.json").read_text(encoding="utf-8"))
    paired = {}
    generations = {name: read_rows(output / f"{name}_generations.jsonl") for name in ("v2", "v3")}
    for suite in ("v2", "v3"):
        before = {row["case"]["id"]: row for row in generations["v2"] if row["suite"] == suite}
        after = {row["case"]["id"]: row for row in generations["v3"] if row["suite"] == suite}
        assert before.keys() == after.keys()
        paired[suite] = {"fixed": [key for key in before if not before[key]["passed"] and after[key]["passed"]],
                         "regressed": [key for key in before if before[key]["passed"] and not after[key]["passed"]],
                         "remaining_failures": [
                             {"id": key, "category": row["case"]["category"],
                              "user": row["case"]["scenario"]["user_message"],
                              "expected": row["case"]["expected"], "actual": row["text"]}
                             for key, row in after.items() if not row["passed"]]}
    workflow_rows = {name: read_rows(output / f"workflow_{name}.jsonl") for name in ("v2", "v3")}
    diagnostics_summary = {}
    for name, rows in workflow_rows.items():
        diagnostics_summary[name] = []
        for key in dict.fromkeys(row["trajectory_id"] for row in rows if row["suite"] == "diagnostic"):
            selected = [row for row in rows if row["trajectory_id"] == key]
            diagnostics_summary[name].append({"id": key, "passed": all(row["passed"] for row in selected),
                                              "turns": len(selected), "turns_exact": sum(row["passed"] for row in selected)})
    training = read_rows(ROOT / "runs/trainpilot_agent_lora_v3/metrics.jsonl")
    result = {
        "model_summary": models, "workflow_summary": workflows, "paired_static_cases": paired,
        "diagnostic_trajectories": diagnostics_summary,
        "training_epochs": [row for row in training if row["event"] == "epoch"],
        "gradient_accumulation": {"micro_batch_size": 1, "accumulation_steps": 8,
                                  "effective_batch_size": 8, "last_group_samples": 5,
                                  "updates_per_epoch": 220, "total_updates": 440,
                                  "weighting": "按本组有效assistant监督token数加权；首个micro-batch清零，最后一个裁剪梯度并更新。"},
        "scope": "新增144条静态题与60条运行轨迹来自同一批测试场景，不能相加当成独立测试数；另13条诊断流程不参与训练。",
        "database_scope": "原终端图替换数据库节点为明确不可用提示；只测理解、路由及本地状态，不代表真实落库或权限校验成功率。",
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if (output / "identity_diagnostic.json").exists():
        result["identity_diagnostic"] = json.loads((output / "identity_diagnostic.json").read_text(encoding="utf-8"))
    save(output / "evaluation_summary.json", result)
    print(json.dumps({"models": {name: {suite: f"{r['passed']}/{r['total']}" for suite, r in suites.items()}
                                 for name, suites in models.items()}, "workflows": workflows,
                      "diagnostics": diagnostics_summary}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("models", "workflows", "summary"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/lora-v3-eval-20261023")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--agent-project", type=Path)
    args = parser.parse_args()
    if args.mode == "models":
        model_evaluation(args)
    elif args.mode == "summary":
        comparison(args)
    else:
        if args.agent_project is None:
            parser.error("workflows需要--agent-project")
        asyncio.run(workflow_evaluation(args))


if __name__ == "__main__":
    main()
