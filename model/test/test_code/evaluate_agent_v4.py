"""V4只读评测：旧业务回归、提示词对照、两轮adapter及实际终端Agent。

输出只写runs；不训练、不改Agent源文件、不访问外部模型或业务数据库。
freeze在看生成结果前固定本助手逐条新编题。models完成后可serve给另一Python环境跑flows。
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime
from http.server import HTTPServer
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from model.dataset.build_agent_data import canonical, read_jsonl, render_prompt
from model.test.test_code.evaluate_agent_v3 import diagnostics

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "model/dataset/datasets/trainpilot_agent"
OUT = ROOT / "runs/lora-v4-eval-20261023"
VARIANTS = ("v3_prompt4", "v4_epoch1", "v4_epoch2")


def save(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def seed(kind):
    if kind == "submit_leave":
        draft = {"leave_type": "事假", "start_date": "2027-09-06", "end_date": "2027-09-07", "reason": "家中电路维修"}
        text = "请核对未提交草稿：事假，2027-09-06至2027-09-07，原因为家中电路维修。请确认执行或放弃。"
    elif kind == "approve":
        draft = {"app_id": "leave_6fa803c29b15", "decision": "rejected", "decision_label": "驳回", "comments": "材料还需补齐"}
        text = "待确认审批：申请leave_6fa803c29b15拟驳回，意见是材料还需补齐。现在尚未执行，请确认或放弃。"
    else:
        draft = {"app_id": "leave_5b17d29ae803"}
        text = "待确认撤回已提交申请leave_5b17d29ae803。目前没有执行撤回，请确认执行或放弃撤回。"
    return {"messages": [{"role": "assistant", "content": text}],
            "pending_action": {"action_type": kind, "draft": draft},
            "conversation_context": {"accumulated_extracted_info": draft if kind == "submit_leave" else {}}}


def new_cases(rules):
    rows = []
    def add(category, user, intent, fields=None, state=None, today="2027-08-23"):
        state = state or {}
        scene = {"current_date": today, "user_message": user, "history": state.get("messages", []),
                 "pending_action": state.get("pending_action"), "summary": state.get("summary", "")}
        expected = {"intent": intent, "extracted_info": fields or {}}
        rows.append({"id": f"manual-{len(rows)+1:02d}", "category": category, "mode": "intent_parse",
                     "group_id": "manual-new-evaluation-20261023", "scenario": scene, "expected": expected,
                     "messages": [{"role": "user", "content": render_prompt(rules, scene)},
                                  {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False, separators=(",", ":"))}]})
    for command, label in (("已核验，执行上面待我确认的动作。", "confirm_action"),
                           ("把当前准备执行的动作放弃掉，这次不办。", "cancel_action")):
        for kind in (None, "submit_leave", "approve", "cancel_leave"):
            add("state_counterfactual", command, label if kind else "clarify", state=seed(kind) if kind else None)
    add("no_pending_after_cancel", "刚才说的确认执行。", "clarify", state={"messages": [{"role": "assistant", "content": "上一项待办已取消，目前没有待确认动作。"}]})
    add("withdrawal_boundary", "不是放弃撤回，我要确认把原申请撤掉。", "confirm_action", state=seed("cancel_leave"))
    add("withdrawal_boundary", "别撤掉原单，只把本次撤回计划终止。", "cancel_action", state=seed("cancel_leave"))
    add("withdrawal_boundary", "撤回草稿已看过，按这项撤回内容执行就行。", "confirm_action", state=seed("cancel_leave"))
    add("withdrawal_boundary", "原申请留着，我不再同意执行撤回。", "cancel_action", state=seed("cancel_leave"))
    leave = seed("submit_leave")["pending_action"]["draft"]
    for user, change in (("首日写2027-09-07，别的项目用原来的。", {"start_date": "2027-09-07"}),
                         ("止日改为2027-09-08，假别、起日与事由保留。", {"end_date": "2027-09-08"}),
                         ("两个日期一起更正成2027-09-13至2027-09-14，别删掉之前的事由。", {"start_date": "2027-09-13", "end_date": "2027-09-14"}),
                         ("这次按年假来填，原日期及维修事由都保留。", {"leave_type": "年假"}),
                         ("事由用“房屋水管检修”，其他三项不要变。", {"reason": "房屋水管检修"}),
                         ("我还没确认提交，只是把事由从电路维修更正为“办理过户手续”。", {"reason": "办理过户手续"})):
        add("complete_field_edit", user, "submit_leave", {**leave, **change}, seed("submit_leave"))
    add("summary_evidence", "最后一天延一天到2027-09-08，其余按摘要里未提交的那份。", "submit_leave",
        {**leave, "end_date": "2027-09-08"}, {"summary": "未提交事假：2027-09-06至2027-09-07；原因家中电路维修。"})
    add("approval_edit", "决定不变，意见重新写“请补充日期证明”。", "approve",
        {"app_id": "leave_6fa803c29b15", "decision": "驳回", "comments": "请补充日期证明"}, seed("approve"))
    add("approval_edit", "审核材料后改成通过，原编号和意见仍照草稿。", "approve",
        {"app_id": "leave_6fa803c29b15", "decision": "通过", "comments": "材料还需补齐"}, seed("approve"))
    add("comment_span", "对leave_491ac72d6be3作通过处理；给出的意见正文是“已补全相关证明”。", "approve",
        {"app_id": "leave_491ac72d6be3", "decision": "通过", "comments": "已补全相关证明"})
    add("advice_not_approval", "我不是让你批准leave_c593be71602a，先说说审批建议。", "get_suggestion", {"app_id": "leave_c593be71602a"})
    for text in ("本轮撤回对象是leave_9b0528fc731d，眼前那项审批先不动。",
                 "不要动leave_6fa803c29b15，撤回自己已提交的leave_9b0528fc731d。",
                 "撤回已提交的leave_9b0528fc731d；不是leave_6fa803c29b15。"):
        add("object_selection", text, "delete_application", {"app_id": "leave_9b0528fc731d"}, seed("approve"))
    add("ambiguous_object", "其中那个撤销一下。", "clarify", state={"messages": [{"role": "assistant", "content": "已提及leave_9b0528fc731d与leave_491ac72d6be3，但没有确定本次目标。"}]})
    add("multi_task", "替我拟请假申请，同时把待我处理的审批列表调出来。", "clarify")
    add("missing_fields", "需要事假，原因是办理过户手续，暂时没法说起止。", "submit_leave", {"leave_type": "事假", "reason": "办理过户手续"})
    add("missing_fields", "申请从2027-09-06到2027-09-07的假，事由家中检修，假别还在考虑。", "submit_leave",
        {"start_date": "2027-09-06", "end_date": "2027-09-07", "reason": "家中检修"})
    add("calendar_boundary", "后天只请当天的年假，理由是集中休息。", "submit_leave",
        {"leave_type": "年假", "start_date": "2028-02-29", "end_date": "2028-02-29", "reason": "集中休息"}, today="2028-02-27")
    add("calendar_boundary", "明天和后天请事假，原因为家中检修。", "submit_leave",
        {"leave_type": "事假", "start_date": "2028-01-01", "end_date": "2028-01-02", "reason": "家中检修"}, today="2027-12-31")
    add("policy_not_action", "“确认提交请假”在操作流程里是哪一步？我是询问含义。", "policy_qa")
    add("query_status", "先看看本人已发起的休假单目前的办理情况。", "query_status")
    add("query_pending", "让我核对现在轮到我处理的请假审批清单。", "query_pending")
    for question in ("你在这套助手系统里用什么名字？", "你到底是MiniMind还是TrainPilot？", "能先说明一下你的身份，再开始吗？"):
        add("agent_identity", question, "general_chat")
    return rows


def manual_workflows():
    def turn(user, intent, fields=None, pending=None, draft=None, response=None):
        nodes = {"submit_leave": "submit_leave", "confirm_action": "execute_pending", "cancel_action": "clear_pending", "clarify": "clarify", "general_chat": "general_chat"}
        item = {"user": user, "expected": {"intent": intent, "extracted_info": fields or {}},
                "expected_nodes": [nodes[intent]], "expected_pending_type": pending}
        if draft:
            item["expected_draft"] = draft
        if response:
            item["response_contains"] = response
        return item
    first = {"leave_type": "年假", "start_date": "2027-09-06", "end_date": "2027-09-07", "reason": "返乡探亲"}
    changed = {**first, "start_date": "2027-09-13", "end_date": "2027-09-14"}
    workflows = [{"id": "new_collect_and_both_dates", "turns": [
        turn("我想请年假，先别替我填时间和理由。", "submit_leave", {"leave_type": "年假"}),
        turn("定在2027年9月6日至7日，理由是返乡探亲。", "submit_leave", first, "submit_leave", first),
        turn("刚才说早了，都往后推一周，改成2027-09-13到2027-09-14。", "submit_leave", changed, "submit_leave", changed),
        turn("这份核对好了，请执行当前提交待办。", "confirm_action", pending="submit_leave"),
    ]}, {"id": "new_cancel_then_recollect", "initial_state": seed("submit_leave"), "turns": [
        turn("现在这份草稿作废，不要提交。", "cancel_action"),
        turn("重新准备病假，起止还没有确定。", "submit_leave", {"leave_type": "病假"}),
        turn("原因先填门诊复查，日期仍待定。", "submit_leave", {"leave_type": "病假", "reason": "门诊复查"}),
        turn("只休2027年9月6日这一天。", "submit_leave", {"leave_type": "病假", "start_date": "2027-09-06", "end_date": "2027-09-06", "reason": "门诊复查"}, "submit_leave"),
    ]}]
    for i, question in enumerate(("你在这套助手系统里用什么名字？", "你到底是MiniMind还是TrainPilot？", "能先说明一下你的身份，再开始吗？")):
        workflows.append({"id": f"new_identity_{i}", "turns": [turn(question, "general_chat", response=["TrainPilot"])]})
    workflows.append({"id": "new_withdrawal_confirm", "initial_state": seed("cancel_leave"), "turns": [
        turn("不是放弃撤回，我要确认把原申请撤掉。", "confirm_action", pending="cancel_leave")]})
    workflows.append({"id": "new_withdrawal_abandon", "initial_state": seed("cancel_leave"), "turns": [
        turn("别撤掉原单，只把本次撤回计划终止。", "cancel_action")]})
    for row in workflows:
        row.update(split="manual_new", kind=row["id"], current_date="2027-08-23", role="employee")
    return workflows


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    rules = (DATA / "intent_prompt_v4.txt").read_text(encoding="utf-8")
    suites = {}
    for version in ("v2", "v3", "v4"):
        cases = []
        for row in read_jsonl(DATA / version / "test.jsonl"):
            if row["mode"] != "intent_parse" and version != "v4":
                continue
            row = deepcopy(row)
            if row["mode"] == "intent_parse":
                row["messages"][0]["content"] = render_prompt(rules, row["scenario"])
            cases.append(row)
        suites[version] = cases
    suites["manual_new"] = new_cases(rules)
    training_inputs = {canonical(row["messages"][:-1]) for version in ("v2", "v3", "v4") for split in ("train", "val", "test")
                       for row in read_jsonl(DATA / version / f"{split}.jsonl")}
    assert all(canonical(row["messages"][:-1]) not in training_inputs for row in suites["manual_new"])
    flows = []
    for version in ("v3", "v4"):
        source = [r for r in read_jsonl(DATA / version / "trajectories.jsonl") if r["split"] == "test"]
        if version == "v3":
            source = list({r["kind"]: r for r in source}.values())  # 旧场景每类一条；静态旧集仍全部测试。
        for row in source:
            row = deepcopy(row)
            row.update(split=version + "_synthetic", id=version + "/" + row["id"])
            flows.append(row)
    flows.extend(diagnostics())
    flows.extend(manual_workflows())
    save(output / "cases.json", suites)
    save(output / "workflow_cases.json", flows)
    save(output / "plan.json", {"frozen_before_generation": True, "variants": VARIANTS,
                                "checkpoint_selection": "epoch2事先按服务器V4验证loss更低选择，仍并列展示两轮测试结果。",
                                "manual_new_scope": "训练后、查看本轮输出前固定的本助手逐条新编题；没有调用训练模板生成器，不是第三方真实用户测试。",
                                "prompt": "所有本轮模型使用V4协议；V3旧协议结果另引用已保存基线，不混作同一条件。",
                                "database_scope": "原终端Agent，数据库节点返回明确不可用；不验证真实执行、权限或落库。",
                                "suites": {name: len(rows) for name, rows in suites.items()}, "workflow_trajectories": len(flows)})
    print(json.dumps({name: len(rows) for name, rows in suites.items()}), flush=True)


def totals(rows):
    structured = [r for r in rows if r["case"]["mode"] == "intent_parse"]
    scored = [r for r in rows if r["passed"] is not None]
    categories = {}
    for row in structured:
        item = categories.setdefault(row["case"]["category"], {"passed": 0, "total": 0})
        item["total"] += 1
        item["passed"] += bool(row["passed"])
    return {"total": len(rows), "structured_total": len(structured), "structured_exact": sum(r["passed"] for r in structured),
            "strict_json": sum(r["strict_json"] for r in structured), "intent_correct": sum(r["intent_ok"] for r in structured),
            "fields_exact": sum(r["fields_exact"] for r in structured), "manual_review": len(rows) - len(scored),
            "truncated": sum(r["stop_reason"] != "eos" for r in rows), "categories": categories}


def models(args):
    import datasets
    import torch
    from safetensors.torch import load_file
    from model.dataset.dataset import QwenSFTEncoder
    from model.test.test_code.evaluate_agent_lora import score_business
    from model.test.test_code.evaluate_qwen_sft import generate, sha256, supervised_stats
    from model.test.test_code.qwen_agent_api import load_pair, make_handler
    output = args.output_dir
    assert not (output / "model_summary.json").exists()
    cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    directories = {"v3_prompt4": ROOT / "runs/trainpilot_agent_lora_v3/adapter",
                   "v4_epoch1": ROOT / "runs/trainpilot_agent_lora_v4/epoch_01/adapter",
                   "v4_epoch2": ROOT / "runs/trainpilot_agent_lora_v4/epoch_02/adapter"}
    state = torch.load(ROOT / "runs/trainpilot_agent_lora_v4/checkpoint.pt", map_location="cpu", weights_only=True)
    final = load_file(str(ROOT / "runs/trainpilot_agent_lora_v4/adapter/adapter_model.safetensors"))
    epoch2 = load_file(str(directories["v4_epoch2"] / "adapter_model.safetensors"))
    assert state["progress"]["completed_epochs"] == 2 and state["progress"]["global_step"] == 1212
    assert state["weights"].keys() == final.keys() == epoch2.keys()
    assert all(torch.equal(value, state["weights"][key]) and torch.equal(value, epoch2[key]) for key, value in final.items())
    training = json.loads((ROOT / "runs/trainpilot_agent_lora_v4/run_manifest.json").read_text(encoding="utf-8"))
    assert training["data"]["source_sha256"] == sha256(DATA / "v4/train.jsonl")
    assert training["data"]["validation_source_sha256"] == sha256(DATA / "v4/val.jsonl")
    save(output / "artifact_check.json", {"progress": state["progress"], "tensors": len(final),
                                         "checkpoint_final_epoch2_identical": True, "training_manifest": training})
    del state, final, epoch2
    torch.set_num_threads(2)
    print("开始加载Full SFT底座，仅BF16推理，不加载优化器。", flush=True)
    model, tokenizer, metadata = load_pair(ROOT / "runs/qwen-fullsft/checkpoint.pt", ROOT / "runs/qwen-base-reference", directories["v3_prompt4"])
    for variant in ("v4_epoch1", "v4_epoch2"):
        model.load_adapter(str(directories[variant]), adapter_name=variant, is_trainable=False, autocast_adapter_dtype=False)
    adapter_names = {"v3_prompt4": "default", "v4_epoch1": "v4_epoch1", "v4_epoch2": "v4_epoch2"}
    metadata.update(adapters={name: {"path": str(path), "sha256": sha256(path / "adapter_model.safetensors")} for name, path in directories.items()},
                    cases_sha256=sha256(output / "cases.json"), max_new_tokens=256, prompt_sha256=sha256(DATA / "intent_prompt_v4.txt"))
    save(output / "model_metadata.json", metadata)
    encoder = QwenSFTEncoder(tokenizer, 1536)
    summaries = {}
    model.eval()
    with torch.inference_mode():
        for variant in VARIANTS:
            model.set_adapter(adapter_names[variant])
            summaries[variant] = {}
            with (output / f"{variant}.jsonl").open("w", encoding="utf-8") as stream:
                for suite, records in cases.items():
                    rows, losses, correct, count = [], 0.0, 0, 0
                    for index, case in enumerate(records, 1):
                        ids = tokenizer.apply_chat_template(case["messages"][:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                        result = generate(model, tokenizer, ids, "cuda", 256)
                        if case["mode"] == "intent_parse":
                            score = score_business(case, result["text"])
                        else:
                            score = {"passed": None, "mentions_trainpilot": "trainpilot" in result["text"].lower(), "grading": "自然回复保留逐条语义复核，不要求逐字匹配。"}
                        row = {"variant": variant, "suite": suite, "case": case, **result, **score}
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        rows.append(row)
                        encoded = encoder(case)
                        inputs = torch.tensor([encoded["input_ids"]], device="cuda")
                        labels = torch.tensor([encoded["labels"]], device="cuda")
                        logits = model(input_ids=inputs, use_cache=False).logits
                        loss, matches, n = supervised_stats(logits, labels)
                        losses += loss
                        correct += matches
                        count += n
                        del logits, inputs, labels
                        if index == 1 or index % 25 == 0 or index == len(records):
                            print(f"{variant} {suite} {index}/{len(records)} exact={sum(bool(r['passed']) for r in rows)}", flush=True)
                    summaries[variant][suite] = {**totals(rows), "supervised_loss": losses / count,
                                                  "teacher_forced_token_accuracy": correct / count, "supervised_tokens": count}
                    save(output / "model_summary.json", summaries)
    metadata["checkpoint_sha256_after"] = sha256(ROOT / "runs/qwen-fullsft/checkpoint.pt")
    assert metadata["checkpoint_sha256_after"] == metadata["checkpoint_sha256"]
    for name, path in directories.items():
        assert sha256(path / "adapter_model.safetensors") == metadata["adapters"][name]["sha256"]
    metadata["weights_unchanged"] = True
    save(output / "model_metadata.json", metadata)
    print("静态评测完成", flush=True)
    if args.serve:
        def infer(messages, limit, variant):
            model.set_adapter(adapter_names[variant])
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            with torch.inference_mode():
                result = generate(model, tokenizer, ids, "cuda", min(limit, 256))
            with (output / "api_requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"variant": variant, "messages": messages, **result}, ensure_ascii=False) + "\n")
            return result
        aliases = {"trainpilot-" + name: (lambda m, n, v=name: infer(m, n, v)) for name in VARIANTS}
        handler = make_handler(aliases.pop("trainpilot-v4_epoch2"), "trainpilot-v4_epoch2", aliases)
        print(f"READY http://127.0.0.1:{args.port}/v1", flush=True)
        with HTTPServer(("127.0.0.1", args.port), handler) as server:
            server.serve_forever()


async def flows(args):
    import logging
    logging.disable(logging.INFO)
    sys.path.insert(0, str(args.agent_project.resolve()))
    from app.domain.user.models import UserContext
    from app.capabilities.policy_engine import validate_leave_draft
    from app.workflows import intent_protocol
    from langchain_openai import ChatOpenAI
    from local_runtime.chat_cli import create_terminal_graph, run_turn
    output = args.output_dir
    assert not (output / "workflow_summary.json").exists()
    cases = json.loads((output / "workflow_cases.json").read_text(encoding="utf-8"))
    rules = (DATA / "intent_prompt_v4.txt").read_text(encoding="utf-8")
    summaries = {}
    sources = ("app/workflows/intent_protocol.py", "app/workflows/intent_protocol.txt", "app/workflows/chat_nodes/recognition.py",
               "app/workflows/chat_nodes/routing.py", "app/workflows/chat_nodes/response.py", "app/workflows/chat_nodes/leave.py",
               "app/workflows/chat_nodes/chat.py", "app/workflows/chat_nodes/execute.py", "local_runtime/chat_cli.py")
    hashes = {name: hashlib.sha256((args.agent_project / name).read_bytes()).hexdigest() for name in sources}
    save(output / "workflow_environment.json", {"agent_sources_sha256": hashes,
                                              "candidate_prompt_sha256": hashlib.sha256(rules.encode()).hexdigest(),
                                              "prompt_override": "process_memory_only", "database_writes": 0})
    for variant in VARIANTS:
        llm = ChatOpenAI(model="trainpilot-" + variant, base_url=f"http://127.0.0.1:{args.port}/v1", api_key="local-evaluation", temperature=0, timeout=180, max_retries=0)
        rows = []
        with (output / f"workflow_{variant}.jsonl").open("w", encoding="utf-8") as stream:
            for number, case in enumerate(cases, 1):
                graph = create_terminal_graph(llm)
                thread = variant + "/" + case["id"]
                if case.get("initial_state"):
                    await graph.aupdate_state({"configurable": {"thread_id": thread}}, deepcopy(case["initial_state"]))
                fixed_now = datetime.fromisoformat(case["current_date"])
                class Clock:
                    @staticmethod
                    def now():
                        return fixed_now
                user = UserContext(user_id="evaluation_account", name="", role=case.get("role", "employee"))
                for index, item in enumerate(case["turns"], 1):
                    trace = []
                    api_path = output / "api_requests.jsonl"
                    offset = api_path.stat().st_size if api_path.exists() else 0
                    with patch.object(intent_protocol, "RULES", rules), patch("app.workflows.chat_nodes.recognition.datetime", Clock), patch(
                        "app.workflows.chat_nodes.leave.validate_leave_draft", side_effect=lambda raw: validate_leave_draft(raw, today=fixed_now.date())
                    ):
                        state = await run_turn(graph, thread, item["user"], user, trace.append)
                    with api_path.open("rb") as api_stream:
                        api_stream.seek(offset)
                        requests = [json.loads(line) for line in api_stream.read().splitlines() if line]
                    try:
                        raw_payload = json.loads(requests[0]["text"])
                    except (ValueError, IndexError):
                        raw_payload = None
                    pending = state.get("pending_action") or {}
                    nodes = [t.removeprefix("[节点] ") for t in trace if t.startswith("[节点] ")]
                    checks = {"intent": state.get("intent") == item["expected"]["intent"],
                              "raw_protocol": raw_payload == item["expected"],
                              "fields_exact": state.get("extracted_info") == item["expected"]["extracted_info"],
                              "route": all(node in nodes for node in item["expected_nodes"]),
                              "pending": pending.get("action_type") == item["expected_pending_type"],
                              "draft": all(pending.get("draft", {}).get(k) == v for k, v in item.get("expected_draft", {}).items())}
                    accumulated = state.get("conversation_context", {}).get("accumulated_extracted_info", {})
                    if "expected_accumulated" in item:
                        checks["accumulated"] = accumulated == item["expected_accumulated"]
                    if item.get("response_contains"):
                        checks["response"] = all(text.lower() in state.get("response", "").lower() for text in item["response_contains"])
                    row = {"variant": variant, "suite": case["split"], "trajectory_id": case["id"], "turn": index,
                           "user": item["user"], "expected": item["expected"],
                           "actual": {"intent": state.get("intent"), "extracted_info": state.get("extracted_info")},
                           "response": state.get("response"), "pending": pending, "accumulated": accumulated,
                           "model_outputs": [request["text"] for request in requests],
                           "checks": checks, "passed": all(checks.values()),
                           "state_passed": all(value for key, value in checks.items() if key not in {"intent", "raw_protocol"}), "trace": trace}
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    stream.flush()
                print(f"workflow {variant} {number}/{len(cases)} {case['id']} passed={all(r['passed'] for r in rows if r['trajectory_id']==case['id'])}", flush=True)
        summaries[variant] = {}
        for suite in dict.fromkeys(row["suite"] for row in rows):
            selected = [row for row in rows if row["suite"] == suite]
            ids = {row["trajectory_id"] for row in selected}
            summaries[variant][suite] = {"turns": len(selected), "turns_exact": sum(r["passed"] for r in selected),
                                         "turns_state_passed": sum(r["state_passed"] for r in selected),
                                         "trajectories_state_passed": sum(all(r["state_passed"] for r in selected if r["trajectory_id"] == uid) for uid in ids),
                                         "trajectories": len(ids), "trajectories_passed": sum(all(r["passed"] for r in selected if r["trajectory_id"] == uid) for uid in ids)}
        save(output / "workflow_summary.json", summaries)
    assert hashes == {name: hashlib.sha256((args.agent_project / name).read_bytes()).hexdigest() for name in sources}


def report(output):
    import statistics
    summaries = json.loads((output / "model_summary.json").read_text(encoding="utf-8"))
    workflows = json.loads((output / "workflow_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "model_metadata.json").read_text(encoding="utf-8"))
    assert set(summaries) == set(VARIANTS) and metadata["weights_unchanged"]
    for variant in VARIANTS:
        assert set(summaries[variant]) == {"v2", "v3", "v4", "manual_new"}
    rows = {variant: read_jsonl(output / f"{variant}.jsonl") for variant in VARIANTS}
    old_dir = ROOT / "runs/lora-v3-eval-20261023"
    old_meta = json.loads((old_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert old_meta["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert old_meta["adapters"]["v3"]["sha256"] == metadata["adapters"]["v3_prompt4"]["sha256"]
    old = json.loads((old_dir / "model_summary.json").read_text(encoding="utf-8"))["v3"]
    paired = {}
    for suite in ("v2", "v3", "v4", "manual_new"):
        before = {r["case"]["id"]: r for r in rows["v3_prompt4"] if r["suite"] == suite and r["case"]["mode"] == "intent_parse"}
        after = {r["case"]["id"]: r for r in rows["v4_epoch2"] if r["suite"] == suite and r["case"]["mode"] == "intent_parse"}
        assert before.keys() == after.keys()
        paired[suite] = {"fixed": [uid for uid in before if not before[uid]["passed"] and after[uid]["passed"]],
                         "regressed": [uid for uid in before if before[uid]["passed"] and not after[uid]["passed"]]}
    failures = [{"suite": r["suite"], "id": r["case"]["id"], "category": r["case"]["category"],
                 "user": r["case"]["scenario"]["user_message"], "context": {k: v for k, v in r["case"]["scenario"].items() if k != "user_message"},
                 "expected": r["case"]["expected"], "actual": r["text"], "truncated": r["stop_reason"] != "eos"}
                for r in rows["v4_epoch2"] if r["passed"] is False]
    replies = [{"variant": variant, "id": r["case"]["id"], "mode": r["case"]["mode"],
                "category": r["case"]["category"], "reference": r["case"]["messages"][-1]["content"], "actual": r["text"]}
               for variant in VARIANTS for r in rows[variant] if r["case"]["mode"] != "intent_parse"]
    workflow_failures = [r for r in read_jsonl(output / "workflow_v4_epoch2.jsonl") if not r["passed"]]
    first_failures = list({r["trajectory_id"]: next(item for item in workflow_failures if item["trajectory_id"] == r["trajectory_id"])
                           for r in workflow_failures}.values())
    selected = rows["v4_epoch2"]
    speed = {"gpu": metadata["gpu"], "dtype": metadata["dtype"], "decode": "greedy",
             "median_output_tokens_per_second": statistics.median(r["tokens_per_second"] for r in selected),
             "median_first_token_seconds": statistics.median(r["first_token_seconds"] for r in selected),
             "peak_allocated_mib": max(r["peak_allocated_mib"] for r in selected),
             "note": "本机顺序评测，含prefill；首token为推理函数内部计时。API缓冲完整生成后发送，不代表用户端流式首字延迟；不是服务器或并发基准。"}
    coverage = json.loads((output / "training_coverage_audit.json").read_text(encoding="utf-8"))
    training_events = read_jsonl(ROOT / "runs/trainpilot_agent_lora_v4/metrics.jsonl")
    summary = {"selected_checkpoint": "v4_epoch2", "selection_basis": "在生成测试输出前按服务器验证loss选择epoch2。",
               "models": summaries, "workflows": workflows, "v3_cached_old_prompt": old,
               "paired_same_v4_prompt": paired, "remaining_failures": failures,
               "workflow_remaining_failures": workflow_failures, "workflow_first_failure_per_trajectory": first_failures,
               "natural_replies": replies, "local_inference_observations": speed,
               "training_coverage_audit": coverage,
               "training_epochs": [r for r in training_events if r["event"] == "epoch"],
               "scope": "旧测试已用于开发回归；V4合成题和对应图流程不是独立的两份证据。本助手逐条新编题在本次输出前固定，仍非第三方真实用户评测。",
               "database_scope": "只测终端图中的意图、路由、本地草稿与状态；未验证数据库业务执行和权限。"}
    if (output / "diagnostic_probes.json").exists():
        summary["posthoc_diagnostic_probes"] = json.loads((output / "diagnostic_probes.json").read_text(encoding="utf-8"))
    if (output / "natural_reply_review.json").exists():
        summary["natural_reply_review"] = json.loads((output / "natural_reply_review.json").read_text(encoding="utf-8"))
    save(output / "evaluation_summary.json", summary)
    lines = ["# TrainPilot V4 评测", "", "日期：2026-10-23。只读评测，未训练或修改权重。结果保存在本目录，不修改历史测试日志。", "",
             "## 结论与后续优先级", "",
             "V4显著改善了本项目的结构化业务适配，但尚未达到全业务可靠执行的目标。第2轮在旧业务、身份和本批Agent轨迹上更好；第1轮在V4合成题和本次逐条新编题上更好，因此不能仅凭更低验证loss断言第2轮全面最优。两轮adapter均应保留。", "",
             "Agent轨迹严格通过数：V3配V4提示词29/37，第1轮34/37，第2轮35/37。第2轮两条失败分别是：待撤回状态下将‘确认取消这张申请’误判为取消操作；将‘别撤掉原单，只把本次撤回计划终止’误判为澄清。另有静态题把放弃撤回判成执行撤回、使用旧申请编号、改写未要求修改的审批意见、相对日期算错。", "",
             "第2轮的12条身份/招呼/称呼/角色/纠正身份样例均提到TrainPilot；开放回复仍有额外别名、未经核验的权限描述等问题，不能只因出现品牌名称就算回复完全正确。", "",
             "1. 先用独立于训练模板的真实说法补齐确认/取消、当前对象与旧对象、只改指定字段的验收题，优先处理可能反向执行的错误。请用户或Agent开发者参与出题；本次新题与训练数据均由同一助手编写。", 
             "2. 对齐模型与Agent的业务协议：列明合法假别、区分任务已知但字段缺失与对象不明确、让程序核验申请编号来源和日期。当前采用完整已知字段协议，本次不擅自改为增量字段。", 
             "3. 用对照表达补充数据并做小规模消融：保留其他任务回放，增加撤回/放弃撤回和旧字段保持的多样说法。日期、编号变体不能代替表达多样性。先预览再生成，尚未开始新版本构造或训练。", 
             "4. 任何提示词或数据改动后重跑同一套回归，再以未参与调参的真实业务题验收。原Agent此次只在内存中切换到V4提示词，正式接入时需要保持协议一致。数据库写入、权限与完整线上交易仍未验证。", "",
             "事后探针中，补充待办展示历史、当前原文编号候选和合法假别枚举分别修复了所选单例；补充日期锚点、撤回待办解释仍未修复所选单例。这些结果用于定位问题，不计入主成绩，也不能代替完整回归。", "",
             "## 权重与评测条件", "", "两轮训练完成；最终adapter、epoch_02和断点权重逐张量一致。使用同一Full SFT底座、BF16、贪心解码、最大256个新token。",
             "在查看测试输出前，按服务器验证loss选定第2轮作为主报告；第1轮仍并列展示。所有本轮结构化请求使用V4提示词；V3旧提示词成绩只作单独对照。", "",
             "V3与V4的训练数据、更新次数不同（440与1212）；这里是版本效果对比，不能作为只改变数据构造方式的单变量消融。", "",
             "## 结构化输出结果", "", "严格比较意图和完整字段，同时检查原始JSON；不通过修复JSON增加得分。", "",
             "|模型|V2旧业务|V3多轮静态请求|V4合成留出|本次本助手逐条新编题|", "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        values = [f"{summaries[variant][suite]['structured_exact']}/{summaries[variant][suite]['structured_total']}" for suite in ("v2", "v3", "v4", "manual_new")]
        lines.append("|" + variant + "|" + "|".join(values) + "|")
    lines += ["", "### 第2轮数值指标", "", "|测试组|监督交叉熵|Teacher-forced token准确率|可解析JSON对象|", "|---|---:|---:|---:|"]
    for suite, r in summaries["v4_epoch2"].items():
        lines.append(f"|{suite}|{r['supervised_loss']:.6f}|{r['teacher_forced_token_accuracy']:.4%}|{r['strict_json']}/{r['structured_total']}|")
    lines += ["", "监督token指标包含重复JSON键、标点及聊天模板的空think前缀；业务判断以完整意图和字段联合正确为准，不能用token准确率代替任务成功率。", ""]
    lines += ["### 数据设计的已知限制", "",
              f"自然回复监督token占训练监督token的{coverage['natural_reply_token_fraction']:.2%}；总体loss主要反映结构化任务。身份的直接问答与Agent实际提示模式需要分别检查。",
              f"withdraw_confirm和withdraw_abandon各{coverage['withdraw_confirm']['records']}条训练请求，但每类只有{coverage['withdraw_confirm']['distinct_current_utterances']}种当前用户句式；日期和编号变化不是独立表达覆盖。",
              "表达多样性不足和任务比例可能影响未见说法的泛化，这是有数据依据的假设；尚未通过补表达或调整比例的消融实验建立因果结论。", ""]
    lines += ["", f"同权重V3旧提示词基线：V2业务{old['v2']['passed']}/{old['v2']['total']}，V3场景{old['v3']['passed']}/{old['v3']['total']}。缓存基线的底座与V3 adapter SHA256已核对。", "",
              "## Agent实际闭环", "", "运行原LangGraph终端图，临时在内存中使用V4提示词，不修改Agent源文件。数据库节点返回明确不可用。原模型输出不合规范，即使图能够容错解析也计为未通过。", "",
              "|模型|场景组|原始协议及状态全通过的轨迹|框架落点与字段状态通过的轨迹|严格逐轮通过|", "|---|---|---:|---:|---:|"]
    for variant, suites in workflows.items():
        for suite, r in suites.items():
            lines.append(f"|{variant}|{suite}|{r['trajectories_passed']}/{r['trajectories']}|{r['trajectories_state_passed']}/{r['trajectories']}|{r['turns_exact']}/{r['turns']}|")
    lines += ["", "严格列要求模型原始JSON、意图、字段、路由及状态全部正确。框架状态列允许解析容错或路由纠正意图，但仍要求提取字段、目标节点、待办及被指定检查的草稿/回复正确；两者均不是数据库交易成功率。", ""]
    lines += ["", "静态题与部分闭环来自同一批合成轨迹，不能相加后当成独立样本。本助手逐条新编题未调用训练模板生成器，但不能证明所有真实输入都能正确处理。", "",
              "## 主报告剩余失败", ""]
    for item in failures:
        lines += [f"### {item['suite']} / {item['id']} / {item['category']}", "", "用户：" + item["user"], "",
                  "上下文：", "", "```json", json.dumps(item["context"], ensure_ascii=False), "```", "",
                  "预期：", "", "```json", json.dumps(item["expected"], ensure_ascii=False), "```", "",
                  "实际：", "", "```text", item["actual"], "```", ""]
    if not failures:
        lines += ["本批结构化题没有失败；这不等于开放场景完美。", ""]
    lines += ["## Agent失败轨迹的首个错误", "", "逐轮结果可能包含早先错误导致的后续状态偏离；这里先展示每条失败轨迹的第一个错误，不把每个后续错误都认作独立根因。", ""]
    for item in first_failures:
        lines += [f"### {item['trajectory_id']} / 第{item['turn']}轮", "", "用户：" + item["user"], "",
                  "检查：" + json.dumps(item["checks"], ensure_ascii=False), "", "预期：", "", "```json", json.dumps(item["expected"], ensure_ascii=False), "```", "",
                  "实际：", "", "```json", json.dumps(item["actual"], ensure_ascii=False), "```", "", "Agent回复：" + str(item["response"]), ""]
    if not first_failures:
        lines += ["本批闭环轨迹均通过；数据库写入与权限仍未验证。", ""]
    lines += ["## 自然回复原文（第2轮）", "", "开放回复不要求逐字等于参考答案，不混入结构化准确率。", ""]
    for item in replies:
        if item["variant"] == "v4_epoch2":
            lines += [f"### {item['id']} / {item['category']} / {item['mode']}", "", "参考：" + item["reference"], "", "实际：", "", "```text", item["actual"], "```", ""]
    lines += ["## 本机推理观测", "", "```json", json.dumps(speed, ensure_ascii=False, indent=2), "```", "",
              "详细生成、完整上下文、计时和错误均保存在各variant的JSONL；工作流的每轮输出及状态见workflow_*.jsonl。", ""]
    if "natural_reply_review" in summary:
        lines += ["## 自然回复语义复核", "", "```json", json.dumps(summary["natural_reply_review"], ensure_ascii=False, indent=2), "```", ""]
    if "posthoc_diagnostic_probes" in summary:
        lines += ["## 事后诊断探针", "", "以下输入改动在发现错误后设计，仅用于定位提示词/上下文增强的可能作用，不计入主测试成绩，也不是新的独立验收。", ""]
        for probe in summary["posthoc_diagnostic_probes"]["probes"]:
            lines += [f"### {probe['name']}", "", "用户：" + probe["user"], "", "改动：" + probe["change"], "",
                      "原输出：", "", "```text", probe["before"], "```", "", "增强后的输出：", "", "```text", probe["after"], "```", "",
                      "增强后按原标签评分：" + str(probe["passed"]), ""]
    (output / "evaluation_report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(json.dumps({"selected": "v4_epoch2", "remaining_failures": len(failures), "workflow": workflows}, ensure_ascii=False, indent=2))


def probes(args):
    from datetime import date, timedelta
    import re
    from urllib.request import Request, urlopen
    rows = read_jsonl(args.output_dir / "v4_epoch2.jsonl")
    failures = [r for r in rows if r["passed"] is False]
    rules = (DATA / "intent_prompt_v4.txt").read_text(encoding="utf-8")
    plans = []
    for category, name, change in (
        ("confirm_action", "confirmation_history", "仅补充待办的人工自然语言展示历史，保留原待办和用户输入。"),
        ("pending_cancel_existing", "current_message_ids", "只列出当前用户原文匹配到的编号候选；不改变业务对象或原标签。"),
        ("relative", "calendar_anchors", "在上下文增加由程序计算的今天/明天/后天日期映射。"),
        ("withdraw_abandon", "pending_action_description", "为撤回待办补充其业务含义，区分取消待办与撤回原申请。"),
        ("missing_fields", "leave_type_enum", "补充现有业务支持的六种假别及未知假别省略规则。"),
    ):
        row = next((r for r in failures if r["case"]["category"] == category and (
            category != "missing_fields" or "leave_type" not in r["case"]["expected"]["extracted_info"])), None)
        if row is None:
            continue
        context = deepcopy(row["case"]["scenario"])
        user = context.pop("user_message")
        if category == "confirm_action":
            pending = context["pending_action"]
            context["history"] = [{"role": "assistant", "content": "目前仅等待确认的动作：" + pending["action_type"] + "；待办内容：" + json.dumps(pending["draft"], ensure_ascii=False) + "。确认表示执行该待办，取消表示放弃该待办。"}]
        elif category == "pending_cancel_existing":
            context["current_message_app_ids"] = re.findall(r"leave_[A-Za-z0-9_]+", user)
        elif category == "relative":
            today = date.fromisoformat(context["current_date"])
            context["relative_date_anchors"] = {key: (today + timedelta(days=offset)).isoformat() for offset, key in enumerate(("今天", "明天", "后天"))}
        elif category == "withdraw_abandon":
            context["pending_action"]["action_description"] = "准备撤回一份已提交申请；放弃此待办表示保留原申请，不是执行撤回。"
        else:
            context["allowed_leave_types"] = ["年假", "病假", "事假", "婚假", "产假", "丧假"]
        extra_rule = "\nleave_type只允许allowed_leave_types中的具体假别；尚未选定具体假别时省略该字段，不能使用泛称‘请假’代替。\n" if category == "missing_fields" else ""
        messages = [{"role": "user", "content": rules + extra_rule + "\n给定上下文：\n" + json.dumps(context, ensure_ascii=False) + "\n用户输入：\n" + user}]
        plans.append({"name": name, "change": change, "case_id": row["case"]["id"], "user": user,
                      "before": row["text"], "expected": row["case"]["expected"], "messages": messages})
    save(args.output_dir / "diagnostic_probe_plan.json", {"posthoc": True, "excluded_from_main_score": True, "probes": plans})
    for item in plans:
        data = json.dumps({"model": "trainpilot-v4_epoch2", "messages": item["messages"], "max_tokens": 256, "temperature": 0}).encode()
        with urlopen(Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=data, headers={"Content-Type": "application/json"}), timeout=180) as response:
            item["after"] = json.load(response)["choices"][0]["message"]["content"]
        try:
            item["passed"] = json.loads(item["after"]) == item["expected"]
        except ValueError:
            item["passed"] = False
        print(item["name"], item["passed"], flush=True)
    save(args.output_dir / "diagnostic_probes.json", {"posthoc": True, "excluded_from_main_score": True, "probes": plans})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "models", "flows", "probes", "report"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--agent-project", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.output_dir)
    elif args.mode == "models":
        models(args)
    elif args.mode == "report":
        report(args.output_dir)
    elif args.mode == "probes":
        probes(args)
    else:
        asyncio.run(flows(args))


if __name__ == "__main__":
    main()
