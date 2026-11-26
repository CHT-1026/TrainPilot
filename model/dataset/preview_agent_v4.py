"""V4 人工样例预览：用脚本化标签检查原 Agent，不生成正式训练集。

Agent 环境：python -m model.dataset.preview_agent_v4 --agent-project PATH
训练环境：python -m model.dataset.preview_agent_v4 --check-tokenizer PATH
预期答案是人工定义的设计样例，不是任何模型的测评结果。
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "model/dataset/datasets/trainpilot_agent/v4_preview"
FIELDS = {"leave_type", "start_date", "end_date", "reason", "app_id", "decision", "comments"}
INTENTS = {"submit_leave", "delete_application", "approve", "query_status", "query_pending",
           "policy_qa", "get_suggestion", "confirm_action", "cancel_action", "general_chat", "clarify"}
NODES = {"submit_leave": "submit_leave", "delete_application": "delete", "approve": "approve",
         "query_status": "query", "query_pending": "query_pending", "policy_qa": "retrieve",
         "get_suggestion": "suggestion", "confirm_action": "execute_pending",
         "cancel_action": "clear_pending", "general_chat": "general_chat", "clarify": "clarify"}
DB_NODES = {"delete", "approve", "query", "query_pending", "retrieve", "suggestion", "execute_pending"}

CANDIDATE_RULES = """你是 TrainPilot 的意图解析模块。你只理解需求并输出JSON，不执行业务操作。
只输出两个顶层字段：intent和extracted_info。extracted_info必须是JSON对象，不能是字符串。不要输出解释、Markdown或工具调用标记。
意图定义：
submit_leave=申请请假或补充修改请假草稿；delete_application=撤销已提交的申请；approve=要求通过或驳回申请；query_status=查询本人申请；query_pending=查询待我审批的申请；policy_qa=询问企业资料、制度或流程；get_suggestion=请求审批建议而不执行审批；confirm_action=确认执行当前待确认动作；cancel_action=放弃当前待确认动作；general_chat=普通对话及询问本助手的名称、身份和能力边界；clarify=需求或操作对象不明确，需要澄清。
规则：
1. extracted_info仅允许leave_type、start_date、end_date、reason、app_id、decision、comments。没有依据的字段省略，不填null、空字符串或占位符。不要输出姓名、权限、working_days、requires_hr或执行结果。
2. 输出当前任务的完整已知字段，依据当前输入、最近历史、摘要和待确认草稿。用户明确更新的值覆盖旧值；没有修改的相关字段保留。新任务不得继承已取消任务的字段；与当前任务无关的历史编号和字段不得混入。
3. 日期使用YYYY-MM-DD。明确的今天、明天、后天依据current_date换算；日期含糊时只提取其他确定字段，不猜测起止日期。请假天数、有效性和权限交由业务系统校验。
4. 确认执行当前待确认动作输出confirm_action；放弃该动作输出cancel_action，两者extracted_info均为{}。没有待确认动作且没有明确新业务目标时，孤立的确认或取消输出clarify。确认撤回申请与放弃撤回操作含义相反，不按单个关键词判断。
5. 明确撤回另一个已提交编号时输出delete_application并提取实际目标app_id；不要选被用户否定的编号。对象含糊且无法从上下文唯一确定时输出clarify。
6. 请求建议是get_suggestion，要求通过或驳回是approve。decision只用“通过”或“驳回”；comments只提取实际意见正文。不要根据用户自述推断权限，也不要把询问如何审批当作执行审批。
7. 同时要求多个独立业务时输出clarify和{}，请由Agent澄清本轮任务；同一任务的补充、修改及确认不算多个独立业务。
8. 普通对话、身份询问、制度咨询以及无法明确需求时，没有相关字段就输出{}。引用或讨论一句业务指令不等于要求执行该指令。
"""


def save(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pending(kind):
    """人工前置状态；编号仅为测试标识，不代表数据库中的真实申请。"""
    if kind == "submit_leave":
        draft = {"leave_type": "事假", "start_date": "2026-11-16", "end_date": "2026-11-17", "reason": "办理证件"}
        text = "当前请假草稿：事假，2026-11-16至2026-11-17，原因办理证件，尚未提交。"
    elif kind == "approve":
        draft = {"app_id": "leave_preview_1701", "decision": "approved", "decision_label": "通过", "comments": "证明已核对"}
        text = "待确认审批：leave_preview_1701，拟通过，意见证明已核对，尚未执行。"
    else:
        draft = {"app_id": "leave_preview_1702"}
        text = "待确认撤回申请leave_preview_1702，尚未执行撤回。"
    return {"pending_action": {"action_id": "act_preview", "action_type": kind, "draft": draft},
            "messages": [{"role": "assistant", "content": text}],
            "conversation_context": {"accumulated_extracted_info": draft if kind == "submit_leave" else {}}}


def definitions():
    """每种边界先给少量人工预览，用户批准后才扩展表达族及正式划分。"""
    scenes = []

    def turn(uid, text, intent, info=None, **extra):
        return {"id": uid, "user": text, "expected": {"intent": intent, "extracted_info": info or {}}, **extra}

    def one(uid, group, text, intent, info=None, initial=None, note="", **extra):
        scenes.append({"id": uid, "coverage": group, "current_date": "2026-11-02",
                       "initial_state": initial or {}, "note": note,
                       "turns": [turn(uid, text, intent, info, **extra)]})

    # 同一表达配不同真实状态，不能仅靠“确认／取消”关键词分类。
    for kind in (None, "submit_leave", "approve", "cancel_leave"):
        suffix = kind or "none"
        one("confirm_" + suffix, "确认状态矩阵", "核对过了，就按当前待办执行。",
            "confirm_action" if kind else "clarify", initial=pending(kind) if kind else None)
        one("cancel_" + suffix, "取消状态矩阵", "当前准备的这项操作先不办了。",
            "cancel_action" if kind else "clarify", initial=pending(kind) if kind else None)
    one("withdraw_confirm", "撤回与放弃撤回", "这张申请确定要撤回，现在执行待确认的撤回操作。", "confirm_action", initial=pending("cancel_leave"))
    one("withdraw_abandon", "撤回与放弃撤回", "保留原申请，放弃刚准备的撤回动作。", "cancel_action", initial=pending("cancel_leave"))
    one("approve_confirm", "审批与确认", "审批意见都已检查，照待确认版本执行，不用重拟。", "confirm_action", initial=pending("approve"))

    base = pending("submit_leave")["pending_action"]["draft"]
    updates = [
        ("start", "开始日期往后改一天到2026年11月17日，其余字段照旧。", {"start_date": "2026-11-17"}),
        ("end", "这份草稿的最后一天改成2026年11月18日，别的保留。", {"end_date": "2026-11-18"}),
        ("both", "整段时间换为2026年11月23日至11月24日，假别和事由沿用。", {"start_date": "2026-11-23", "end_date": "2026-11-24"}),
        ("type", "日期和事由没变，假别改选年假。", {"leave_type": "年假"}),
        ("reason", "事由不再写办理证件，改写成家中设备检修，其他项不动。", {"reason": "家中设备检修"}),
        ("negation", "暂不提交，也不是取消草稿，只把事由换成办理房屋交接。", {"reason": "办理房屋交接"}),
    ]
    for uid, text, change in updates:
        one("modify_" + uid, "完整已知字段", text, "submit_leave", {**base, **change}, initial=pending("submit_leave"),
            expected_draft={**base, **change})
    history = {"messages": [{"role": "user", "content": "请整理事假，2026-11-16至2026-11-17，办理证件用。"},
                            {"role": "assistant", "content": "已记录这些信息，请继续核对。"}]}
    one("history_only", "旧简化历史回归", "两天都往后顺延一周，假别和事由不变。", "submit_leave",
        {**base, "start_date": "2026-11-23", "end_date": "2026-11-24"}, initial=history)
    one("summary_only", "可见摘要恢复", "结束日延至2026-11-18，其余按照摘要中这份申请。", "submit_leave",
        {**base, "end_date": "2026-11-18"}, initial={"summary": "当前仅有一份未提交事假草稿：2026-11-16至2026-11-17，事由办理证件。"},
        note="仅使用模型输入中确实可见的摘要；不训练从被裁掉且未摘要的历史恢复隐藏信息。")

    one("approval_comment", "审批字段正文", "请将leave_preview_1801拟为通过，审批意见填写“已补齐必要材料”。", "approve",
        {"app_id": "leave_preview_1801", "decision": "通过", "comments": "已补齐必要材料"})
    one("approval_missing_id", "缺失字段", "这项我打算驳回，意见写请补齐日期，编号还没拿到。", "approve",
        {"decision": "驳回", "comments": "请补齐日期"}, note="数据库节点未接入，只检查模型不编造app_id及正确路由。")
    one("approval_missing_decision", "缺失字段", "先准备审批leave_preview_1802，决定我还没有选。", "approve", {"app_id": "leave_preview_1802"})
    one("approval_edit", "审批完整已知字段", "原意见保留，结论改成驳回。", "approve",
        {"app_id": "leave_preview_1701", "decision": "驳回", "comments": "证明已核对"}, initial=pending("approve"))
    one("advice", "建议与执行", "对leave_preview_1803先给出审批建议，不要批准或驳回。", "get_suggestion", {"app_id": "leave_preview_1803"})
    one("advice_missing_id", "缺失字段", "我需要一份具体申请的审批建议，但还没找到单号。", "get_suggestion")
    one("policy_not_approve", "建议与执行", "审批中通常什么情形应该驳回？我是在咨询流程，不是在处理某张单。", "policy_qa")
    one("quoted_command", "引用不执行", "操作说明里的“撤回申请”指的是什么流程？只是问含义。", "policy_qa")

    for uid, text in (
        ("other_only", "当前审批先挂着，我要撤回的是自己已提交的leave_preview_1901。"),
        ("other_negative", "撤回leave_preview_1901，不是待确认审批里的leave_preview_1701。"),
        ("other_reverse", "leave_preview_1701不要动，真正要撤回的是leave_preview_1901。"),
    ):
        one(uid, "操作对象与干扰编号", text, "delete_application", {"app_id": "leave_preview_1901"}, initial=pending("approve"))
    one("delete_missing_id", "缺失字段", "我准备撤销已经提交的请假申请，单号稍后给你。", "delete_application")
    one("ambiguous_object", "对象歧义", "把它撤回吧。", "clarify", initial={"messages": [
        {"role": "assistant", "content": "上一轮提到两个申请：leave_preview_1902和leave_preview_1903；尚未确定本次操作对象。"}]})
    one("mine", "查询本人申请", "打开我已提交的请假申请列表让我看看。", "query_status")
    one("pending_list", "查询待审批", "看看有哪些申请正等我来审批。", "query_pending")
    one("policy", "企业制度检索", "病假申请需要附哪些证明，按现有制度查一下。", "policy_qa",
        note="训练的是送到知识检索节点，不编造制度答案；当前没有知识库内容。")
    one("multi_task", "多任务先澄清", "帮我新建请假申请，再把等待我审批的列表查出来。", "clarify",
        note="用户已确认单轮先澄清；现有Agent只返回统一澄清文案，专门询问任务顺序留待集成。")
    one("unclear", "意图歧义", "把那个事情处理一下，具体是哪件我还没想好。", "clarify")

    one("missing_all", "缺失字段", "我打算请假，具体信息接着再填。", "submit_leave")
    one("missing_dates", "缺失字段", "申请事假办理证件，起止日期还定不下来。", "submit_leave", {"leave_type": "事假", "reason": "办理证件"})
    one("missing_reason", "缺失字段", "2026-11-23到2026-11-24申请年假，原因我还没写。", "submit_leave",
        {"leave_type": "年假", "start_date": "2026-11-23", "end_date": "2026-11-24"})
    one("missing_type", "缺失字段", "2026年11月23日至24日需要请假，原因为办理房屋交接，假别还没选。", "submit_leave",
        {"start_date": "2026-11-23", "end_date": "2026-11-24", "reason": "办理房屋交接"})
    one("invalid_dates", "提取与业务校验分开", "事假从2026-11-24到2026-11-23，原因办理证件。", "submit_leave",
        {"leave_type": "事假", "start_date": "2026-11-24", "end_date": "2026-11-23", "reason": "办理证件"},
        response_contains=["未通过校验"], note="忠实提取用户给出的顺序，日期合法性由现有业务校验拒绝，不擅自调换。")
    for uid, today, text, info in (
        ("month", "2026-11-30", "明天到后天请事假，事由为办理证件。", {"leave_type": "事假", "start_date": "2026-12-01", "end_date": "2026-12-02", "reason": "办理证件"}),
        ("year", "2026-12-31", "明天当天请年假，原因安排休息。", {"leave_type": "年假", "start_date": "2027-01-01", "end_date": "2027-01-01", "reason": "安排休息"}),
    ):
        one("relative_" + uid, "相对日期跨边界", text, "submit_leave", info)
        scenes[-1]["current_date"] = today
    one("uncertain_dates", "不猜测日期", "下个月找两天请事假办证，具体哪两天还没定。", "submit_leave", {"leave_type": "事假", "reason": "办证"})

    scenes.append({"id": "collect_update_confirm", "coverage": "连续多轮办理", "current_date": "2026-11-02", "turns": [
        turn("flow_a1", "先开一份年假草稿，其他内容我接着说。", "submit_leave", {"leave_type": "年假"}),
        turn("flow_a2", "日期填2026-11-23到2026-11-24，事由填返乡探亲。", "submit_leave", {"leave_type": "年假", "start_date": "2026-11-23", "end_date": "2026-11-24", "reason": "返乡探亲"}),
        turn("flow_a3", "改用2026-11-30到2026-12-01，之前的假别和事由照旧。", "submit_leave", {"leave_type": "年假", "start_date": "2026-11-30", "end_date": "2026-12-01", "reason": "返乡探亲"}),
        turn("flow_a4", "这份内容没问题，执行待确认提交。", "confirm_action"),
    ]})
    scenes.append({"id": "cancel_then_new", "coverage": "取消后开启新任务", "current_date": "2026-11-02", "initial_state": pending("submit_leave"), "turns": [
        turn("flow_b1", "办理证件这份先作废，不提交这份草稿。", "cancel_action"),
        turn("flow_b2", "另开一份病假，日期和事由等我重新提供。", "submit_leave", {"leave_type": "病假"}),
        turn("flow_b3", "2026-11-23当天，事由写门诊复查。", "submit_leave", {"leave_type": "病假", "start_date": "2026-11-23", "end_date": "2026-11-23", "reason": "门诊复查"}),
    ]})

    # 先识别为general_chat，再监督原Agent实际的回复请求；同一场景两条监督记录。
    replies = [
        ("identity_name", "请告诉我你在这个终端里的称呼。", "我叫 TrainPilot，是一名企业业务助手。"),
        ("identity_role", "初次见面，介绍一下你的身份和分工吧。", "我是 TrainPilot，负责理解业务需求、整理信息和组织回答；实际业务操作由系统执行。"),
        ("greeting", "第一次来用，先向你打个招呼。", "你好，我是 TrainPilot。你可以直接告诉我想办理的事项。"),
        ("identity_correction", "这位助手也叫MiniMind吗？", "我叫 TrainPilot，是这里的企业业务助手。"),
        ("missing_information", "如果我没说清必要信息，你会补出一个答案吗？", "不会。我会询问缺少的信息，不会编造日期、原因或申请编号。"),
        ("execution_claim", "没有系统执行结果，你能直接告诉我已经办好了吗？", "不能。只有获得实际执行结果后，才能说明办理结果；业务操作还需要相应的确认和权限校验。"),
    ]
    for uid, text, reply in replies:
        one(uid, "身份与实际回复模式", text, "general_chat", response=reply)
    return scenes


def specification():
    return {
        "version": "v4_preview", "status": "pending_user_review", "assistant_name": "TrainPilot",
        "scope": "11种意图、7字段协议；模型理解与回复，Agent路由和业务执行。",
        "confirmed": {"output_policy": "complete_known_fields", "multi_task": "clarify_then_ask",
                      "clear_field": "interface_not_supported_no_execution_training", "general_ability": "not_primary_objective"},
        "prompt_change": "仅预览candidate提示词，内存替换检查；尚未修改Agent运行文件。后续部署必须训练/运行提示词同版。",
        "replay_plan": [
            "逐条按规则审查V2训练样本：恢复V3过度删除的6类共336条候选，不因简化历史整类删除。",
            "其余797条旧业务候选和V3的960条新业务请求纳入审核；不是未经检查直接拼接。",
            "重建同一V4提示词输入，来源ID和排除理由可追溯；旧文件、权重保持不变。",
            "通用数学练习不纳入本轮；身份样例分别覆盖识别JSON、Agent实际回复及直接对话。",
        ],
        "expansion": [
            "每类使用不同句式族，包含口语、省略、否定、纠正、同义表达；日期/编号替换不计作新语义覆盖。",
            "正常业务与反例成对；状态矩阵覆盖无待办/提交/审批/撤回，字段更新覆盖起日/止日/双日期/假别/事由。",
            "操作对象覆盖只提目标ID、双ID顺序互换、否定干扰ID、上下文唯一对象和对象歧义。",
            "多轮包括追问补齐、反复修改、取消后重开、可见摘要恢复；不要求恢复输入中不存在的信息。",
            "6种现有假别均覆盖；审批结论通过/驳回及缺失ID/结论/意见的组合均覆盖。",
            "按任务族和监督token数统计比例，单列旧业务、新边界和回复模式；不以大量重复模板凑样本量。",
        ],
        "integration_gaps": [
            "字段清空、半天时段、多任务执行序列、批量审批不在当前协议内；不训练伪执行成功。",
            "general_chat和clarify节点当前清除pending_action；如需暂聊/澄清后保留草稿，需单独改Agent状态逻辑。",
            "未取消旧草稿就直接另起不完整新草稿时，现有合并逻辑仍可能带入旧字段；本轮先覆盖明确取消后重开，直接切换需Agent重置任务状态。",
            "多任务当前只返回统一澄清文案；专门询问先办哪项，需要Agent回复调整。",
            "终端替换了数据库节点。查询/权限/检索依据/真正提交撤回审批必须接数据库另验收。",
            "当前查询节点是列表查询，不支持模型参数驱动的任意日期筛选、批量或单号查询筛选。",
            "摘要生成节点未纳入本轮训练；仅覆盖已有可见摘要，敏感字段被摘要省略后不能要求模型猜出。",
            "注入拦截属于现有代码能力，不把拦截结果冒充模型能力；保留程序级检查。",
        ],
        "split_and_acceptance": [
            "原V2/V3 val/test和已知失败题仅作开发回归，不复制原题入训练；新的独立表达族冻结为最终验收。",
            "完整轨迹、同一模板族不跨划分；相同短句在不同状态下可以用于反事实对照。",
            "预览属于设计材料，不作为独立测试；新验收题在正式训练数据冻结后单独制作、封存。",
            "分别汇报旧业务、新场景、独立新题；静态请求与由同一请求组成的闭环不能算两份独立证据。",
            "目标：验收集JSON合法率100%；确认/放弃/对象选择零错误；其余字段与意图严格联合准确率至少98%；多轮全轨迹至少95%。这些是待验证目标。",
            "身份识别和实际回复分别检查，不能只看含system身份声明的直接对话。",
        ],
        "experiment_plan": [
            "固定Full SFT底座、同一LoRA结构；保留V2/V3基线。",
            "先评估V3+旧提示词与V3+候选提示词，区分提示词收益；再训练新V4 adapter与候选提示词组合。",
            "保存各epoch adapter，依据旧业务、新边界验证联合选择，不只看总体平均loss；最终验收集不参与选模型。",
            "如比较不同混合比例，固定实际优化步数和学习率日程，记录样本与监督token数量。",
        ],
        "not_guaranteed": "有限验收不能证明所有真实输入完美；出现失败时按失败类型补训练族，并保留新的独立验收题。",
    }


class ScriptedModel:
    def __init__(self):
        self.expected = None
        self.reply = None
        self.calls = []

    def invoke(self, prompt):
        assert "给定上下文：" in prompt, "预览未脚本化摘要或其他额外模型任务"
        self.calls.append(("intent_parse", prompt))
        return SimpleNamespace(content=json.dumps(self.expected, ensure_ascii=False, separators=(",", ":")))

    async def ainvoke(self, prompt):
        assert self.reply is not None
        self.calls.append(("agent_reply", prompt))
        return SimpleNamespace(content=self.reply)

    async def astream(self, prompt):
        yield await self.ainvoke(prompt)


async def capture(agent_project, output, refresh=False):
    sys.path.insert(0, str(agent_project))
    from app.capabilities.policy_engine import validate_leave_draft
    from app.domain.user.models import UserContext
    from app.workflows import intent_protocol
    from local_runtime.chat_cli import create_terminal_graph, run_turn
    logging.disable(logging.INFO)

    assert not output.exists() or refresh, "已有V4预览目录，请先审核现有内容；明确更新预览使用--refresh"
    output.mkdir(parents=True, exist_ok=True)
    scenes, spec = definitions(), specification()
    (output / "intent_prompt_candidate.txt").write_bytes(CANDIDATE_RULES.encode("utf-8"))
    save(output / "spec.json", spec)
    save(output / "scenarios.json", {"status": "pending_user_review", "trajectories": scenes})
    rows, evidence = [], []
    user = UserContext(user_id="preview_account", name="", role="employee")
    for scene in scenes:
        teacher = ScriptedModel()
        graph = create_terminal_graph(teacher)
        thread = "preview-v4-" + scene["id"]
        if scene.get("initial_state"):
            await graph.aupdate_state({"configurable": {"thread_id": thread}}, deepcopy(scene["initial_state"]))
        fixed_now = datetime.fromisoformat(scene["current_date"])

        class Clock:
            @staticmethod
            def now():
                return fixed_now

        for index, item in enumerate(scene["turns"], 1):
            before = (await graph.aget_state({"configurable": {"thread_id": thread}})).values
            before_pending = deepcopy(before.get("pending_action"))
            expected = item["expected"]
            assert set(expected) == {"intent", "extracted_info"} and expected["intent"] in INTENTS
            assert set(expected["extracted_info"]) <= FIELDS
            assert all(isinstance(v, str) and v for v in expected["extracted_info"].values())
            assert expected["extracted_info"].get("decision", "通过") in {"通过", "驳回"}
            teacher.expected, teacher.reply = expected, item.get("response")
            calls, trace = len(teacher.calls), []
            with patch.object(intent_protocol, "RULES", CANDIDATE_RULES), patch("app.workflows.chat_nodes.recognition.datetime", Clock), patch(
                "app.workflows.chat_nodes.leave.validate_leave_draft", side_effect=lambda raw: validate_leave_draft(raw, today=fixed_now.date())
            ):
                state = await run_turn(graph, thread, item["user"], user, trace.append)
            captured = teacher.calls[calls:]
            assert [mode for mode, _ in captured] == (["intent_parse", "agent_reply"] if teacher.reply else ["intent_parse"])
            node = NODES[expected["intent"]]
            checks = {"intent": state["intent"] == expected["intent"], "fields": state["extracted_info"] == expected["extracted_info"],
                      "route": f"[节点] {node}" in trace,
                      "response": all(part in state["response"] for part in item.get("response_contains", []))}
            if teacher.reply:
                checks["scripted_reply"] = state["response"] == teacher.reply
            if node in DB_NODES:
                checks["database_unavailable"] = "没有执行数据库操作" in state["response"]
                # 本地数据库占位节点保留待办，不可据此声称生产节点执行成功。
                checks["pending_preserved_by_stub"] = state.get("pending_action") == before_pending
            if expected["intent"] in {"cancel_action", "clarify", "general_chat"}:
                checks["pending_cleared"] = state.get("pending_action") is None
            if expected["intent"] == "cancel_action":
                checks["accumulated_cleared"] = state["conversation_context"].get("accumulated_extracted_info") == {}
            if expected["intent"] == "submit_leave":
                info = expected["extracted_info"]
                complete = {"leave_type", "start_date", "end_date", "reason"} <= set(info)
                if complete and item["id"] != "invalid_dates":
                    checks["full_draft"] = (state.get("pending_action") or {}).get("action_type") == "submit_leave" and all(
                        state["pending_action"]["draft"].get(k) == v for k, v in info.items())
                else:
                    checks["no_pending_incomplete_or_invalid"] = state.get("pending_action") is None
                checks["accumulated_known_fields"] = state["conversation_context"].get("accumulated_extracted_info") == info
            if item.get("expected_draft"):
                draft = (state.get("pending_action") or {}).get("draft", {})
                checks["draft"] = all(draft.get(k) == v for k, v in item["expected_draft"].items())
            assert all(checks.values()), (item["id"], checks)
            for mode, prompt in captured:
                answer = json.dumps(expected, ensure_ascii=False, separators=(",", ":")) if mode == "intent_parse" else teacher.reply
                row = {"id": item["id"] + "/" + mode, "trajectory_id": scene["id"], "turn": index,
                       "mode": mode, "coverage": scene["coverage"], "status": "pending_user_review",
                       "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}
                if mode == "intent_parse":
                    context = json.loads(prompt.split("\n给定上下文：\n", 1)[1].split("\n用户输入：\n", 1)[0])
                    row.update(expected=expected, scenario={**context, "user_message": item["user"]})
                rows.append(row)
            current = state.get("pending_action")
            evidence.append({"id": item["id"], "checks": checks, "nodes": [t[5:] for t in trace if t.startswith("[节点] ")],
                             "observed_response": state["response"],
                             "observed_pending": {"action_type": current["action_type"], "draft": current.get("draft")} if current else None,
                             "note": scene.get("note", ""), "execution_scope": "人工标签推进原终端Agent；无真实模型调用，数据库节点不可用。"})

    # 直接问答的身份与原Agent单条human模板分开记录，避免只训练system声明一种输入。
    for uid, system in (("direct_identity_no_system", None), ("direct_identity_with_system", "你是 TrainPilot，一名企业业务助手。")):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": "请用一小句话说明这位助手的名字。"},
            {"role": "assistant", "content": "我叫 TrainPilot，是一名企业业务助手。"}]
        rows.append({"id": uid, "mode": "direct_reply", "coverage": "直接对话身份", "status": "pending_user_review", "messages": messages})

    old_prompts = set()
    for version in ("v2", "v3"):
        for split in ("val", "test"):
            for line in (ROOT / f"model/dataset/datasets/trainpilot_agent/{version}/{split}.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                old_prompts.add(json.dumps(record["messages"][:-1], ensure_ascii=False, sort_keys=True))
    assert not any(json.dumps(row["messages"][:-1], ensure_ascii=False, sort_keys=True) in old_prompts for row in rows)
    assert {row["expected"]["intent"] for row in rows if row["mode"] == "intent_parse"} == INTENTS
    files = ("app/workflows/intent_protocol.py", "app/workflows/intent_protocol.txt", "app/workflows/chat_nodes/recognition.py",
             "app/workflows/chat_nodes/response.py", "app/workflows/chat_nodes/leave.py", "app/workflows/chat_nodes/chat.py",
             "app/workflows/chat_nodes/query.py", "app/workflows/chat_nodes/approval.py", "app/workflows/chat_nodes/routing.py",
             "app/workflows/chat_nodes/summarize.py", "app/routes/chat.py", "app/capabilities/policy_engine.py", "local_runtime/chat_cli.py")
    artifact = {"status": "pending_user_review", "note": "仅人工预览及代码链路检查，不是模型测评，尚无正式train/val/test。",
                "candidate_prompt_sha256": sha(output / "intent_prompt_candidate.txt"),
                "preview_generator_sha256": sha(Path(__file__)),
                "scenario_sha256": sha(output / "scenarios.json"), "agent_sources_sha256": {name: sha(agent_project / name) for name in files},
                "modes": dict(Counter(row["mode"] for row in rows)), "sample_count": len(rows),
                "external_model_calls": 0, "database_writes": 0, "runtime_checks_passed": True,
                "old_heldout_exact_input_overlap": 0, "samples": rows, "runtime_evidence": evidence}
    save(output / "samples.json", artifact)
    review(output, spec, scenes, rows, evidence)
    print(json.dumps({key: value for key, value in artifact.items() if key not in {"samples", "runtime_evidence", "agent_sources_sha256"}}, ensure_ascii=False, indent=2))


def review(output, spec, scenes, rows, evidence):
    lines = ["# TrainPilot V4 数据覆盖与样例预览", "", "状态：待用户审核。没有生成V4正式训练集，没有训练模型，也没有修改Agent运行文件。", "",
             "下列输出为人工编写的目标答案。脚本把这些答案传给原Agent，检查解析、路由、本地状态及回复接口；通过不代表模型已经掌握能力。", "",
             "## 目标与已确认边界", "", "统一身份TrainPilot；完整已知字段；多任务先clarify；字段清空暂列接口待支持。主目标是项目业务能力。", "",
             "验收目标：JSON合法率100%；确认/放弃/操作对象关键题零错误；其余意图和字段联合准确率≥98%；多轮全轨迹成功率≥95%。有限验收不能保证全部真实输入完美。", "",
             "## 覆盖清单", "", "|覆盖项|本次预览识别请求数|正式扩展要求|", "|---|---:|---|"]
    counts = Counter(row["coverage"] for row in rows if row["mode"] == "intent_parse")
    for name, count in counts.items():
        lines.append(f"|{name}|{count}|不同表达族、不同状态及反例；按轨迹分组|")
    lines += ["", "六种假别、双审批结论、所有缺失字段组合将在正式扩展时补齐；当前样例只用于审核规则，不宣称已完成全部覆盖。", "",
              "## 数据配方与实验", ""]
    for heading, key in (("旧样本与新样本", "replay_plan"), ("扩展方式", "expansion"), ("划分与验收", "split_and_acceptance"), ("训练对照", "experiment_plan"), ("需Agent配合的事项", "integration_gaps")):
        lines += [f"### {heading}", ""] + ["- " + item for item in spec[key]] + [""]
    lines += ["候选意图提示词仅保存在本预览目录，检查时内存替换；确认后需让训练和实际Agent使用相同版本。提示词变化本身要单独做对照。", "",
              "## 人工样例", "", f"共{len(rows)}条监督请求，覆盖原Agent的识别JSON、实际自然回复输入，以及有/无system的直接身份问答。完整输入见samples.json。", ""]
    evidence_by_id = {item["id"]: item for item in evidence}
    for row in rows:
        lines += [f"### {row['id']} · {row['mode']} · {row['coverage']}", ""]
        if row["mode"] == "intent_parse":
            context = row["scenario"]
            lines += ["上下文（模型实际可见）：", "", "```json", json.dumps({k: v for k, v in context.items() if k != "user_message"}, ensure_ascii=False, separators=(",", ":")), "```", "",
                      "用户：" + context["user_message"], "", "预期输出：", "", "```json", row["messages"][-1]["content"], "```", ""]
            item = evidence_by_id[row["id"].split("/")[0]]
            lines += ["原Agent检查：" + "、".join(item["checks"]) + "通过；业务节点：" + " → ".join(item["nodes"]), ""]
            if item["note"]:
                lines += ["说明：" + item["note"], ""]
        else:
            lines += ["实际输入：", "", "```text", "\n".join(m["content"] for m in row["messages"][:-1]), "```", "", "预期回复：" + row["messages"][-1]["content"], ""]
    lines += ["## 审核后再执行", "", "1. 按通过的规则审查旧样本并扩展正式数据，输出来源及覆盖统计。",
              "2. 冻结新的表达族划分，检查冲突、泄漏、长度和assistant监督范围。",
              "3. 独立核验提示词对照与关键状态题，再准备服务器训练包；保留V2/V3权重。", ""]
    (output / "review.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def check_tokenizer(path, output):
    from transformers import AutoTokenizer
    from model.dataset.dataset import QwenSFTEncoder
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    encoder = QwenSFTEncoder(tokenizer, max_length=1536)
    data = json.loads((output / "samples.json").read_text(encoding="utf-8"))
    lengths, targets = [], []
    assistant_end = "<|im_end|>"
    assistant_end_id = tokenizer.convert_tokens_to_ids(assistant_end)
    for row in data["samples"]:
        encoded = encoder(row)
        full = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=False)
        assert encoded["input_ids"] == full, (row["id"], "被截断或模板不一致")
        supervised = [label for label in encoded["labels"] if label != -100]
        # Base的eos是endoftext；聊天模板的assistant结束符是im_end，之后还有非监督换行。
        assert supervised[-1] == assistant_end_id, (row["id"], "assistant结束符缺少监督")
        text = tokenizer.decode(supervised)
        answer = row["messages"][-1]["content"] + assistant_end
        assert text.endswith(answer), (row["id"], "答案正文不完整")
        assert text[:-len(answer)] in {"", "<think>\n\n</think>\n\n"}, (row["id"], "监督混入其他正文")
        n = sum(label != -100 for label in encoded["labels"][1:])
        assert n > 0
        lengths.append(len(full))
        targets.append(n)
    result = {"samples": len(lengths), "max_tokens": max(lengths), "min_supervised_tokens": min(targets),
              "total_supervised_tokens": sum(targets), "max_length": 1536, "truncated": 0,
              "assistant_end_token": assistant_end, "assistant_end_id": assistant_end_id,
              "base_eos_token": tokenizer.eos_token,
              "note": "仅编码完整性与assistant结束符检查，不是模型效果测试。Qwen模板自动添加空think前缀，不含人工思维链。",
              "samples_sha256": sha(output / "samples.json")}
    save(output / "encoding_check.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-project", type=Path)
    parser.add_argument("--check-tokenizer", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="重新生成尚未批准的预览，不用于正式数据")
    args = parser.parse_args()
    if args.check_tokenizer:
        check_tokenizer(args.check_tokenizer.resolve(), args.output_dir.resolve())
    else:
        asyncio.run(capture(args.agent_project.resolve(), args.output_dir.resolve(), args.refresh))


if __name__ == "__main__":
    main()
