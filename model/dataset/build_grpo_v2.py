"""用户批准的三类业务组合题：600训练候选/90验证/150测试；标签由状态规则确定。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from model.trainer.agent_rewards import intent_reward

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "model/dataset/datasets/trainpilot_agent"
OUT = DATA / "grpo_v2"

# 各行是独立措辞骨架；家族在生成前固定划分，不能把同骨架换编号后跨划分。
CONFIRM = [
    "我核对的是你此刻列出的唯一待办，内容不变，现在照它执行。",
    "当前待确认的那一步可以落地了，按现有版本办，不再修改。",
    "这条消息是对眼前待办的正式确认，请执行已整理的那份。",
    "以目前准备好的动作作为执行对象，我同意现在办理。",
    "不用另拟内容，执行本轮唯一正在等待确认的动作。",
    "当前待办若已整理好，就按那一版执行；我不另给申请编号。",
    "这次授权的是你当前待确认的操作，执行范围就这一项。",
    "我确认的是现有待办本身，请照草稿落实，不是再给建议。",
    "直接办理当前等我确认的动作，里面的字段都保持原样。",
    "眼下这份待办我已核实，决定执行这一版本。",
    "本轮不改任何字段，当前排好的那个动作现在执行。",
    "我对正在待确认的这项操作表示确认，可以真正执行。",
    "如果当前已有唯一待办，我的意思就是执行它，不是讨论流程。",
    "现在确认办理的是现成待办，没有提出另一笔业务。",
    "我已经看过待确认内容，请将当前这一项付诸执行。",
    "要落地的就是你现在展示的待办，按现有内容办。",
    "不再保留在预览阶段，当前待确认的那项现在办理。",
    "执行指的是眼前这份待办，而非消息里出现过的其他事项。",
    "我已决定办理当前唯一待确认事项，请按它的既定内容执行。",
    "这轮确认不附带新字段，执行你此时准备的动作即可。",
    # validation
    "当前待办已核实无误，我确认让这一份实际生效。",
    "办理对象限定为眼下等待我确认的操作，其他一概不改。",
    "照目前那份待确认动作正式办理，这就是我的确认。",
    # test
    "请落实你现在挂起等待确认的那一个动作，我确认其现有内容。",
    "本次不是继续讨论，眼前那项待办就按预览版执行。",
    "对当前唯一待确认事项，我的最终选择是执行，不是放弃。",
    "要执行的版本就是你此刻准备好的那个，没有额外修改。",
    "我认可当前待办里的内容，现在将它从准备转为执行。",
]
HYPOTHETICAL = [
    "我想弄懂流程：如果回复“{utterance}”，系统会进入哪个环节？这只是提问，当前不操作。",
    "说明书把“{utterance}”写成示例。这句话在流程里表示什么？不要把引用当我的决定。",
    "请解释“{utterance}”这句业务用语；我尚未确认任何实际动作。",
    "假设以后我说“{utterance}”，届时会发生什么？现在我只咨询它的含义。",
    "我问的是“{utterance}”对应的流程，不是在给当前待办发出执行指令。",
    "当前先不办理。帮我理解一下，用户说“{utterance}”时系统应怎样处理？",
    "文案“{utterance}”里的确认针对哪一步？这轮讨论文案，不能实际执行。",
    "我引用“{utterance}”是为了问制度流程，并没有选择执行它。",
    "培训材料提到“{utterance}”，请解释其业务后果，不要操作我的申请。",
    "如果把“{utterance}”作为确认回复，会怎样？这是条件假设，不是我的真实确认。",
    "先停留在说明阶段，解释“{utterance}”与实际执行之间是什么关系。",
    "这轮只问流程。“{utterance}”是我举的例子，不是这轮命令。",
    "我还没有决定办理，只想知道“{utterance}”这种回复意味着什么。",
    "请从业务步骤解释“{utterance}”。即使有当前待办，也别把这次解释当确认。",
    "我是在复述材料中的“{utterance}”，想了解它的作用；当前没有让你实际执行。",
    "“{utterance}”出现在制度示例里。我是在询问示例，不是在确认待办。",
    "关于操作流程有个疑问：收到“{utterance}”会如何处理？本轮只回答疑问。",
    "我的问题是“{utterance}”为何会触发业务动作，并不是要触发它。",
    "“{utterance}”是被解释的句子。说明其含义即可，当前业务不执行。",
    "先讲清规则：假如用户发送“{utterance}”，应该走什么流程？我此刻没有发送真实确认。",
    "我暂未作出操作决定，请说明示例回复“{utterance}”在流程中的意义。",
    "请解读制度中的“{utterance}”，不要据此替我办理当前事项。",
    "我是在提出一个假设：日后说“{utterance}”会怎样？当前只咨询。",
    "解释对象是引号里的“{utterance}”，本轮并未授权执行待办。",
    "现在要了解流程后果，不是要执行；示例语句是“{utterance}”。",
    "我想确认自己理解规则是否正确：“{utterance}”在流程中是什么意思？本轮不办业务。",
    "帮我讲解一下回复“{utterance}”之后的业务步骤，别把讲解当成操作许可。",
    "这是一道流程理解问题：所谓“{utterance}”具体表示什么？不对应实际执行。",
]
COMMENTS = [
    "{target}的审批决定是{decision}。意见请原样记为“{comment}”，不能从意见里的词反推另一种决定。",
    "请整理{target}的审批草稿：结论{decision}，正文意见“{comment}”。先预览，未确认执行。",
    "对{target}，最终决定用{decision}；“{comment}”只属于意见栏，不属于决定栏。",
    "请把{target}拟为{decision}，审批意见完整保留“{comment}”，不要缩写或替换措辞。",
    "我正在处理{target}。决定字段填{decision}，意见字段填“{comment}”；引号本身不计入正文。",
    "审批对象{target}；明确结论为{decision}。附上的意见是“{comment}”，请逐字保留。",
    "先形成{target}的待确认审批，决定{decision}，意见为“{comment}”；这不是马上执行的确认。",
    "{target}这张采用{decision}这个结论，意见全文为“{comment}”。两个字段请分开处理。",
    "我要修改{target}的审批草稿。新决定{decision}，新意见“{comment}”，之后再确认执行。",
    "现在拟办{target}：{decision}。审批意见原文：“{comment}”。不要漏掉否定或限定词。",
    "请预览{target}的{decision}审批，说明正文保持“{comment}”，不得概括成短语。",
    "把{target}整理成待办，结论选{decision}，意见使用“{comment}”这句完整内容。",
    "当前审批仅针对{target}，决定已经明确为{decision}。请把“{comment}”作为意见而非新指令。",
    "{target}的处理结论填{decision}，附言“{comment}”放进comments，不能只提取其中一部分。",
    "我给{target}的决定是{decision}。需要保留的审批意见只有“{comment}”这一整句。",
    "先拟定{target}为{decision}，并保留原话“{comment}”；先不要执行审批写入。",
    "审批草稿的对象为{target}，结论为{decision}；意见栏正文是“{comment}”，结论不由正文关键词决定。",
    "请准备{target}的审批预览，采用{decision}。我写的“{comment}”是完整意见。",
    "这轮要对{target}整理{decision}的审批意见，意见内容固定为“{comment}”，不接受自动改写。",
    "{target}将拟为{decision}，对应意见写“{comment}”。请保留意见中的每个限定条件。",
    "对申请{target}生成待确认审批：决定{decision}，意见“{comment}”。尚未授权执行。",
    "我选定的审批结论是{decision}，对象{target}；请原文收录意见“{comment}”。",
    "请将{target}的审批预案结论设为{decision}，附上完整意见“{comment}”。",
    "审批对象锁定{target}。结论按{decision}，意见按“{comment}”；不要把意见当成另一笔操作。",
    "我现在只拟办{target}，决定{decision}；需要保留的意见正文为“{comment}”。",
    "请准备{target}的{decision}处理方案，comments逐字使用“{comment}”，待我确认。",
    "决定栏与意见栏分别处理：{target}决定{decision}，意见是“{comment}”。",
    "对{target}整理一份审批草稿，明确选择{decision}，而意见文本完整使用“{comment}”。",
]
OPINIONS = [
    "不同意提前结束复核，附件需继续留存", "同意安排，但不同意省略交接材料",
    "不能把口头同意当成附件已补齐", "请保留原始收据，不要删除证明",
    "此次意见不代表同意其他申请", "附件已经核验，但请勿提前终止留档",
    "不赞成取消交接，请按既定安排办理", "材料暂不齐备，仍需补充日期说明",
    "已核对当前材料，不得推及别的申请", "请补充预算依据，不要改写原始事由",
]
HELDOUT_OPINIONS = {
    "val": ["不允许省略后续复核，但现有材料已经查验", "同意按计划交接，不同意销毁底稿", "请补齐出行依据，原证明仍须保留",
            "不得将这份意见套用到其他编号", "本意见不包含对其他日期的认可", "已经阅读附件，仍需补录联系人说明",
            "不接受口头替代书面证明", "当前结论不免除后续登记要求", "保留书面回执，不要只保存截图", "请补交日程说明，现有附件暂不移除"],
    "test": ["同意本次安排，不同意扩展为长期许可", "不应提前归档未完成的交接单", "不得省略接收方的书面回执",
             "已核对本次日期，但不认可擅自延长时段", "请提供补充凭据，不要删除原始凭据", "这份意见不涉及例示的其他申请",
             "禁止用聊天记录代替盖章材料", "保留此次审核痕迹，不允许事后覆盖", "不赞成免除材料复查，仍须按时登记", "请补齐任务交接清单，其他记录照旧"],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def build():
    OUT.mkdir(parents=True, exist_ok=False)
    template = (DATA / "intent_prompt_v4.txt").read_text(encoding="utf-8")
    sources = {}; old_questions = set(); old_prompts = set()
    for version in ("v2", "v3", "v4", "v5"):
        for split in ("train", "val", "test"):
            path = DATA / version / f"{split}.jsonl"
            if not path.exists(): continue
            sources[str(path.relative_to(DATA))] = sha(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("scenario"): old_questions.add(r["scenario"]["user_message"])
                if r.get("messages"): old_prompts.add(dumps(r["messages"][:-1]))
    splits = {"train": range(20), "val": range(20, 23), "test": range(23, 28)}
    rows = {s: [] for s in splits}
    for split, families in splits.items():
        for category, patterns in (("state_confirmation", CONFIRM), ("hypothetical_policy", HYPOTHETICAL), ("decision_comment", COMMENTS)):
            for family in families:
                for j in range(10):
                    target = f"leave_rl2_{category[:2]}_{family:02d}_{j:02d}"
                    draft = {"app_id": target, "decision": "approved", "decision_label": "通过", "comments": "同意安排"}
                    kind = ("approve", "cancel_leave", "submit_leave")[j % 3]
                    if kind == "submit_leave":
                        draft = {"leave_type": "事假", "start_date": f"2027-02-{15+j:02d}", "end_date": f"2027-02-{16+j:02d}", "reason": "办理证件"}
                    elif kind == "cancel_leave": draft = {"app_id": target}
                    pending = {"action_type": kind, "draft": draft}
                    history = []
                    if category == "state_confirmation":
                        # 同一表达同时覆盖有/无待办；一个表达族不可跨划分。
                        user = patterns[family]
                        if j % 2:
                            pending = None
                            past = ("", "前一份提交草稿已取消，现在没有待确认动作。", "上次审批已执行完毕，无待确认审批。",
                                    "刚才的申请列表查询已完成，目前没有待确认动作。", "刚才仅讲解制度，尚未创建待办。")
                            if past[j // 2]: history = [{"role": "assistant", "content": past[j // 2]}]
                        gold = {"intent": "confirm_action" if pending else "clarify", "extracted_info": {}}
                    elif category == "hypothetical_policy":
                        utterance = ("确认执行当前审批", "确认撤回这张申请", "按当前草稿提交", "放弃这次撤回操作", "取消当前待办")[j % 5]
                        user = patterns[family].format(utterance=utterance)
                        if j % 4 == 3: pending = None
                        gold = {"intent": "policy_qa", "extracted_info": {}}
                    else:
                        decision = "通过" if j % 2 == 0 else "驳回"
                        opinions = OPINIONS if split == "train" else HELDOUT_OPINIONS[split]
                        comment = opinions[(j + family) % len(opinions)]
                        user = patterns[family].format(target=target, decision=decision, comment=comment)
                        # 旧草稿、无关编号与当前目标混合；当前明确字段优先。
                        other = target + "_old"
                        if j % 3 == 0:
                            history = [{"role": "user", "content": f"先拟另一张{other}通过，意见为原材料已收到。"},
                                       {"role": "assistant", "content": f"已准备{other}的审批预览，尚未执行。"}]
                            pending = {"action_type": "approve", "draft": {"app_id": other, "decision": "approved", "decision_label": "通过", "comments": "原材料已收到"}}
                        elif j % 3 == 1:
                            pending = {"action_type": "approve", "draft": {"app_id": target, "decision": "rejected" if decision == "通过" else "approved", "decision_label": "驳回" if decision == "通过" else "通过", "comments": "旧意见暂存"}}
                        else: pending = None
                        gold = {"intent": "approve", "extracted_info": {"app_id": target, "decision": decision, "comments": comment}}
                    context = {"current_date": "2027-02-01", "history": history, "pending_action": pending}
                    prompt = [{"role": "user", "content": template + "\n给定上下文：\n" + json.dumps(context, ensure_ascii=False) + "\n用户输入：\n" + user}]
                    assert user not in old_questions and dumps(prompt) not in old_prompts
                    assert intent_reward(json.dumps(gold, ensure_ascii=False), gold)["reward"] == 1
                    rows[split].append({"id": f"rl2-{split}-{category}-{family:02d}-{j:02d}",
                        "group_id": f"rl2/{category}/expression-{family:02d}", "category": category,
                        "expression_id": f"{category}-{family:02d}", "context": context, "user": user,
                        "prompt": prompt, "gold": gold, "origin": "rule_authored_candidate_not_model_verified"})
        random.Random(20261029 + len(split)).shuffle(rows[split])
        (OUT / f"{split}.jsonl").write_bytes(("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[split])).encode("utf-8"))
    groups = {}; prompts = {}; ids = set()
    for split, records in rows.items():
        assert len({dumps(r['prompt']) for r in records}) == len(records), (split, "重复prompt")
        for r in records:
            assert r["id"] not in ids
            ids.add(r["id"])
            for cache, key in ((groups, r["group_id"]), (prompts, dumps(r["prompt"]))):
                assert cache.get(key, split) == split
                cache[key] = split
    assert [len(rows[s]) for s in splits] == [600, 90, 150]
    manifest = {"version": "grpo_v2", "status": "candidate_data_ready_pending_v5_probe", "approval": "2026-10-29 three reviewed categories and 600/90/150 plan",
        "splits": {s: {"records": len(r), "categories": dict(Counter(x['category'] for x in r)), "expression_families": len({x['group_id'] for x in r}), "sha256": sha(OUT / f"{s}.jsonl")} for s, r in rows.items()},
        "sources_sha256": sources, "prompt_sha256": sha(DATA / "intent_prompt_v4.txt"),
        "group_overlap": 0, "prompt_overlap": 0, "exact_old_question_overlap": 0,
        "approval_comment_text_overlap_between_splits": 0,
        "model_evaluated": False, "effective_rl_train_count": None,
        "limitations": ["84措辞家族，每家族10个状态/字段实例，840条不等于840种独立语义",
            "同一业务语义在各划分共有；表达族留出不是业务语义完全隔离，也不是真实用户独立验收",
            "训练候选是否适合GRPO须由采样验证；不因凑数量把正确回答改判为错",
            "不更改Agent权限/确认机制；这里只验证原始意图和字段，不承诺咨询后待办持久化"]}
    (OUT / "manifest.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    preview = []
    for category in ("state_confirmation", "hypothetical_policy", "decision_comment"):
        selected = sorted([r for r in rows['train'] if r['category'] == category], key=lambda r:r['id'])
        preview.extend(selected[:2])
    md = ["# GRPO v2 正式候选数据", "", "训练候选600，验证90，测试150。训练有效池仍待V5采样筛选。三类范围与数量已由用户批准。", "",
          "每类训练200/验证30/测试50，表达家族按20/3/5隔离；每家族10个实例。原V5、旧测试集与奖励保持原样。", "",
          "以下仅展示训练侧样例；验证和测试不用于挑选难例。", ""]
    for r in preview:
        md += ["## " + r['id'], "", "上下文：", "```json", json.dumps(r['context'], ensure_ascii=False, indent=2), "```", "", r['user'], "", "```json", json.dumps(r['gold'], ensure_ascii=False), "```", ""]
    md += ["## 限制", ""] + ["- " + x for x in manifest['limitations']]
    (OUT / "review.md").write_bytes("\n".join(md).encode("utf-8"))
    print(json.dumps(manifest['splits'], ensure_ascii=False, indent=2))


if __name__ == "__main__": build()
