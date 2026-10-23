"""会话级实体、主题与可覆盖时点（S21）。

这些字段只用于追问补全，不是证据；回答仍必须重新检索并重新授权。
"""
from __future__ import annotations

import re

from enterprise_rag.query import extract_time

_MODEL = re.compile(r"\b(?:LT|Lantu)[-A-Za-z0-9]*[A-Za-z0-9]+(?:[-A-Za-z0-9]+)*\b")
_DEPARTMENT = re.compile(r"[\u4e00-\u9fff]{2,8}(?:部(?!门)|中心|研究院|事业部|办公室|科|组)")
_PERSON = re.compile(r"(?:负责人|主管|经理|总监|上级|下属)(?:是|为)?\s*([\u4e00-\u9fff]{2,4})")
_QUOTED = re.compile(r"[《“\"]([^》”\"]{2,30})[》”\"]")
_FOLLOW_UP = re.compile(r"^(?:那|那么|这个|那个|它|他们|还有|另外|再|上述|前者|后者|呢)|^.{0,10}呢[？?]?$")
_TIME = re.compile(r"(?:19|20)\d{2}\s*年|[一二三四]季度|上半年|下半年|本月|上月|今年|去年|前年")


def extract_entities(text: str) -> list[str]:
    entities = []
    entities.extend(_MODEL.findall(text or ""))
    entities.extend(_QUOTED.findall(text or ""))
    entities.extend(_DEPARTMENT.findall(text or ""))
    for match in _PERSON.finditer(text or ""):
        entities.append(match.group(1))
    return list(dict.fromkeys(item for item in entities if item))


def update_dialog_state(previous: dict | None, question: str, extracted_info: dict | None = None,
                        intent: str = "", explicit_as_of: str | None = None) -> dict:
    state = dict(previous or {})
    text = (question or "").strip()
    explicit_entities = extract_entities(text)
    follow_up = bool(_FOLLOW_UP.search(text))
    # 实体属于当前主题，不能把上一型号永久累积后再随意选第一个。
    entities = explicit_entities or (list(state.get("entities") or []) if follow_up else [])
    as_of, _ = extract_time(text)
    if explicit_as_of:
        as_of = explicit_as_of
    topic = state.get("topic", "")
    entity_changed = bool(explicit_entities and set(explicit_entities) != set(state.get("entities") or []))
    topic_changed = bool(text and (not follow_up or entity_changed))
    if topic_changed:
        topic = text[:60]
        # 新主题且用户没有给新时点时，不继承上一主题的时间约束。
        if not _TIME.search(text) and not explicit_as_of:
            as_of = ""
    elif not as_of and not _TIME.search(text):
        # 同主题追问未给新时点时，保留该主题已有的时间约束。
        as_of = state.get("as_of", "")
    state.update({
        "topic": topic,
        "entities": entities[-20:],
        "as_of": as_of or "",
        "last_intent": intent or state.get("last_intent", ""),
        "last_question": text[:120],
    })
    return state


def apply_context(question: str, messages: list[dict], dialog_state: dict | None) -> tuple[str, str | None]:
    state = dialog_state or {}
    rewritten = question
    inherited_as_of = None
    if _FOLLOW_UP.search(question.strip()):
        # 省略主语的追问要消费已跟踪实体，否则“那它的尺寸呢”会丢失型号。
        anchor = state.get("topic", "")
        if not anchor:
            # 老会话可能只有消息历史；只从用户的具体问题恢复主题。
            anchor = next((str(m.get("content", ""))[:180] for m in reversed(messages or [])
                           if m.get("role") == "user" and m.get("content") != question
                           and not _FOLLOW_UP.search(str(m.get("content", "")))), "")
        if _TIME.search(question):
            # 新时点优先，旧问句中的年份不能反过来成为检索的首个时点。
            anchor = _TIME.sub("", anchor)
        entities = state.get("entities") or []
        if anchor and anchor != question and not extract_entities(question):
            rewritten = f"{question}（上下文：{anchor}）"
        elif len(entities) == 1 and entities[0] not in rewritten:
            rewritten = f"{question}（指代：{entities[0]}）"
        if not _TIME.search(question) and state.get("as_of"):
            inherited_as_of = state["as_of"]
    return rewritten, inherited_as_of
