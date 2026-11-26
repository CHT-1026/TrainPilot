"""Qwen3 Hermes工具调用解析；只转换合法完整标记，不执行模型生成的代码。"""
import json
import re
import uuid


def parse_tool_output(text, *, bare_names=()):
    thinking = re.findall(r"<think>(.*?)</think>", text, re.S)
    if text.count("<think>") != text.count("</think>"):
        raise ValueError("incomplete_thinking_block")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if bare_names and "<tool_call>" not in text:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if (isinstance(value, dict) and set(value) == {"name", "arguments"}
                    and isinstance(value["name"], str) and value["name"] in bare_names
                    and isinstance(value["arguments"], dict)):
                text = "<tool_call>" + text + "</tool_call>"
    blocks = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S))
    if text.count("<tool_call>") != len(blocks) or text.count("</tool_call>") != len(blocks):
        raise ValueError("incomplete_tool_call")
    calls = []
    for block in blocks:
        value = json.loads(block[1])
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ValueError("invalid_tool_call_object")
        if not isinstance(value["name"], str) or not value["name"] or not isinstance(value["arguments"], dict):
            raise ValueError("invalid_tool_call_arguments")
        calls.append({"id": "call_" + uuid.uuid4().hex[:16], "type": "function", "function": {
            "name": value["name"], "arguments": json.dumps(value["arguments"], ensure_ascii=False)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    # Empty think is a template artifact; nonempty thinking is preserved rather than executed.
    content = re.sub(r"<think>\s*</think>", "", content).strip()
    return {"role": "assistant", "content": content or None,
            **({"reasoning_content": "\n".join(thinking)} if any(t.strip() for t in thinking) else {}),
            **({"tool_calls": calls} if calls else {})}


def template_messages(messages):
    """OpenAI arguments字符串 -> Qwen模板要求的对象，不改变原始请求。"""
    converted = []
    for message in messages:
        item = dict(message)
        if item.get("content") is None:
            item["content"] = ""
        if item.get("tool_calls"):
            item["tool_calls"] = []
            for call in message["tool_calls"]:
                function = dict(call["function"])
                if isinstance(function["arguments"], str):
                    function["arguments"] = json.loads(function["arguments"])
                item["tool_calls"].append({**call, "function": function})
        converted.append(item)
    return converted
