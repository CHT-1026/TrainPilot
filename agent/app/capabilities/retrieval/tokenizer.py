"""轻量中文检索分词：ASCII 词 + 汉字二元组。"""
import re

_ASCII_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = _ASCII_RE.findall(text)
    for seq in _CJK_RE.findall(text):
        if len(seq) == 1:
            tokens.append(seq)
        else:
            tokens.extend(seq[index : index + 2] for index in range(len(seq) - 1))
    return tokens
