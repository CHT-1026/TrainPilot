"""GraphRAG 基础数据结构与名称归一化。"""
from __future__ import annotations

import re
import unicodedata

ENTITY_TYPES = ("company", "person", "department", "product", "customer", "project", "document", "family")

_PUNCT = re.compile(r"[\s\-—_·•,，。；;:：()（）\[\]【】<>《》\"'“”‘’/\\|]+")


def normalize_name(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    return _PUNCT.sub("", text)


def entity_id(entity_type: str, name: str) -> str:
    return f"{entity_type}:{normalize_name(name)}"


def relation_key(source: str, target: str, relation: str) -> str:
    return f"{source}|{relation}|{target}"


def entity_payload(entity_type: str, name: str, **kwargs) -> dict:
    payload = {
        "id": entity_id(entity_type, name),
        "type": entity_type,
        "name": name,
        "aliases": [],
        "doc_ids": [],
        "attributes": {},
    }
    payload.update(kwargs)
    return payload


def relation_payload(source: str, target: str, relation: str, **kwargs) -> dict:
    payload = {
        "source": source,
        "target": target,
        "relation": relation,
        "doc_id": "",
        "chunk_id": "",
        "section": "",
        "evidence": "",
        "confidence": 0.8,
    }
    payload.update(kwargs)
    return payload
