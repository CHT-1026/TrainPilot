"""实体对齐：别名归一、父子名称消歧、保守合并。"""
from __future__ import annotations

import re

from enterprise_rag.graph.schema import entity_id, normalize_name

COMPANY_ALIASES = (
    "澜图智能装备股份有限公司",
    "澜图智能",
    "澜图",
    "LANTU",
    "lantu",
)

TITLE_SUFFIXES = ("先生", "女士", "老师", "同志", "总", "工")
ROLE_WORDS = {
    "公司", "企业", "集团", "部门", "中心", "研究所", "办公室", "委员会", "工厂",
    "项目", "产品", "客户", "团队", "员工", "负责人", "总经理", "副总裁", "总监",
    "经理", "主管", "组长", "所长", "厂长", "工程师", "董事长", "主任", "部长",
    "本文件", "资料", "以上", "其中", "同时", "此外", "因此",
}

ALIAS_OVERRIDES = {
    "澜图智能": "company:澜图智能装备股份有限公司",
    "澜图": "company:澜图智能装备股份有限公司",
    "lantu": "company:澜图智能装备股份有限公司",
}


def strip_title(name: str) -> str:
    name = (name or "").strip()
    for suffix in TITLE_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix) + 1:
            name = name[: -len(suffix)]
    return name


def canonical_id(entity_type: str, name: str) -> str:
    name = strip_title(name)
    normalized = normalize_name(name)
    if normalized in {normalize_name(alias) for alias in COMPANY_ALIASES}:
        return ALIAS_OVERRIDES.get(normalized.lower(), entity_id("company", "澜图智能装备股份有限公司"))
    return entity_id(entity_type, name)


def is_probable_person(name: str) -> bool:
    name = strip_title(name)
    if not (2 <= len(name) <= 4):
        return False
    if name in ROLE_WORDS:
        return False
    if any(word in name for word in ("公司", "企业", "部门", "中心", "项目", "产品", "客户", "资料")):
        return False
    return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,4}", name))


def merge_aliases(primary: str, aliases: list[str]) -> list[str]:
    seen = {normalize_name(primary)}
    merged = []
    for alias in aliases:
        normalized = normalize_name(alias)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(alias)
    return merged
