"""规则优先的实体与关系抽取，全部结果绑定原文片段。

高精度优先：部门只认组织文档中的标准名称，人员只认作者、明确任命句、
汇报关系句；避免把“本办法/研发中心”等短语误抽成人名或部门。
"""
from __future__ import annotations

import re

from enterprise_rag.graph.align import canonical_id, is_probable_person, merge_aliases, strip_title
from enterprise_rag.graph.schema import entity_payload, relation_payload

NAME = r"[\u4e00-\u9fff]{2,4}"
MODEL_RE = re.compile(r"\b(?:LT|Lantu)[-A-Za-z0-9]*[A-Za-z0-9]+(?:[-A-Za-z0-9]+)*\b")
ORG_RE = re.compile(r"([\u4e00-\u9fff]{2,12}(?:集团|股份有限公司|有限公司|项目))")
TITLE = r"(?:CTO|CFO|COO|CEO|总裁|副总裁|总监|总经理|董事长|董事|主任|部长|经理|组长|所长|厂长|工程师|主管)"
ROLE_STOP = {
    "部门", "中心", "研究所", "办公室", "委员会", "工厂", "公司", "企业", "项目",
    "产品", "客户", "团队", "员工", "负责人", "总经理", "副总裁", "总监", "经理",
    "主管", "组长", "所长", "厂长", "工程师", "董事长", "主任", "部长", "本文件",
    "以上", "其中", "同时", "此外", "因此", "各部门", "该部门", "本部门", "分管",
    "直接", "核心", "现场", "渠道", "履约", "回款", "漏洞", "高级", "隔级主管",
    "生命周期", "第三方", "独立董事", "过程质量", "整体经营", "经理全面",
}
GENERIC_ORG = {"本项目", "该项目", "项目", "客户", "公司", "企业", "集团", "供应商"}

KNOWN_DEPARTMENTS = {
    "研发中心", "产品部", "销售中心", "客户成功部", "供应链中心", "制造中心",
    "质量部", "财务部", "人力资源部", "法务合规部", "信息技术部", "售后服务部",
    "战略投资部", "市场部", "行政部", "国际业务部", "董事会办公室", "审计委员会",
    "提名委员会", "薪酬与考核委员会", "战略委员会", "合规委员会", "信息安全委员会",
    "质量安全委员会", "产品决策委员会", "机器人本体研究所", "控制器研究所",
    "机器视觉研究所", "工艺应用研究所", "具身智能研究所", "测试中心", "苏州工厂",
    "远程诊断中心", "欧洲技术中心", "华东大区", "华南大区", "华北大区", "西部大区",
    "半导体事业部", "智能工厂事业部", "客户服务中心", "备件中心",
}

KNOWN_PROJECTS = {
    "恒芯动力电池项目", "澜岳汽车零部件白车身焊装项目", "星禾电子精密组装项目",
    "晶拓微电子晶圆搬运项目", "海翎家电喷涂与打磨项目", "顺捷供应链仓储拣选项目",
    "麦丰食品后端包装项目", "康泽制药无菌灌装项目", "鼎重机械结构件焊接项目",
    "昇达电梯门板装配项目", "曜阳光能电池片串焊项目", "苏州二期扩产项目",
    "芯动力电池电芯叠片与物流项目", "恒锐精密机床上下料项目",
}
KNOWN_CUSTOMERS = {
    "恒芯动力电池", "澜岳汽车零部件", "星禾电子", "晶拓微电子", "海翎家电",
    "顺捷供应链", "麦丰食品", "康泽制药", "鼎重机械", "昇达电梯", "曜阳光能", "恒锐精密",
    "恒芯", "澜岳", "星禾", "晶拓", "海翎", "顺捷", "麦丰", "康泽", "鼎重", "昇达", "曜阳",
}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。！？；\n]", text or "") if part.strip()]


def _add(entity_map: dict, entity: dict) -> None:
    current = entity_map.get(entity["id"])
    if current is None:
        entity_map[entity["id"]] = entity
        return
    current["aliases"] = merge_aliases(current["name"], current.get("aliases", []) + entity.get("aliases", []))
    current["doc_ids"] = sorted(set(current.get("doc_ids", [])) | set(entity.get("doc_ids", [])))
    current["attributes"].update({k: v for k, v in entity.get("attributes", {}).items() if v})


def extract_entities(chunk: dict, known_persons: set[str] | None = None) -> list[dict]:
    text = chunk.get("text", "")
    title = chunk.get("title", "")
    doc_id = chunk.get("doc_id", "")
    entities: dict[str, dict] = {}
    known_persons = known_persons or set()

    doc_entity = entity_payload(
        "document", title, aliases=[doc_id], doc_ids=[doc_id],
        attributes={
            "doc_id": doc_id,
            "version": chunk.get("version", ""),
            "effective_date": chunk.get("effective_date", ""),
            "doc_family": chunk.get("doc_family", ""),
            "category": chunk.get("category", ""),
            "department": chunk.get("department", ""),
            "author": chunk.get("author", ""),
        },
    )
    _add(entities, doc_entity)

    family = chunk.get("doc_family") or ""
    if family:
        _add(entities, entity_payload("family", family, doc_ids=[doc_id],
                                      attributes={"doc_family": family}))

    author = strip_title(chunk.get("author", ""))
    if is_probable_person(author):
        _add(entities, entity_payload("person", author, doc_ids=[doc_id]))

    department = (chunk.get("department") or "").strip()
    if department in KNOWN_DEPARTMENTS:
        _add(entities, entity_payload("department", department, doc_ids=[doc_id]))

    lowered = (text + "\n" + title).lower()
    if any(alias.lower() in lowered for alias in ("澜图智能", "澜图", "lantu")):
        _add(entities, entity_payload("company", "澜图智能装备股份有限公司",
                                      aliases=["澜图智能", "澜图", "LANTU"], doc_ids=[doc_id]))

    for name in KNOWN_DEPARTMENTS:
        if name in text:
            _add(entities, entity_payload("department", name, doc_ids=[doc_id]))

    for match in MODEL_RE.finditer(text):
        name = match.group(0).strip("-")
        if len(name) < 3:
            continue
        _add(entities, entity_payload("product", name, doc_ids=[doc_id]))

    for name in KNOWN_PROJECTS:
        if name in text:
            _add(entities, entity_payload("project", name, doc_ids=[doc_id]))

    for name in KNOWN_CUSTOMERS:
        if len(name) >= 3 and name in text:
            _add(entities, entity_payload("customer", name, doc_ids=[doc_id]))

    for name in known_persons:
        if name in text:
            _add(entities, entity_payload("person", name, doc_ids=[doc_id]))

    return list(entities.values())


def _entity_names(entities: list[dict]) -> list[tuple[str, str]]:
    pairs = [(entity["id"], entity["name"]) for entity in entities if entity["type"] != "document"]
    return sorted(pairs, key=lambda item: len(item[1]), reverse=True)


def extract_relations(chunk: dict, entities: list[dict]) -> list[dict]:
    text = chunk.get("text", "")
    doc_id = chunk.get("doc_id", "")
    chunk_id = chunk.get("chunk_id", "")
    section = chunk.get("section", "")
    doc_entity = canonical_id("document", chunk.get("title", ""))
    relations: dict[str, dict] = {}
    names = _entity_names(entities)
    person_names = {name for entity in entities if entity["type"] == "person" for name in [entity["name"]]}
    department = (chunk.get("department") or "").strip()

    def add(source: str, target: str, relation: str, evidence: str, confidence: float = 0.85) -> None:
        if not source or not target or source == target:
            return
        payload = relation_payload(source, target, relation, doc_id=doc_id, chunk_id=chunk_id,
                                   section=section, evidence=evidence[:240], confidence=confidence)
        relations[f"{source}|{relation}|{target}"] = payload

    def match_entities(fragment: str, types: tuple[str, ...]) -> list[str]:
        return [entity_id for entity_id, name in names
                if entity_id.split(":", 1)[0] in types and name in fragment]

    family = chunk.get("doc_family") or ""
    if family:
        add(doc_entity, canonical_id("family", family), "version_of", f"文档族：{family}", 1.0)
    if department in KNOWN_DEPARTMENTS:
        add(doc_entity, canonical_id("department", department), "belongs_to", f"所属部门：{department}", 0.95)
        author = strip_title(chunk.get("author", ""))
        if is_probable_person(author):
            add(canonical_id("person", author), canonical_id("department", department), "belongs_to",
                f"作者：{author}；所属部门：{department}", 0.9)

    for sentence in _sentences(text):
        for match in re.finditer(rf"({NAME})(?:是|为)({NAME})的(?:直属)?上级", sentence):
            boss, person = match.group(1), match.group(2)
            if boss in person_names and person in person_names:
                add(canonical_id("person", boss), canonical_id("person", person), "manages", sentence, 0.95)
        for match in re.finditer(rf"({NAME})的(?:直属)?上级(?:是|为)({NAME})", sentence):
            person, boss = match.group(1), match.group(2)
            if boss in person_names and person in person_names:
                add(canonical_id("person", boss), canonical_id("person", person), "manages", sentence, 0.95)
        for match in re.finditer(rf"({NAME})(?:直接)?(?:向|对)({NAME})汇报", sentence):
            person, boss = match.group(1), match.group(2)
            if boss in person_names and person in person_names:
                add(canonical_id("person", boss), canonical_id("person", person), "manages", sentence, 0.95)
        for match in re.finditer(rf"({NAME})(?:管理|带领|领导)({NAME})", sentence):
            boss, person = match.group(1), match.group(2)
            if boss in person_names and person in person_names:
                add(canonical_id("person", boss), canonical_id("person", person), "manages", sentence, 0.9)

        for match in re.finditer(rf"(?:^|[，。；、\s])({NAME})(?:同时)?(?:担任|负责)([^，。；\n]{{1,40}})", sentence):
            person, fragment = match.group(1), match.group(2)
            if person not in person_names:
                continue
            for target in match_entities(fragment, ("department", "product", "project", "customer")):
                add(canonical_id("person", person), target, "responsible_for", sentence, 0.85)
            if "项目" in fragment:
                for entity_id, name in names:
                    if entity_id.startswith("project:") and name in fragment:
                        add(canonical_id("person", person), entity_id, "responsible_for", sentence, 0.85)

        for match in re.finditer(rf"(?:配套|采用|搭载|集成)([^，。；\n]{{1,40}})", sentence):
            fragment = match.group(1)
            for target in match_entities(fragment, ("product",)):
                add(doc_entity, target, "uses", sentence, 0.75)

        for match in re.finditer(rf"({NAME}(?:集团|股份有限公司|有限公司|项目))", sentence):
            org = match.group(1)
            if org in GENERIC_ORG:
                continue
            org_id = canonical_id("customer" if not org.endswith("项目") else "project", org)
            add(doc_entity, org_id, "mentions_org", sentence, 0.7)

    return list(relations.values())
