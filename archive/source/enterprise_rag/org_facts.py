"""Deterministic organization-chart facts: counts, membership, additions, versions.

The org chart is a small structured document family. Quoting two version lists is
more reliable than asking a model to remember which department was added or
removed. This module only covers org_chart-family chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_rag.citations import chunk_body

_DEPT_LIST = re.compile(r"设\s*(\d+)\s*个一级部门[：:]([^。]+)。")
_UPGRADE = re.compile(r"([^。；\s]+?)由([^。；]*?)升级为一级部门")


@dataclass
class OrgVersion:
    doc_id: str
    title: str
    version: str
    updated: str
    effective: str
    year: str
    departments: list[str]
    chunk_id: str
    index: int
    text: str


def parse_versions(chunks: list[dict]) -> list[OrgVersion]:
    versions: list[OrgVersion] = []
    seen: set[tuple[str, str]] = set()
    for index, chunk in enumerate(chunks, 1):
        doc_id = str(chunk.get("doc_id") or "")
        title = str(chunk.get("title") or "")
        if "org_chart" not in doc_id and "组织架构" not in title:
            continue
        text = chunk_body(chunk)
        match = _DEPT_LIST.search(text)
        if not match:
            continue
        key = (doc_id, chunk.get("chunk_id") or "")
        if key in seen:
            continue
        seen.add(key)
        departments = [item.strip() for item in re.split(r"[、，,]", match.group(2)) if item.strip()]
        year_match = re.search(r"(20\d{2})", title) or re.search(r"(\d{4})", text[:80])
        versions.append(
            OrgVersion(
                doc_id=doc_id,
                title=title or doc_id,
                version=str(chunk.get("version") or ""),
                updated=str(chunk.get("doc_updated_at") or "")[:10],
                effective=str(chunk.get("effective_date") or "")[:10],
                year=year_match.group(1) if year_match else "",
                departments=departments,
                chunk_id=str(chunk.get("chunk_id") or ""),
                index=index,
                text=text,
            )
        )
    return sorted(versions, key=lambda item: (item.effective or "9999", item.version))


def _citation(version: OrgVersion) -> str:
    return f" [{version.index}]"


def _wrong_year(question: str, versions: list[OrgVersion]) -> str | None:
    years = re.findall(r"(20\d{2})", question)
    known = {version.year for version in versions if version.year}
    if not years:
        return None
    asked = [year for year in years if year not in known]
    latest = versions[-1]
    if asked and re.search(r"版|更新", question) and len(set(years)) == 1:
        return (
            f"当前资料中没有{asked[0]}版组织架构；最新为{latest.title}"
            f"（{latest.version}，更新时间为{latest.updated}）。{_citation(latest)}"
        )
    asserted = re.search(
        r"还是\s*(20\d{2})\s*版吗|是\s*(20\d{2})\s*年.{0,6}?(更新|发布|生效|改版)的吗|更新于\s*(20\d{2})\s*年",
        question,
    )
    if asserted:
        asked_year = next(group for group in asserted.groups() if group)
        if latest.year and asked_year != latest.year:
            return (
                f"不是。最新组织架构为{latest.title}（{latest.version}），"
                f"更新时间为{latest.updated}。{_citation(latest)}"
            )
    return None


def answer_org_question(question: str, chunks: list[dict]) -> tuple[str, list[int], float] | None:
    versions = parse_versions(chunks)
    if not versions:
        return None
    latest = versions[-1]
    correction = _wrong_year(question, versions)
    if correction:
        return correction, [latest.index], 1.0

    years = [year for year in dict.fromkeys(re.findall(r"(20\d{2})", question))
             if year in {version.year for version in versions}]
    by_year = {version.year: version for version in versions if version.year}

    if len(versions) >= 2 and re.search(r"旧版|新版", question) and re.search(r"哪一版|什么版本|分别", question):
        older, newer = versions[0], versions[-1]
        return (
            f"旧版为《{older.title}》（{older.version}，更新于{older.updated}）{_citation(older)}；"
            f"新版为《{newer.title}》（{newer.version}，更新于{newer.updated}）{_citation(newer)}。",
            [older.index, newer.index],
            1.0,
        )

    net = re.search(r"(多|少|增加|减少|净增|净减)了?\s*(几|多少)个一级部门", question)
    if net and len(years) >= 2:
        subject, base = by_year[years[0]], by_year[years[1]]
        diff = len(subject.departments) - len(base.departments)
        word = "多" if diff > 0 else "少"
        return (
            f"{subject.title}较{base.title}{word}{abs(diff)}个一级部门"
            f"（{len(base.departments)}→{len(subject.departments)}）。"
            f"{_citation(subject)}{_citation(base)}",
            [subject.index, base.index],
            1.0,
        )

    if re.search(
        r"(?:多少|几)个(?:不同|不重复)?的?(?:一级)?部门|"
        r"(?:一级)?部门(?:的)?(?:总数|数量|共有几个|有几个|共有多少个)",
        question,
    ):
        if len(years) >= 2:
            picks = [by_year[year] for year in years[:2]]
            answer = "；".join(
                f"{version.title}设{len(version.departments)}个一级部门{_citation(version)}"
                for version in picks
            )
            return answer + "。", [version.index for version in picks], 1.0
        target = by_year[years[0]] if years else latest
        return (
            f"{target.title}设{len(target.departments)}个一级部门。{_citation(target)}",
            [target.index],
            1.0,
        )

    if re.search(r"新增|增加", question) and re.search(r"部门|一级部门", question):
        older = versions[0]
        added = [name for name in latest.departments if name not in older.departments]
        if added:
            return (
                f"{latest.title}较{older.title}新增一级部门：{'、'.join(added)}。"
                f"{_citation(latest)}{_citation(older)}",
                [latest.index, older.index],
                1.0,
            )

    if re.search(r"不再|取消|移除|减少|撤销|少了", question) and re.search(r"部门|一级部门", question):
        older = versions[0]
        removed = [name for name in older.departments if name not in latest.departments]
        if removed:
            return (
                f"{latest.title}较{older.title}减少一级部门：{'、'.join(removed)}。"
                f"{_citation(latest)}{_citation(older)}",
                [latest.index, older.index],
                1.0,
            )

    known_names = sorted({name for version in versions for name in version.departments},
                         key=len, reverse=True)
    mentioned = next((name for name in known_names if name in question), "")
    if mentioned and re.search(r"一级部门吗|什么级别|哪一级|存在吗|有没有|是否|属于一级部门", question):
        target = by_year.get(years[0], latest) if years else latest
        if mentioned in target.departments:
            return (
                f"是，{mentioned}在{target.title}中为一级部门。{_citation(target)}",
                [target.index],
                1.0,
            )
        upgrade = _UPGRADE.search(target.text)
        if target is latest and upgrade and upgrade.group(1) == mentioned:
            parent = upgrade.group(2).strip()
            origin = f"由{parent}升级为一级部门" if parent else "升级为一级部门"
            return (
                f"{mentioned}在{latest.title}中为一级部门，{origin}。{_citation(latest)}",
                [latest.index],
                1.0,
            )
        return (
            f"不是，{mentioned}不在{target.title}的一级部门清单中。{_citation(target)}",
            [target.index],
            1.0,
        )

    if re.search(r"哪个部门.{0,6}升级|升级为一级部门", question):
        upgrade = _UPGRADE.search(latest.text)
        if upgrade:
            parent = upgrade.group(2).strip()
            origin = f"由{parent}升级为一级部门" if parent else "升级为一级部门"
            return (
                f"{upgrade.group(1)}{origin}。{_citation(latest)}",
                [latest.index],
                1.0,
            )
    return None
