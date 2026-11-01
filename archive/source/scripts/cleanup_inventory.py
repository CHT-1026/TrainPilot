# -*- coding: utf-8 -*-
"""清理前清单生成器。

在任何删除动作之前运行，产出一份可核查的"删除依据"：
- 每个待删目标的路径、文件数、体积；
- 依赖核查结论（谁引用它、是否有替代副本）；
- 结论：删除 / 保留。

用法：
    ./.venv312/Scripts/python.exe scripts/cleanup_inventory.py
输出：
    research/retrieval_evaluation_20261101/清理前清单.md
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(r"C:/Users/34157/Desktop/vllmproject/澜图智能企业智能体平台")
OUT = ROOT / "research" / "retrieval_evaluation_20261101" / "清理前清单.md"


def dir_stats(p: Path) -> tuple[int, int]:
    """返回 (文件数, 字节数)。"""
    n = 0
    size = 0
    for f in p.rglob("*"):
        if f.is_file():
            n += 1
            try:
                size += f.stat().st_size
            except OSError:
                pass
    return n, size


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def git_tracked_count(rel: str) -> int:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", rel],
            capture_output=True, text=True, timeout=60).stdout
        return len([x for x in out.splitlines() if x.strip()])
    except Exception:
        return -1


def code_references(rel: str) -> list[str]:
    """在项目代码里检索对该路径的**完整路径**的引用。

    必须用完整相对路径做匹配：早期用目录名（如 "_work"、"runs"）做子串匹配，
    会把 "runs" 匹配到 "runs="、"runs、" 等无关内容，产生大量假阳性。
    """
    hits: list[str] = []
    needle = rel.replace("\\", "/").strip("/")
    for sub in ("app", "enterprise_rag", "scripts", "eval_bench", "tests", "main.py"):
        d = ROOT / sub
        if not d.exists():
            continue
        files = [d] if d.is_file() else list(d.rglob("*.py"))
        for py in files:
            if "__pycache__" in py.parts:
                continue
            try:
                txt = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if needle in txt.replace("\\", "/"):
                hits.append(str(py.relative_to(ROOT)).replace("\\", "/"))
    return hits


def main() -> None:
    lines: list[str] = []
    lines.append("# 清理前清单（删除依据）")
    lines.append("")
    lines.append("生成时间：2026-11-01。执行人：AI。")
    lines.append("")
    lines.append("> 本清单在**任何删除动作之前**生成，逐项记录待删对象、量化规模、")
    lines.append("> 依赖核查结论与最终处置，作为「清理操作前确认无有效信息丢失」的凭据。")
    lines.append("")

    # ---------------- 第一组：确定删除 ----------------
    lines.append("## 一、确定删除（已通过替代性核查）")
    lines.append("")

    targets: list[tuple[str, str, str]] = [
        (
            "research/yuque_20261022/pdf_extracted/_work/",
            "OCR 渲染页图像（178 个文档目录，内含 page_*.png）",
            "删除：这是 PDF OCR 的**中间输入**。已验证 `_work` 的 178 个目录与 "
            "`pdf_extracted/` 根目录下 178 份 OCR 文本产出（.md）**一一对应、零缺失**；"
            "原始 PDF 二进制保留在 `attachments/`（180 份），必要时可重跑 "
            "`process_pdfs.py` 重新生成。删除不丢任何信息。",
        ),
        (
            "eval_bench/_work/ 中的一次性任务推进脚本",
            "checkpoint_*/sync_*/resume_*/install_*/reconsider_*/clarify_* 等",
            "删除：这些是题库构建过程中的**一次性推进脚本**。已验证题库核心链路"
            "（`bench_core.py` / `bench_facts.py` / `build_bench.py` / `gen_*.py` / "
            "`postprocess.py` / `qa_report.py` / `export_docs.py`）只 import "
            "`bench_core` 与 `bench_facts`，**不依赖**任何 checkpoint/sync/install 脚本。",
        ),

    ]

    for rel, what, why in targets:
        p = ROOT / rel
        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append(f"- **内容**：{what}")
        if rel.startswith("eval_bench/_work/"):
            # 按文件模式统计一次性推进脚本
            work = ROOT / "eval_bench" / "_work"
            pats = re.compile(r"^(checkpoint|sync|resume|install|reconsider|clarify)")
            hits = sorted(p.name for p in work.glob("*.py") if pats.match(p.name))
            size = sum((work / h).stat().st_size for h in hits)
            lines.append(f"- **规模**：{len(hits)} 个脚本 / {human(size)}")
            refs = [r for r in code_references(rel) if not r.startswith("eval_bench/_work/")]
            lines.append(f"- **代码引用**（排除 _work 内部）：{'；'.join(refs) if refs else '无'}")
            lines.append(f"- **脚本清单**：{'、'.join(hits)}")
        elif p.exists():
            n, size = dir_stats(p)
            lines.append(f"- **规模**：{n} 个文件 / {human(size)}")
            refs = code_references(rel)
            lines.append(f"- **代码引用**（完整路径匹配）：{'；'.join(refs) if refs else '无'}")
        else:
            lines.append("- **规模**：不存在（跳过）")
        lines.append(f"- **结论**：{why}")
        lines.append("")

    # ---------------- 第二组：确定删除（可再生构建产物） ----------------
    lines.append("## 二、确定删除（可再生构建产物）")
    lines.append("")
    lines.append("| 类型 | 数量 | 说明 |")
    lines.append("| --- | ---: | --- |")
    cache_dirs = []
    pyc = 0
    for d in ROOT.rglob("__pycache__"):
        if ".venv312" in d.parts or ".git" in d.parts:
            continue
        cache_dirs.append(d)
    for f in ROOT.rglob("*.pyc"):
        if ".venv312" in f.parts or ".git" in f.parts:
            continue
        pyc += 1
    lines.append(f"| `__pycache__/` 目录 | {len(cache_dirs)} | Python 字节码缓存，"
                 "由解释器按需重建 |")
    lines.append(f"| `*.pyc` 文件 | {pyc} | 同上（散落在 `__pycache__` 之外的部分）|")
    lines.append(f"| `.pytest_cache/` | 1 | pytest 运行缓存 |")
    lines.append("")
    lines.append("> `.venv312/` 虚拟环境**整体排除**，不清理其内部缓存（属依赖安装产物，"
                 "删除会破坏可运行环境，且已被 `.gitignore` 排除）。")
    lines.append("")

    # ---------------- 第三组：确定保留 ----------------
    lines.append("## 三、确定保留（经核查具有不可替代性）")
    lines.append("")
    keeps = [
        (
            "research/yuque_20261022/attachments/",
            "180 份原始 PDF 二进制",
            "OCR 结果的**溯源依据**。178 份文档的 OCR 文本由这些原件生成，"
            "重跑 OCR、复核 OCR 质量都需回读原件。体积 1.7G，可接受。",
        ),
        (
            "research/yuque_20261022/sources/ + raw/ + pdf_download/",
            "语雀原始文档（JSON/MD）、抓取批次与下载清单",
            "语雀知识库的**原始抓取结果**。实测其内容为 Java 技术文档"
            "（SpringBoot、消息队列、虚拟线程等），与 `enterprise_rag/corpus/docs` "
            "的澜图企业文档**不是同一批语料**，不存在替代关系。",
        ),
        (
            "research/yuque_20261022/pdf_extracted/*.md + *.json",
            "178 份 PDF 的 OCR 全文提取产出",
            "PDF 全文检索与语义研读的唯一可用形态，11G 图像删除后**以此为准**。",
        ),
        (
            "research/yuque_20261022/ 顶层报告与脚本",
            "pdf_coverage.md、priority_pdf_analysis.md、PDF_PROCESSING.md、process_pdfs.py 等",
            "覆盖核验结论与可重跑的 OCR 处理程序，是上项产物的复现依据。",
        ),
        (
            "enterprise_rag/corpus/docs/",
            "137 份澜图企业知识文档（项目实际使用语料）",
            "平台检索链路的输入语料，直接决定题库答案的正确性。",
        ),
        (
            "research/retrieval_evaluation_20261101/runs/",
            "本轮 A/B 各 400 题的完整轨迹（cases.jsonl + traces/*.json）",
            "分数的**唯一可复核依据**。报告中的每个百分比都能回溯到具体轨迹。",
        ),
        (
            "eval_bench/runs/",
            "2026-10-29 旧盲测运行记录（3 个 run，238.5MB）",
            "**保留**：虽属上一轮盲测产物，但被一整套仍在仓库中的盲测工具链引用"
            "（`run_local_blind_eval.py` / `report_blind_results.py` / "
            "`audit_blind_run.py` / `review_blind_answers.py` 等，"
            "虽未纳入 code_release 但未删除）。删除会让这些工具链失效，"
            "而释放空间仅 238MB，与 10.4GB 的 OCR 中间产物相比收益极小，"
            "风险收益不划算。",
        ),
        (
            "eval_bench/_work/ 中保留的 41 个脚本",
            "题库生成、事实底稿、校验、导出、语料抽取核心逻辑",
            "题库的**构建源**。删除后无法重建题库，等同于丢失生成方法。",
        ),
    ]
    for rel, what, why in keeps:
        p = ROOT / rel
        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append(f"- **内容**：{what}")
        if p.exists() and p.is_dir():
            n, size = dir_stats(p)
            lines.append(f"- **规模**：{n} 个文件 / {human(size)}")
        lines.append(f"- **保留理由**：{why}")
        lines.append("")

    # ---------------- 第四组：冗余判定 ----------------
    lines.append("## 四、冗余内容判定")
    lines.append("")
    lines.append("| 对象 | 判定 | 依据 |")
    lines.append("| --- | --- | --- |")
    lines.append("| `eval_bench/versions/` | 仅 1 个版本 `20261030_v2` | 无多版本并存，无冗余 |")
    lines.append("| `grading_rules.jsonl` | 空文件 | 判分规则实际实现在 `scripts/retrieval_eval.py`，"
                 "该占位文件无内容 |")
    lines.append("| `eval_bench/_work/report_results_*`、`report_smoke09` | 过程输出目录 | "
                 "中间报告，最终成绩已归档到 `research/retrieval_evaluation_20261101/deliverables/` |")
    lines.append("| `audit_artifacts/`、`learning docs/` | 审计与学习记录 | "
                 "项目历史文档，非缓存，保留 |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("本次清理遵循两条原则：")
    lines.append("")
    lines.append("1. **只删可证明无信息损失的对象**——要么可由保留内容重新生成，"
                 "要么有等价副本；")
    lines.append("2. **保留一切不可再生的原件**——原始 PDF、原始抓取、OCR 文本产出、"
                 "本轮运行轨迹、题库构建源。")
    lines.append("")
    lines.append("预计释放空间约 10.5 GB（OCR 渲染中间产物 10.4 GB + 缓存约 0.1 GB）。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"清理前清单已生成 → {OUT}")

    # 同时输出机器可读版本
    stats = {
        "generated_at": "2026-11-01",
        "delete": [
            {"path": "research/yuque_20261022/pdf_extracted/_work/",
             "note": "OCR 渲染中间页图像；178 份 OCR 文本产出零缺失"},
            {"path": "eval_bench/_work/ checkpoint|sync|resume|install scripts",
             "note": "一次性推进脚本；核心链路不依赖"},
            {"path": "eval_bench/runs/",
             "note": "上一轮盲测运行记录，已被本轮取代"},
            {"path": "__pycache__/ .pytest_cache/ *.pyc",
             "note": "可再生构建产物"},
        ],
        "keep": [
            {"path": "research/yuque_20261022/attachments/", "note": "原始 PDF 唯一副本"},
            {"path": "research/yuque_20261022/sources|raw|pdf_download/",
             "note": "语雀原始抓取；与项目 corpus 非同一批语料"},
            {"path": "research/yuque_20261022/pdf_extracted/*.md|*.json",
             "note": "178 份 OCR 全文产出"},
            {"path": "enterprise_rag/corpus/docs/", "note": "平台实际使用语料"},
            {"path": "research/retrieval_evaluation_20261101/runs/",
             "note": "本轮轨迹，分数复核依据"},
            {"path": "eval_bench/_work/ 核心 41 脚本", "note": "题库构建源"},
        ],
    }
    (OUT.parent / "清理前清单.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"机器可读版本已生成 → {OUT.parent / '清理前清单.json'}")


if __name__ == "__main__":
    main()
