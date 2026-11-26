"""整理已完成的V5评测与实际工作流；保留原始分数、数据问题和人工复核边界。"""
import json
from collections import defaultdict
from pathlib import Path

from model.test.test_code.evaluate_agent_v5 import ROOT, OUT, read, rows


def ratio(n, total):
    return f"{n}/{total}（{n / total:.1%}）" if total else "不适用"


def link(path, label):
    return f"[{label}](<{Path(path).resolve().as_posix()}>)"


def main():
    output = OUT
    plan, metadata = read(output / "plan.json"), read(output / "model_metadata.json")
    variants = {v: rows(output / f"{v}.jsonl") for v in plan["variants"]}
    final = variants["v5_epoch2"]
    assert len(final) == plan["full_final_cases"]
    assert all(len(variants[v]) == plan["matched_panel_cases"] for v in ("v5_epoch1", "v4_epoch2"))
    audit = read(output / "label_audit.json")
    excluded = {r["id"] for r in audit["issues"] if r["split"] == "test"}
    valid = [r for r in final if r["case"]["id"] not in excluded]
    workflow = read(output / "rag_tools/rag_results.json")
    tools = read(output / "rag_tools/tool_results.json")
    assert len(workflow["oracle"]) == 18 and len(workflow["graph"]) == 6 and len(tools) == 25
    semantic = read(output / "rag_tools/semantic_review_v5.json")
    artifact = read(output / "artifact_check.json")
    training = artifact["epoch_metrics"]
    lines = ["# TrainPilot LoRA V5 测评记录", "", "日期：2026-10-24。评测对象：原 Qwen3-1.7B Full SFT 底座加载 V5 LoRA；本地 BF16 贪心生成，不训练、不修复模型输出。", "",
             "## 主要结论", "", "V5已建立基本业务协议与RAG回答格式；最终第二轮略优于第一轮固定子集，保留第二轮作为当前实验基线。尚不代表可靠完成真实多轮业务。",
             "业务完整匹配341/355；原18题RAG的格式及编号范围18/18合法，但事实与具体引用共同成立14/18；15条业务严格轨迹13/15。真实6轮RAG仅3轮达到本次完整判据，其中一轮是正确拒答。",
             "下一步先修框架实体误拦和跟进路由，再修正问题题及补真实多片段引用、表格定位和原文字段保留，之后做有可验证奖励的RL对照。", "",
             "## 评测范围与可追溯性", "", "最终adapter全量695题；第一轮及V4在推理前固定的165题分层子集对照。另复测18道原始RAG题、6轮实际RAG工作流、15条业务执行轨迹（25轮）。",
             "业务回归及旧RAG题已用于开发诊断；V5留出题主要由规则生成，并非独立真实用户验收或公开基准。", "",
             f"权重核对：最终adapter、第二轮adapter与训练断点的{artifact['tensors']}个参数张量完全一致；completed_epochs=2、global_step=2068；训练/验证文件哈希吻合。",
             f"底座SHA256：`{metadata['checkpoint_sha256']}`。", "",
             "| 训练轮次 | 训练loss | 验证loss |", "|---|---:|---:|"]
    for epoch in ("epoch_01", "epoch_02"):
        r = training[epoch]
        lines.append(f"|{epoch}|{r['train_loss']:.6f}|{r['validation']['loss']:.6f}|")
    lines += ["", "第二轮训练loss继续下降，验证改善较小；训练/验证loss不能代替自由生成和业务验收。", "",
              "## 数据质量审计", "", f"在复核输出时发现{len(excluded)}道测试题的表格行键不唯一：收入构成表中“行业”“产品线”各有多行，题目却只用第一列定位。所有这类题依据证据和标签统一标记，不按模型对错挑选。训练/验证集未发现同类行键问题。",
              "原始695题分数保留；另外报告剔除这36题后的659题。该清理为事后数据审计，不伪称预注册指标。",
              f"另发现{audit['table_coverage_gap']['tables_without_separator']}个资料表没有Markdown分隔行，原构造器跳过首条实际数据行，造成覆盖缺口。未在此次测试中修改训练数据或权重。", "",
              "## 最终模型：业务与RAG分开统计", "",
              "| 项目 | 结果 |", "|---|---:|"]
    business = [r for r in final if r["kind"] == "business"]
    for label, key in (("业务完整意图及字段匹配", "passed"), ("业务JSON可解析", "strict_json"),
                       ("业务意图正确", "intent_ok"), ("业务字段完全一致", "fields_exact")):
        lines.append(f"|{label}|{ratio(sum(bool(r[key]) for r in business), len(business))}|")
    for mode in ("enterprise_rag", "oa_rag"):
        original = [r for r in final if r["case"]["mode"] == mode]
        selected = [r for r in valid if r["case"]["mode"] == mode]
        lines.append(f"|{mode}：原始题格式/引用编号合法|{ratio(sum(r['schema_and_refs'] for r in original),len(original))}|")
        lines.append(f"|{mode}：有效题格式/引用编号合法|{ratio(sum(r['schema_and_refs'] for r in selected),len(selected))}|")
        lines.append(f"|{mode}：有效题参考文本及引用匹配|{ratio(sum(r['reference_exact'] for r in selected),len(selected))}|")
    lines += [f"|生成达到上限|{ratio(sum(r['stop_reason']!='eos' for r in final),len(final))}|", "",
              "reference_exact对措辞、事实和来源一起严格匹配，只做一致性指标；改写可能正确，格式/编号合法也可能事实错误，均不等同语义正确率。OA静态结构检查还会在真实工作流中由原verify_answer再次验证。", "",
              "### 有限RAG题的内容复核", ""]
    static_review = read(output / "static_semantic_review.json")
    lines += [f"排除36道不唯一答案题后，共{static_review['total']}道RAG题，内容及引用复核通过{static_review['reviewed_pass']}道。这是本会话助手的事后有限题复核，非独立人工验收，不能泛化为真实业务总体准确率。",
              "数值正确的计算改写与仅引号变化不判为事实错误；差值缺单位、术语损坏或改变指标含义不计完整通过。12道部分覆盖题全部选择整题拒答，另设明确问法探针诊断。", ""]
    historical = read(output / "business_historical_comparison.json")
    lines += [f"同一批355道业务输入与V4历史记录逐条核对一致：V4为{historical['v4_pass']}/355，V5为{historical['v5_pass']}/355；修复{len(historical['fixed'])}题，新增回归{len(historical['regressed'])}题。历史对照与本次新跑固定子集分别记录。", "",
              "## 同题、同设置的固定子集对照", "", "下表只比较预先固定子集；剔除同一批不唯一表格题。", "",
              "| 版本 | 业务完整匹配 | 企业RAG格式/编号 | 企业RAG参考一致 | OA格式/编号 | OA参考一致 |", "|---|---:|---:|---:|---:|---:|"]
    for variant, all_rows in variants.items():
        selected = [r for r in all_rows if r["case"]["id"] in set(plan["panel_ids"]) and r["case"]["id"] not in excluded]
        b = [r for r in selected if r["kind"] == "business"]
        e = [r for r in selected if r["case"]["mode"] == "enterprise_rag"]
        o = [r for r in selected if r["case"]["mode"] == "oa_rag"]
        values = [ratio(sum(r['passed'] for r in b),len(b)), ratio(sum(r['schema_and_refs'] for r in e),len(e)),
                  ratio(sum(r['reference_exact'] for r in e),len(e)), ratio(sum(r['schema_and_refs'] for r in o),len(o)),
                  ratio(sum(r['reference_exact'] for r in o),len(o))]
        lines.append("|"+variant+"|"+"|".join(values)+"|")
    lines += ["", "## 原始RAG题：内容与接口复核", "", "以下18题与V4基线使用相同输入证据和提示词。关键词自动检查仅作辅助，下表结合引用证据逐条复核。复核者为本会话助手，不是独立人工裁判。", "",
              "| 题号 | 模型格式/编号 | 自动候选通过 | 复核结果 |", "|---|---|---|---|"]
    notes = {r["id"]: r for r in semantic["oracle"]}
    for r in workflow["oracle"]:
        c = r["case"]
        lines.append(f"|{c['id']}|{r['schema_and_refs']}|{r['auto_pass']}|{notes[c['id']]['review']}|")
    lines += ["", f"格式及引用编号范围：18/18；逐条事实与具体来源复核：{semantic['oracle_supported']}/18。编号存在不等于所引片段支持断言。", ""]
    lines += ["", "## 原Agent真实检索及业务执行", "",
              "使用原LangGraph、检索排序、权限/版本规则、业务service与repository。索引由既有语料快照提供；业务持久层用隔离SQLite适配，不代表MySQL锁、并发、HTTP认证或Redis持久化验证。", "",
              "| RAG工作流轮次 | 原始意图 | 最终路由 | 生成方式 | 结果复核 |", "|---|---|---|---|---|"]
    graph_notes = {f"{r['id']}/{r['turn']}": r["review"] for r in semantic["graph"]}
    for r in workflow["graph"]:
        calls = r["model_calls"]
        try:
            intent = json.loads(calls[0]["response"]["choices"][0]["message"]["content"]).get("intent")
        except (ValueError, KeyError, IndexError, AttributeError):
            intent = "解析失败"
        state = r["state"]
        lines.append(f"|{r['id']}/{r['turn']}|{intent}|{state.get('intent')}|{state.get('answer_method')}|{graph_notes[r['id']+'/'+str(r['turn'])]}|")
    tool_groups = defaultdict(list)
    for r in tools:
        tool_groups[r["id"]].append(r)
    lines += ["", f"业务严格轨迹：{ratio(sum(all(r['passed'] for r in group) for group in tool_groups.values()),len(tool_groups))}；轮次：{ratio(sum(r['passed'] for r in tools),len(tools))}。", "",
              "业务回执由程序根据执行结果生成；不算模型原生function calling。确定性推理、抽取式兜底和程序拒答均与模型生成分开记录。", "",
              "## 训练与部署输入的差异", "",
              "V5的3424条RAG训练样本只提供0/1/2段证据，正确引用仅覆盖编号1和2；本次旧RAG题会提供最多10段资料。离线短证据表现不能直接代替多段检索后的表现。该差异是可能原因，不据此单独断言所有错误来源。",
              "身份/职责18道有限题已保留原文：明确称呼、招呼和纠正名称的9题均使用TrainPilot；无system的职责概括一题泛化到日程等项目外能力，其余边界回答基本符合预期。", "",
              "## 失败样例与下一步", ""]
    for r in [r for r in business if not r["passed"]][:8]:
        lines += [f"### {r['case']['category']}", "", r["case"]["scenario"]["user_message"], "", "期望：", "```json",
                  json.dumps(r["case"]["expected"],ensure_ascii=False), "```", "实际：", "```text", r["text"], "```", ""]
    lines += semantic["conclusions"]
    lines += ["", "## 完整记录", "", link(output / "plan.json", "冻结方案与子集清单"), "",
              link(output / "label_audit.json", "数据标签审计"), "", link(output / "v5_epoch2.jsonl", "全量输入与原始生成"), "",
              link(output / "static_semantic_review.json", "有限RAG题内容复核"), "",
              link(output / "probe_results.json", "明确实体及金标准证据诊断输出"), "",
              link(output / "framework_replay.json", "正确回答被实体校验误拦的重放记录"), "",
              link(output / "rag_tools/model_calls.json", "真实Agent每次模型请求和输出"), "",
              link(output / "rag_tools/tool_results.json", "工具执行与数据库状态"), "",
              link(output / "rag_workflow_example.md", "LT-R20实际工作流示例"), ""]
    path = ROOT / "model/test/test_log/lora_sft/test_trainpilot_agent_lora_v5_2026-10-24.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
