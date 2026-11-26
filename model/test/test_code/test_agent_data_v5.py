"""V5 数据的可验证标签、来源隔离和真实训练编码检查；不加载大模型。"""
from copy import deepcopy
import json
from pathlib import Path
import re
import unittest

from model.dataset.build_agent_v5 import (
    DATA, OUTPUT, SPLITS, canonical, context, corpus_pairs, read_rows, sha, structural_check,
)


class V5DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {s: read_rows(OUTPUT / f"{s}.jsonl") for s in SPLITS}
        cls.manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
        cls.sources = json.loads((OUTPUT / "sources.json").read_text(encoding="utf-8"))

    def test_sources_hashes_splits_and_all_v4_replay(self):
        checks = structural_check(OUTPUT)
        self.assertEqual(checks["rows"], sum(map(len, self.rows.values())))
        for split in SPLITS:
            original = read_rows(DATA / "v4" / f"{split}.jsonl")
            self.assertEqual(sha(DATA / "v4" / f"{split}.jsonl"), self.manifest["v4_source_sha256"][split])
            replay = {r["source_id"]: r for r in self.rows[split] if r["origin"] == f"v4_{split}_replay"}
            self.assertEqual(len(replay), len(original))
            for row in original:
                self.assertEqual(replay[row["id"]]["messages"], row["messages"])

    def test_actual_corpus_in_every_split_and_no_historical_eval_sources(self):
        excluded = set(self.sources["excluded_historical_eval_families"])
        for split, rows in self.rows.items():
            for mode in ("enterprise_rag", "oa_rag"):
                real = [r for r in rows if r["mode"] == mode and r["origin"] == "corpus_extractive_qa"]
                self.assertGreater(len(real), 0, (split, mode))
                families = {c.get("doc_family") or c["doc_id"] for r in real for c in r["evidence"]}
                self.assertTrue(families.isdisjoint(excluded))
                if mode == "oa_rag":
                    self.assertGreaterEqual(len(families), 2, split)

    def test_real_reference_answers_are_exact_extracts_or_table_cells(self):
        documented_invalid = []
        for split, rows in self.rows.items():
            for row in rows:
                if row["origin"] != "corpus_extractive_qa":
                    continue
                source = row["evidence"][row["expected_claims"][0]["sources"][0] - 1]
                possible = {tuple(texts) for _, texts, category in corpus_pairs(source, row["mode"].split("_")[0])
                            if category == row["category"]}
                answer = tuple(c["text"] for c in row["expected_claims"])
                if answer not in possible:
                    # V5 原始文件冻结保留；新版生成器不再承认这些非唯一行键标签。
                    self.assertEqual(split, "test")
                    self.assertEqual(source["chunk_id"], "lantu_finance_revenue_breakdown#c000")
                    self.assertEqual(row["category"], "table_lookup")
                    documented_invalid.append(row["id"])
        self.assertEqual(len(documented_invalid), 36, "新增标签错误不能混入已知的36条历史问题")

    def test_numeric_prefix_is_not_mistaken_for_a_list_marker(self):
        chunk = {"chunk_id": "decimal-fixture", "title": "测试", "section": "时效", "text": "测试 · 时效\n1.5 小时内完成设备检查并且记录结果供审核。"}
        pairs = list(corpus_pairs(chunk, "enterprise"))
        self.assertIn("1.5 小时", pairs[0][1][0])

    def test_counterfactual_references_and_calculations(self):
        required = {"calculation", "two_hop", "injection", "conflict", "empty", "irrelevant", "missing_premise", "distractor", "partial"}
        for rows in self.rows.values():
            selected = [r for r in rows if r["origin"] == "controlled_evidence"]
            self.assertEqual({r["category"] for r in selected}, required)
            for row in selected:
                category, claims = row["category"], row["expected_claims"]
                if category in {"conflict", "empty", "irrelevant", "missing_premise"}:
                    self.assertFalse(claims)
                else:
                    self.assertTrue(claims)
                    if category in {"calculation", "two_hop"}:
                        self.assertEqual(claims[0]["sources"], [1, 2])
                    if category == "calculation":
                        a, b, difference = map(int, re.search(r"（(\d+)-(\d+)=(\d+)）", claims[0]["text"]).groups())
                        self.assertEqual(a - b, difference)
                        evidence = " ".join(c["text"] for c in row["evidence"])
                        self.assertIn(f"{a} kg", evidence)
                        self.assertIn(f"{b} kg", evidence)
                    if category in {"injection", "distractor"}:
                        claim = claims[0]
                        self.assertIn(claim["text"], row["evidence"][claim["sources"][0] - 1]["text"])

    def test_structural_checks_reject_out_of_range_reference(self):
        rows = deepcopy(self.rows)
        row = next(r for r in rows["train"] if r.get("expected_claims"))
        row["expected_claims"][0]["sources"] = [999]
        with self.assertRaises(AssertionError):
            structural_check(OUTPUT, rows, self.manifest)

    def test_new_training_code_uses_only_explicit_train_validation(self):
        launcher = (OUTPUT.parents[3] / "trainer/run_trainpilot_lora.sh").read_text(encoding="utf-8")
        self.assertIn('v5) DATA_CHECK_MODULE="model.dataset.build_agent_v5"', launcher)
        self.assertIn("MAX_LENGTH=2048", launcher)
        self.assertNotIn('--data-path "$DATA_DIR/test.jsonl"', launcher)


if __name__ == "__main__":
    unittest.main()
