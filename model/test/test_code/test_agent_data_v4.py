"""V4数据的业务不变量与来源检查；不加载模型、不调用API。"""
import hashlib
import json
from pathlib import Path
import unittest

from model.dataset.build_agent_data import INTENTS, canonical, read_jsonl, render_prompt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "model/dataset/datasets/trainpilot_agent"


class V4DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {split: read_jsonl(DATA / "v4" / f"{split}.jsonl") for split in ("train", "val", "test")}
        cls.manifest = json.loads((DATA / "v4/manifest.json").read_text(encoding="utf-8"))

    def test_hashes_and_approved_scope(self):
        self.assertEqual(self.manifest["training_start"], "original_full_sft_with_fresh_lora")
        self.assertEqual(json.loads((DATA / "v4_preview/approval.json").read_text(encoding="utf-8"))["status"], "user_approved")
        for name, expected in self.manifest["inputs_sha256"].items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), expected, name)
        for split, info in self.manifest["outputs"].items():
            self.assertEqual(hashlib.sha256((DATA / "v4" / info["file"]).read_bytes()).hexdigest(), info["sha256"])
            self.assertEqual(len(self.rows[split]), info["records"])

    def test_old_business_restored_once_and_no_test_replay(self):
        replay = [r for r in self.rows["train"] if r["origin"].endswith("train_replay")]
        keys = {(r["source_file"], r["source_id"]) for r in replay}
        self.assertEqual(len(keys), len(replay))
        self.assertEqual(sum(r["origin"] == "v2_train_replay" for r in replay), 1133)
        self.assertEqual(sum(r["origin"] == "v3_train_replay" for r in replay), 960)
        for category in ("history_fill", "history_modify", "confirm_action", "cancel_action", "pending_other_task", "pending_cancel_existing"):
            self.assertEqual(sum(r["origin"] == "v2_train_replay" and r["category"] == category for r in replay), 56)
        self.assertFalse(any(r["category"] == "natural" for r in self.rows["train"]))
        protected = {render_prompt("", r["scenario"]) for version in ("v2", "v3") for split in ("val", "test")
                     for r in read_jsonl(DATA / version / f"{split}.jsonl") if "expected" in r}
        for r in self.rows["train"]:
            if "expected" in r:
                self.assertNotIn(render_prompt("", r["scenario"]), protected)

    def test_inputs_families_and_whole_trajectories_disjoint(self):
        inputs, groups, trajectories = set(), {}, {}
        for split, rows in self.rows.items():
            for r in rows:
                key = canonical(r["messages"][:-1])
                self.assertNotIn(key, inputs, r["id"])
                inputs.add(key)
                for collection, value in ((groups, r["group_id"]), (trajectories, r.get("trajectory_id"))):
                    if value:
                        self.assertEqual(collection.get(value, split), split)
                        collection[value] = split
            self.assertEqual({r["expected"]["intent"] for r in rows if "expected" in r}, INTENTS)

    def test_all_leave_and_approval_missing_field_combinations(self):
        for category, fields in (("leave_subset", ("leave_type", "start_date", "end_date", "reason")),
                                 ("approval_subset", ("app_id", "decision", "comments"))):
            for split, rows in self.rows.items():
                subset = [r for r in rows if r["category"] == category]
                masks = {sum(1 << i for i, field in enumerate(fields) if field in r["expected"]["extracted_info"]) for r in subset}
                self.assertEqual(masks, set(range(1 << len(fields))), split)
        for split, rows in self.rows.items():
            subset = [r for r in rows if r["category"] == "leave_subset"]
            self.assertEqual({r["expected"]["extracted_info"]["leave_type"] for r in subset if "leave_type" in r["expected"]["extracted_info"]},
                             {"年假", "病假", "事假", "婚假", "产假", "丧假"})

    def test_same_utterance_changes_meaning_with_pending_state(self):
        groups = {}
        for r in self.rows["train"]:
            if r["category"] not in {"confirm", "cancel"}:
                continue
            s = r["scenario"]
            groups.setdefault((s["user_message"], s["current_date"]), []).append(r)
        mixed = 0
        for records in groups.values():
            absent = [r for r in records if not r["scenario"]["pending_action"]]
            if not absent:
                continue
            mixed += 1
            self.assertTrue(all(r["expected"] == {"intent": "clarify", "extracted_info": {}} for r in absent))
            active = [r for r in records if r["scenario"]["pending_action"]]
            self.assertEqual({r["scenario"]["pending_action"]["action_type"] for r in active}, {"submit_leave", "approve", "cancel_leave"})
            self.assertTrue(all(r["expected"]["intent"] in {"confirm_action", "cancel_action"} and r["expected"]["extracted_info"] == {} for r in active))
        self.assertGreaterEqual(mixed, 8)

    def test_modification_retains_unmodified_visible_fields(self):
        changed = {"modify_start": {"start_date"}, "modify_end": {"end_date"}, "modify_both": {"start_date", "end_date"},
                   "modify_type": {"leave_type"}, "modify_reason": {"reason"}, "negated_modification": {"reason"}}
        checked = 0
        for r in self.rows["train"]:
            if r["category"] not in changed or not r["scenario"]["pending_action"]:
                continue
            original = r["scenario"]["pending_action"]["draft"]
            fields = r["expected"]["extracted_info"]
            self.assertEqual(set(fields), {"leave_type", "start_date", "end_date", "reason"})
            for key in set(fields) - changed[r["category"]]:
                self.assertEqual(fields[key], original[key], r["id"])
            checked += 1
        self.assertGreater(checked, 50)

    def test_cancel_and_new_task_do_not_inherit_old_fields(self):
        trajectories = read_jsonl(DATA / "v4/trajectories.jsonl")
        links = {r["id"] for rows in self.rows.values() for r in rows if r["origin"] == "v4_agent_runtime"}
        self.assertEqual(links, {step["sample_id"] for t in trajectories for step in t["turns"]})
        for t in trajectories:
            self.assertTrue(all(all(step["runtime_checks"].values()) for step in t["turns"]))
            if t["kind"] == "cancel_then_new":
                self.assertIsNone(t["turns"][0]["observed_pending"])
                self.assertEqual(t["turns"][0]["observed_accumulated"], {})
                self.assertEqual(set(t["turns"][1]["observed_accumulated"]), {"leave_type"})

    def test_identity_has_parser_actual_response_and_direct_modes(self):
        for split, rows in self.rows.items():
            rows = [r for r in rows if r["origin"] == "v4_agent_identity"]
            self.assertEqual({r["mode"] for r in rows}, {"intent_parse", "agent_reply", "direct_reply"})
            for r in rows:
                if r["mode"] == "intent_parse":
                    self.assertEqual(r["expected"], {"intent": "general_chat", "extracted_info": {}})
                elif r["category"] in {"identity_name", "identity_role", "identity_correction", "identity_greeting"}:
                    self.assertIn("TrainPilot", r["messages"][-1]["content"])
                if r["mode"] == "agent_reply":
                    self.assertEqual(r["messages"][0]["role"], "user")
                    self.assertIn("系统已知结果：", r["messages"][0]["content"])

    def test_encoding_summary_matches_frozen_data(self):
        result = json.loads((DATA / "v4/encoding_check.json").read_text(encoding="utf-8"))
        self.assertEqual(self.manifest["status"], "encoding_checked_pending_training")
        self.assertLessEqual(result["max_tokens"], result["max_length"])
        self.assertEqual(result["truncated"], 0)
        self.assertEqual(result["zero_targets"], 0)
        for split, info in result["splits"].items():
            self.assertEqual(info["records"], len(self.rows[split]))
            self.assertEqual(result["inputs_sha256"][split], self.manifest["outputs"][split]["sha256"])


if __name__ == "__main__":
    unittest.main()
