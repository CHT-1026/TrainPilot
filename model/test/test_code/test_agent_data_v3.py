"""V3业务数据验收：检查取消后的继承、修改后的字段和操作对象边界。"""
import hashlib
import json
import unittest

from model.dataset.build_agent_v3 import DATA_ROOT, KINDS, audit, make_trajectory, parameters, read_jsonl


class AgentDataV3Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = DATA_ROOT / "v3"
        cls.splits = {split: read_jsonl(cls.root / f"{split}.jsonl") for split in ("train", "val", "test")}
        cls.trajectories = read_jsonl(cls.root / "trajectories.jsonl")
        cls.manifest = json.loads((cls.root / "manifest.json").read_text(encoding="utf-8"))

    def test_file_hashes_and_v2_baseline_remain_unchanged(self):
        for name, info in self.manifest["outputs"].items():
            self.assertEqual(hashlib.sha256((self.root / info["file"]).read_bytes()).hexdigest(), info["sha256"])
            self.assertEqual(len(self.splits[name]), info["records"])
        for name, digest in self.manifest["inputs_sha256"].items():
            self.assertEqual(hashlib.sha256((DATA_ROOT / name).read_bytes()).hexdigest(), digest)
        for info in self.manifest["evidence"].values():
            self.assertEqual(hashlib.sha256((self.root / info["file"]).read_bytes()).hexdigest(), info["sha256"])
        v2 = json.loads((DATA_ROOT / "v2/manifest.json").read_text(encoding="utf-8"))
        for info in v2["outputs"].values():
            self.assertEqual(hashlib.sha256((DATA_ROOT / "v2" / info["file"]).read_bytes()).hexdigest(), info["sha256"])

    def test_replay_is_only_selected_v2_training_records_without_label_rewrites(self):
        original = {row["id"]: row for row in read_jsonl(DATA_ROOT / "v2/train.jsonl")}
        replay = [row for row in self.splits["train"] if row["origin"] == "v2_train_replay"]
        self.assertEqual(len(replay), 797)
        self.assertFalse(any(row["origin"] == "v2_train_replay" for split in ("val", "test") for row in self.splits[split]))
        for row in replay:
            source = original[row["source_id"]]
            self.assertEqual(row["messages"], source["messages"])
            self.assertEqual(row.get("expected"), source.get("expected"))
            self.assertNotIn(row["category"], {"natural", "confirm_action", "cancel_action", "history_modify", "history_fill"})

    def test_whole_trajectories_and_expression_families_do_not_cross_splits(self):
        audit(self.splits, self.trajectories)
        for split in self.splits:
            rows = [row for row in self.splits[split] if row["origin"] == "agent_runtime_v3"]
            self.assertEqual({row["category"] for row in rows}, set(KINDS))
            for trajectory in [item for item in self.trajectories if item["split"] == split]:
                actual = sorted(row["turn"] for row in rows if row["trajectory_id"] == trajectory["id"])
                self.assertEqual(actual, list(range(1, len(trajectory["turns"]) + 1)))

    def test_cancel_clears_old_fields_before_a_new_application(self):
        trajectories = [item for item in self.trajectories if item["kind"] == "cancel_then_new"]
        self.assertEqual(len(trajectories), 104)
        for item in trajectories:
            first, cancel, new = item["turns"]
            self.assertEqual(cancel["expected"], {"intent": "cancel_action", "extracted_info": {}})
            self.assertIsNone(cancel["observed_pending"])
            self.assertEqual(cancel["observed_accumulated"], {})
            self.assertEqual(set(new["expected"]["extracted_info"]), {"leave_type"})
            self.assertNotEqual(first["expected"]["extracted_info"]["leave_type"], new["expected"]["extracted_info"]["leave_type"])
            self.assertEqual(new["observed_accumulated"], new["expected"]["extracted_info"])

    def test_revision_preserves_other_fields_and_confirmation_is_not_resubmission(self):
        for trajectory in self.trajectories:
            if trajectory["kind"] not in {"collect_modify_confirm", "side_query_resume"}:
                continue
            previous, revised, confirm = trajectory["turns"][-3:]
            old_fields = previous["observed_pending"]["draft"]
            fields = revised["expected"]["extracted_info"]
            self.assertEqual(set(fields), {"leave_type", "start_date", "end_date", "reason"})
            for name in ("leave_type", "start_date", "reason"):
                self.assertEqual(fields[name], old_fields[name])
            self.assertNotEqual(fields["end_date"], old_fields["end_date"])
            self.assertEqual(confirm["expected"], {"intent": "confirm_action", "extracted_info": {}})
            self.assertIn("execute_pending", confirm["observed_nodes"])
            self.assertIn("没有执行数据库操作", confirm["observed_response"])
            if trajectory["kind"] == "side_query_resume":
                self.assertEqual(previous["observed_accumulated"], {})

    def test_confirm_cancel_approval_and_withdrawal_are_both_covered(self):
        combinations = set()
        for trajectory in self.trajectories:
            if trajectory["kind"] != "pending_decision":
                continue
            kind = trajectory["initial_state"]["pending_action"]["action_type"]
            turn = trajectory["turns"][0]
            intent = turn["expected"]["intent"]
            combinations.add((kind, intent))
            self.assertEqual(turn["expected"]["extracted_info"], {})
            if intent == "cancel_action":
                self.assertIsNone(turn["observed_pending"])
        self.assertEqual(combinations, {(kind, intent) for kind in ("approve", "cancel_leave")
                                        for intent in ("confirm_action", "cancel_action")})

    def test_explicit_other_object_is_not_copied_from_pending_or_history(self):
        for split, rows in self.splits.items():
            for row in rows:
                if row["category"] != "other_object":
                    continue
                target = row["expected"]["extracted_info"]["app_id"]
                self.assertEqual(row["expected"]["intent"], "delete_application")
                self.assertIn(target, row["scenario"]["user_message"])
                self.assertNotEqual(target, row["scenario"]["pending_action"]["draft"].get("app_id"))

    def test_same_seed_rebuilds_definitions_and_no_initial_turn_collisions(self):
        family = json.loads((DATA_ROOT / "scenarios_v3.json").read_text(encoding="utf-8"))["families"][0]
        dates = set()
        for index in range(20):
            values = parameters(20261022, family["id"], "collect_modify_confirm", index)
            self.assertEqual(values, parameters(20261022, family["id"], "collect_modify_confirm", index))
            self.assertNotIn(values["today"], dates)
            dates.add(values["today"])
            item = make_trajectory(family, "collect_modify_confirm", values, index)
            self.assertEqual(len(item["turns"]), 4)
            self.assertEqual(item["turns"][-1]["expected"]["intent"], "confirm_action")


if __name__ == "__main__":
    unittest.main()
