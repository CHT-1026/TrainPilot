"""新候选划分与混合组筛选的回归；不调用模型，不改标签。"""
import hashlib
import json
from pathlib import Path
import unittest

from model.trainer.agent_rewards import intent_reward
from model.trainer.qwen_grpo import select_probe_pool

ROOT = Path(__file__).resolve().parents[3]


class DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ROOT / "model/dataset/datasets/trainpilot_agent/grpo_v2"
        cls.rows = {s: [json.loads(l) for l in (cls.root / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
                    for s in ("train", "val", "test")}

    def test_counts_hashes_and_split_disjointness(self):
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        groups, prompts = set(), set()
        for split, n in (("train", 600), ("val", 90), ("test", 150)):
            rows = self.rows[split]
            self.assertEqual(len(rows), n)
            self.assertEqual(hashlib.sha256((self.root / f"{split}.jsonl").read_bytes()).hexdigest(), manifest["splits"][split]["sha256"])
            g = {r["group_id"] for r in rows}
            p = {json.dumps(r["prompt"], sort_keys=True) for r in rows}
            self.assertEqual(len(p), n)
            self.assertTrue(groups.isdisjoint(g)); self.assertTrue(prompts.isdisjoint(p))
            groups.update(g); prompts.update(p)

    def test_same_words_require_current_pending_state(self):
        for split, rows in self.rows.items():
            for row in rows:
                if row["category"] != "state_confirmation": continue
                expected = "confirm_action" if row["context"]["pending_action"] else "clarify"
                self.assertEqual(row["gold"], {"intent": expected, "extracted_info": {}})

    def test_comment_wrong_object_decision_or_dropped_negation_cannot_pass(self):
        for row in self.rows['train']:
            if row['category'] != 'decision_comment': continue
            gold = row['gold']; fields = gold['extracted_info']
            for key, value in (("app_id", "leave_wrong"), ("comments", fields['comments'][1:]),
                               ("decision", "驳回" if fields['decision'] == '通过' else '通过')):
                payload = {**gold, "extracted_info": {**fields, key: value}}
                self.assertEqual(intent_reward(json.dumps(payload, ensure_ascii=False), gold)['reward'], 0.)


class ProbePoolChecks(unittest.TestCase):
    def test_selection_rechecks_raw_answers_and_ignores_old_probe(self):
        gold = {"intent": "general_chat", "extracted_info": {}}
        rows = [{"id": x, "gold": gold} for x in ("a", "b", "c")]
        good = {"text": json.dumps(gold), "ended": True, "reward": 1.}
        bad = {"text": '{"intent":"clarify","extracted_info":{}}', "ended": True, "reward": 1.}
        records = [{"probe_id": "new", "id": "a", "answers": [good, bad]},
                   {"probe_id": "new", "id": "b", "answers": [good, good]},
                   {"probe_id": "old", "id": "c", "answers": [good, bad]}]
        self.assertEqual([r['id'] for r in select_probe_pool(rows, records, "new")], ['a'])

    def test_unknown_or_holdout_id_is_not_allowed_into_pool(self):
        with self.assertRaises(KeyError):
            select_probe_pool([], [{"probe_id": "new", "id": "test_id", "answers": []}], "new")


if __name__ == "__main__": unittest.main()
