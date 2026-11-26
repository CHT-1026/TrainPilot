"""采样诊断边界检查，不加载项目模型，不把RAG措辞差异当作RL奖励。"""
import json
from pathlib import Path
import unittest

from model.test.test_code.probe_agent_diagnosis import case_seed, classify_groups, score_answer, ROOT


class DiagnosisChecks(unittest.TestCase):
    def test_wrong_fields_and_truncation_never_get_positive_task_reward(self):
        case = {"mode": "intent_parse", "expected": {"intent": "approve", "extracted_info": {"app_id": "leave_a", "decision": "通过"}}}
        incomplete = '{"intent":"approve","extracted_info":{"decision":"通过"}}'
        self.assertEqual(score_answer(case, incomplete, True)["task_reward"], 0)
        correct = json.dumps(case["expected"], ensure_ascii=False)
        self.assertEqual(score_answer(case, correct, True)["task_reward"], 1)
        self.assertEqual(score_answer(case, correct, False)["task_reward"], 0)

    def test_rag_reference_match_is_not_semantic_reward(self):
        case = {"mode": "enterprise_rag", "expected_claims": [], "evidence": []}
        score = score_answer(case, '{"refused":true,"claims":[]}', True)
        self.assertTrue(score["reference_exact"])
        self.assertIsNone(score["task_reward"])
        self.assertNotIn("passed", score)

    def test_group_routes_distinguish_signal_from_truncation(self):
        def groups(values, stop="eos"):
            return [{"answers": [{"task_reward": x, "stop_reason": stop} for x in values]}]
        self.assertEqual(classify_groups(groups([0, 1, 0, 1]))["route"], "sampled_correct_and_incorrect_rl_candidate")
        self.assertEqual(classify_groups(groups([0]*4))["route"], "no_success_observed_review_prompt_labels_or_sft")
        self.assertEqual(classify_groups(groups([1]*4))["route"], "stable_under_observed_sampling")
        self.assertEqual(classify_groups(groups([0]*4,"max_new_tokens"))["route"], "fix_completion_limit_before_training")
        self.assertEqual(classify_groups(groups([None]*4))["route"], "manual_reward_design")
        self.assertEqual(classify_groups([])["route"], "greedy_only_prompt_ablation")

    def test_seed_is_stable_and_variant_independent(self):
        self.assertEqual(case_seed("same_case",42), case_seed("same_case",42))
        self.assertNotEqual(case_seed("same_case",42), case_seed("same_case",43))
        self.assertNotEqual(case_seed("same_case",42), case_seed("other_case",42))

    def test_frozen_runtime_cases_do_not_send_assistant_targets(self):
        path = ROOT / "runs/agent-diagnosis-20261030/runtime_cases.jsonl"
        records = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records),238)
        self.assertEqual(sum(r["sampling_enabled"] for r in records),214)
        self.assertTrue(all(r["source_split"] in ("train","development") for r in records))
        for row in records:
            self.assertNotEqual(row["inference_messages"][-1]["role"],"assistant")
            if row["case"]["mode"] == "intent_parse" and row["sampling_enabled"]:
                self.assertEqual(row["prompt_variant"],"agent_default_trainpilot")
        self.assertEqual(len({r["id"] for r in records}),238)


if __name__ == "__main__":
    unittest.main()
