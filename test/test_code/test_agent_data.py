"""专项数据的场景隔离和关键标签检查；不调用模型。"""
import random
import unittest

from dataset.build_agent_data import DATA_ROOT, INTENTS, build_dataset, build_row, canonical, make_context


class AgentDataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, cls.checks = build_dataset()
        cls.prompt = (DATA_ROOT / "intent_prompt_v2.txt").read_text(encoding="utf-8")

    def test_reproducible_counts_intent_coverage_and_disjoint_groups(self):
        second, _ = build_dataset()
        self.assertEqual(canonical(self.splits), canonical(second))
        self.assertEqual({k: len(v) for k, v in self.splits.items()}, {"train": 1200, "val": 150, "test": 150})
        seen_groups, seen_messages = set(), set()
        for rows in self.splits.values():
            groups = {r["group_id"] for r in rows}
            messages = {r["scenario"]["user_message"] for r in rows}
            self.assertFalse(groups & seen_groups)
            self.assertFalse(messages & seen_messages)
            self.assertEqual({r["expected"]["intent"] for r in rows if "expected" in r}, INTENTS)
            seen_groups.update(groups)
            seen_messages.update(messages)

    def row(self, kind, text, **overrides):
        context = make_context(random.Random(3))
        context.update(today="2026-12-31", leave_type="事假", start_date="2027-01-05", end_date="2027-01-06",
                       new_start="2027-01-12", new_end="2027-01-13", reason="办理证件", app_id="leave_Z731",
                       relative_day="明天", relative_date="2027-01-01")
        context.update(overrides)
        return build_row(kind, 0, context, text, self.prompt, "fixture")

    def test_relative_date_crosses_year_and_does_not_copy_current_date(self):
        row = self.row("leave_relative", "明天请一天{leave_type}，因为{reason}。")
        self.assertEqual(row["expected"], {"intent": "submit_leave", "extracted_info": {
            "leave_type": "事假", "start_date": "2027-01-01", "end_date": "2027-01-01", "reason": "办理证件"}})
        self.assertIn('"current_date": "2026-12-31"', row["messages"][0]["content"])

    def test_missing_slots_are_not_filled_from_context_generator(self):
        row = self.row("missing_dates", "请{leave_type}，因为{reason}，时间还没定。")
        self.assertEqual(row["expected"]["extracted_info"], {"leave_type": "事假", "reason": "办理证件"})
        row = self.row("approve_missing_decision", "我想审批{app_id}，还没决定结果。")
        self.assertEqual(row["expected"]["extracted_info"], {"app_id": "leave_Z731"})

    def test_history_update_and_cancel_targets_are_distinct(self):
        row = self.row("history_modify", "时间改成{new_start}到{new_end}，其余不变。")
        self.assertEqual(row["expected"]["extracted_info"], {
            "leave_type": "事假", "start_date": "2027-01-12", "end_date": "2027-01-13", "reason": "办理证件"})
        row = self.row("cancel_action", "放弃当前草稿。")
        self.assertEqual(row["expected"], {"intent": "cancel_action", "extracted_info": {}})
        row = self.row("pending_cancel_existing", "取消已提交申请{app_id}，不是草稿。")
        self.assertEqual(row["expected"], {"intent": "delete_application", "extracted_info": {"app_id": "leave_Z731"}})


if __name__ == "__main__":
    unittest.main()
