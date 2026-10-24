"""Absent provider usage must remain unknown, even when known-token sum is zero."""
import unittest
from types import SimpleNamespace
from enterprise_rag.budget import RequestBudget, model_usage_available


class UsageCompletenessTest(unittest.TestCase):
    def test_empty_or_partial_usage_is_unknown_but_explicit_zero_is_known(self):
        for usage in (None, SimpleNamespace(), SimpleNamespace(total_tokens=10), SimpleNamespace(prompt_tokens=2)):
            self.assertFalse(model_usage_available(usage))
        self.assertTrue(model_usage_available(SimpleNamespace(prompt_tokens=0,completion_tokens=0)))
    def test_failed_and_usage_less_attempts_are_explicitly_unknown(self):
        budget=RequestBudget()
        budget.charge_model(stage='answer');budget.note_call_result(ok=False)
        budget.charge_model(stage='answer');budget.note_model_usage(available=False)
        result=budget.snapshot()
        self.assertEqual(result['model_calls'],2)
        self.assertFalse(result['model_usage_complete'])
        self.assertEqual(result['unknown_model_usage_calls'],2)

    def test_reported_zero_usage_is_known_but_does_not_hide_failed_attempt(self):
        budget=RequestBudget();budget.charge_model();budget.note_call_result(ok=False)
        budget.charge_model();budget.note_model_usage(available=True)
        self.assertEqual(budget.snapshot()['unknown_model_usage_calls'],1)
