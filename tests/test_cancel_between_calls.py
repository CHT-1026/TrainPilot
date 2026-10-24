"""Cancellation during a failed provider attempt must prevent paid retries."""
import threading
import unittest
from unittest.mock import patch
from enterprise_rag import answer, pipeline
from enterprise_rag.budget import RequestBudget, RequestCancelled


def retry_probe():
    cancelled=threading.Event(); calls=[]
    chunk={"chunk_id":"product#0","doc_id":"product","title":"工业机器人应用", "text":"工业机器人用于搬运和装配。",
           "score":.9,"visibility":"all","section":"应用","doc_family":"product","version":"v1"}
    retrieved={"chunks":[chunk],"debug":{"top_score":.9},"query":{"original":"产品用途","rewritten":"产品用途","matched_synonyms":[],"expansions":[],"time_label":""},
               "as_of":None,"terms":[],"candidates":1,"scope":{}}
    def failed(*args,**kwargs):
        calls.append("provider");cancelled.set();raise RuntimeError("simulated provider failure")
    error=None
    with patch.object(pipeline,"retrieve",return_value=retrieved),patch.object(answer,"_llm_answer",side_effect=failed):
        try:pipeline.query("产品适合哪些应用？",use_llm=True,cancel_check=cancelled.is_set)
        except Exception as exc:error=type(exc).__name__
    return {"provider_attempts":len(calls),"error":error}


class CallCancellationTest(unittest.TestCase):
    def test_cancellation_after_failed_attempt_blocks_answer_retry(self):
        self.assertEqual(retry_probe(),{"provider_attempts":1,"error":"RequestCancelled"})

    def test_budget_cancellation_blocks_new_model_and_embedding_permissions(self):
        cancelled=threading.Event();budget=RequestBudget();budget.cancel_check=cancelled.is_set
        self.assertTrue(budget.can_call_model());self.assertTrue(budget.can_call_embedding())
        cancelled.set()
        for permit in (budget.can_call_model,budget.can_call_embedding):
            with self.assertRaises(RequestCancelled):permit()
        self.assertEqual(budget.stopped_reason,"cancelled")
