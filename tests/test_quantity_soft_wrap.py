"""PDF layout repair cannot create a quantity, subject, or statistical year."""
import unittest

from enterprise_rag.answer import _extractive_answer
from enterprise_rag.facts import factual_clauses
from enterprise_rag.verification import verify_answer_facts


def row(text, title="", doc="evidence", score=.8):
    return {"text":text,"title":title,"doc_id":doc,"chunk_id":doc+"#0","score":score}


class QuantitySoftWrapTest(unittest.TestCase):
    def test_wrapped_number_unit_and_percent_preserve_explicit_year(self):
        source=row("2023年研发人员1,540\n人，占员工总数33.7\n%。")
        result=verify_answer_facts("研发情况？","2023年研发人员1,540人，占员工总数33.7% [1]。",[source])
        self.assertTrue(result["verified"])

    def test_wrapping_does_not_cross_new_year(self):
        source=row("2023年研发人员1,540\n人。\n2024年研发人员1,680人。")
        result=verify_answer_facts("2024年人数？","2024年研发人员1,540人 [1]。",[source])
        self.assertFalse(result["verified"])
        self.assertEqual(len(factual_clauses(source["text"])),2)

    def test_people_and_device_units_do_not_join(self):
        source=row("公司有3人\n负责13台机器人。")
        result=verify_answer_facts("机器人多少台？","公司有3台机器人 [1]。",[source])
        self.assertFalse(result["verified"])

    def test_a_new_named_subject_is_not_a_unit_continuation(self):
        source=row("北辰公司有3人。\n南星公司有13台机器人。")
        result=verify_answer_facts("北辰公司设备？","北辰公司有13台机器人 [1]。",[source])
        self.assertFalse(result["verified"])
        self.assertEqual(factual_clauses("编号3\n人事部有13人。"),["编号3","人事部有13人。"])

    def test_undated_percentage_does_not_inherit_later_metric_year(self):
        source=row("研发人员1,540\n人，占员工总数33.7%，2023年研发投入2.9亿元。")
        result=verify_answer_facts("2023研发？","2023年研发人员1,540人，占员工总数33.7% [1]。",[source])
        self.assertFalse(result["verified"])
        self.assertIn("33.7%",{item["token"] for item in result["numeric_missing"]})

    def test_extractor_keeps_independent_year_scope_and_deduplicates_headcount(self):
        sources=[row("研发人员1,540\n人，占员工总数33.7%，2023年研发投入2.9亿元。",score=.95),
                 row("研发人员1,540人。","2023年研发组织",doc="dated",score=.85),
                 row("研发人员 1,540 人。",doc="summary",score=.9)]
        answer, indices, _=_extractive_answer("2023年研发投入和研发人员分别是多少？",sources)
        self.assertIn("2.9亿元",answer)
        self.assertEqual(answer.replace(" ","").count("研发人员1,540人"),1)
        self.assertNotIn("33.7%",answer)
        # Rebind source indices to the returned citation order just as generate_answer does.
        self.assertTrue(verify_answer_facts("2023研发？",answer,sources)["verified"])


if __name__=="__main__":
    unittest.main()
