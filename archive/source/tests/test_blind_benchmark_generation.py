"""Regressions against named facts in the local primary corpus; no model calls."""
from pathlib import Path
import sys
import unittest

WORK = Path(__file__).resolve().parents[1]/'eval_bench/_work'
sys.path.insert(0, str(WORK))
import gen_a_org
import gen_k_cross
import gen_h_support_legal
import gen_g_tech
import gen_i_company_strategy
import gen_j_meeting
sys.path.pop(0)


class BenchmarkGenerationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.org = {row.question: row for row in gen_a_org.build('A')}
        cls.cross = gen_k_cross.build('K')

    def test_roles_belong_to_the_named_person(self):
        answer = self.org['程知远的直接上级是谁？该上级在公司担任什么职务？'].answer
        self.assertEqual(answer, '程知远的直接上级是沈亦舟。沈亦舟的职务是CTO，负责研发体系。')
        chair = self.org['沈亦舟的直接上级是谁？该上级在公司担任什么职务？']
        self.assertIn('顾承澜的职务是董事长兼总经理', chair.answer)
        self.assertIn('lantu_org_executives', chair.expected_doc_ids)
        self.assertIn('（控制器研究所所长）', self.org['林晓在 LT-C900 项目中的角色是什么？她的行政上级是谁？'].answer)

    def test_functional_branches_are_not_flattened_into_a_fake_chain(self):
        answer = self.org['蒋雯的上级的上级是谁？请给出完整汇报路径。'].answer
        self.assertIn('顾承澜 → 陈依然 → 蒋雯', answer)
        self.assertNotIn('陈依然的上级的上级是谁？请给出完整汇报路径。', self.org)
        self.assertNotIn('邵岚的上级的上级是谁？请给出完整汇报路径。', self.org)

    def test_direct_reports_include_all_branches_and_do_not_skip_a_level(self):
        root = self.org['顾承澜的直接下属有哪些？沈亦舟是否在其中？'].answer
        self.assertEqual(root, '顾承澜的直接下属有沈亦舟、罗嘉树、韩立恒、周敏仪、陈依然、邵岚。沈亦舟在其中。')
        self.assertEqual(self.org['沈亦舟的直接下属有哪些？程知远是否在其中？'].answer,
                         '沈亦舟的直接下属有程知远。程知远在其中。')

    def test_version_difference_discloses_the_removed_list_entry(self):
        question = '2024 版与 2025 版的一级部门清单差异具体是什么？请完整列出新增与升级两种变化。'
        answer = self.org[question].answer
        self.assertIn('战略投资部未列入2025版一级部门名单', answer)
        self.assertNotIn('其余一级部门名称在两版中保持一致', answer)

    def test_cross_document_question_preserves_the_direction_of_the_source_relation(self):
        item = next(row for row in self.cross if row.question.startswith('组织架构 2025 版写明销售中心'))
        self.assertIn('罗嘉树是唐婧的上级', item.question)
        self.assertNotIn('唐婧是罗嘉树的上级', item.question)

    def test_direct_administrative_manager_is_distinguished_from_the_supervising_executive(self):
        answer = self.org['质量部负责人是谁？他的直接上级是谁？质量部与研发中心的分管高管是否相同？'].answer
        self.assertIn('行政直接上级是总经理顾承澜', answer)
        self.assertIn('COO 韩立恒分管', answer)

    def test_approval_name_does_not_invent_an_unstated_higher_level(self):
        item = next(row for row in gen_h_support_legal.build('H') if row.question.startswith('出口管制指引的专项审批要求'))
        self.assertIn('不能仅凭名称作此推断', item.answer)
        self.assertNotIn('且审批层级更高', item.answer)

    def test_partial_year_progress_does_not_decide_full_year_failure(self):
        item = next(row for row in gen_i_company_strategy.build('I') if row.question.startswith('2025 年度经营计划的目标'))
        self.assertIn('不能据前三季度数据判定全年未达标', item.answer)

    def test_an_earlier_review_cannot_be_the_postmortem_of_a_later_service_event(self):
        item = next(row for row in gen_j_meeting.build('J') if row.question.startswith('恒芯项目的复盘会纪要与 48 小时'))
        self.assertIn('2025-03-18', item.answer)
        self.assertIn('2025-04-08', item.answer)
        self.assertIn('不能认定', item.answer)
        self.assertNotIn('可串成同一条时间线', item.answer)

    def test_a_proposed_process_mapping_is_not_an_invented_document_rule(self):
        item = next(row for row in gen_g_tech.build('G') if row.question.startswith('技术路线图 2026 年目标中的'))
        self.assertIn('实施建议', item.answer)
        self.assertIn('文档未给出', item.answer)


if __name__ == '__main__':
    unittest.main()
