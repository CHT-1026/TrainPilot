"""组织架构确定性事实：数量、增减、版本前提与升级来源。"""
import unittest

from enterprise_rag.org_facts import answer_org_question

DEPTS_2024 = "研发中心、产品部、销售中心、供应链中心、制造中心、质量部、财务部、人力资源部、法务合规部、信息技术部、战略投资部"
DEPTS_2025 = "研发中心、产品部、销售中心、客户成功部、供应链中心、制造中心、质量部、财务部、人力资源部、法务合规部、信息技术部、售后服务部"


def chunk(doc_id: str, year: int, departments: str, index: int) -> dict:
    title = f"澜图智能组织架构（{year}版）"
    return {
        "chunk_id": f"{doc_id}#c000",
        "doc_id": doc_id,
        "title": title,
        "version": "v1.0" if year == 2024 else "v2.0",
        "effective_date": "2024-01-01" if year == 2024 else "2025-04-01",
        "doc_updated_at": "2024-01-15T10:00:00" if year == 2024 else "2025-04-10T10:00:00",
        "text": f"{title} · 一、一级部门 {year} 年公司设 {len(departments.split('、'))} 个一级部门：{departments}。",
        "score": 0.5,
    }


OLD = chunk("lantu_org_chart_2024", 2024, DEPTS_2024, 2)
NEW = chunk("lantu_org_chart_2025", 2025, DEPTS_2025, 1)


def answer(question: str, chunks=(NEW, OLD)):
    result = answer_org_question(question, list(chunks))
    return result[0] if result else ""


class OrgFactsTest(unittest.TestCase):
    def test_counts_by_year(self):
        text = answer("2024版和2025版组织架构各有多少个一级部门")
        self.assertIn("11个", text)
        self.assertIn("12个", text)

    def test_added_department(self):
        text = answer("2024版到2025版新增了哪个一级部门")
        self.assertIn("客户成功部", text)
        self.assertIn("售后服务部", text)

    def test_removed_department(self):
        self.assertIn("战略投资部", answer("2025版比2024版少了哪个部门"))

    def test_net_difference(self):
        self.assertIn("多1个", answer("2025版组织架构比2024版多了几个一级部门"))
        self.assertIn("少1个", answer("2024版组织架构比2025版少几个一级部门"))

    def test_wrong_year_is_corrected(self):
        self.assertIn("没有2026版", answer("公司架构是2026年更新的吗"))
        self.assertIn("2025-04-10", answer("公司架构更新于2023年吗"))

    def test_membership_is_version_scoped(self):
        self.assertIn("不在", answer("客户成功部在2024版组织架构中存在吗"))
        self.assertIn("不是", answer("售后服务部在2024版是一级部门吗"))
        self.assertIn("是，客户成功部", answer("客户成功部在2025版是一级部门吗"))

    def test_single_year_uses_requested_version(self):
        self.assertIn("设11个一级部门", answer("2024版组织架构有多少个一级部门"))
        self.assertIn("设12个一级部门", answer("2025版组织架构有多少个一级部门"))

    def test_unknown_question_returns_none(self):
        self.assertIsNone(answer_org_question("研发中心负责什么", [NEW, OLD]))

    def test_upgrade_parent_comes_from_source(self):
        upgraded = dict(
            NEW,
            text=(
                "澜图智能组织架构（2025版） · 一、一级部门 2025 年 4 月起公司设 12 个一级部门："
                + DEPTS_2025 + "。 售后服务部由技术中心下属部门升级为一级部门。"
            ),
        )
        text = answer("哪个部门升级为一级部门", [upgraded, OLD])
        self.assertIn("技术中心", text)
        self.assertNotIn("销售中心下属部门升级", text)


if __name__ == "__main__":
    unittest.main()
