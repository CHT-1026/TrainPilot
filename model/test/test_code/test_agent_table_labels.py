"""数据标签的可核验性回归；不重建或改写既有 V5 训练/测试数据。"""
import unittest

from model.dataset.build_agent_v5 import corpus_pairs


def table(text):
    return {"chunk_id": "isolated-table", "title": "隔离测试表", "section": "参数",
            "text": "隔离测试表 · 参数\n" + text}


class TableLabelChecks(unittest.TestCase):
    def pairs(self, text):
        return list(corpus_pairs(table(text), "enterprise"))

    def test_first_data_row_without_separator_is_preserved(self):
        rows = self.pairs("| 型号 | 负载 |\n| TP-A | 20kg |\n| TP-B | 30kg |")
        self.assertEqual([r[0]["entity"] for r in rows], ["TP-A", "TP-B"])

    def test_separator_is_skipped_and_single_data_row_supported(self):
        rows = self.pairs("| 型号 | 负载 |\n| :--- | ---: |\n| TP-A | 20kg |")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0]["value"], "20kg")

    def test_repeated_category_requires_specific_row_key(self):
        rows = self.pairs("| 分类 | 项目 | 收入 |\n| 行业 | 汽车 | 10亿 |\n| 行业 | 光伏 | 20亿 |")
        self.assertEqual({r[0]["entity"] for r in rows}, {"分类=行业、项目=汽车", "分类=行业、项目=光伏"})
        self.assertEqual({r[0]["field"] for r in rows}, {"收入"})

    def test_ambiguous_duplicate_key_is_not_assigned_an_arbitrary_value(self):
        rows = self.pairs("| 分类 | 项目 | 收入 |\n| 行业 | 汽车 | 10亿 |\n| 行业 | 汽车 | 20亿 |")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
