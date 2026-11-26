"""V5 框架回归：别名、追问和路由边界；不调用模型、不连接业务库。"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.app.domain.user.models import UserContext
from agent.app.workflows.chat_nodes.policy_qa import create_policy_nodes
from agent.app.workflows.chat_nodes.routing import create_routing_nodes
from agent.app.workflows.dialog_state import apply_context, extract_entities, update_dialog_state
from agent.enterprise_rag.answer import _focus_supported, generate_answer, ModelAnswer
from agent.enterprise_rag.retrieval import _rerank


def chunk(text, title="澜图智能差旅费用报销标准（2025年7月修订版）"):
    return {"doc_id": "travel", "chunk_id": "travel#1", "title": title,
            "text": text, "score": .85}


class CompanyAliasChecks(unittest.TestCase):
    def test_known_company_aliases_use_cited_title(self):
        for name in ("澜图公司", "澜图智能", "澜图智能装备股份有限公司"):
            with self.subTest(name=name):
                self.assertTrue(_focus_supported(
                    name + "的住宿标准是多少？", [chunk("一线城市住宿标准600元/晚。")], True))

    def test_correct_model_answer_is_no_longer_refused(self):
        with patch("agent.enterprise_rag.answer._llm_answer", return_value=ModelAnswer(
            "一线城市住宿标准600元/晚 [1]。", [1], 1., "frozen-v5", []
        )):
            result = generate_answer("澜图公司住宿标准是多少？",
                                     [chunk("一线城市住宿标准600元/晚。")], use_llm=True)
        self.assertFalse(result["refused"])
        self.assertEqual(result["method"], "llm")
        self.assertIn("600", result["answer"])

    def test_other_company_and_similar_names_do_not_borrow_evidence(self):
        for name in ("北辰公司", "星澜图公司", "澜图智能子公司", "澜图智能控股有限公司"):
            with self.subTest(name=name):
                self.assertFalse(_focus_supported(name + "的住宿标准是多少？",
                                                  [chunk("住宿标准600元/晚。")], True))

    def test_alias_does_not_borrow_another_company_or_model(self):
        self.assertFalse(_focus_supported("澜图公司住宿标准？",
                                          [chunk("北辰公司住宿标准600元/晚。", "北辰公司差旅制度")], True))
        self.assertFalse(_focus_supported("澜图公司LT-R30的负载？",
                                          [chunk("LT-R20负载20kg。", "澜图智能产品参数")], True))

    def test_uncited_company_name_cannot_authorize_a_claim(self):
        # 模型实际只引了北辰的块；上下文里其他块的澜图名称不能替它放行。
        self.assertFalse(_focus_supported("澜图公司住宿标准？",
                                          [chunk("住宿标准600元/晚。", "北辰公司差旅制度")], True))

    def test_product_manual_can_inherit_known_namespace_owner(self):
        source = dict(chunk("LT-R20负载20kg。", "LT-R系列产品手册"), namespace="lantu")
        self.assertTrue(_focus_supported("澜图公司LT-R20负载？", [source], True))
        self.assertFalse(_focus_supported("北辰公司LT-R20负载？", [source], True))
        source["namespace"] = "other"
        self.assertFalse(_focus_supported("澜图公司LT-R20负载？", [source], True))

    def test_namespace_does_not_replace_an_explicit_foreign_subject(self):
        source = dict(chunk("北辰公司住宿标准600元/晚。", "差旅制度"), namespace="lantu")
        self.assertFalse(_focus_supported("澜图公司住宿标准？", [source], True))


class FollowupRoutingChecks(unittest.TestCase):
    def setUp(self):
        self.route = create_routing_nodes()["task_router"]
        self.dialog = update_dialog_state(None, "澜图智能LT-R20的负载是多少？", intent="policy_qa")

    def state(self, text, **kwargs):
        return {"intent": "general_chat", "user_message": text, "dialog_state": self.dialog,
                "messages": [], **kwargs}

    def test_model_number_next_to_chinese_is_tracked(self):
        self.assertEqual(extract_entities("澜图智能LT-R20的负载？"), ["LT-R20"])
        self.assertEqual(extract_entities("LT-R30 Pro的负载？"), ["LT-R30 Pro"])
        self.assertEqual(extract_entities("ALT-R20B的负载？"), [])

    def test_chinese_adjacent_model_keeps_pro_variant_penalty(self):
        chunks = [dict(chunk("LT-R20 Pro臂展1750mm。", "LT-R20 Pro参数"), chunk_id="pro"),
                  dict(chunk("LT-R20臂展1700mm。", "LT-R20参数"), chunk_id="base")]
        ranked = _rerank(chunks, ["LT-R20", "臂展"],
                         SimpleNamespace(term_idf={}, versioned_families=set()), None, False, "LT-R20的臂展呢？")
        scores = {c["chunk_id"]: c["features"]["model_scope_penalty"] for c in ranked}
        self.assertEqual(scores, {"base": 0., "pro": -.30})

    def test_followup_routes_to_fresh_retrieval(self):
        state = self.state("那它的臂展呢？")
        routed = {**state, **self.route(state)}
        self.assertEqual(routed["current_task"], "policy_qa")
        dialog = update_dialog_state(routed["dialog_state"], state["user_message"], intent="policy_qa")
        question, _ = apply_context(state["user_message"], [], dialog)
        self.assertIn("LT-R20", question)
        self.assertIn("臂展", question)
        self.assertNotIn("负载", question)

    def test_no_context_or_topic_change_does_not_force_rag(self):
        for state in (self.state("那它的臂展呢？", dialog_state={}),
                      self.state("你好"), self.state("换个话题，那它是谁呢？"),
                      self.state("那1加1等于多少？"),
                      self.state("翻译这句话：那它的臂展呢？")):
            with self.subTest(text=state["user_message"]):
                self.assertEqual(self.route(state)["current_task"], "general_chat")

    def test_non_knowledge_turn_expires_previous_rag_routing(self):
        chat = self.route(self.state("你好"))
        next_state = self.state("那它的臂展呢？", dialog_state=chat["dialog_state"])
        self.assertEqual(self.route(next_state)["current_task"], "general_chat")

    def test_pending_confirmation_and_injection_clarification_keep_precedence(self):
        for intent in ("confirm_action", "cancel_action", "clarify", "submit_leave", "approve"):
            state = self.state("那它的臂展呢？", intent=intent)
            self.assertEqual(self.route(state)["intent"], intent)
        state = self.state("那它的臂展呢？", pending_action={"action_type": "approve"})
        self.assertEqual(self.route(state)["current_task"], "general_chat")

    def test_current_records_and_public_information_keep_source_routing(self):
        self.assertEqual(self.route(self.state("那我的申请状态呢？"))["current_task"], "query")
        self.assertEqual(self.route(self.state("那它的股价呢？"))["current_task"], "clarify")

    def test_new_entity_does_not_reuse_old_model(self):
        dialog = update_dialog_state(self.dialog, "那LT-R30的臂展呢？", intent="policy_qa")
        question, _ = apply_context("那LT-R30的臂展呢？", [], dialog)
        self.assertNotIn("LT-R20", question)
        self.assertEqual(dialog["entities"], ["LT-R30"])

    def test_oa_followup_preserves_namespace_and_current_actor(self):
        user = UserContext(user_id="isolated", name="隔离用户", role="employee", department="技术部")
        dialog = update_dialog_state(None, "入职满3年的员工年假多少天？", intent="policy_qa")
        state = self.state("那满5年呢？", dialog_state=dialog, intent="policy_qa", user=user.model_dump())
        settings = SimpleNamespace(kb_top_k=6, kb_min_score=.1, rag_tool_calls=False)
        with patch("agent.app.workflows.chat_nodes.policy_qa.get_settings", return_value=settings), \
             patch("agent.enterprise_rag.retrieval.retrieve") as enterprise, \
             patch("agent.app.workflows.chat_nodes.policy_qa.retrieve", return_value=[]) as oa:
            result = create_policy_nodes(None)["retrieve"](state)
        enterprise.assert_not_called()
        self.assertEqual(oa.call_args.args[1].user_id, user.user_id)
        self.assertIn("年假", result["retrieval_question"])
        self.assertEqual(result["knowledge_namespace"], "oa")


if __name__ == "__main__":
    unittest.main()
