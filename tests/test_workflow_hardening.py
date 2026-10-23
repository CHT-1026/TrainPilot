"""对话/工作流硬化：状态版本、来源路由、工具契约、会话状态、记忆与流式阶段。"""
import unittest
from unittest.mock import patch

from app.domain.common.results import ToolResult
from app.domain.common.tool_contract import validate_action, validate_execution
from app.domain.user.models import UserContext
from app.integrations.mcp_adapter import authorize
from app.workflows.chat_nodes.execute import create_execution_nodes
from app.workflows.chat_nodes.routing import create_routing_nodes
from app.workflows.dialog_state import apply_context, update_dialog_state
from app.workflows.fact_memory import (
    MemoryUnavailable,
    correct_fact,
    forget_fact,
    list_facts,
    remember_fact,
)
from app.workflows.retention import retention_plan
from app.workflows.source_policy import (
    SOURCE_BUSINESS,
    SOURCE_DOCUMENT,
    SOURCE_PUBLIC,
    classify_source,
    suggested_task,
)
from enterprise_rag.skills import match_skills, render_skill_block

USER = UserContext(user_id="u_flow", name="流程用户", role="employee", department="技术部")


class StateVersionAndContractTest(unittest.TestCase):
    def test_stale_draft_is_rejected_before_service_call(self):
        nodes = create_execution_nodes()
        state = {
            "pending_action": {
                "action_id": "act_1", "action_type": "submit_leave",
                "draft": {"leave_type": "事假"}, "draft_version": 1,
            },
            "draft_version": 2,
            "user": USER.model_dump(),
        }
        with patch("app.workflows.chat_nodes.execute.leave_service.submit") as submit:
            result = nodes["execute_pending"](state)
        submit.assert_not_called()
        self.assertEqual(result["last_result"]["code"], "STALE_DRAFT")
        self.assertIsNone(result["pending_action"])

    def test_matching_draft_version_executes_once(self):
        nodes = create_execution_nodes()
        state = {
            "pending_action": {
                "action_id": "act_2", "action_type": "submit_leave",
                "draft": {"leave_type": "事假"}, "draft_version": 1,
            },
            "draft_version": 1,
            "user": USER.model_dump(),
        }
        with patch("app.workflows.chat_nodes.execute.leave_service.submit",
                   return_value=ToolResult.success(message="已提交")) as submit:
            result = nodes["execute_pending"](state)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(result["task_result"], "已提交")

    def test_contract_rejects_unknown_action_and_empty_action_id(self):
        self.assertFalse(validate_action("delete_everything", {}).ok)
        self.assertEqual(validate_execution("submit_leave", {}, "").code, "INVALID_ARGS")
        self.assertTrue(validate_execution("submit_leave", {}, "act_1").ok)
        self.assertFalse(validate_action("submit_leave", {"leave_type": "事假"}).ok)


class SourcePolicyAndRoutingTest(unittest.TestCase):
    def test_business_record_question_is_not_document_question(self):
        self.assertEqual(classify_source("我的请假申请进度到哪了？")["kind"], SOURCE_BUSINESS)
        self.assertEqual(suggested_task("我的请假申请进度到哪了？"), "query")
        self.assertEqual(suggested_task("待审批列表有哪些？"), "query_pending")

    def test_public_question_has_no_external_fallback(self):
        source = classify_source("帮我搜索一下今天的行业新闻")
        self.assertEqual(source["kind"], SOURCE_PUBLIC)
        self.assertFalse(source["allowed"])
        self.assertEqual(suggested_task("帮我搜索一下今天的行业新闻"), "clarify")

    def test_policy_question_stays_document(self):
        self.assertEqual(classify_source("年假有几天？")["kind"], SOURCE_DOCUMENT)

    def test_router_overrides_misclassified_business_intent(self):
        nodes = create_routing_nodes()
        state = {"intent": "policy_qa", "user_message": "我的申请进度到哪了？"}
        result = nodes["task_router"](state)
        self.assertEqual(result["current_task"], "query")
        self.assertEqual(result["source_kind"], SOURCE_BUSINESS)


class DialogStateTest(unittest.TestCase):
    def test_follow_up_inherits_explicit_as_of_only(self):
        dialog = update_dialog_state(None, "2024年差旅住宿标准是多少？", intent="policy_qa")
        question, as_of = apply_context("那再说说住宿吧？", [], dialog)
        self.assertEqual(as_of, "2024-12-31")
        self.assertIn("那再说说住宿吧", question)

    def test_follow_up_with_new_time_overrides_previous_as_of(self):
        dialog = update_dialog_state(None, "2024年差旅住宿标准是多少？")
        _, as_of = apply_context("那2025年呢？", [], dialog)
        self.assertIsNone(as_of)

    def test_entities_and_topic_are_tracked(self):
        dialog = update_dialog_state(None, "LT-R20 的负载是多少？")
        self.assertIn("LT-R20", dialog["entities"])
        self.assertIn("LT-R20", dialog["topic"])


class RetentionAndSkillsTest(unittest.TestCase):
    def test_summary_window_and_checkpoint_ttl_are_separate(self):
        messages = [{"role": "user", "content": str(index)} for index in range(14)]
        plan = retention_plan(messages, 0)
        self.assertTrue(plan["should_summarize"])
        self.assertEqual(plan["summarize_range"], [0, 6])
        self.assertEqual(plan["checkpoint_ttl_days"], 30)

    def test_skills_load_only_on_matching_question(self):
        skills = match_skills("请假需要哪些信息？")
        self.assertTrue(skills)
        self.assertIn("请假", render_skill_block(skills))
        self.assertEqual(match_skills("公司总部在哪里？"), [])


class AdapterAndMemoryTest(unittest.TestCase):
    def test_mcp_adapter_is_read_only_and_role_scoped(self):
        manager = UserContext(user_id="m", name="主管", role="manager", department="技术部")
        self.assertTrue(authorize(manager, "hr_directory", "lookup_employee").allowed)
        self.assertFalse(authorize(USER, "hr_directory", "lookup_employee").allowed)
        self.assertFalse(authorize(manager, "hr_directory", "update_employee").allowed)

    def test_fact_memory_is_user_scoped_and_deletable(self):
        from tests.workflow_fakes import TransactionRedis

        fake = TransactionRedis()
        with patch("app.workflows.fact_memory._client", return_value=fake):
            first = remember_fact(USER, "我的常用项目是澜图一号")
            self.assertEqual(len(list_facts(USER)), 1)
            corrected = correct_fact(USER, first["fact_id"], "我的常用项目是澜图二号")
            self.assertIn("二号", corrected["text"])
            self.assertTrue(forget_fact(USER, first["fact_id"]))
            self.assertEqual(list_facts(USER), [])

    def test_fact_memory_failure_is_not_reported_as_success(self):
        class BrokenRedis:
            def hgetall(self, key):
                raise RuntimeError("redis down")

            def hdel(self, key, field):
                raise RuntimeError("redis down")

        with patch("app.workflows.fact_memory._client", return_value=BrokenRedis()):
            with self.assertRaises(MemoryUnavailable):
                list_facts(USER)
            with self.assertRaises(MemoryUnavailable):
                forget_fact(USER, "mem_missing")


class ResponseRoutingTest(unittest.TestCase):
    def test_corrected_policy_task_is_not_overwritten_by_model(self):
        from app.workflows.chat_nodes.response import create_response_nodes

        class FakeLLM:
            async def ainvoke(self, prompt):
                return type("Response", (), {"content": "模型改写"})()
            async def astream(self, prompt):
                yield type("Delta", (), {"content": "模型改写"})()

        import asyncio
        node = create_response_nodes(FakeLLM())["response_generator"]
        output = asyncio.run(node({
            "intent": "general_chat", "current_task": "policy_qa",
            "task_result": "已核验答案 [1]。", "user_message": "年假有几天",
            "messages": [], "conversation_context": {},
        }))
        self.assertEqual(output["response"], "已核验答案 [1]。")


class StreamingPhaseTest(unittest.TestCase):
    def test_retrieve_and_answer_emit_status_phases(self):
        from app.workflows.chat_nodes.policy_qa import create_policy_nodes

        events = []

        class FakeWriter:
            def __call__(self, payload):
                events.append(payload)

        class FakeLLM:
            async def ainvoke(self, prompt):
                return type("Response", (), {"content": "年假为 5 天 [1]。"})()

        state = {
            "user_message": "年假有几天",
            "messages": [],
            "user": USER.model_dump(),
            "retrieved_chunks": [{
                "chunk_id": "c1", "doc_id": "d1", "title": "年假制度", "section": "年假",
                "text": "年假制度 · 年假\n年假为 5 天。", "version": "v1.0", "score": 0.9,
            }],
            "knowledge_namespace": "oa",
            "retrieval_question": "年假有几天",
            "source_kind": "document",
        }
        with patch("app.workflows.chat_nodes.policy_qa.get_stream_writer", return_value=FakeWriter()):
            nodes = create_policy_nodes(FakeLLM())
            import asyncio
            output = asyncio.run(nodes["answer"](state))
        stages = [event.get("stage") for event in events if event.get("type") == "status"]
        self.assertIn("verifying", stages)
        self.assertIn("verified", stages)
        self.assertEqual(events[-1]["mode"], "verified")
        self.assertEqual(output["citations"][0]["doc_id"], "d1")


if __name__ == "__main__":
    unittest.main()
