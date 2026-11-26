"""已批准的TrainPilot身份、OA部分回答及逐句引用约束。"""
import asyncio
from types import SimpleNamespace
import unittest

from agent.app.capabilities.retrieval.answer import ANSWER_PROMPT, REFUSAL_TEXT, verify_answer
from agent.app.workflows.chat_nodes.response import create_response_nodes
from agent.enterprise_rag.prompts import answer_system_prompt, PROMPT_VERSIONS


class TrainPilotPromptChecks(unittest.TestCase):
    def test_rag_identity_and_partial_rule(self):
        self.assertTrue(answer_system_prompt().startswith('你是 TrainPilot'))
        self.assertTrue(ANSWER_PROMPT.startswith('你是 TrainPilot'))
        self.assertIn('资料仅覆盖部分问题时',ANSWER_PROMPT)
        self.assertIn('整题没有可支持的答案时',ANSWER_PROMPT)
        self.assertIn('部分覆盖时先回答有证据的部分',answer_system_prompt())
        self.assertEqual(PROMPT_VERSIONS['answer'],'answer-v4.0-trainpilot-tools-20261102')

    def test_partial_answer_keeps_sentence_citations(self):
        chunks=[{'text':'演练记录\n额定负载72 kg。','chunk_id':'preview-only','title':'演练资料'}]
        valid='额定负载72 kg。[1]\n当前资料未提供采购价格。[1]'
        self.assertTrue(verify_answer(valid,chunks)[2])
        self.assertFalse(verify_answer('额定负载72 kg；当前资料未提供采购价格。[1]',chunks)[2])
        self.assertFalse(verify_answer('额定负载72 kg。[2]',chunks)[2])
        self.assertTrue(verify_answer(REFUSAL_TEXT,chunks)[2])

    def test_live_response_node_uses_trainpilot_and_execution_boundary(self):
        class Capture:
            async def ainvoke(self,prompt):
                self.prompt=prompt
                return SimpleNamespace(content='我是TrainPilot。')
        model=Capture()
        node=create_response_nodes(model)['response_generator']
        state={'intent':'general_chat','current_task':'general_chat','user_message':'介绍你的职责',
               'messages':[],'conversation_context':{},'summary':''}
        asyncio.run(node(state))
        self.assertIn('你是 TrainPilot，一名企业业务助手',model.prompt)
        self.assertIn('不要声称已经执行任何写操作',model.prompt)
        self.assertIn('用户消息：介绍你的职责',model.prompt)


if __name__=='__main__':unittest.main()
