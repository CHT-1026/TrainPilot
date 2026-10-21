"""节点工厂：组装所有业务节点。"""
from langchain_openai import ChatOpenAI

from app.workflows.chat_nodes.approval import create_approval_nodes
from app.workflows.chat_nodes.chat import create_chat_nodes
from app.workflows.chat_nodes.delete import create_delete_nodes
from app.workflows.chat_nodes.execute import create_execution_nodes
from app.workflows.chat_nodes.leave import create_leave_nodes
from app.workflows.chat_nodes.policy_qa import create_policy_nodes
from app.workflows.chat_nodes.query import create_query_nodes
from app.workflows.chat_nodes.recognition import create_recognition_nodes
from app.workflows.chat_nodes.response import create_response_nodes
from app.workflows.chat_nodes.routing import create_routing_nodes
from app.workflows.chat_nodes.summarize import create_summarize_node


def create_all_nodes(llm: ChatOpenAI):
    nodes = {}
    nodes.update(create_recognition_nodes(llm))
    nodes.update(create_routing_nodes())
    nodes.update(create_leave_nodes())
    nodes.update(create_approval_nodes())
    nodes.update(create_query_nodes())
    nodes.update(create_policy_nodes(llm))
    nodes.update(create_delete_nodes())
    nodes.update(create_execution_nodes())
    nodes.update(create_chat_nodes())
    nodes.update(create_response_nodes(llm))
    nodes.update(create_summarize_node(llm))
    return nodes
