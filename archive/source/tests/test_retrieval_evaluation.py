import asyncio
import json
from pathlib import Path

import pytest

from app.evaluation.retrieval_corpus import CorpusTools, build_snapshot, write_json
from app.evaluation.retrieval_grading import equal_value, grade
from app.evaluation.retrieval_reporting import report_runs
from app.evaluation.retrieval_runtime import DEFAULT_BUDGET, MCPTools, execute, run_case


@pytest.mark.parametrize("expected,actual,ok", [
    (True, 1, False), (False, 0, False), (1, True, False), (0, False, False),
    ({"x": None}, {}, False), ({"x": 1}, {"x": 1, "wrong": 2}, False),
    (None, "null", False), (1, "1", False), ([1, 2], [2, 1], False),
    (0.02, 0.09, False), (0.02, 0.020, True), (-5, 5, False),
    ({"unit": "亿元", "v": 52}, {"unit": "万元", "v": 52}, False),
    ({"甲": 30, "乙": 90}, {"甲": 90, "乙": 30}, False),
    ({"allowed": True}, {"allowed": False}, False),
    ({"x": None}, {"x": None}, True), ([{"x": 3}], [{"x": 3}], True),
    ({"x": [None, False, 0.002]}, {"x": [None, False, 0.002]}, True),
])
def test_typed_semantics(expected, actual, ok):
    assert equal_value(expected, actual) is ok


def fixture_trace():
    return {"qid": "x", "status": "answered", "answer": {"status": "answerable", "answer": {"count": 4},
        "citations": [{"doc_id": "a", "locator": "line:1@0", "quote": "甲乙"},
                      {"doc_id": "b", "locator": "page:1@0", "quote": "丙丁"}]},
        "tool_calls_detail": [{"turn": 0, "result": {"hits": [
            {"doc_id": "a", "locator": "line:1@0", "text": "甲乙"},
            {"doc_id": "b", "locator": "page:1@0", "text": "丙丁"}]}}]}


def fixture_gold():
    return {"qid": "x", "retrieval_rubric": {"status": "answerable", "accepted_statuses": ['answerable'], "expected": {"count": 4},
        "evidence_groups": [[{"doc_id": "a", "quote": "甲乙"}], [{"doc_id": "b", "quote": "丙丁"}]]}}


def test_all_required_evidence():
    trace, gold = fixture_trace(), fixture_gold()
    assert grade(trace, gold)["success"]
    trace["answer"]["citations"].pop()
    assert not grade(trace, gold)["success"]


@pytest.mark.parametrize("change", ["unseen", "locator", "wrong_quote", "only_list", "not_delivered", "previous_turn"])
def test_provenance_failures(change):
    trace, gold = fixture_trace(), fixture_gold()
    if change == "unseen":
        trace["tool_calls_detail"] = []
    elif change == "locator":
        trace["answer"]["citations"][0]["locator"] = "line:999"
    elif change == "wrong_quote":
        trace["answer"]["citations"][0]["quote"] = "无关"
    elif change == "only_list":
        trace["tool_calls_detail"][0]["result"] = {"documents": [{"doc_id": "a"}, {"doc_id": "b"}]}
    elif change == "not_delivered":
        trace["tool_calls_detail"][0]["visible"] = False
    else:
        gold["retrieval_rubric"]["evidence_after_turn"] = 1
    assert not grade(trace, gold)["success"]


def test_open_answer_remains_pending():
    assert grade(fixture_trace(), {"answer": "相似文本"})["correct"] is None


@pytest.fixture
def workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "doc.md").write_text("# 公司\n部门：研发部、销售部\n", encoding="utf-8")
    ws = tmp_path / "workspace"
    build_snapshot(source, ws)
    return ws


@pytest.mark.parametrize("args", [{"query": "公司", "top_k": "10"}, {"query": "公司", "top_k": 1.5},
    {"query": "公司", "top_k": True}, {"query": "公司", "top_k": -1},
    {"query": "公司", "top_k": 11}, {"query": ""}, {"query": "公司", "surprise": 1}])
def test_schema_not_silent_coercion(workspace, args):
    assert CorpusTools(workspace).run("search", args)["error"] == "invalid_arguments"


def test_tools_and_snapshot_boundaries(workspace):
    tools = CorpusTools(workspace)
    assert tools.run("search", {"query": "研发部"})["hits"]
    # JSON Schema integer includes integral JSON numbers. Normalize only AFTER validation.
    assert tools.run("search", {"query": "研发部", "top_k": 1.0})['hits']
    assert tools.run("read", {"doc_id": "../../private/gold"})["error"] == "unknown_document"
    assert tools.run("run_command", {"command": "read gold"})["error"] == "unknown_tool"
    page1 = tools.run("read", {"doc_id": "doc", "limit": 1})
    assert page1["next_offset"] == 1
    assert tools.run("read", {"doc_id": "doc", "offset": 1})["segments"][0]["locator"] == "line:2@0"
    with (workspace / "segments.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Snapshot changed"):
        CorpusTools(workspace)


def test_mcp_preserves_local_schema_and_original_argument_types(workspace):
    from app.evaluation.retrieval_corpus import TOOL_SPECS
    cases = [('search', {'query': '研发部', 'top_k': value}) for value in ['3', True, 1.5, 3, 3.0]]
    cases += [('read', {'doc_id': 'doc', 'offset': 1.0, 'limit': 1.0}),
        ('read', {'doc_id': 'doc', 'limit': True}), ('list', {}),
        ('list', {'keyword': '公司', 'extra': 1}), ('run_command', {'command': 'read gold'})]

    async def check():
        local = CorpusTools(workspace)
        async with MCPTools(workspace) as remote:
            discovered = (await remote.session.list_tools()).tools
            assert {t.name:t.input_schema for t in discovered} == {
                spec['function']['name']:spec['function']['parameters'] for spec in TOOL_SPECS}
            for name, arguments in cases:
                assert await remote.run(name, arguments) == local.run(name, arguments)
    asyncio.run(check())


def reply(content="", tool_calls=None, reasoning=None, finish="stop", usage=True):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"response": {"model": "test", "choices": [{"message": message, "finish_reason": finish}],
            "usage": {"completion_tokens": 10, "total_tokens": 50} if usage else None}}


CONF = {"slot": "A", "model": "test", "base_url": "http://test", "api_key": "secret"}
FINAL = json.dumps({"status": "answerable", "answer": {"count": 2}, "citations": []})


@pytest.mark.parametrize("mode,expected", [("reasoning_only", "invalid_final_format"),
    ("length", "output_truncated"), ("bad_json", "invalid_final_format"),
    ("missing_usage", "usage_unavailable"), ("http_error", "api_error")])
def test_terminal_protocol(workspace, mode, expected):
    def invoke(*args):
        if mode == "reasoning_only":
            return reply(reasoning=FINAL)
        if mode == "length":
            return reply(FINAL, finish="length")
        if mode == "bad_json":
            return reply("<tool_call>fake</tool_call>")
        if mode == "missing_usage":
            return reply(FINAL, usage=False)
        return {"error": "http_error", "http_status": 403}
    result = asyncio.run(run_case({"qid": "x", "question": "检索"}, CONF, workspace, DEFAULT_BUDGET, invoke=invoke))
    assert result["status"] == expected
    assert result["answer"] is None


def test_native_continuation_and_multiturn(workspace):
    call = {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"doc_id":"doc"}'}}
    records = iter([reply(tool_calls=[call], reasoning="private protocol content"), reply(FINAL), reply(tool_calls=[call]), reply(FINAL)])
    observed = []
    def invoke(conf, messages, remaining, timeout):
        observed.append(json.loads(json.dumps(messages)))
        return next(records)
    trace = asyncio.run(run_case({"qid": "x", "turns": ["查今年", "改成去年"]}, CONF, workspace, DEFAULT_BUDGET, invoke=invoke))
    assert trace["status"] == "answered"
    assert len(trace["turn_answers"]) == 2
    assert [t["turn"] for t in trace["tool_calls_detail"]] == [0, 1]
    assert observed[1][-1]["role"] == "tool"
    assert observed[1][-2]["reasoning_content"] == "private protocol content"
    assert trace["completion_tokens"] == 40


def test_total_round_budget_across_turns(workspace):
    trace = asyncio.run(run_case({"qid": "x", "turns": ["1", "2"]}, CONF, workspace,
        {**DEFAULT_BUDGET, "rounds": 1}, invoke=lambda *a: reply(FINAL)))
    assert len(trace["rounds"]) == 1
    assert trace["answer"] is None
    assert trace["status"] == "round_budget_exhausted"


def test_resume_contract_rejects_change(workspace, tmp_path):
    run = tmp_path / "run"
    asyncio.run(execute([], CONF, workspace, run, DEFAULT_BUDGET, "in_process"))
    manifest_before = (run / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="Resume contract changed"):
        asyncio.run(execute([], CONF, workspace, run, {**DEFAULT_BUDGET, "rounds": 1}, "in_process", True))
    assert (run / "manifest.json").read_bytes() == manifest_before


def test_report_all_errors_and_missing_retained(tmp_path):
    run = tmp_path / "D"
    write_json(run / "manifest.json", {"cases": ["x", "y"]})
    write_json(run / "traces/x.json", {"qid": "x", "status": "api_error", "answer": None})
    gold = tmp_path / "gold.jsonl"
    gold.write_text('{"qid":"x"}\n{"qid":"y"}\n', encoding="utf-8")
    summary = report_runs([run], gold, tmp_path / "report")["runs"][0]
    assert summary["planned"] == 2 and summary["recorded"] == 1
    assert summary["verified_failures"] == 1 and summary["not_run"] == 1


def test_real_mcp_roundtrip(workspace):
    async def check():
        async with MCPTools(workspace) as tools:
            result = await tools.run("search", {"query": "研发部"})
            assert result["hits"][0]["doc_id"] == "doc"
            assert (await tools.run("read", {"doc_id": "doc"}))["segments"]
    asyncio.run(check())


def test_absolute_deadline_cancels_request(workspace):
    cancelled = []
    async def stalled(*args):
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.append(True)
    result = asyncio.run(run_case({'qid': 'x', 'question': '检索'}, CONF, workspace,
        {**DEFAULT_BUDGET, 'case_seconds': 0.02}, invoke=stalled))
    assert result['status'] == 'timeout'
    assert cancelled == [True]


def test_partial_answer_and_evidence_saved_before_next_request(workspace, tmp_path):
    checkpoint = tmp_path / 'partial.json'
    call = {'id': 'c', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"doc_id":"doc"}'}}
    replies = iter([reply(tool_calls=[call]), reply(FINAL)])
    def invoke(*args):
        next_reply = next(replies)
        if (next_reply['response']['choices'][0]['message'].get('content')):
            saved = json.loads(checkpoint.read_text(encoding='utf-8'))
            assert saved['tool_calls_detail'][0]['result']['segments']
            assert saved['rounds'][0]['response']['choices'][0]['message']['tool_calls']
        return next_reply
    result = asyncio.run(run_case({'qid': 'x', 'question': '检索'}, CONF, workspace, DEFAULT_BUDGET,
        invoke=invoke, checkpoint=checkpoint))
    assert result['status'] == 'answered'


def test_resume_interrupted_is_not_automatically_retried(workspace, tmp_path):
    run = tmp_path / 'run'
    cases = [{'qid': 'x', 'question': '检索'}]
    from app.evaluation.retrieval_runtime import contract
    write_json(run / 'manifest.json', {'contract': contract(cases, CONF, workspace, DEFAULT_BUDGET, 'in_process')})
    write_json(run / 'partial/x.json', {'qid': 'x', 'status': 'unfinished', 'answer': None, 'rounds': []})
    asyncio.run(execute(cases, CONF, workspace, run, DEFAULT_BUDGET, 'in_process', True))
    result = json.loads((run / 'traces/x.json').read_text(encoding='utf-8'))
    assert result['status'] == 'interrupted'


def test_http_absolute_timeout_closes_without_worker_thread():
    from app.evaluation.retrieval_runtime import completion
    async def check():
        async def stall(reader, writer):
            await reader.read(100000)
            await reader.read()
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(stall, '127.0.0.1', 0)
        try:
            port = server.sockets[0].getsockname()[1]
            conf = {**CONF, 'slot': 'D', 'base_url': f'http://127.0.0.1:{port}/v1', 'runtime_meta':{'n_ctx':16384}}
            result = await completion(conf, [{'role': 'user', 'content': 'timeout fixture'}], 10, 0.05)
            assert result['error'] == 'TimeoutError'
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(check())


@pytest.mark.parametrize('input_tokens,should_send', [(2,True),(10,False),(11,False)])
def test_local_context_budget_before_generation(monkeypatch,input_tokens,should_send):
    import httpx
    from app.evaluation.retrieval_runtime import completion
    original = httpx.AsyncClient
    sent = []
    def handle(req):
        if req.url.path == '/apply-template':
            return httpx.Response(200,json={'prompt':'formatted prompt'})
        if req.url.path == '/tokenize':
            return httpx.Response(200,json={'tokens':list(range(input_tokens))})
        sent.append(json.loads(req.content))
        response = reply(FINAL)['response']
        response['usage']={'completion_tokens':1,'total_tokens':input_tokens+1}
        return httpx.Response(200,json=response)
    monkeypatch.setattr('app.evaluation.retrieval_runtime.httpx.AsyncClient',
        lambda **kw:original(transport=httpx.MockTransport(handle),**kw))
    conf={**CONF,'slot':'D','base_url':'http://fixture.invalid/v1','runtime_meta':{'n_ctx':10}}
    result=asyncio.run(completion(conf,[{'role':'user','content':'x'}],8192,1))
    assert result['model_request_sent'] is should_send
    if should_send:
        assert sent[0]['max_tokens'] == 8
    else:
        assert result['error']=='context_budget_exhausted' and sent == []
