"""Direct Chat Completions transport. No chat graph, RAG, tools or gold I/O."""
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request

from pydantic import BaseModel, Field

from app.evaluation.contracts import BlindQuestion, Prediction


class CompletionMessage(BaseModel):
    content: str | None
    reasoning_content: str | None = None


class CompletionChoice(BaseModel):
    message: CompletionMessage
    finish_reason: str | None


class CompletionReply(BaseModel):
    choices: list[CompletionChoice] = Field(min_length=1)
    usage: dict | None = None
    model: str | None = None
    id: str | None = None


class ChatCompletionsClient:
    def __init__(self, identity: dict, api_key: str, system_prompt: str):
        self.identity = identity
        self.api_key = api_key
        self.system_prompt = system_prompt

    def answer(self, question: BlindQuestion) -> Prediction:
        payload = {'model': self.identity['model'], 'temperature': self.identity['temperature'],
                   self.identity['token_limit_parameter']: self.identity['max_tokens'], 'stream': False,
                   'messages': [{'role': 'system', 'content': self.system_prompt},
                                {'role': 'user', 'content': question.question}],
                   **self.identity['request_options']}
        encoded = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(self.identity['base_url'] + '/chat/completions', data=encoded,
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key})
        started = time.monotonic()
        text, error, finish_reason, usage = '', None, None, None
        response_model, response_id, error_detail, reasoning_chars = None, None, None, 0
        try:
            with urllib.request.urlopen(request, timeout=self.identity['timeout_seconds']) as response:
                reply = CompletionReply.model_validate(json.load(response))
            choice = reply.choices[0]
            text = choice.message.content or ''
            finish_reason = choice.finish_reason
            usage = reply.usage
            response_model, response_id = reply.model, reply.id
            reasoning_chars = len(choice.message.reasoning_content or '')
            status = 'completed' if finish_reason == 'stop' and text.strip() else 'incomplete'
        except urllib.error.HTTPError as exc:
            status, error = 'api_error', f'HTTP_{exc.code}'
            error_detail = exc.read().decode('utf-8', errors='replace').replace(self.api_key, '[REDACTED]')[:1200]
        except (OSError, http.client.IncompleteRead) as exc:
            status, error = 'api_error', type(exc).__name__
            error_detail = str(exc).replace(self.api_key, '[REDACTED]')[:1200]
        except ValueError as exc:
            status, error, text = 'api_error', 'INVALID_RESPONSE_' + type(exc).__name__, ''
        return Prediction(qid=question.qid, status=status, text=text, error=error,
            finish_reason=finish_reason, request_sha256=hashlib.sha256(encoded).hexdigest(),
            latency_seconds=round(time.monotonic()-started, 6), usage=usage,
            response_model=response_model, response_id=response_id, reasoning_chars=reasoning_chars,
            error_detail=error_detail)
