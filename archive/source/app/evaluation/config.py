"""One configuration boundary, independent of the platform's active chat model."""
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / 'config/llm_evaluation.json'
ModelLabel = Literal['A', 'B', 'C', 'D']


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ModelSlot(ConfigModel):
    slot: str
    base_url_env: str
    api_key_env: str
    model_env: str
    token_limit_parameter: Literal['max_tokens', 'max_completion_tokens'] = 'max_tokens'
    request_options: dict = Field(default_factory=dict)

    @field_validator('request_options')
    @classmethod
    def provider_options_only(cls, value):
        if set(value) & {'model', 'messages', 'tools', 'stream', 'max_tokens', 'max_completion_tokens', 'temperature'}:
            raise ValueError('提供方扩展参数不能覆盖模型、题目、工具或统一推理条件')
        return value

    def public_status(self) -> dict:
        return {'slot': self.slot, 'model': os.getenv(self.model_env, ''),
                'base_url_configured': bool(os.getenv(self.base_url_env)),
                'api_key_configured': bool(os.getenv(self.api_key_env)),
                'required_env': [self.base_url_env, self.api_key_env, self.model_env]}


class EvaluationConfig(ConfigModel):
    execution_enabled: bool = False
    mode: Literal['closed_book'] = 'closed_book'
    question_bank: str
    answer_bank: str
    grading_rules: str
    runs_dir: str
    seed: int
    question_order: Literal['seeded_shuffle', 'type_then_seeded_shuffle'] = 'seeded_shuffle'
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    models: dict[ModelLabel, ModelSlot]

    def path(self, value: str) -> Path:
        return (ROOT / value).resolve()

    def inference_parameters(self) -> dict:
        return self.model_dump(include={'temperature', 'max_tokens', 'timeout_seconds'})


def load_config() -> EvaluationConfig:
    return EvaluationConfig.model_validate_json(CONFIG_PATH.read_text(encoding='utf-8'))


def resolve_model(config: EvaluationConfig, label: ModelLabel) -> tuple[dict, str]:
    slot = config.models[label]
    values = {name: os.getenv(name, '').strip() for name in
              (slot.base_url_env, slot.api_key_env, slot.model_env)}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError('请填写配置项：' + ', '.join(missing))
    identity = {'slot': slot.slot, 'base_url': values[slot.base_url_env].rstrip('/'),
                'model': values[slot.model_env], 'token_limit_parameter': slot.token_limit_parameter,
                'request_options': slot.request_options, **config.inference_parameters()}
    return identity, values[slot.api_key_env]
