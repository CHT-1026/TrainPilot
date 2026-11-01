"""Preparation-only HTTP contract; batch execution remains unavailable in this stage."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.domain.user.models import UserContext
from app.evaluation.config import ModelLabel, load_config
from app.evaluation.files import load_jsonl
from app.routes.auth import get_user_context

router = APIRouter(prefix='/api/evaluations', tags=['模型盲测（准备阶段）'])


def evaluation_admin(user: UserContext = Depends(get_user_context)) -> UserContext:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail='仅管理员可查看测评配置')
    return user


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    models: list[ModelLabel] = Field(default_factory=lambda: ['A', 'B'], min_length=1, max_length=4)


@router.get('/plan')
async def evaluation_plan(user: UserContext = Depends(evaluation_admin)):
    config = load_config()
    questions = load_jsonl(config.path(config.question_bank))
    return {'code': 200, 'data': {'stage': 'interface_preparation', 'api_execution_available': False,
        'cli_execution_enabled': config.execution_enabled, 'mode': config.mode,
        'question_count': len(questions), 'models': {label: slot.public_status() for label, slot in config.models.items()},
        'answer_visible_to_models': False, 'final_accuracy': None,
        'grading': 'offline_exact_or_reviewed_alias_then_anonymous_human_review'}}


@router.post('/runs', status_code=501)
async def evaluation_run_placeholder(payload: EvaluationRunRequest, user: UserContext = Depends(evaluation_admin)):
    raise HTTPException(status_code=501, detail='本阶段只预留测评接口，不创建任务、不调用模型；后续通过llm_blind_eval.py显式执行')
