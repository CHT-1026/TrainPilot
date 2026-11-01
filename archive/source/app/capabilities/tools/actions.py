"""Single-claim write execution. Unknown side effects are never automatically replayed."""
import json
import time

from redis.exceptions import RedisError

from app.capabilities.tools.registry import ToolRegistry
from app.capabilities.tools.local import ToolError
from app.database.connection import get_async_redis_client
from app.domain.common.results import ToolResult


async def execute_confirmed(registry: ToolRegistry, pending: dict, draft_version: int) -> ToolResult:
    if pending["draft_version"] != draft_version or pending["expires_at"] < time.time():
        return ToolResult.failure("STALE_TOOL_DRAFT", "工具草稿已更新或超过确认期限，请重新提出操作")
    try:
        draft = pending["draft"]
        call = registry.prepare(draft["name"], draft["arguments"])
    except ToolError as exc:
        return ToolResult.failure(exc.code, str(exc))
    if pending["binding"] != registry.action_binding(call):
        return ToolResult.failure("STALE_TOOL_DRAFT", "工具目标或定义已变化，请重新核对后发起操作")
    key = f"tool_action:{registry.actor.user_id}:{pending['action_id']}"
    try:
        client = get_async_redis_client()
        acquired = await client.set(key, json.dumps({"status": "running"}), nx=True, ex=86400)
        if not acquired:
            stored = json.loads(await client.get(key))
            if stored["status"] == "finished":
                return ToolResult.model_validate(stored["result"])
            return ToolResult.failure("ACTION_OUTCOME_UNKNOWN", "该操作已开始但没有确定回执，请核对实际结果；未重复执行")
    except RedisError:
        return ToolResult.failure("ACTION_STORE_UNAVAILABLE", "操作记录存储不可用，工具未执行")
    result = await registry.execute(call)
    try:
        await client.set(key, json.dumps({"status": "finished", "result": result.to_dict()}, ensure_ascii=False), ex=86400)
    except RedisError:
        return ToolResult.failure("ACTION_OUTCOME_UNKNOWN", "工具已返回，但回执保存失败；请核对结果，不要重复执行")
    return result
