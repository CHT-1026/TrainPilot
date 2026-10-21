"""
澜图智能企业智能体平台 — FastAPI 主程序
"""
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.metrics import inc, observe
from app.database.connection import (
    close_all_connections,
    close_async_redis,
    get_redis_client,
    get_sync_db_connection,
)
from app.models.exceptions import BusinessException
from enterprise_rag.embedding import EmbeddingUnavailableError, EmbeddingIntegrityError

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

settings = get_settings()

if settings.tracing_enabled:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    logger.info("LangSmith 追踪已启用")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_sync_db_connection()
    logger.info("MySQL数据库连接池初始化完成")
    await init_graphs()
    logger.info("工作流图初始化完成")
    yield
    await close_async_redis()
    close_all_connections()
    logger.info("数据库连接已关闭")


app = FastAPI(
    title="澜图智能企业智能体平台",
    description="企业知识问答、制度检索、请假与审批一体化的智能体平台",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)


async def init_graphs():
    from app.workflows.chat_agent import get_chat_graph

    await get_chat_graph()


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.trace_id = trace_id
    started = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - started
    response.headers["X-Trace-ID"] = trace_id
    inc(
        "http_requests_total",
        {"method": request.method, "path": request.url.path, "status": response.status_code},
    )
    observe("http_request_seconds", duration, {"path": request.url.path})
    logger.info(
        "request trace_id=%s method=%s path=%s status=%s duration=%.3fs",
        trace_id,
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


from app.routes.approvals import router as approvals_router
from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.routes.kb import router as kb_router
from app.routes.metrics import router as metrics_router
from app.routes.users import router as users_router
from enterprise_rag.api import router as rag_router

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(approvals_router)
app.include_router(kb_router)
app.include_router(metrics_router)
app.include_router(users_router)
app.include_router(rag_router)


@app.exception_handler(BusinessException)
async def business_exception_handler(request: Request, exc: BusinessException):
    logger.warning(
        "业务异常: %s | %s | path=%s | method=%s",
        exc.code,
        exc.message,
        request.url.path,
        request.method,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message, "success": False},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        errors.append(
            {"field": ".".join(str(loc) for loc in error["loc"]), "message": error["msg"]}
        )
    logger.warning("请求验证失败: %s", errors)
    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "请求参数验证失败",
            "errors": errors,
            "success": False,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("未捕获异常: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "服务器内部错误",
            "success": False,
        },
    )


@app.exception_handler(EmbeddingUnavailableError)
@app.exception_handler(EmbeddingIntegrityError)
async def embedding_exception_handler(request: Request, exc: Exception):
    logger.error("知识索引不可用: %s", type(exc).__name__)
    return JSONResponse(status_code=503, content={"code": "RAG_INDEX_UNAVAILABLE",
        "message": "知识索引暂不可用，请检查向量配置并重新入库。", "success": False})


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "message": "澜图智能企业智能体平台运行中",
    }


@app.get("/api/ready")
async def ready():
    checks = {}
    healthy = True

    try:
        conn = get_sync_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            checks["mysql"] = "ok"
        finally:
            conn.close()
    except Exception as exc:
        checks["mysql"] = f"error: {exc}"
        healthy = False

    try:
        get_redis_client().ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"
        healthy = False

    try:
        from app.capabilities.retrieval.index import index_stats

        stats = await asyncio.to_thread(index_stats)
        checks["knowledge_base"] = stats
        if not stats.get("chunks"):
            healthy = False
    except Exception as exc:
        checks["knowledge_base"] = f"error: {exc}"
        healthy = False

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ready" if healthy else "degraded", "checks": checks},
    )


if __name__ == "__main__":
    import uvicorn

    print("[START] 启动澜图智能企业智能体平台 v3.0 ...")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
