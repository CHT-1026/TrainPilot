"""企业级 RAG 配置：与 OA 的 default 命名空间隔离，默认使用离线可复现向量。"""
import os
from pathlib import Path

# backend/enterprise_rag
PACKAGE_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = PACKAGE_ROOT.parent

# 语料命名空间：与 OA 制度库（default）彻底隔离
RAG_NAMESPACE = os.getenv("RAG_NAMESPACE", "lantu")
CORPUS_DIR = Path(os.getenv("RAG_CORPUS_DIR", str(PACKAGE_ROOT / "corpus" / "docs")))
MANIFEST_PATH = Path(os.getenv("RAG_MANIFEST_PATH", str(PACKAGE_ROOT / "corpus" / "manifest.jsonl")))
EVAL_DIR = PACKAGE_ROOT / "eval"
REPORT_DIR = Path(os.getenv("RAG_REPORT_DIR", str(EVAL_DIR / "reports")))

# 检索参数
TOP_K = int(os.getenv("RAG_TOP_K", "8"))
CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "60"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
BM25_WEIGHT = float(os.getenv("RAG_BM25_WEIGHT", "1.0"))
VECTOR_WEIGHT = float(os.getenv("RAG_VECTOR_WEIGHT", "1.0"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.18"))
# 版本对比问题需要同时召回旧版与新版，单块分数通常低于事实型问题。
MIN_SCORE_COMPARISON = float(os.getenv("RAG_MIN_SCORE_COMPARISON", "0.12"))
MAX_CHUNKS_PER_DOC = int(os.getenv("RAG_MAX_CHUNKS_PER_DOC", "4"))

# 向量
EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "hash").strip().lower()
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "384"))
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "lantu-hash-384" if EMBEDDING_PROVIDER == "hash" else "text-embedding-v3")
EMBEDDING_VERSION = os.getenv("RAG_EMBEDDING_VERSION", "1")
EMBEDDING_BATCH = int(os.getenv("RAG_EMBEDDING_BATCH", "16"))

# 显式 false 禁止模型生成；向量是否联网由 provider 与请求 offline 控制。
_llm_mode = os.getenv("RAG_USE_LLM", "auto").strip().lower()
USE_LLM = bool(os.getenv("OPENAI_API_KEY")) if _llm_mode == "auto" else _llm_mode in {"1", "true", "yes"}
LLM_MODEL = os.getenv("RAG_LLM_MODEL", os.getenv("OPENAI_MODEL", "deepseek-v4.1-flash"))
LLM_TIMEOUT = float(os.getenv("RAG_LLM_TIMEOUT", "60"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "18000"))
REASONING_TOP_K = int(os.getenv("RAG_REASONING_TOP_K", "32"))

# ---- 请求预算与可观测性 ------------------------------------------------
REQUEST_TIMEOUT_SECONDS = float(os.getenv("RAG_REQUEST_TIMEOUT", "90"))
MAX_QUERY_VARIANTS = int(os.getenv("RAG_MAX_QUERY_VARIANTS", "8"))
MAX_EMBEDDING_CALLS = int(os.getenv("RAG_MAX_EMBEDDING_CALLS", "8"))
MAX_MODEL_CALLS = int(os.getenv("RAG_MAX_MODEL_CALLS", "4"))
# 输出预算按任务收敛：综合题优先保证在客户端窗口内完成，避免长 JSON 被截断。
MAX_TOKENS_MEDIUM = int(os.getenv("RAG_MAX_TOKENS_MEDIUM", "1800"))
MAX_TOKENS_COMPLEX = int(os.getenv("RAG_MAX_TOKENS_COMPLEX", "2600"))
MAX_TOKENS_SYNTHESIS = int(os.getenv("RAG_MAX_TOKENS_SYNTHESIS", "2200"))
QUERY_CACHE_SIZE = int(os.getenv("RAG_QUERY_CACHE_SIZE", "256"))
QUERY_CACHE_TTL_SECONDS = int(os.getenv("RAG_QUERY_CACHE_TTL", "900"))

# ---- 检索增强开关（默认保持既有基线，实验项显式开启） -------------------
MULTI_REPRESENTATION = os.getenv("RAG_MULTI_REPRESENTATION", "false").lower() in {"1", "true", "yes"}
NEIGHBOR_EXPANSION = os.getenv("RAG_NEIGHBOR_EXPANSION", "false").lower() in {"1", "true", "yes"}
NEIGHBOR_WINDOW = int(os.getenv("RAG_NEIGHBOR_WINDOW", "1"))
CONTEXT_COMPRESSION = os.getenv("RAG_CONTEXT_COMPRESSION", "false").lower() in {"1", "true", "yes"}
CROSS_ENCODER_ENABLED = os.getenv("RAG_CROSS_ENCODER", "false").lower() in {"1", "true", "yes"}
HYDE_ENABLED = os.getenv("RAG_HYDE_ENABLED", "false").lower() in {"1", "true", "yes"}
MAX_HOPS = int(os.getenv("RAG_MAX_HOPS", "2"))
ANN_CHUNK_THRESHOLD = int(os.getenv("RAG_ANN_CHUNK_THRESHOLD", "20000"))
RAG_MAX_CONCURRENT = int(os.getenv("RAG_MAX_CONCURRENT", "8"))

# ---- 答案生成与拒绝 ------------------------------------------------------
FACT_CHECK_REFUSE_ON_CONFLICT = os.getenv(
    "RAG_FACT_CHECK_REFUSE_ON_CONFLICT", "false"
).lower() in {"1", "true", "yes"}

# ---- 模型路由：按任务复杂度选择模型，未配置时沿用 LLM_MODEL ---------------
MODEL_ROUTE_SIMPLE = os.getenv("RAG_MODEL_SIMPLE", "")
MODEL_ROUTE_COMPLEX = os.getenv("RAG_MODEL_COMPLEX", "")

# 权限：企业语料默认全员可读，财务/HR/管理层文档按 OA 规则收敛
DEFAULT_VISIBILITY = "all"

