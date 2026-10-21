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

# 有模型配置时默认使用生成式；显式 false 保留离线抽取/确定性推理模式。
_llm_mode = os.getenv("RAG_USE_LLM", "auto").strip().lower()
USE_LLM = bool(os.getenv("OPENAI_API_KEY")) if _llm_mode == "auto" else _llm_mode in {"1", "true", "yes"}
LLM_MODEL = os.getenv("RAG_LLM_MODEL", os.getenv("OPENAI_MODEL", "deepseek-v4.1-flash"))
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = float(os.getenv("RAG_LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("RAG_LLM_MAX_RETRIES", "0"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "18000"))
REASONING_TOP_K = int(os.getenv("RAG_REASONING_TOP_K", "32"))

# 权限：企业语料默认全员可读，财务/HR/管理层文档按 OA 规则收敛
DEFAULT_VISIBILITY = "all"
