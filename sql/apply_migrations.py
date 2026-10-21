"""幂等地应用 Phase 0 数据库迁移。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe sql\\apply_migrations.py
优先读取 .env.local，其次 .env。
"""
from pathlib import Path

import pymysql
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def load_values() -> dict:
    for name in (".env.local", ".env"):
        path = ROOT / name
        if path.exists():
            values = dotenv_values(path)
            if values.get("MYSQL_HOST"):
                return values
    raise RuntimeError("未找到包含 MYSQL_HOST 的 .env.local 或 .env")


def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS total FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s AND column_name = %s
        """,
        (table, column),
    )
    return int(cursor.fetchone()["total"]) > 0


def index_exists(cursor, table: str, index: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS total FROM information_schema.statistics
        WHERE table_schema = DATABASE() AND table_name = %s AND index_name = %s
        """,
        (table, index),
    )
    return int(cursor.fetchone()["total"]) > 0


def main() -> None:
    values = load_values()
    conn = pymysql.connect(
        host=values["MYSQL_HOST"],
        port=int(values.get("MYSQL_PORT", 3306)),
        user=values["MYSQL_USER"],
        password=values["MYSQL_PASSWORD"],
        database=values["MYSQL_DATABASE"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
    applied = []
    try:
        with conn.cursor() as cursor:
            additions = [
                ("approval_step", "VARCHAR(16) NOT NULL DEFAULT 'manager' AFTER hr_approval"),
                ("version", "INT NOT NULL DEFAULT 0 AFTER approval_step"),
                ("idempotency_key", "VARCHAR(128) DEFAULT NULL AFTER version"),
                ("submitted_at", "DATETIME DEFAULT NULL AFTER idempotency_key"),
                ("cancelled_at", "DATETIME DEFAULT NULL AFTER submitted_at"),
            ]
            for column, definition in additions:
                if not column_exists(cursor, "leave_applications", column):
                    cursor.execute(
                        f"ALTER TABLE leave_applications ADD COLUMN {column} {definition}"
                    )
                    applied.append(f"add column {column}")

            if not column_exists(cursor, "users", "manager_id"):
                cursor.execute(
                    "ALTER TABLE users ADD COLUMN manager_id VARCHAR(64) DEFAULT NULL "
                    "COMMENT '直属主管 username' AFTER position"
                )
                applied.append("add users.manager_id")

            if not column_exists(cursor, "leave_applications", "manager_id"):
                cursor.execute(
                    "ALTER TABLE leave_applications ADD COLUMN manager_id VARCHAR(64) DEFAULT NULL "
                    "COMMENT '提交时的直属主管快照' AFTER employee_role"
                )
                applied.append("add leave_applications.manager_id")

            if not index_exists(cursor, "leave_applications", "uk_leave_idem"):
                cursor.execute(
                    "ALTER TABLE leave_applications ADD UNIQUE KEY uk_leave_idem (employee_id, idempotency_key)"
                )
                applied.append("add unique key uk_leave_idem")

            cursor.execute(
                "UPDATE leave_applications SET approval_step = 'done' "
                "WHERE status IN ('approved', 'rejected', 'finished') AND approval_step <> 'done'"
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_audit (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    application_id VARCHAR(64) NOT NULL,
                    action VARCHAR(32) NOT NULL,
                    actor_id VARCHAR(64) NOT NULL,
                    actor_role VARCHAR(32) NOT NULL,
                    from_status VARCHAR(32),
                    to_status VARCHAR(32),
                    comment TEXT,
                    trace_id VARCHAR(64),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_app (application_id),
                    INDEX idx_actor (actor_id),
                    INDEX idx_created (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idempotency_key VARCHAR(128) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    endpoint VARCHAR(64) NOT NULL,
                    response_json JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    INDEX idx_expires (expires_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_documents (
                    id VARCHAR(64) PRIMARY KEY,
                    title VARCHAR(200) NOT NULL,
                    source_path VARCHAR(500),
                    doc_type VARCHAR(32),
                    version VARCHAR(32),
                    effective_date DATE,
                    status VARCHAR(16) NOT NULL DEFAULT 'active',
                    visibility VARCHAR(32) NOT NULL DEFAULT 'all',
                    department VARCHAR(64) DEFAULT NULL,
                    checksum CHAR(64),
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_status (status),
                    INDEX idx_visibility (visibility)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id VARCHAR(64) PRIMARY KEY,
                    doc_id VARCHAR(64) NOT NULL,
                    section VARCHAR(200),
                    ord INT NOT NULL,
                    text MEDIUMTEXT NOT NULL,
                    token_count INT,
                    embedding_model VARCHAR(64),
                    embedding_version VARCHAR(32),
                    INDEX idx_doc (doc_id),
                    INDEX idx_doc_ord (doc_id, ord)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            applied.append("ensure audit, idempotency and knowledge-base tables")

            # Phase 4：企业级 RAG 多命名空间与文档元数据
            document_columns = [
                ("namespace", "VARCHAR(32) NOT NULL DEFAULT 'default' COMMENT '语料命名空间' AFTER id"),
                ("author", "VARCHAR(64) DEFAULT NULL COMMENT '作者' AFTER department"),
                ("doc_family", "VARCHAR(128) DEFAULT NULL COMMENT '文档族' AFTER author"),
                ("category", "VARCHAR(64) DEFAULT NULL COMMENT '文档类型' AFTER doc_family"),
                ("format", "VARCHAR(16) DEFAULT NULL COMMENT '文件格式' AFTER doc_type"),
                ("doc_created_at", "DATETIME DEFAULT NULL COMMENT '文档创建时间' AFTER checksum"),
                ("doc_updated_at", "DATETIME DEFAULT NULL COMMENT '文档更新时间' AFTER doc_created_at"),
            ]
            for column, definition in document_columns:
                if not column_exists(cursor, "kb_documents", column):
                    cursor.execute(f"ALTER TABLE kb_documents ADD COLUMN {column} {definition}")
                    applied.append(f"add kb_documents.{column}")

            chunk_columns = [
                ("chunk_type", "VARCHAR(16) NOT NULL DEFAULT 'text' COMMENT 'text/table' AFTER section"),
                ("embedding_json", "MEDIUMTEXT DEFAULT NULL COMMENT '向量JSON' AFTER token_count"),
                ("embedding_dim", "INT DEFAULT NULL COMMENT '向量维度' AFTER embedding_json"),
            ]
            for column, definition in chunk_columns:
                if not column_exists(cursor, "kb_chunks", column):
                    cursor.execute(f"ALTER TABLE kb_chunks ADD COLUMN {column} {definition}")
                    applied.append(f"add kb_chunks.{column}")

            if not index_exists(cursor, "kb_documents", "idx_namespace_status"):
                cursor.execute(
                    "ALTER TABLE kb_documents ADD INDEX idx_namespace_status (namespace, status)"
                )
                applied.append("add index idx_namespace_status")
            if not index_exists(cursor, "kb_documents", "idx_doc_family"):
                cursor.execute(
                    "ALTER TABLE kb_documents ADD INDEX idx_doc_family (namespace, doc_family)"
                )
                applied.append("add index idx_doc_family")

            cursor.execute(
                "UPDATE kb_documents SET namespace = 'default' WHERE namespace IS NULL OR namespace = ''"
            )
            cursor.execute(
                "UPDATE kb_documents SET format = doc_type WHERE format IS NULL OR format = ''"
            )
            cursor.execute(
                "UPDATE kb_documents SET doc_updated_at = updated_at WHERE doc_updated_at IS NULL"
            )
    finally:
        conn.close()

    print({"applied": applied, "database": values["MYSQL_DATABASE"]})


if __name__ == "__main__":
    main()
