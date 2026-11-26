-- Phase 4 企业级 RAG：多租户命名空间、完整文档元数据、向量字段
-- 说明：本文件为迁移参考；实际执行使用 sql/apply_migrations.py（幂等）。
ALTER TABLE kb_documents
    ADD COLUMN namespace VARCHAR(32) NOT NULL DEFAULT 'default' COMMENT '语料命名空间' AFTER id,
    ADD COLUMN author VARCHAR(64) DEFAULT NULL COMMENT '作者' AFTER department,
    ADD COLUMN doc_family VARCHAR(128) DEFAULT NULL COMMENT '文档族（同族多版本）' AFTER author,
    ADD COLUMN category VARCHAR(64) DEFAULT NULL COMMENT '文档类型（company/org/...）' AFTER doc_family,
    ADD COLUMN format VARCHAR(16) DEFAULT NULL COMMENT 'md/docx/pdf/csv/xlsx/txt' AFTER doc_type,
    ADD COLUMN doc_created_at DATETIME DEFAULT NULL COMMENT '文档创建时间' AFTER checksum,
    ADD COLUMN doc_updated_at DATETIME DEFAULT NULL COMMENT '文档更新时间' AFTER doc_created_at,
    ADD INDEX idx_namespace_status (namespace, status),
    ADD INDEX idx_doc_family (namespace, doc_family);

ALTER TABLE kb_chunks
    ADD COLUMN chunk_type VARCHAR(16) NOT NULL DEFAULT 'text' COMMENT 'text/table' AFTER section,
    ADD COLUMN embedding_json MEDIUMTEXT DEFAULT NULL COMMENT '向量（JSON 数组）' AFTER token_count,
    ADD COLUMN embedding_dim INT DEFAULT NULL COMMENT '向量维度' AFTER embedding_json;
