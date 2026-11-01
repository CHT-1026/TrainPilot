-- Phase 1 知识库表（幂等）
CREATE TABLE IF NOT EXISTS kb_documents (
    id VARCHAR(64) PRIMARY KEY COMMENT '文档ID',
    title VARCHAR(200) NOT NULL COMMENT '标题',
    source_path VARCHAR(500) COMMENT '源文件路径',
    doc_type VARCHAR(32) COMMENT 'md/txt/docx/pdf',
    version VARCHAR(32) COMMENT '制度版本',
    effective_date DATE COMMENT '生效日期',
    status VARCHAR(16) NOT NULL DEFAULT 'active' COMMENT 'active/superseded',
    visibility VARCHAR(32) NOT NULL DEFAULT 'all' COMMENT 'all/manager/hr/finance/admin',
    department VARCHAR(64) DEFAULT NULL COMMENT '限定部门',
    checksum CHAR(64) COMMENT '内容校验',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_visibility (visibility)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识库文档';

CREATE TABLE IF NOT EXISTS kb_chunks (
    id VARCHAR(64) PRIMARY KEY COMMENT '块ID',
    doc_id VARCHAR(64) NOT NULL COMMENT '文档ID',
    section VARCHAR(200) COMMENT '章节',
    ord INT NOT NULL COMMENT '块序号',
    text MEDIUMTEXT NOT NULL COMMENT '块文本（含标题前缀）',
    token_count INT COMMENT '检索词数',
    embedding_model VARCHAR(64) COMMENT '预留向量模型',
    embedding_version VARCHAR(32) COMMENT '预留向量版本',
    INDEX idx_doc (doc_id),
    INDEX idx_doc_ord (doc_id, ord)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识库文本块';
