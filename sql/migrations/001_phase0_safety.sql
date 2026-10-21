-- Phase 0 安全地基迁移（仅用于已存在的数据库）
-- 新建数据库请直接使用 sql/init_mysql.sql。
-- 本文件按顺序执行；apply_migration.py 会跳过已存在的列/表。

ALTER TABLE leave_applications
    ADD COLUMN approval_step VARCHAR(16) NOT NULL DEFAULT 'manager' COMMENT '当前审批环节（manager/hr/done）' AFTER hr_approval,
    ADD COLUMN version INT NOT NULL DEFAULT 0 COMMENT '乐观锁版本号' AFTER approval_step,
    ADD COLUMN idempotency_key VARCHAR(128) DEFAULT NULL COMMENT '幂等键' AFTER version,
    ADD COLUMN submitted_at DATETIME DEFAULT NULL COMMENT '提交时间' AFTER idempotency_key,
    ADD COLUMN cancelled_at DATETIME DEFAULT NULL COMMENT '取消时间' AFTER submitted_at,
    ADD UNIQUE KEY uk_leave_idem (employee_id, idempotency_key);

UPDATE leave_applications
SET approval_step = 'done'
WHERE status IN ('approved', 'rejected', 'finished');

CREATE TABLE IF NOT EXISTS approval_audit (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键',
    application_id VARCHAR(64) NOT NULL COMMENT '申请ID',
    action VARCHAR(32) NOT NULL COMMENT '动作（submit/approve/reject/cancel）',
    actor_id VARCHAR(64) NOT NULL COMMENT '操作者ID',
    actor_role VARCHAR(32) NOT NULL COMMENT '操作者角色',
    from_status VARCHAR(32) COMMENT '变更前状态',
    to_status VARCHAR(32) COMMENT '变更后状态',
    comment TEXT COMMENT '意见/说明',
    trace_id VARCHAR(64) COMMENT '链路ID',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_app (application_id),
    INDEX idx_actor (actor_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='审批审计流水';

CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key VARCHAR(128) PRIMARY KEY COMMENT '幂等键',
    user_id VARCHAR(64) NOT NULL COMMENT '用户ID',
    endpoint VARCHAR(64) NOT NULL COMMENT '业务动作',
    response_json JSON COMMENT '上次响应',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    expires_at DATETIME COMMENT '过期时间',
    INDEX idx_expires (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='写操作幂等键';
