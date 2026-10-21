-- ============================================
-- 澜图智能企业智能体平台 - MySQL数据库初始化脚本
-- ============================================
-- 创建日期: 2026-05-22
-- 说明: 创建用户表及请假、报销、离职申请相关表
-- ============================================

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS xiaodi_smart_oa 
DEFAULT CHARACTER SET utf8mb4 
DEFAULT COLLATE utf8mb4_unicode_ci;

USE xiaodi_smart_oa;

-- ============================================
-- 1. 用户表
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户ID',
    username VARCHAR(64) NOT NULL UNIQUE COMMENT '用户名（登录账号）',
    password VARCHAR(256) NOT NULL COMMENT '密码（bcrypt加密存储）',
    name VARCHAR(64) NOT NULL COMMENT '姓名',
    role VARCHAR(32) NOT NULL COMMENT '角色（admin/manager/finance/hr/employee）',
    department VARCHAR(64) COMMENT '部门',
    position VARCHAR(64) COMMENT '职位',
    manager_id VARCHAR(64) DEFAULT NULL COMMENT '直属主管 username',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_username (username) COMMENT '用户名索引',
    INDEX idx_role (role) COMMENT '角色索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- 插入预设用户数据（所有密码均为 admin123 的bcrypt哈希）
-- 注意：bcrypt哈希中的 $ 符号在SQL中不需要转义
INSERT INTO users (username, password, name, role, department, position) VALUES
('admin', '$2b$12$mEoLQu6mOJUlZiUXW57xS.pea4M2j57afrvAMAGRFyZWccKNnrdnq', '系统管理员', 'admin', '管理部', '系统管理员'),
('manager_zhang', '$2b$12$mEoLQu6mOJUlZiUXW57xS.pea4M2j57afrvAMAGRFyZWccKNnrdnq', '张主管', 'manager', '技术部', '部门主管'),
('manager_li', '$2b$12$mEoLQu6mOJUlZiUXW57xS.pea4M2j57afrvAMAGRFyZWccKNnrdnq', '李经理', 'manager', '财务部', '部门经理'),
('finance_wang', '$2b$12$mEoLQu6mOJUlZiUXW57xS.pea4M2j57afrvAMAGRFyZWccKNnrdnq', '王财务', 'finance', '财务部', '财务专员'),
('hr_liu', '$2b$12$mEoLQu6mOJUlZiUXW57xS.pea4M2j57afrvAMAGRFyZWccKNnrdnq', '刘HR', 'hr', '人事部', 'HR专员'),
('employee_zhao', '$2b$12$mEoLQu6mOJUlZiUXW57xS.pea4M2j57afrvAMAGRFyZWccKNnrdnq', '赵工', 'employee', '技术部', '高级工程师'),
('employee_qian', '$2b$12$mEoLQu6mOJUlZiUXW57xS.pea4M2j57afrvAMAGRFyZWccKNnrdnq', '钱工', 'employee', '技术部', '工程师');

-- ============================================
-- 2. 请假申请表
-- ============================================
CREATE TABLE IF NOT EXISTS leave_applications (
    id VARCHAR(64) PRIMARY KEY COMMENT '申请ID（UUID）',
    employee_id VARCHAR(64) NOT NULL COMMENT '员工ID',
    employee_name VARCHAR(64) NOT NULL COMMENT '员工姓名',
    employee_role VARCHAR(32) NOT NULL COMMENT '员工角色（employee/manager/finance/hr/admin）',
    manager_id VARCHAR(64) DEFAULT NULL COMMENT '提交时的直属主管快照',
    department VARCHAR(64) DEFAULT NULL COMMENT '部门',
    leave_type VARCHAR(32) NOT NULL COMMENT '请假类型（年假/病假/事假/婚假/产假/丧假）',
    leave_days INT NOT NULL COMMENT '请假天数',
    start_date DATE NOT NULL COMMENT '开始日期',
    end_date DATE NOT NULL COMMENT '结束日期',
    reason TEXT COMMENT '请假原因',
    status VARCHAR(32) DEFAULT 'pending' COMMENT '状态（pending/approved/rejected/cancelled/finished）',
    manager_approval VARCHAR(32) DEFAULT 'pending' COMMENT '主管审批（pending/approved/rejected/not_required）',
    hr_approval VARCHAR(32) DEFAULT 'pending' COMMENT 'HR审批（pending/approved/rejected/not_required）',
    approval_step VARCHAR(16) NOT NULL DEFAULT 'manager' COMMENT '当前审批环节（manager/hr/done）',
    version INT NOT NULL DEFAULT 0 COMMENT '乐观锁版本号',
    idempotency_key VARCHAR(128) DEFAULT NULL COMMENT '幂等键',
    submitted_at DATETIME DEFAULT NULL COMMENT '提交时间',
    cancelled_at DATETIME DEFAULT NULL COMMENT '取消时间',
    comments TEXT COMMENT '审批意见',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_employee (employee_id) COMMENT '员工ID索引',
    INDEX idx_status (status) COMMENT '状态索引',
    INDEX idx_created (created_at) COMMENT '创建时间索引',
    UNIQUE KEY uk_leave_idem (employee_id, idempotency_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='请假申请表';

-- ============================================
-- 3. 报销申请表
-- ============================================
CREATE TABLE IF NOT EXISTS reimbursement_applications (
    id VARCHAR(64) PRIMARY KEY COMMENT '申请ID（UUID）',
    employee_id VARCHAR(64) NOT NULL COMMENT '员工ID',
    employee_name VARCHAR(64) NOT NULL COMMENT '员工姓名',
    employee_role VARCHAR(32) NOT NULL COMMENT '员工角色（employee/manager/finance/hr/admin）',
    department VARCHAR(64) DEFAULT NULL COMMENT '部门',
    amount DECIMAL(10,2) NOT NULL COMMENT '报销金额',
    category VARCHAR(32) NOT NULL COMMENT '报销类别（transport/food/accommodation/office）',
    description TEXT COMMENT '报销描述',
    receipt_info VARCHAR(256) DEFAULT NULL COMMENT '票据信息',
    status VARCHAR(32) DEFAULT 'pending' COMMENT '状态（pending/approved/rejected）',
    manager_approval VARCHAR(32) DEFAULT 'pending' COMMENT '主管审批（pending/approved/rejected）',
    finance_approval VARCHAR(32) DEFAULT 'pending' COMMENT '财务审批（pending/approved/rejected）',
    comments TEXT COMMENT '审批意见',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_employee (employee_id) COMMENT '员工ID索引',
    INDEX idx_status (status) COMMENT '状态索引',
    INDEX idx_created (created_at) COMMENT '创建时间索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='报销申请表';

-- ============================================
-- 4. 离职申请表
-- ============================================
CREATE TABLE IF NOT EXISTS resignation_applications (
    id VARCHAR(64) PRIMARY KEY COMMENT '申请ID（UUID）',
    employee_id VARCHAR(64) NOT NULL COMMENT '员工ID',
    employee_name VARCHAR(64) NOT NULL COMMENT '员工姓名',
    employee_role VARCHAR(32) NOT NULL COMMENT '员工角色（employee/manager/finance/hr/admin）',
    department VARCHAR(64) DEFAULT NULL COMMENT '部门',
    position VARCHAR(64) DEFAULT NULL COMMENT '职位',
    reason TEXT COMMENT '离职原因',
    status VARCHAR(32) DEFAULT 'pending' COMMENT '状态（pending/validating/approved/rejected）',
    handover_status VARCHAR(32) DEFAULT 'pending' COMMENT '工作交接状态',
    asset_return_status VARCHAR(32) DEFAULT 'pending' COMMENT '资产归还状态',
    it_account_status VARCHAR(32) DEFAULT 'pending' COMMENT 'IT账号注销状态',
    hr_interview_status VARCHAR(32) DEFAULT 'pending' COMMENT 'HR面谈状态',
    comments TEXT COMMENT '审批意见',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_employee (employee_id) COMMENT '员工ID索引',
    INDEX idx_status (status) COMMENT '状态索引',
    INDEX idx_created (created_at) COMMENT '创建时间索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='离职申请表';

-- ============================================
-- 5. 审批审计（只追加）
-- ============================================
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

-- ============================================
-- 6. 幂等键
-- ============================================
CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key VARCHAR(128) PRIMARY KEY COMMENT '幂等键',
    user_id VARCHAR(64) NOT NULL COMMENT '用户ID',
    endpoint VARCHAR(64) NOT NULL COMMENT '业务动作',
    response_json JSON COMMENT '上次响应',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    expires_at DATETIME COMMENT '过期时间',
    INDEX idx_expires (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='写操作幂等键';

-- ============================================
-- 7. 知识库文档与文本块
-- ============================================
CREATE TABLE IF NOT EXISTS kb_documents (
    id VARCHAR(64) PRIMARY KEY COMMENT '文档ID',
    namespace VARCHAR(32) NOT NULL DEFAULT 'default' COMMENT '语料命名空间（default=OA / lantu=企业RAG）',
    title VARCHAR(200) NOT NULL COMMENT '标题',
    source_path VARCHAR(500) COMMENT '源文件路径',
    doc_type VARCHAR(32) COMMENT 'md/txt/docx/pdf/csv/xlsx',
    format VARCHAR(16) DEFAULT NULL COMMENT '文件格式',
    version VARCHAR(32) COMMENT '文档版本',
    effective_date DATE COMMENT '生效日期',
    status VARCHAR(16) NOT NULL DEFAULT 'active' COMMENT 'active/superseded',
    visibility VARCHAR(32) NOT NULL DEFAULT 'all' COMMENT 'all/manager/hr/finance/admin',
    department VARCHAR(64) DEFAULT NULL COMMENT '所属部门',
    author VARCHAR(64) DEFAULT NULL COMMENT '作者',
    doc_family VARCHAR(128) DEFAULT NULL COMMENT '文档族（同族多版本）',
    category VARCHAR(64) DEFAULT NULL COMMENT '文档类型（company/org/product/...）',
    checksum CHAR(64) COMMENT '内容校验',
    doc_created_at DATETIME DEFAULT NULL COMMENT '文档创建时间',
    doc_updated_at DATETIME DEFAULT NULL COMMENT '文档更新时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_visibility (visibility),
    INDEX idx_namespace_status (namespace, status),
    INDEX idx_doc_family (namespace, doc_family)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识库文档';

CREATE TABLE IF NOT EXISTS kb_chunks (
    id VARCHAR(64) PRIMARY KEY COMMENT '块ID',
    doc_id VARCHAR(64) NOT NULL COMMENT '文档ID',
    section VARCHAR(200) COMMENT '章节',
    chunk_type VARCHAR(16) NOT NULL DEFAULT 'text' COMMENT 'text/table',
    ord INT NOT NULL COMMENT '块序号',
    text MEDIUMTEXT NOT NULL COMMENT '块文本（含标题前缀）',
    token_count INT COMMENT '检索词数',
    embedding_json MEDIUMTEXT COMMENT '向量（JSON 数组）',
    embedding_dim INT COMMENT '向量维度',
    embedding_model VARCHAR(64) COMMENT '向量模型',
    embedding_version VARCHAR(32) COMMENT '向量版本',
    INDEX idx_doc (doc_id),
    INDEX idx_doc_ord (doc_id, ord)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识库文本块';

-- ============================================
-- 完成提示
-- ============================================
SELECT '数据库初始化完成！' AS message;
