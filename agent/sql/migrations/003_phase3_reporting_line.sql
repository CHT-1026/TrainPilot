-- Phase 3 汇报线权限（可空；为空时回退部门权限）
ALTER TABLE users
    ADD COLUMN manager_id VARCHAR(64) DEFAULT NULL COMMENT '直属主管 username' AFTER position;

ALTER TABLE leave_applications
    ADD COLUMN manager_id VARCHAR(64) DEFAULT NULL COMMENT '提交时的直属主管快照' AFTER employee_role;
