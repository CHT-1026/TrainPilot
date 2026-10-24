"""Real service/SQL integration using connection-local temporary tables only.

Run explicitly: .venv312/Scripts/python.exe -B -X utf8 tests/probe_mysql_leave_temporary.py
No repository or SQL mocks; no model calls; no permanent business writes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
TABLES = ("leave_applications", "approval_audit", "idempotency_keys")
OUT = ROOT / "research/project_assessment_20261023/revision2/workflow_ui/service_mysql_temp_20261024"


def digest_tables(conn):
    """Read-only observer sees permanent tables, never the temporary shadows."""
    conn.rollback()
    result = {}
    with conn.cursor() as cursor:
        for table in (*TABLES, "users"):
            cursor.execute(f"SELECT * FROM `{table}`")
            rows = sorted(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) for row in cursor.fetchall())
            result[table] = {"rows": len(rows), "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest()}
    conn.rollback()
    return result


def main():
    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env.local", override=True)
    import pymysql
    from app.core.config import get_settings
    from app.domain.leave import repository
    from app.domain.user.models import UserContext
    import app.services.leave_service as service_module

    OUT.mkdir(parents=True, exist_ok=True)
    run = uuid4().hex[:10]
    report = {"run_id": run, "mode": "real_mysql_temporary_tables", "repository_mocked": False,
              "sql_mocked": False, "model_calls": 0, "cases": [], "cleanup": {},
              "scope": "Single dedicated connection with three empty CREATE TEMPORARY TABLE LIKE shadows. Separate read-only observer fingerprints permanent tables. User contexts are synthetic; users table is read only."}
    conn = observer = None
    created = []
    fail_next_commit = False
    sql_failure_count = 0
    transactions = {"commit": 0, "rollback": 0}
    phase = "connections"
    expected_connection_id = None
    isolation_lost = False
    report["isolation_checks"] = []

    class IsolationLost(RuntimeError):
        pass

    def native_connection():
        settings = get_settings()
        # No DBUtils wrapper, ping, pool, or reconnect path can replace this session.
        return pymysql.connect(host=settings.mysql_host, port=settings.mysql_port,
            user=settings.mysql_user, password=settings.mysql_password,
            database=settings.mysql_database, charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor, autocommit=False,
            connect_timeout=5, read_timeout=15, write_timeout=15)

    def guard(stage):
        nonlocal isolation_lost
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT CONNECTION_ID() AS id")
                actual_id = cursor.fetchone()["id"]
                if actual_id != expected_connection_id:
                    raise IsolationLost("Physical MySQL session changed")
                for table in TABLES:
                    cursor.execute(f"SHOW CREATE TABLE `{table}`")
                    ddl = next(v for k, v in cursor.fetchone().items() if k.lower().startswith("create"))
                    if "CREATE TEMPORARY TABLE" not in ddl.upper():
                        raise IsolationLost("Temporary shadow missing; all service access refused")
            report["isolation_checks"].append({"stage": stage, "connection_id": actual_id, "three_temporary_shadows": True})
        except BaseException:
            isolation_lost = True
            report["isolation_checks"].append({"stage": stage, "blocked": True})
            raise

    def record(cid, description, actual, passed):
        report["cases"].append({"id": cid, "description": description, "actual": actual, "pass": bool(passed)})
        if isolation_lost and cid != "DB29":
            raise IsolationLost("Abort remaining cases after isolation failure")
        if not passed:
            raise AssertionError(f"{cid} failed; stop this run before further service operations")

    def counts():
        guard("counts")
        with conn.cursor() as cursor:
            result = {}
            for table in TABLES:
                cursor.execute(f"SELECT COUNT(*) AS n FROM `{table}`")
                result[table] = int(cursor.fetchone()["n"])
            return result

    def state(app_id):
        guard("state")
        return repository.get_application(conn, app_id)

    @contextmanager
    def same_connection():
        guard("service_read")
        yield conn

    @contextmanager
    def same_transaction():
        nonlocal fail_next_commit, sql_failure_count
        guard("before_begin")
        conn.begin()
        try:
            yield conn
            guard("before_commit")
            if fail_next_commit:
                fail_next_commit = False
                sql_failure_count += 1
                # A real MySQL error after actual service SQL but before COMMIT.
                with conn.cursor() as cursor:
                    cursor.execute("SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'TEMP_PROBE_FORCE_ROLLBACK'")
            conn.commit()
            transactions["commit"] += 1
        except BaseException:
            try:
                conn.rollback()
                transactions["rollback"] += 1
            finally:
                guard("after_rollback")
            raise

    def capture(cid, description, result, passed):
        record(cid, description, result.to_dict(), passed)
        return result

    try:
        conn, observer = native_connection(), native_connection()
        report["connection_mode"] = "native_pymysql_dedicated_no_pool_no_ping_no_reconnect"
        with conn.cursor() as cursor:
            cursor.execute("SELECT DATABASE() AS db, VERSION() AS version, CONNECTION_ID() AS id")
            runtime = cursor.fetchone()
            expected_connection_id = runtime["id"]
            report["mysql_version"] = runtime["version"]
            schema = runtime["db"]
        with observer.cursor() as cursor:
            cursor.execute("SELECT CONNECTION_ID() AS id")
            assert cursor.fetchone()["id"] != runtime["id"], "Observer must use a different physical connection"
        report["permanent_before"] = digest_tables(observer)
        report["preflight"] = {}
        phase = "schema_preflight"
        with conn.cursor() as cursor:
            for table in TABLES:
                phase = "schema_preflight:" + table
                cursor.execute("SELECT ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s", (schema, table))
                engine = cursor.fetchone()["ENGINE"]
                cursor.execute("SELECT COUNT(*) AS n FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND REFERENCED_TABLE_NAME IS NOT NULL", (schema, table))
                foreign_keys = int(cursor.fetchone()["n"])
                cursor.execute("SELECT COUNT(*) AS n FROM information_schema.TRIGGERS WHERE EVENT_OBJECT_SCHEMA=%s AND EVENT_OBJECT_TABLE=%s", (schema, table))
                triggers = int(cursor.fetchone()["n"])
                report["preflight"][table] = {"engine": engine, "foreign_keys": foreign_keys, "triggers": triggers}
                if engine != "InnoDB" or foreign_keys or triggers:
                    raise RuntimeError("Temporary LIKE cannot faithfully cover this table's transactional constraints")
            for table in TABLES:
                phase = "create_temporary:" + table
                # MySQL rejects target LIKE itself with 1066 even for a shadow.
                # Both stages are TEMPORARY; LIKE still copies the real schema.
                staging = f"probe_{run}_{table}"
                cursor.execute(f"CREATE TEMPORARY TABLE `{staging}` LIKE `{table}`")
                created.append(staging)
                cursor.execute(f"CREATE TEMPORARY TABLE `{table}` LIKE `{staging}`")
                created.append(table)
                cursor.execute(f"DROP TEMPORARY TABLE `{staging}`")
                created.remove(staging)
                cursor.execute(f"SHOW CREATE TABLE `{table}`")
                ddl = next(v for k, v in cursor.fetchone().items() if k.lower().startswith("create"))
                if "CREATE TEMPORARY TABLE" not in ddl.upper():
                    raise RuntimeError("Temporary shadow verification failed; no service write permitted")
        record("DB01", "三张同名连接级临时表均为空且确认为TEMPORARY", counts(), all(n == 0 for n in counts().values()))
        phase = "service_cases"

        department = "临时集成隔离部"
        manager = UserContext(user_id=f"temp_mgr_{run}", name="临时主管", role="manager", department=department)
        employee = UserContext(user_id=f"temp_emp_{run}", name="临时员工", role="employee", department=department, manager_id=manager.user_id)
        wrong_manager = UserContext(user_id=f"temp_wrong_{run}", name="临时其他主管", role="manager", department=department)
        outsider = UserContext(user_id=f"temp_other_{run}", name="临时其他员工", role="employee", department=department)
        hr = UserContext(user_id=f"temp_hr_{run}", name="临时HR", role="hr", department=department)
        start = date.today() + timedelta(days=14)
        while start.weekday() != 0: start += timedelta(days=1)
        def draft(weeks, annual=False):
            day = start + timedelta(weeks=weeks)
            return {"leave_type": "年假" if annual else "事假", "start_date": day.isoformat(),
                    "end_date": (day + timedelta(days=4) if annual else day).isoformat(),
                    "reason": f"TEMP ONLY integration {run} week {weeks}"}
        service = service_module.leave_service
        with patch.object(service_module, "connection", same_connection), patch.object(service_module, "transaction", same_transaction):
            submitted = service.submit(employee, draft(0), idempotency_key=f"{run}:submit", trace_id=run)
            capture("DB02", "真实submit写入申请/审计/幂等", submitted, submitted.ok and submitted.data["application"]["status"] == "pending")
            if not submitted.ok: raise RuntimeError("Primary submit failed")
            app = submitted.data["application"]["app_id"]
            before = counts()
            repeat = service.submit(employee, draft(0), idempotency_key=f"{run}:submit", trace_id=run)
            capture("DB03", "重复提交action返回同一申请且三表无新增", repeat, repeat.ok and repeat.data["application"]["app_id"] == app and counts() == before)
            preview = service.preview_approval(manager, app, "通过", "临时审批预览")
            capture("DB04", "preview只读且不修改三表", preview, preview.ok and counts() == before)
            for cid, actor in (("DB05", employee), ("DB06", wrong_manager)):
                result = service.approve(actor, app, "通过", "不应执行", idempotency_key=f"{run}:unauthorized")
                capture(cid, "本人/非直属同部门主管越权审批均无写入", result, not result.ok and result.code == "FORBIDDEN" and counts() == before and state(app)["version"] == 0)
            pending = service.pending(manager)
            capture("DB07", "直属主管pending使用真实JOIN/条件可见临时申请", pending, pending.ok and pending.data["total"] == 1)
            other_pending = service.pending(wrong_manager)
            capture("DB08", "非直属同部门主管pending不可见", other_pending, other_pending.ok and other_pending.data["total"] == 0)
            approved = service.approve(manager, app, "通过", "临时审批通过", idempotency_key=f"{run}:approve", trace_id=run)
            capture("DB09", "主管审批完成且版本递增", approved, approved.ok and approved.data["application"]["status"] == "approved" and state(app)["version"] == 1)
            before = counts()
            repeat = service.approve(manager, app, "通过", "重复审批", idempotency_key=f"{run}:approve")
            capture("DB10", "重复审批action回放成功但无重复审计/迁移", repeat, repeat.ok and counts() == before and state(app)["version"] == 1)
            result = service.approve(wrong_manager, app, "通过", "相同key跨actor", idempotency_key=f"{run}:approve")
            capture("DB11", "同action key不向其他actor回放成功", result, not result.ok and counts() == before)
            result = service.approve(manager, app, "通过", "新action重复状态", idempotency_key=f"{run}:fresh")
            capture("DB12", "新action不能二次审批已完成状态", result, not result.ok and result.code == "CONFLICT" and counts() == before)
            result = service.cancel(employee, app)
            capture("DB13", "已批准申请不能取消", result, not result.ok and result.code == "CONFLICT" and counts() == before)
            mine, audit = service.mine(employee), service.audit(employee, app)
            record("DB14", "mine/audit读取真实结果与顺序", {"mine": mine.to_dict(), "audit": audit.to_dict()}, mine.ok and mine.data["total"] == 1 and audit.ok and [x["action"] for x in audit.data["items"]] == ["submit", "approve"])
            result = service.audit(outsider, app)
            capture("DB15", "他人审计查询拒绝", result, not result.ok and result.code == "FORBIDDEN")

            rejection = service.submit(employee, draft(1), idempotency_key=f"{run}:reject_submit")
            rejected_id = rejection.data["application"]["app_id"]
            result = service.preview_approval(manager, rejected_id, "驳回", "")
            capture("DB16", "驳回必须填写意见", result, not result.ok and result.code == "INVALID_ARGS")
            rejected = service.approve(manager, rejected_id, "驳回", "临时驳回意见", idempotency_key=f"{run}:reject")
            capture("DB17", "真实驳回状态迁移", rejected, rejected.ok and state(rejected_id)["status"] == "rejected")
            before = counts()
            result = service.approve(manager, rejected_id, "驳回", "重复", idempotency_key=f"{run}:reject")
            capture("DB18", "驳回action重复不重复写", result, result.ok and counts() == before)

            cancellation = service.submit(employee, draft(2), idempotency_key=f"{run}:cancel_submit")
            cancelled_id = cancellation.data["application"]["app_id"]
            preview = service.preview_cancel(employee, cancelled_id)
            before = counts()
            result = service.cancel(outsider, cancelled_id, idempotency_key=f"{run}:wrong_cancel")
            record("DB19", "取消预览及他人取消拒绝", {"preview": preview.to_dict(), "forbidden": result.to_dict()}, preview.ok and not result.ok and result.code == "FORBIDDEN" and counts() == before)
            result = service.cancel(employee, cancelled_id, idempotency_key=f"{run}:cancel")
            capture("DB20", "本人真实取消", result, result.ok and state(cancelled_id)["status"] == "cancelled" and state(cancelled_id)["version"] == 1)
            before = counts()
            result = service.cancel(employee, cancelled_id, idempotency_key=f"{run}:cancel")
            capture("DB21", "取消action重复不重复写", result, result.ok and counts() == before)

            annual = service.submit(employee, draft(4, annual=True), idempotency_key=f"{run}:hr_submit")
            annual_id = annual.data["application"]["app_id"]
            manager_result = service.approve(manager, annual_id, "通过", "主管通过", idempotency_key=f"{run}:hr_manager")
            record("DB22", "五天年假主管后进入HR环节", manager_result.to_dict(), manager_result.ok and state(annual_id)["status"] == "pending" and state(annual_id)["approval_step"] == "hr")
            before = counts()
            result = service.approve(manager, annual_id, "通过", "主管不得代替HR")
            capture("DB23", "主管不能越过HR环节", result, not result.ok and result.code == "FORBIDDEN" and counts() == before)
            result = service.approve(hr, annual_id, "通过", "HR通过", idempotency_key=f"{run}:hr_approve")
            capture("DB24", "HR真实审批完成双环节", result, result.ok and state(annual_id)["status"] == "approved" and state(annual_id)["version"] == 2)

            before = counts();fail_next_commit = True
            result = service.submit(employee, draft(6), idempotency_key=f"{run}:rollback_submit")
            capture("DB25", "真实MySQL SIGNAL失败后submit三表整体回滚", result, not result.ok and result.code == "TOOL_FAILED" and result.retryable and counts() == before)
            retry = service.submit(employee, draft(6), idempotency_key=f"{run}:rollback_submit")
            capture("DB26", "回滚后可用原action重新提交", retry, retry.ok)
            retry_id = retry.data["application"]["app_id"]
            before = counts();fail_next_commit = True
            result = service.approve(manager, retry_id, "通过", "应整体回滚", idempotency_key=f"{run}:rollback_approve")
            capture("DB27", "真实MySQL SIGNAL失败后审批状态/审计/幂等整体回滚", result, not result.ok and result.code == "TOOL_FAILED" and counts() == before and state(retry_id)["status"] == "pending" and state(retry_id)["version"] == 0)
            result = service.approve(manager, retry_id, "通过", "回滚后重试", idempotency_key=f"{run}:rollback_approve")
            capture("DB28", "回滚审批可用原action重试", result, result.ok and state(retry_id)["version"] == 1)
            report["temporary_final_counts"] = counts()
            # Fault-inject a missing TEMP shadow after business cases are done.
            # The guarded transaction body contains no writes even if the guard fails.
            with conn.cursor() as cursor: cursor.execute("DROP TEMPORARY TABLE `idempotency_keys`")
            created.remove("idempotency_keys")
            report["guard_test_temporary_drops"] = 1
            blocked = False
            try:
                with same_transaction(): pass
            except IsolationLost:
                blocked = True
            record("DB29", "TEMP标识丢失后事务入口拒绝继续，禁止落向正式表", {"blocked_before_begin": blocked}, blocked)
        report["injected_real_mysql_failures"] = sql_failure_count
        report["transactions"] = transactions
    except Exception as exc:
        # No credentials, connection strings, or formal business rows in reports.
        errno = exc.args[0] if exc.args and isinstance(exc.args[0], int) else None
        report["error"] = {"type": type(exc).__name__, "stage": phase, "mysql_errno": errno,
                           "capability_denied": errno in {1044, 1045, 1142, 1227}}
    finally:
        if conn is not None:
            cleanup_errors = []
            try:
                conn.rollback()
            except Exception as exc:
                cleanup_errors.append({"stage": "rollback", "type": type(exc).__name__})
            for table in reversed(created):
                try:
                    with conn.cursor() as cursor: cursor.execute(f"DROP TEMPORARY TABLE IF EXISTS `{table}`")
                except Exception as exc: cleanup_errors.append({"table": table, "type": type(exc).__name__})
            drop_errors = sum("table" in item for item in cleanup_errors)
            report["cleanup"]["temporary_tables_dropped"] = len(created) - drop_errors + report.get("guard_test_temporary_drops", 0)
            report["cleanup"]["errors"] = cleanup_errors
            conn.close()
            report["cleanup"]["dedicated_connection_closed"] = True
        if observer is not None:
            try:
                report["permanent_after"] = digest_tables(observer)
                report["permanent_unchanged"] = report.get("permanent_before") == report["permanent_after"]
            except Exception as exc:
                report["cleanup"]["observer_error"] = type(exc).__name__
            finally:
                observer.close()
                report["cleanup"]["observer_connection_closed"] = True
        report["summary"] = {"total": len(report["cases"]), "pass": sum(x["pass"] for x in report["cases"]), "fail": sum(not x["pass"] for x in report["cases"])}
        report["passed"] = (report["summary"]["total"] == 29 and report["summary"]["fail"] == 0
                            and "error" not in report and report.get("permanent_unchanged") is True
                            and report["cleanup"].get("temporary_tables_dropped") == 3
                            and not report["cleanup"].get("errors"))
        archive = OUT / "runs" / run
        archive.mkdir(parents=True, exist_ok=False)
        (archive / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "passed": report["passed"], "permanent_unchanged": report.get("permanent_unchanged"), "cleanup": report["cleanup"], "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
