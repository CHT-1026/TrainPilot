"""用户/汇报线管理 HTTP 冒烟：管理员权限、设置与回退。"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.getenv("OA_BASE", "http://127.0.0.1:8080")


def api(path, token=None, payload=None, method=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with urllib.request.urlopen(
        urllib.request.Request(BASE + path, data=data, headers=headers, method=method), timeout=30
    ) as response:
        return json.load(response)


def login(username):
    return api("/api/auth/login", payload={"username": username, "password": "admin123"})["data"]["token"]


def main():
    admin = login("admin")
    employee = login("employee_qian")
    old_manager = None
    try:
        users = api("/api/users", admin)["data"]
        target = next(item for item in users if item["username"] == "employee_qian")
        old_manager = target.get("manager_id")
        print("list users", len(users))

        try:
            api("/api/users", employee)
            raise AssertionError("employee should not list users")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403, exc.code
            print("employee list blocked")

        updated = api(
            "/api/users/employee_qian/manager",
            admin,
            {"manager_id": "manager_zhang"},
            method="PUT",
        )["data"]
        assert updated["manager_id"] == "manager_zhang", updated
        print("set manager ok")

        try:
            api(
                "/api/users/employee_qian/manager",
                admin,
                {"manager_id": "employee_zhao"},
                method="PUT",
            )
            raise AssertionError("non-manager should be rejected")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400, exc.code
            print("non-manager rejected")

        try:
            api("/api/users/no_such_user/manager", admin, {"manager_id": None}, method="PUT")
            raise AssertionError("missing user should be 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, exc.code
            print("missing user rejected")

        restored = api(
            "/api/users/employee_qian/manager",
            admin,
            {"manager_id": old_manager},
            method="PUT",
        )["data"]
        assert restored["manager_id"] == old_manager, restored
        print("restore manager ok")
        print("USER_ADMIN_PASS")
    finally:
        api(
            "/api/users/employee_qian/manager",
            admin,
            {"manager_id": old_manager},
            method="PUT",
        )


if __name__ == "__main__":
    main()
