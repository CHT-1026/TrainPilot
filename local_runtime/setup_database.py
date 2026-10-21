"""Initialize only the local WSL demo database, preserving existing records."""
import json
import pathlib
import subprocess
import sys

from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parents[1]


def mysql(sql):
    result = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--exec", "mysql",
         "--batch", "--skip-column-names", "--default-character-set=utf8mb4", "-u", "root"],
        input=sql.encode("utf-8"), capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=90,
    )
    if result.returncode:
        # SQL errors can echo statements; do not include statements or passwords in logs.
        raise RuntimeError("Local MySQL initialization failed (exit %s)." % result.returncode)
    return result.stdout.decode("utf-8", errors="replace").strip()


def main():
    values = dotenv_values(ROOT / ".env.local")
    if values.get("MYSQL_HOST") not in ("127.0.0.1", "localhost"):
        raise RuntimeError("This setup only supports a local database.")
    if values.get("MYSQL_DATABASE") != "xiaodi_smart_oa":
        raise RuntimeError("Unexpected local database name.")
    username = values.get("MYSQL_USER", "")
    if username != "oa_local":
        raise RuntimeError("Expected the dedicated oa_local database account.")
    password = values.get("MYSQL_PASSWORD", "")
    if not password:
        raise RuntimeError("Local database password is missing.")
    exists = mysql("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='xiaodi_smart_oa' AND table_name='users';")
    seeded = False
    if exists == "0" or mysql("SELECT COUNT(*) FROM xiaodi_smart_oa.users;") == "0":
        mysql((ROOT / "sql" / "init_mysql.sql").read_text(encoding="utf-8-sig"))
        seeded = True
    escaped = password.replace("\\", "\\\\").replace("'", "''")
    mysql(
        "CREATE USER IF NOT EXISTS 'oa_local'@'%' IDENTIFIED BY '" + escaped + "';\n"
        "ALTER USER 'oa_local'@'%' IDENTIFIED BY '" + escaped + "';\n"
        "GRANT ALL PRIVILEGES ON xiaodi_smart_oa.* TO 'oa_local'@'%';\n"
    )
    users = mysql("SELECT COUNT(*) FROM xiaodi_smart_oa.users;")
    leaves = mysql("SELECT COUNT(*) FROM xiaodi_smart_oa.leave_applications;")
    print(json.dumps({"local_database": "ready", "seeded_now": seeded,
                      "users": int(users), "leave_records": int(leaves)}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
