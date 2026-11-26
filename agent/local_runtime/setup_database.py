"""Create an isolated demo database, cloning local OA data on first use only."""
import subprocess
from dotenv import dotenv_values
from agent.local_runtime.demo_config import AGENT_ROOT, load_config


def wsl_command(*args, sql=None):
    command = ["wsl.exe", "-d", load_config()["wsl_distribution"], "-u", "root", "--exec", *args]
    result = subprocess.run(command, input=sql.encode("utf-8") if sql is not None else None,
        capture_output=True, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"本机数据库命令失败（{args[0]}，exit={result.returncode}），未输出含凭据的 SQL。")
    return result.stdout.decode("utf-8", errors="replace").strip()


def mysql(sql):
    return wsl_command("mysql", "--batch", "--skip-column-names", "--default-character-set=utf8mb4", sql=sql)


def main():
    values = dotenv_values(AGENT_ROOT / ".env.local")
    if (values.get("MYSQL_HOST") != "127.0.0.1" or values.get("MYSQL_DATABASE") != "trainpilot_demo"
            or values.get("MYSQL_USER") != "trainpilot_local"):
        raise ValueError("初始化仅用于独立的 trainpilot_demo 本机数据库。")
    exists = mysql("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='trainpilot_demo' AND table_name='users';")
    if exists == "0":
        mysql("CREATE DATABASE IF NOT EXISTS trainpilot_demo CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        source = mysql("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='xiaodi_smart_oa' AND table_name='users';")
        if source != "0":
            dump = wsl_command("mysqldump", "--single-transaction", "--no-tablespaces", "--skip-add-drop-table", "xiaodi_smart_oa")
            mysql("USE trainpilot_demo;\n" + dump)
        else:
            sql = (AGENT_ROOT / "sql/init_mysql.sql").read_text(encoding="utf-8-sig")
            mysql(sql.replace("xiaodi_smart_oa", "trainpilot_demo"))
    password = values["MYSQL_PASSWORD"].replace("\\", "\\\\").replace("'", "''")
    mysql("CREATE USER IF NOT EXISTS 'trainpilot_local'@'%' IDENTIFIED BY '" + password + "';\n"
          "ALTER USER 'trainpilot_local'@'%' IDENTIFIED BY '" + password + "';\n"
          "GRANT ALL PRIVILEGES ON trainpilot_demo.* TO 'trainpilot_local'@'%';")
    print("独立演示数据库 trainpilot_demo 已就绪；原 xiaodi_smart_oa 数据保留。")


if __name__ == "__main__":
    main()
