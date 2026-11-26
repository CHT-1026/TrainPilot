# -*- coding: utf-8 -*-
"""归档内容隐私与凭据泄漏扫描。

在推送前后运行，确认将要上传的内容里没有真实凭据与个人隐私。

检查项：
  1. 敏感文件名（.env / token / secret / *.pem / *.key …）
  2. 供应商 API Key 格式（OpenAI / Anthropic / GitHub / AWS / Google / Slack / JWT）
  3. 本地 .env.local 中的**真实密钥值**逐一比对（这是最有效的一项）
  4. 弱口令与默认口令
  5. 本机绝对路径（暴露用户名）
  6. 邮箱地址（区分开源项目作者与个人邮箱）

用法：
    ./.venv312/Scripts/python.exe scripts/scan_archive_secrets.py [--repo code_release] [--ref HEAD]
退出码：0 = 干净；1 = 发现真实凭据泄漏
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:/Users/34157/Desktop/vllmproject/澜图智能企业智能体平台")

# 各供应商 API Key 的真实格式
KEY_FORMATS = [
    ("OpenAI",            r"sk-[A-Za-z0-9]{20,}"),
    ("OpenAI/百炼 ws",     r"sk-ws-[A-Za-z0-9_.-]{20,}"),
    ("Anthropic",         r"sk-ant-[A-Za-z0-9_-]{20,}"),
    ("GitHub PAT",        r"ghp_[A-Za-z0-9]{36}"),
    ("GitHub App",        r"ghs_[A-Za-z0-9]{36}"),
    ("GitHub OAuth",      r"gho_[A-Za-z0-9]{36}"),
    ("GitHub fine-grained", r"github_pat_[A-Za-z0-9_]{50,}"),
    ("AWS AccessKey",     r"AKIA[A-Z0-9]{16}"),
    ("Google API",        r"AIza[0-9A-Za-z_-]{35}"),
    ("Slack",             r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ("LangSmith",         r"lsv2_[A-Za-z0-9_]{20,}"),
    ("Bearer",            r"[Bb]earer [A-Za-z0-9._-]{30,}"),
]

SENSITIVE_FILENAMES = re.compile(
    r"(^\.env$)|(^[\w.-]*\.env\.(local|prod|dev)$)|(token)|(secret)|(credential)|"
    r"(\.pem$)|(\.key$)|(\.p12$)|(\.pfx$)|(\.netrc$)|(\.npmrc$)|(id_rsa)",
    re.I)

# 明显是测试/示例的弱口令，不算泄漏
BENIGN_PASSWORDS = {
    "admin123", "password", "123456", "test", "changeme", "secret",
    "MOCK_PASSWORD_EXPOSED", "fake-test-key", "zzsmoketoken",
    "xiaodi-oa-default-secret-key",
}

# 开源项目作者的公开邮箱域名（出现在依赖包元数据里属正常）
UPSTREAM_DOMAINS = re.compile(
    r"@(github|gmail|google|mozilla|python\.org|pypi|django|pydantic|readthedocs|"
    r"opentelemetry|consilium|json-schema|linuxfoundation|certifi|urllib3|docker|"
    r"microsoft|redhat|canonical|stackoverflow|debian|fedoraproject|numpy|scipy|"
    r"pandas|matplotlib|spdx|osdev|qiita|freebsd|netbsd|openstack|ansible|golang|"
    r"rust-lang|kubernetes|openai|anthropic|GrayVines|du\.nham\.ca|vctrn\.dev|"
    r"glicksoftware|skippinet|countzero|davidhewitt|jpadilla|mcp\.resource|mcp\.tool|"
    r"jolibois|r67)\b", re.I)


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True, timeout=300)
    return r.stdout


def tree_files(repo: Path, ref: str, prefix: str) -> list[str]:
    out = git(repo, "ls-tree", "-r", "--name-only", ref, prefix)
    return [x for x in out.splitlines() if x.strip()]


def read_blob(repo: Path, ref: str, path: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"],
                       capture_output=True, timeout=120)
    return r.stdout.decode("utf-8", errors="replace")


def load_env_secrets(env_path: Path) -> list[tuple[str, str]]:
    """从本地 .env / .env.local 提取真实密钥值。"""
    out: list[tuple[str, str]] = []
    if not env_path.exists():
        return out
    key_hint = re.compile(
        r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|DSN)", re.I)
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        k = k.strip()
        # 只要长度够、且键名像凭据的，才当作真密钥
        if len(v) >= 8 and key_hint.search(k):
            if v in BENIGN_PASSWORDS:
                continue
            out.append((k, v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="code_release")
    ap.add_argument("--ref", default="HEAD")
    args = ap.parse_args()
    repo = ROOT / args.repo
    prefix = "archive/"
    if not repo.exists():
        print(f"[FAIL] 仓库不存在：{repo}")
        return 1

    files = tree_files(repo, args.ref, prefix)
    if not files:
        print(f"[FAIL] {args.ref}:{prefix} 下没有文件")
        return 1
    print(f"扫描目标：{repo.name} @ {args.ref} / {prefix}（{len(files)} 个文件）")
    print("=" * 70)

    # 读入全部文本内容
    blobs: list[tuple[str, str]] = []
    for f in files:
        if f.lower().endswith((".png", ".pdf", ".xlsx", ".docx", ".jpg", ".ico")):
            continue
        blobs.append((f, read_blob(repo, args.ref, f)))
    text_all = "\n".join(t for _, t in blobs)
    print(f"已读入 {len(blobs)} 个文本文件（{len(text_all)/1024/1024:.1f} MB）")

    leaks: list[str] = []

    # ---- 1. 敏感文件名 ----
    print("\n[1] 敏感文件名")
    hits = [f for f in files if SENSITIVE_FILENAMES.search(Path(f).name)]
    # 排除明显是代码/配置里提到这些词的正常文件名
    real = [f for f in hits if not Path(f).name.endswith((".py", ".md", ".json"))
            or Path(f).name in (".env.example",)]
    if real:
        print(f"  ⚠ 命中 {len(real)} 个：{real[:8]}")
    else:
        print("  ✅ 无（仅 .env.example 等示例文件）")

    # ---- 2. 供应商 key 格式 ----
    print("\n[2] 供应商 API Key 格式")
    fmt_hits = 0
    for name, pat in KEY_FORMATS:
        m = re.findall(pat, text_all)
        if m:
            fmt_hits += len(m)
            print(f"  ⚠ {name}: {len(m)} 处 —— {m[:2]}")
    if not fmt_hits:
        print("  ✅ 全部 12 类格式均无命中")

    # ---- 3. 本地真实密钥值比对（最有效）----
    print("\n[3] 本地 .env.local 真实密钥值逐一比对")
    secrets = []
    for env in (ROOT / ".env.local", ROOT / ".env"):
        secrets += load_env_secrets(env)
    # 去重：同一值可能在多个文件里出现，保留首个键名
    # 注意：这里必须保持 (键名, 值) 的顺序。
    # 早期写成 {v: k for k, v in secrets} 后再解包，会把键名当值去比对，
    # 造成"密钥泄漏"的假报警。
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for k, v in secrets:
        if v not in seen:
            seen.add(v)
            uniq.append((k, v))
    secrets = uniq
    if not secrets:
        print("  ⚠ 本地未找到真实密钥（无法比对）")
    else:
        leaked = [(k, v) for k, v in secrets if v in text_all]
        for k, v in secrets:
            status = "!! 泄漏" if v in text_all else "未出现"
            print(f"  {k:<34} [{len(v):>3} 字符] {status}")
        if leaked:
            leaks.append(f"{len(leaked)} 个真实密钥值出现在归档中")

    # ---- 4. 弱口令 ----
    print("\n[4] 弱口令 / 默认口令")
    weak = []
    for pat in (r"password[\"']?\s*[:=]\s*[\"']([^\"']{4,})[\"']",
                r"[\"']([a-zA-Z0-9_]+123)[\"']"):
        for m in re.finditer(pat, text_all):
            v = m.group(1)
            if v.lower() not in BENIGN_PASSWORDS and not v.startswith("{"):
                weak.append(v)
    if weak:
        uniq = sorted(set(weak))
        print(f"  ⚠ 出现 {len(uniq)} 种非示例弱口令：{uniq[:8]}")
        print("     （多为本地测试账号固定口令，若目标仓库为公开仓库建议改为环境变量注入）")
    else:
        print("  ✅ 仅示例/测试口令")

    # ---- 5. 本机绝对路径 ----
    print("\n[5] 本机绝对路径")
    paths = sorted(set(re.findall(r"[A-Za-z]:\\Users\\[^\\\s\"']+", text_all)))
    if paths:
        print(f"  ⚠ {len(paths)} 种：{paths[:5]}")
        leaks.append(f"{len(paths)} 种本机绝对路径")
    else:
        print("  ✅ 无")

    # ---- 6. 邮箱 ----
    print("\n[6] 邮箱地址")
    mails = sorted(set(re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text_all)))
    own = [m for m in mails if not UPSTREAM_DOMAINS.search(m)]
    print(f"  共 {len(mails)} 个不同邮箱；其中非上游开源项目作者 {len(own)} 个")
    if own:
        print(f"  ⚠ {own[:10]}")
        print("     （多为项目自身公开联系方式/示例值，请人工确认是否可公开）")
    else:
        print("  ✅ 全部为上游开源项目作者公开邮箱")

    print("\n" + "=" * 70)
    if leaks:
        print("结论：发现需处理的问题")
        for x in leaks:
            print(f"  - {x}")
        return 1
    print("结论：✅ 未发现真实凭据泄漏")
    return 0


if __name__ == "__main__":
    sys.exit(main())
