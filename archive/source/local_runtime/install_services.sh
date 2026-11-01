#!/usr/bin/env bash
set -euo pipefail

# This script is for the already installed Ubuntu-24.04 WSL distribution.
# It installs local database services; it does not contact the course database.
if [ "$(id -u)" -ne 0 ]; then
  echo 'Run this script as root in Ubuntu-24.04.' >&2
  exit 1
fi

curl -fsSL --max-time 30 https://packages.redis.io/gpg -o /tmp/oa-redis-repository.gpg
gpg --dearmor --yes -o /usr/share/keyrings/oa-redis-archive-keyring.gpg /tmp/oa-redis-repository.gpg
chmod 644 /usr/share/keyrings/oa-redis-archive-keyring.gpg
printf '%s\n' 'deb [signed-by=/usr/share/keyrings/oa-redis-archive-keyring.gpg] https://packages.redis.io/deb noble main' > /etc/apt/sources.list.d/oa-redis.list
apt-get update -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server redis
systemctl start mysql
systemctl start redis-server
redis-cli CONFIG SET appendonly yes
redis-cli CONFIG REWRITE
mysql --version
redis-server --version
redis-cli PING
redis-cli COMMAND INFO JSON.SET FT.CREATE
