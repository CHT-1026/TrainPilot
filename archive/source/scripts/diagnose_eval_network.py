"""Read-only DNS/TCP evidence for the two target providers; no model calls."""
import json
import os
from pathlib import Path
import socket
from urllib.parse import urlsplit

hosts = ['api.xiaomimimo.com', 'cn-beijing.maas.qianwenaiapi.com']
proxies = {name: urlsplit(os.environ[name]).hostname for name in
           ('HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY') if os.environ.get(name)}
results = {}
for host in dict.fromkeys(hosts + list(proxies.values())):
    try:
        addresses = sorted({row[4][0] for row in socket.getaddrinfo(host, 443)})
        results[host] = {'addresses': addresses}
        if host in hosts:
            with socket.create_connection((host, 443), timeout=10):
                results[host]['tcp_443'] = 'connected'
    except OSError as exc:
        results[host] = {'error': str(exc)}
receipt = {'dns_and_tcp': results, 'proxy_hosts': proxies, 'no_proxy': os.getenv('NO_PROXY', '')}
path = Path(__file__).resolve().parents[1]/'research/llm_blind_eval_20261029_execution/network_diagnosis.json'
path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
