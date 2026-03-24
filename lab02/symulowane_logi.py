import re
import numpy as np
from datetime import datetime
from collections import defaultdict

pattern = re.compile(r'(?P<ip>\S+).*\[(?P<time>.*?)\]"(?P<method>\S+) (?P<path>\S+).*" (?P<status>\d{3})(?P<size>\d+)')

fake_logs = [
    '192.168.0.10 - - [01/Jan/2025:10:00:01 +0100] "GET / HTTP/1.1" 200 5320 "-" "Mozilla/5.0"',
    '192.168.0.11 - - [01/Jan/2025:10:00:01 +0100] "GET /api HTTP/1.1" 200 1024 "-" "Mozilla/5.0"',
    '10.0.0.1 - - [01/Jan/2025:10:00:02 +0100] "POST /login HTTP/1.1" 302 512 "-" "Mozilla/5.0"',
    '203.0.113.5 - - [01/Jan/2025:10:05:00 +0100] "GET / HTTP/1.1" 503 256 "-" "bot/1.0"',
]

def parse_time(s):
    return datetime.strptime(s.split()[0], "%d/%b/%Y:%H:%M:%S")

buckets = defaultdict(list)
for line in fake_logs:
    m = pattern.search(line)
    if not m: continue
    t = parse_time(m.group("time"))
    buckets[t].append({"ip": m.group("ip"), "status":
int(m.group("status")), "size": int(m.group("size"))})

features = []
for t, rec in sorted(buckets.items()):
    rps = len(rec)
    unique_ips = len(set(r["ip"] for r in rec))
    avg_size = np.mean([r["size"] for r in rec])
    error_rate = np.mean([1 if r["status"] >= 400 else 0 for r in rec])
    features.append([rps, unique_ips, avg_size, error_rate])

X = np.array(features)