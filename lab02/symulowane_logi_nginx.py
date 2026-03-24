import re
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pathlib import Path
import csv

pattern = re.compile(r'(?P<ip>\S+).+?\[(?P<time>.*?)\] "(?P<method>\S+) (?P<path>\S+).*?" (?P<status>\d{3}) ('
                     r'?P<size>\d+).*?"?(?P<ua>.*?)"?$')

fake_logs = [
    '192.168.0.10 - - [01/Jan/2025:10:00:01 +0100] "GET / HTTP/1.1" 200 5320 "-" "Mozilla/5.0"',
    '192.168.0.11 - - [01/Jan/2025:10:00:01 +0100] "GET /api HTTP/1.1" 200 1024 "-" "Mozilla/5.0"',
    '10.0.0.1 - - [01/Jan/2025:10:00:02 +0100] "POST /login HTTP/1.1" 302 512 "-" "Mozilla/5.0"',
    '203.0.113.5 - - [01/Jan/2025:10:05:00 +0100] "GET / HTTP/1.1" 503 256 "-" "bot/1.0"',
]

# szum: zwykły ruch (200,304) w kolejnych minutach
for minute in range(10, 15):
    for i in range(5):
        ip = f'192.168.1.{10 + i}'
        t = f'01/Jan/2025:10:{minute:02d}:0{i} +0100'
        fake_logs.append(f'{ip} - - [{t}] "GET /page{i} HTTP/1.1" 200 {1000 + i*10} "-" "Mozilla/5.0"')
    # kilka 304
    fake_logs.append(f'198.51.100.2 - - [01/Jan/2025:10:{minute:02d}:05 +0100] "GET /static/img.png HTTP/1.1" 304 0 "-" "curl/7.64"')

# Dodajemy "mini-atak": jedna sekunda z dziesiątkami powtórzeń tego samego IP
attack_ip = '203.0.113.9'
attack_time = '01/Jan/2025:10:20:00 +0100'
for k in range(30):
    fake_logs.append(f'{attack_ip} - - [{attack_time}] "GET / HTTP/1.1" 200 1200 "-" "bot/1.2"')

# Kilka losowych requestów POST i różnych UA
fake_logs.extend([
    '10.0.0.2 - - [01/Jan/2025:10:00:30 +0100] "POST /submit HTTP/1.1" 200 850 "-" "Mozilla/5.0"',
    '10.0.0.3 - - [01/Jan/2025:10:01:12 +0100] "POST /api/upload HTTP/1.1" 500 120 "-" "curl/7.64"',
    '198.51.100.3 - - [01/Jan/2025:10:02:45 +0100] "GET /search?q=test HTTP/1.1" 200 2048 "-" "Googlebot/2.1"',
])


def parse_time_minute(s):
    """Parsuje czas z logu i obcina sekundę (agregacja po minutach)."""
    # s przykładowo: '01/Jan/2025:10:00:01 +0100'
    dt = datetime.strptime(s.split()[0], "%d/%b/%Y:%H:%M:%S")
    # obcinamy sekundy -> agregacja po minutach
    return dt.replace(second=0)


def build_features_from_logs(log_lines, ip_freq_threshold=10):
    """
    Parsuje listę linii logów, agreguje po minutach i tworzy macierz cech.
    Zwraca: X (ndarray), feature_names (lista nazw cech)
    Cechy:
      - rps: liczba zapytań w oknie (sekunda -> minuta)
      - unique_ips: liczba unikalnych IP w oknie
      - avg_size: średni rozmiar odpowiedzi
      - error_rate: udział odpowiedzi >=400
      - post_count: liczba żądań POST w oknie
      - unique_paths: liczba różnych ścieżek (path)
      - bot_fraction: odsetek requestów z UA zawierającym 'bot'
      - max_requests_from_one_ip: maksymalna liczba requestów z jednego IP w oknie
      - num_ips_over_threshold: liczba IP w oknie, które mają > ip_freq_threshold requestów w tym oknie
    """
    buckets = defaultdict(list)
    global_ip_counter = Counter()

    for line in log_lines:
        m = pattern.search(line)
        if not m:
            continue
        t = parse_time_minute(m.group('time'))
        ip = m.group('ip')
        method = m.group('method')
        path = m.group('path')
        status = int(m.group('status'))
        size = int(m.group('size'))
        ua = m.group('ua') or ''
        rec = {'ip': ip, 'method': method, 'path': path, 'status': status, 'size': size, 'ua': ua}
        buckets[t].append(rec)
        global_ip_counter[ip] += 1

    feature_names = ['rps', 'unique_ips', 'avg_size', 'error_rate', 'post_count', 'unique_paths', 'bot_fraction', 'max_requests_from_one_ip', 'num_ips_over_threshold']
    features = []
    times = []

    for t, recs in sorted(buckets.items()):
        times.append(t)
        rps = len(recs)
        ips = [r['ip'] for r in recs]
        unique_ips = len(set(ips))
        avg_size = float(np.mean([r['size'] for r in recs])) if recs else 0.0
        error_rate = float(np.mean([1 if r['status'] >= 400 else 0 for r in recs])) if recs else 0.0
        post_count = sum(1 for r in recs if r['method'].upper() == 'POST')
        unique_paths = len(set(r['path'] for r in recs))
        bot_fraction = float(np.mean([1 if ('bot' in (r['ua'] or '').lower()) else 0 for r in recs])) if recs else 0.0
        # per-window IP counts
        per_ip_counts = Counter(ips)
        max_requests_from_one_ip = max(per_ip_counts.values()) if per_ip_counts else 0
        num_ips_over_threshold = sum(1 for cnt in per_ip_counts.values() if cnt > ip_freq_threshold)

        features.append([rps, unique_ips, avg_size, error_rate, post_count, unique_paths, bot_fraction, max_requests_from_one_ip, num_ips_over_threshold])

    X = np.array(features)
    return X, feature_names, times, global_ip_counter


if __name__ == '__main__':
    # Zbuduj cechy z fake_logs
    X, feature_names, times, global_ip_counter = build_features_from_logs(fake_logs, ip_freq_threshold=10)
    print('Liczba bucketów (okien):', X.shape[0])
    print('Nazwy cech:', feature_names)
    print('Macierz cech (każdy wiersz = okno minutowe):')
    print(X)

    # Pokaż kilka statystyk globalnych dotyczących IP
    print('\nTop 5 IP (liczba wystąpień w całym logu):')
    for ip, cnt in global_ip_counter.most_common(5):
        print(ip, cnt)

    # Uruchom demonstracyjnie IsolationForest
    if X.shape[0] >= 2:
        iso = IsolationForest(contamination=0.05, random_state=42)
        labels = iso.fit_predict(X)
        anomaly_indices = np.where(labels == -1)[0]
        print('\nIndeksy anomalii wykryte przez IF:', anomaly_indices)

        # Przygotuj katalog wyników
        out_dir = Path(__file__).with_name('symulowane_logi_results')
        out_dir.mkdir(exist_ok=True)

        # Zapis cech do CSV (czas + cechy)
        csv_path = out_dir / 'log_features.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(['time'] + feature_names)
            for t, row in zip(times, X):
                writer.writerow([t.strftime('%Y-%m-%d %H:%M')] + list(row))

        # Zapis top IP globalnych
        with open(out_dir / 'top_ips.txt', 'w', encoding='utf-8') as f:
            for ip, cnt in global_ip_counter.most_common():
                f.write(f"{ip},{cnt}\n")

        # Wykresy czasowe kilku cech z oznaczeniem anomalii
        times_str = [t.strftime('%H:%M') for t in times]
        plots = [
            ('rps', 0),
            ('unique_ips', 1),
            ('error_rate', 3),
            ('bot_fraction', 6),
            ('max_requests_from_one_ip', 7)
        ]
        for name, idx in plots:
            plt.figure(figsize=(8,3))
            plt.plot(times_str, X[:, idx], marker='o', label=name)
            if anomaly_indices.size > 0:
                plt.scatter([times_str[i] for i in anomaly_indices], X[anomaly_indices, idx], color='red', s=80, label='IF anomaly')
            plt.xticks(rotation=45)
            plt.xlabel('Time (minute)')
            plt.ylabel(name)
            plt.title(f'{name} over time (minute buckets)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'{name}_timeseries.png')
            plt.close()

        print('Zapisano CSV i wykresy w:', out_dir)
    else:
        print('Za mało okien aby uruchomić IF.')
