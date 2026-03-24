import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import csv

# Dane: normalny ruch + atak DDoS
rng = np.random.default_rng(0)
normal_rps = rng.normal(loc=500, scale=30, size=360)
ddos_rps = np.array([3000, 4500, 7200, 8000, 6000, 5500, 4000,
3500])
combined = np.concatenate([normal_rps, ddos_rps])

# Upewnij się, że katalog wyników istnieje
out_dir = os.path.join(os.path.dirname(__file__), 'isolation_forest_results')
os.makedirs(out_dir, exist_ok=True)

# Macierz cech (1D: RPS)
X = combined.reshape(-1, 1)
normal_X = normal_rps.reshape(-1, 1)

# Pomocnicza funkcja rysująca sygnał i oznaczająca anomalia
def plot_with_anomalies(values, anomaly_idx, title, fname, show_attack=True):
    plt.figure(figsize=(10, 3))
    plt.plot(values, label='RPS')
    if len(anomaly_idx) > 0:
        plt.scatter(anomaly_idx, values[anomaly_idx], color='red', label='Anomalie', zorder=5)
    if show_attack:
        # zaznacz końcówkę z atakiem
        attack_start = len(values) - len(ddos_rps)
        plt.axvspan(attack_start - 0.5, len(values) - 0.5, color='orange', alpha=0.1, label='Strefa ataku')
    plt.legend()
    plt.title(title)
    plt.xlabel('Indeks (czas)')
    plt.ylabel('RPS')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()

# 1. Różne wartości contamination
contamination_values = [0.01, 0.05, 0.1, 0.2]
contamination_results = {}
for c in contamination_values:
    iso = IsolationForest(contamination=c, random_state=0)
    labels = iso.fit_predict(X)  # 1 = normalny, -1 = anomalia
    anomaly_indices = np.where(labels == -1)[0]
    contamination_results[c] = anomaly_indices
    print(f'contamination={c}: liczba anomalii = {len(anomaly_indices)}; indeksy={anomaly_indices}')
    plot_with_anomalies(combined, anomaly_indices,
                        title=f'IsolationForest contamination={c} - {len(anomaly_indices)} anomalii',
                        fname=f'if_contamination_{c}.png')

# 2. Uczenie tylko na ruchu normalnym
iso_normal_train = IsolationForest(contamination=0.05, random_state=0)
# uczymy tylko na normalnym ruchu
iso_normal_train.fit(normal_X)
labels_trained_on_normal = iso_normal_train.predict(X)
anomaly_idx_normal_train = np.where(labels_trained_on_normal == -1)[0]
print(f'Wytrenowany na normalnym ruchu (contamination=0.05): liczba anomalii = {len(anomaly_idx_normal_train)}; indeksy={anomaly_idx_normal_train}')
plot_with_anomalies(combined, anomaly_idx_normal_train,
                    title='IF trenowany na normalnym ruchu (contamination=0.05)',
                    fname='if_trained_on_normal.png')

# 3 & 4. Odczyt score'ów decyzji i wykres score'ów
iso_for_scores = IsolationForest(contamination=0.05, random_state=0)
iso_for_scores.fit(X)
scores = iso_for_scores.decision_function(X)  # im wyższy, tym bardziej normalne

# zapisz wykres score'ów
plt.figure(figsize=(10, 3))
plt.plot(scores, label='decision_function score')
attack_start = len(combined) - len(ddos_rps)
plt.axvspan(attack_start - 0.5, len(combined) - 0.5, color='orange', alpha=0.1, label='Strefa ataku')
plt.legend()
plt.title('Decision function scores (cały sygnał)')
plt.xlabel('Indeks (czas)')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'if_scores.png'))
plt.close()

# Zobacz score'y dla końcówki ataku
attack_scores = scores[attack_start:]
print('Score decyzyjny dla punktów ataku:', attack_scores)

# 5. Parametry strukturalne modelu i test stabilności
n_estimators_list = [50, 100, 200]
max_samples_list = [0.5, 1.0]
seeds = list(range(10))  # powtórzenia by sprawdzić stabilność

stability_summary = {}
for n in n_estimators_list:
    for ms in max_samples_list:
        freq = np.zeros(len(combined), dtype=int)
        for s in seeds:
            iso_s = IsolationForest(n_estimators=n, max_samples=ms, contamination=0.05, random_state=s)
            labels_s = iso_s.fit_predict(X)
            freq += (labels_s == -1)
        stability_summary[(n, ms)] = freq
        # zapisz wykres częstotliwości
        plt.figure(figsize=(10, 3))
        plt.bar(np.arange(len(combined)), freq, color='grey')
        plt.title(f'Anomaly frequency over {len(seeds)} runs (n={n}, max_samples={ms})')
        plt.xlabel('Indeks (czas)')
        plt.ylabel('Ilosc wykryc jako anomalia')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'if_stability_n{n}_ms{str(ms).replace(".", "_")}.png'))
        plt.close()
        # wypisz wnioski skrótowo
        consistent_indices = np.where(freq == len(seeds))[0]
        print(f'n={n}, max_samples={ms}: indeksy wykrywane jako anomalie we wszystkich {len(seeds)} uruchomieniach: {consistent_indices}; liczba = {len(consistent_indices)}')

# Zapis podsumowania do CSV
summary_path = os.path.join(out_dir, 'if_summary.csv')
with open(summary_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['experiment', 'param', 'num_anomalies', 'anomaly_indices', 'notes'])
    # contamination results
    for c, idx in contamination_results.items():
        writer.writerow(['contamination', c, len(idx), ' '.join(map(str, idx.tolist())), 'trained on whole combined'])
    # trained on normal
    writer.writerow(['trained_on_normal', 0.05, len(anomaly_idx_normal_train), ' '.join(map(str, anomaly_idx_normal_train.tolist())), 'model trained on normal_X'])
    # scores summary
    writer.writerow(['attack_scores_mean', '', np.mean(attack_scores), '', 'mean decision score on attack segment'])
    # stability
    for (n, ms), freq in stability_summary.items():
        consistent = np.where(freq == len(seeds))[0]
        writer.writerow(['stability_consistent', f'n={n},max_samples={ms}', len(consistent), ' '.join(map(str, consistent.tolist())), f'indices detected as anomaly in all {len(seeds)} runs'])

print('\nWszystkie wykresy zapisane w:', out_dir)
print('Podsumowanie eksperymentu zapisano w:', summary_path)
