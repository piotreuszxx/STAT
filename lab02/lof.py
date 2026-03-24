import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Dane jak poprzednio
rng = np.random.default_rng(0)
normal_rps = rng.normal(loc=500, scale=30, size=360)
ddos_rps = np.array([3000, 4500, 7200, 8000, 6000, 5500, 4000, 3500])
combined = np.concatenate([normal_rps, ddos_rps])
X = combined.reshape(-1, 1)
indices = np.arange(len(combined))

# Przygotowanie katalogu wyników
out_dir = Path(__file__).with_name('local_outlier_factor_results')
out_dir.mkdir(exist_ok=True)

# Najpierw policzemy Isolation Forest (do porównań)
iso = IsolationForest(contamination=0.05, random_state=42)
labels_iso = iso.fit_predict(X)  # 1 = normal, -1 = anomaly
anomaly_indices_iso = np.where(labels_iso == -1)[0]
print("Indeksy anomalii (IF, contamination=0.05):", anomaly_indices_iso)

# Zapisz też wykres z IF (do porównania)
plt.figure(figsize=(10,4))
plt.plot(indices, combined, label='RPS')
if anomaly_indices_iso.size > 0:
    plt.scatter(anomaly_indices_iso, combined[anomaly_indices_iso], color='purple', marker='s', s=80, label='IF anomalies')
plt.xlabel('Czas (indeks)')
plt.ylabel('RPS')
plt.title('IsolationForest (contamination=0.05) - RPS z oznaczonymi anomaliami')
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / 'if_for_comparison.png')
plt.close()

# Eksperymenty LOF z różnymi n_neighbors
n_list = [5, 10, 20, 50]
results = []  # do CSV: dicts

for n_neighbors in n_list:
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
    labels_lof = lof.fit_predict(X)  # 1 normal, -1 anomaly
    anomaly_indices_lof = np.where(labels_lof == -1)[0]
    anomaly_values_lof = combined[anomaly_indices_lof]
    print(f"n_neighbors={n_neighbors}: indeksy anomalie (LOF):", anomaly_indices_lof)

    # Porównanie z IF
    set_if = set(np.where(labels_iso == -1)[0])
    set_lof = set(anomaly_indices_lof)
    common = sorted(set_if & set_lof)
    print(f"Wspólne anomalie IF & LOF (n={n_neighbors}):", common)

    # Wykres: LOF anomalies (czerwone x), IF anomalies (purpurowe kwadraty)
    plt.figure(figsize=(10,4))
    plt.plot(indices, combined, label='RPS')
    if anomaly_indices_lof.size > 0:
        plt.scatter(anomaly_indices_lof, anomaly_values_lof, color='red', marker='x', s=80, label=f'LOF anomalies (n={n_neighbors})')
    if anomaly_indices_iso.size > 0:
        plt.scatter(anomaly_indices_iso, combined[anomaly_indices_iso], color='purple', marker='s', s=60, label='IF anomalies (cont=0.05)')
    plt.xlabel('Czas (indeks)')
    plt.ylabel('RPS')
    plt.title(f'LOF (n_neighbors={n_neighbors}) vs IF - RPS z oznaczonymi anomaliami')
    plt.legend()
    plt.tight_layout()
    fname = out_dir / f'lof_n{n_neighbors}_vs_if.png'
    plt.savefig(fname)
    plt.close()

    results.append({
        'method': 'LOF',
        'n_neighbors': n_neighbors,
        'mode': 'fit_predict',
        'anomaly_count': int(anomaly_indices_lof.size),
        'anomaly_indices': ' '.join(map(str, anomaly_indices_lof.tolist())),
        'common_with_if': ' '.join(map(str, common))
    })

# Eksperyment: LOF w trybie novelty=True (trenujemy na normalnych danych)
lof_novel = LocalOutlierFactor(n_neighbors=20, contamination='auto', novelty=True)

lof_novel.fit(normal_rps.reshape(-1,1))
pred_novel = lof_novel.predict(X)  # 1 normal, -1 anomaly
anomaly_indices_novel = np.where(pred_novel == -1)[0]
print("LOF novelty (trenowane na normal) - indeksy anomalie:", anomaly_indices_novel)

# Wykres dla novelty
plt.figure(figsize=(10,4))
plt.plot(indices, combined, label='RPS')
if anomaly_indices_novel.size > 0:
    plt.scatter(anomaly_indices_novel, combined[anomaly_indices_novel], color='green', marker='x', s=80, label='LOF novelty anomalies')
if anomaly_indices_iso.size > 0:
    plt.scatter(anomaly_indices_iso, combined[anomaly_indices_iso], color='purple', marker='s', s=60, label='IF anomalies (cont=0.05)')
plt.xlabel('Czas (indeks)')
plt.ylabel('RPS')
plt.title('LOF (novelty=True, trained on normal) - RPS z oznaczonymi anomaliami')
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / 'lof_novelty_trained_on_normal_vs_if.png')
plt.close()

# Zapis wyniku
set_if = set(np.where(labels_iso == -1)[0])
set_novel = set(anomaly_indices_novel)
common_novel = sorted(set_if & set_novel)
results.append({
    'method': 'LOF',
    'n_neighbors': 20,
    'mode': 'novelty',
    'anomaly_count': int(anomaly_indices_novel.size),
    'anomaly_indices': ' '.join(map(str, anomaly_indices_novel.tolist())),
    'common_with_if': ' '.join(map(str, common_novel))
})

# Zapis podsumowania do CSV
csv_path = Path(__file__).with_name('local_outlier_factor_results') / 'lof_summary.csv'
csv_path.parent.mkdir(exist_ok=True)
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['method','n_neighbors','mode','anomaly_count','anomaly_indices','common_with_if'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"Wyniki eksperymentu LOF zapisano w katalogu: {out_dir}")
print(f"Podsumowanie eksperymentu zapisano w: {csv_path}")

# Dodatkowy wykres ogólny LOF dla n=20 (standard)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels_lof = lof.fit_predict(X)
anomaly_indices_lof = np.where(labels_lof == -1)[0]
plt.figure(figsize=(10,4))
plt.plot(indices, combined, label='RPS')
if anomaly_indices_lof.size > 0:
    plt.scatter(anomaly_indices_lof, combined[anomaly_indices_lof], color='red', marker='x', s=80, label='LOF anomalies (n=20)')
plt.xlabel('Czas (indeks)')
plt.ylabel('RPS')
plt.title('Local Outlier Factor - RPS z oznaczonymi anomaliami (n=20)')
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / 'lof_rps_n20.png')
plt.close()

print("Skończono eksperymenty LOF.")
