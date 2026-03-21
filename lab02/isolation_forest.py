import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pathlib import Path

# Dane: normalny ruch + atak DDoS
rng = np.random.default_rng(0)
normal_rps = rng.normal(loc=500, scale=30, size=360)
ddos_rps = np.array([3000, 4500, 7200, 8000, 6000, 5500, 4000, 3500])
combined = np.concatenate([normal_rps, ddos_rps])

# Macierz cech (1D: RPS)
X = combined.reshape(-1, 1)

# Model Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=0)
labels_iso = iso.fit_predict(X) # 1 = normalny, -1 = anomalia

# Indeksy i wartości anomalii
anomaly_indices_iso = np.where(labels_iso == -1)[0]
anomaly_values_iso = combined[anomaly_indices_iso]
print("Indeksy anomalii (Isolation Forest):", anomaly_indices_iso)
print("Wartości anomalii (Isolation Forest):", anomaly_values_iso)

# Wykres sygnału RPS z zaznaczonymi anomaliami
indices = np.arange(len(combined))
plt.figure(figsize=(10,4))
plt.plot(indices, combined, label='RPS')
if anomaly_indices_iso.size > 0:
    plt.scatter(anomaly_indices_iso, anomaly_values_iso, color='red', marker='x', s=80, label='Anomalie')
plt.xlabel('Czas (indeks)')
plt.ylabel('RPS')
plt.title('Isolation Forest - RPS z oznaczonymi anomaliami')
plt.legend()
plt.tight_layout()
out_path = Path(__file__).with_name('isolation_forest_rps.png')
plt.savefig(out_path)
plt.close()
print(f"Zapisano wykres do: {out_path}")
