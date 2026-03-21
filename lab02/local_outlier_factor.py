import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from pathlib import Path

# Dane jak poprzednio
rng = np.random.default_rng(0)
normal_rps = rng.normal(loc=500, scale=30, size=360)
ddos_rps = np.array([3000, 4500, 7200, 8000, 6000, 5500, 4000, 3500])
combined = np.concatenate([normal_rps, ddos_rps])
X = combined.reshape(-1, 1)

# Model LOF
lof = LocalOutlierFactor(contamination=0.05)
labels_lof = lof.fit_predict(X) # 1 = normalny, -1 = anomalia
anomaly_indices_lof = np.where(labels_lof == -1)[0]
anomaly_values_lof = combined[anomaly_indices_lof]
print("Indeksy anomalii (LOF):", anomaly_indices_lof)
print("Wartości anomalii (LOF):", anomaly_values_lof)

# Wykres sygnału RPS z zaznaczonymi anomaliami
indices = np.arange(len(combined))
plt.figure(figsize=(10,4))
plt.plot(indices, combined, label='RPS')
if anomaly_indices_lof.size > 0:
    plt.scatter(anomaly_indices_lof, anomaly_values_lof, color='red', marker='x', s=80, label='Anomalie')
plt.xlabel('Czas (indeks)')
plt.ylabel('RPS')
plt.title('Local Outlier Factor - RPS z oznaczonymi anomaliami')
plt.legend()
plt.tight_layout()
out_path = Path(__file__).with_name('lof_rps.png')
plt.savefig(out_path)
plt.close()
print(f"Zapisano wykres do: {out_path}")
