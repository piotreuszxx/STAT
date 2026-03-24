import numpy as np
from sklearn.ensemble import IsolationForest

# Dane: normalny ruch + atak DDoS
rng = np.random.default_rng(0)
normal_rps = rng.normal(loc=500, scale=30, size=360)
ddos_rps = np.array([3000, 4500, 7200, 8000, 6000, 5500, 4000,
3500])
combined = np.concatenate([normal_rps, ddos_rps])

# Macierz cech (1D: RPS)
X = combined.reshape(-1, 1)

# Model Isolation Forest
iso = IsolationForest(contamination=0.2, random_state=0)
labels_iso = iso.fit_predict(X) # 1 = normalny, -1 = anomalia

# Indeksy i wartości anomalii
anomaly_indices_iso = np.where(labels_iso == -1)[0]
anomaly_values_iso = combined[anomaly_indices_iso]
print("Indeksy anomalii (Isolation Forest):",
anomaly_indices_iso)
print("Wartości anomalii (Isolation Forest):",
anomaly_values_iso)