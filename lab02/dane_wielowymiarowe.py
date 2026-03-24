import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
rng = np.random.default_rng(0)
n_normal = 500
n_attack = 50

# --- normal traffic ---
rps_norm = rng.normal(loc=500, scale=50, size=n_normal)
unique_ips_norm = rng.integers(20, 50, size=n_normal)
avg_bytes_norm = rng.normal(loc=15000, scale=2000, size=n_normal)
error_rate_norm = rng.uniform(0.0, 0.03, size=n_normal)

# --- DDoS traffic ---
rps_att = rng.normal(loc=5000, scale=1000, size=n_attack)
unique_ips_att = rng.integers(100, 500, size=n_attack)
avg_bytes_att = rng.normal(loc=5000, scale=1000, size=n_attack)
error_rate_att = rng.uniform(0.1, 0.5, size=n_attack)
X_norm = np.column_stack([rps_norm, unique_ips_norm, avg_bytes_norm, error_rate_norm])
X_att = np.column_stack([rps_att, unique_ips_att, avg_bytes_att, error_rate_att])
X = np.vstack([X_norm, X_att])
y_true = np.hstack([np.zeros(n_normal), np.ones(n_attack)])

# Isolation Forest
iso = IsolationForest(contamination=n_attack / (n_normal + n_attack), random_state=0)
y_pred_iso = (iso.fit_predict(X) == -1).astype(int)

# LOF
lof = LocalOutlierFactor(contamination=n_attack / (n_normal + n_attack))
y_pred_lof = (lof.fit_predict(X) == -1).astype(int)

def scores(name, y_true, y_pred):
    print(name)
    print("precision:", precision_score(y_true, y_pred))
    print("recall :", recall_score(y_true, y_pred))
    print("f1 :", f1_score(y_true, y_pred))
    print()

scores("Isolation Forest", y_true, y_pred_iso)
scores("LOF", y_true, y_pred_lof)