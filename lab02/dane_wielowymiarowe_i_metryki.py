import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from pathlib import Path
import csv

rng = np.random.default_rng(0)

# Funkcja budująca dane na podstawie parametrów
def build_dataset(n_normal=500, n_attack=50, attack_avg_bytes_loc=5000, attack_error_low=0.1, attack_error_high=0.5):
    # normal
    rps_norm = rng.normal(loc=500, scale=50, size=n_normal)
    unique_ips_norm = rng.integers(20, 50, size=n_normal)
    avg_bytes_norm = rng.normal(loc=15000, scale=2000, size=n_normal)
    error_rate_norm = rng.uniform(0.0, 0.03, size=n_normal)

    # attack
    rps_att = rng.normal(loc=5000, scale=1000, size=n_attack)
    unique_ips_att = rng.integers(100, 500, size=n_attack)
    avg_bytes_att = rng.normal(loc=attack_avg_bytes_loc, scale=1000, size=n_attack)
    error_rate_att = rng.uniform(attack_error_low, attack_error_high, size=n_attack)

    X_norm = np.column_stack([rps_norm, unique_ips_norm, avg_bytes_norm, error_rate_norm])
    X_att = np.column_stack([rps_att, unique_ips_att, avg_bytes_att, error_rate_att])
    X = np.vstack([X_norm, X_att])
    y = np.hstack([np.zeros(n_normal), np.ones(n_attack)])
    return X, y

# Funkcja wykonująca pojedyncze uruchomienie modeli i zwracająca metryki
def eval_models(X, y_true, contamination=None):
    n_total = X.shape[0]
    n_attack = int(y_true.sum())
    if contamination is None:
        contamination = n_attack / n_total

    # Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=0)
    y_pred_iso = (iso.fit_predict(X) == -1).astype(int)

    # LOF
    # ensure n_neighbors < n_samples
    n_neighbors = min(20, max(2, X.shape[0] - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred_lof = (lof.fit_predict(X) == -1).astype(int)

    # OneClassSVM
    nu = contamination
    svm = OneClassSVM(gamma='scale', nu=nu)
    try:
        svm.fit(X)
        y_pred_svm = (svm.predict(X) == -1).astype(int)
    except Exception as e:
        # jeśli SVM się wykrzaczy, zwracamy None
        print('OneClassSVM failed:', e)
        y_pred_svm = None

    def scores_calc(y_true, y_pred):
        if y_pred is None:
            return (None, None, None)
        # If there are no positive predictions or true positives, precision/recall may warn; handle gracefully
        try:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            p, r, f1 = None, None, None
        return p, r, f1

    iso_scores = scores_calc(y_true, y_pred_iso)
    lof_scores = scores_calc(y_true, y_pred_lof)
    svm_scores = scores_calc(y_true, y_pred_svm)

    return {
        'iso_pred': y_pred_iso,
        'lof_pred': y_pred_lof,
        'svm_pred': y_pred_svm,
        'iso_scores': iso_scores,
        'lof_scores': lof_scores,
        'svm_scores': svm_scores,
        'contamination': contamination
    }

# Katalog wyników
out_dir = Path(__file__).with_name('dane_wielowymiarowe_results')
out_dir.mkdir(exist_ok=True)

# Lista eksperymentów: (name, n_normal, n_attack, attack_avg_bytes_loc, attack_error_low, attack_error_high)
experiments = [
    ('base', 500, 50, 5000, 0.1, 0.5),
    ('attack_avgbytes_similar', 500, 50, 15000, 0.02, 0.05),
    ('imbalance_stronger', 1000, 10, 5000, 0.1, 0.5),
]

# Zbiór wyników do zapisu CSV
csv_rows = []

for name, n_normal, n_attack, avg_bytes_loc, err_low, err_high in experiments:
    # bez skalowania
    X, y_true = build_dataset(n_normal=n_normal, n_attack=n_attack, attack_avg_bytes_loc=avg_bytes_loc, attack_error_low=err_low, attack_error_high=err_high)
    res = eval_models(X, y_true)
    iso_p, iso_r, iso_f1 = res['iso_scores']
    lof_p, lof_r, lof_f1 = res['lof_scores']
    svm_p, svm_r, svm_f1 = res['svm_scores']

    csv_rows.append({'experiment': name, 'scaled': False, 'n_normal': n_normal, 'n_attack': n_attack,
                     'model': 'IsolationForest', 'precision': iso_p, 'recall': iso_r, 'f1': iso_f1})
    csv_rows.append({'experiment': name, 'scaled': False, 'n_normal': n_normal, 'n_attack': n_attack,
                     'model': 'LOF', 'precision': lof_p, 'recall': lof_r, 'f1': lof_f1})
    csv_rows.append({'experiment': name, 'scaled': False, 'n_normal': n_normal, 'n_attack': n_attack,
                     'model': 'OneClassSVM', 'precision': svm_p, 'recall': svm_r, 'f1': svm_f1})

    # zapis wykresu porównawczego (metryk)
    labels = ['IsolationForest', 'LOF', 'OneClassSVM']
    f1s = [iso_f1 if iso_f1 is not None else 0, lof_f1 if lof_f1 is not None else 0, svm_f1 if svm_f1 is not None else 0]
    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, f1s, color=['purple','red','green'])
    plt.ylim(0,1)
    plt.title(f'F1 scores (no scaling) - {name}')
    for bar, val in zip(bars, f1s):
        plt.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(out_dir / f'f1_{name}_notscaled.png')
    plt.close()

    # ze skalowaniem
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    res_s = eval_models(X_scaled, y_true)
    iso_p_s, iso_r_s, iso_f1_s = res_s['iso_scores']
    lof_p_s, lof_r_s, lof_f1_s = res_s['lof_scores']
    svm_p_s, svm_r_s, svm_f1_s = res_s['svm_scores']

    csv_rows.append({'experiment': name, 'scaled': True, 'n_normal': n_normal, 'n_attack': n_attack,
                     'model': 'IsolationForest', 'precision': iso_p_s, 'recall': iso_r_s, 'f1': iso_f1_s})
    csv_rows.append({'experiment': name, 'scaled': True, 'n_normal': n_normal, 'n_attack': n_attack,
                     'model': 'LOF', 'precision': lof_p_s, 'recall': lof_r_s, 'f1': lof_f1_s})
    csv_rows.append({'experiment': name, 'scaled': True, 'n_normal': n_normal, 'n_attack': n_attack,
                     'model': 'OneClassSVM', 'precision': svm_p_s, 'recall': svm_r_s, 'f1': svm_f1_s})

    labels = ['IsolationForest', 'LOF', 'OneClassSVM']
    f1s_s = [iso_f1_s if iso_f1_s is not None else 0, lof_f1_s if lof_f1_s is not None else 0, svm_f1_s if svm_f1_s is not None else 0]
    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, f1s_s, color=['purple','red','green'])
    plt.ylim(0,1)
    plt.title(f'F1 scores (scaled) - {name}')
    for bar, val in zip(bars, f1s_s):
        plt.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(out_dir / f'f1_{name}_scaled.png')
    plt.close()

# Zapis CSV
csv_path = out_dir / 'dane_wielowymiarowe_metrics.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['experiment','scaled','n_normal','n_attack','model','precision','recall','f1']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)

print('Eksperymenty zakończone. Wyniki zapisane w:', out_dir)
