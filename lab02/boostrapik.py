import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Kompletny skrypt eksperymentalny dla bootstrap CI i detekcji anomalii

# Ustawienia globalne
RNG_SEED = 0
DEFAULT_N_BOOTSTRAP = 2000  # mniejsze domyślne dla szybszych eksperymentów
DEFAULT_N_BOOTSTRAP_ROLL = 500
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(RNG_SEED)

# Generowanie danych: normalny ruch i atak DDoS (domyślnie)
def generate_data(loc=500, scale=30, normal_size=360, attack_values=None, seed=RNG_SEED):
    rng_local = np.random.default_rng(seed)
    normal_rps = rng_local.normal(loc=loc, scale=scale, size=normal_size)
    if attack_values is None:
        attack_values = [3000, 4500, 7200, 8000, 6000, 5500, 4000, 3500]
    ddos_rps = np.array(attack_values)
    combined = np.concatenate([normal_rps, ddos_rps])
    return normal_rps, ddos_rps, combined


# Funkcja bootstrap CI (dla dowolnej statystyki)
def bootstrap_ci(x, stat_func=np.mean, n_bootstrap=DEFAULT_N_BOOTSTRAP, alpha=0.05, seed=1):
    x = np.asarray(x)
    n = len(x)
    rng_local = np.random.default_rng(seed)
    stats = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng_local.integers(0, n, size=n)
        sample = x[idx]
        stats[i] = stat_func(sample)

    lower = np.quantile(stats, alpha/2)
    upper = np.quantile(stats, 1 - alpha/2)
    return lower, upper, stats


# Detekcja anomalii na podstawie jednej (globalnej) górnej granicy CI
def detect_global_anomalies(normal_rps, combined, stat_func=np.mean, alpha=0.05, n_bootstrap=DEFAULT_N_BOOTSTRAP):
    lower, upper, stats = bootstrap_ci(normal_rps, stat_func=stat_func, n_bootstrap=n_bootstrap, alpha=alpha)
    anomalies_idx = [i for i, r in enumerate(combined) if r > upper]
    return lower, upper, stats, anomalies_idx


# Prosta detekcja z oknem (rolling window) — sprawdzamy ostatni punkt w oknie
def rolling_bootstrap_anomalies(signal, window_size=60, step=1, stat_func=np.mean, n_bootstrap=DEFAULT_N_BOOTSTRAP_ROLL, alpha=0.05):
    n = len(signal)
    anomalies = set()
    # Jeśli okno większe niż sygnał — zwróć pustą listę (lub ewentualnie zastosuj globalny próg)
    if window_size > n:
        return []

    for start in range(0, n - window_size + 1, step):
        window = signal[start : start + window_size]
        _, upper, _ = bootstrap_ci(window, stat_func=stat_func, n_bootstrap=n_bootstrap, alpha=alpha, seed= start + 7)
        idx_to_check = start + window_size - 1  # testujemy ostatni punkt okna
        if signal[idx_to_check] > upper:
            anomalies.add(idx_to_check)
    return sorted(anomalies)


# Funkcja pomocnicza do zapisu wyników do CSV
def append_results_csv(path, rows, header=None):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists and header is not None:
            writer.writerow(header)
        writer.writerows(rows)


# Główna funkcja wykonująca eksperymenty
def run_experiments():
    # Generowanie domyślnych danych
    normal_rps, ddos_rps, combined = generate_data()
    n_normal = len(normal_rps)

    # Parametry eksperymentu
    alphas = [0.05, 0.1, 0.01, 0.001]
    stat_funcs = [("mean", np.mean), ("median", np.median), ("p90", lambda x: np.percentile(x, 90))]

    summary_rows = []
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")

    print("=== Eksperyment: domyślne parametry ruchu ===")
    for name, func in stat_funcs:
        for alpha in alphas:
            lower, upper, stats = bootstrap_ci(normal_rps, stat_func=func, n_bootstrap=DEFAULT_N_BOOTSTRAP, alpha=alpha, seed=1)
            anomalies_global = [i for i, r in enumerate(combined) if r > upper]
            anomalies_in_attack = [i for i in anomalies_global if i >= n_normal]
            count_attack = len(anomalies_in_attack)

            confidence = (1 - alpha) * 100
            print(f"stat={name} alpha={alpha}: upper={upper:.2f}, anomalies_total={len(anomalies_global)}, anomalies_in_attack={count_attack}")

            # Histogram bootstrapowych statystyk
            fig_h = plt.figure(figsize=(6,4))
            plt.hist(stats, bins=40, edgecolor="black")
            plt.axvline(upper, color="red", linestyle="--", label=f"upper {confidence:.1f}% CI")
            plt.title(f"Bootstrapowy rozkład stat: {name} alpha={alpha}")
            plt.xlabel("Wartość statystyki")
            plt.legend()
            hist_path = os.path.join(RESULTS_DIR, f"hist_{name}_alpha{alpha}.png")
            fig_h.savefig(hist_path, dpi=150)
            plt.close(fig_h)

            # Wykres sygnału z zaznaczonymi anomaliami (global)
            fig = plt.figure(figsize=(10,4))
            plt.plot(combined, label="RPS")
            plt.scatter(anomalies_global, [combined[i] for i in anomalies_global], color="red", s=30, label="anomalie globalne")
            plt.axhline(upper, color="red", linestyle="--", label=f"górna granica CI ({confidence:.1f}%)")
            plt.xlabel("Czas (sekundy)")
            plt.ylabel("RPS")
            plt.title(f"Detekcja anomalii (global) stat={name} alpha={alpha}")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(RESULTS_DIR, f"signal_global_{name}_alpha{alpha}.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

            # Rolling window — porównanie
            rolling_anoms = rolling_bootstrap_anomalies(combined, window_size=60, step=1, stat_func=func, n_bootstrap=DEFAULT_N_BOOTSTRAP_ROLL, alpha=alpha)

            # Wykres porównawczy: global vs rolling
            fig2 = plt.figure(figsize=(10,4))
            plt.plot(combined, label="RPS")
            if anomalies_global:
                plt.scatter(anomalies_global, [combined[i] for i in anomalies_global], color="red", s=30, label="anomalie globalne")
            if rolling_anoms:
                plt.scatter(rolling_anoms, [combined[i] for i in rolling_anoms], color="orange", s=20, label="anomalie rolling")
            plt.axhline(upper, color="red", linestyle="--", label=f"górna granica CI ({confidence:.1f}%)")
            plt.xlabel("Czas (sekundy)")
            plt.ylabel("RPS")
            plt.title(f"Global vs Rolling stat={name} alpha={alpha}")
            plt.legend()
            plt.tight_layout()
            plot2_path = os.path.join(RESULTS_DIR, f"signal_comp_{name}_alpha{alpha}.png")
            fig2.savefig(plot2_path, dpi=150)
            plt.close(fig2)

            # Zapis do tabeli CSV
            summary_rows.append(["default", name, alpha, f"{upper:.2f}", len(anomalies_global), count_attack, len(rolling_anoms), hist_path, plot_path, plot2_path])

    append_results_csv(csv_path, summary_rows, header=["scenario", "stat", "alpha", "upper", "anomalies_total", "anomalies_in_attack", "anomalies_rolling", "hist", "global_plot", "comparison_plot"])

    # 5. Eksperyment z parametrami ruchu
    print("\n=== Eksperyment: zmienione parametry ruchu (loc=800, scale=100, atak mniejszy) ===")
    normal2, ddos2, combined2 = generate_data(loc=800, scale=100, normal_size=360, attack_values=[1500, 1800, 2200, 2500, 2000, 1700, 1600, 1900], seed=RNG_SEED)
    n_normal2 = len(normal2)
    summary_rows2 = []
    for name, func in stat_funcs:
        for alpha in alphas:
            lower2, upper2, stats2 = bootstrap_ci(normal2, stat_func=func, n_bootstrap=DEFAULT_N_BOOTSTRAP, alpha=alpha, seed=1)
            anomalies_global2 = [i for i, r in enumerate(combined2) if r > upper2]
            anomalies_in_attack2 = [i for i in anomalies_global2 if i >= n_normal2]
            count_attack2 = len(anomalies_in_attack2)
            confidence2 = (1 - alpha) * 100
            print(f"stat={name} alpha={alpha}: upper={upper2:.2f}, anomalies_total={len(anomalies_global2)}, anomalies_in_attack={count_attack2}")

            # Zapis wykresów
            fig = plt.figure(figsize=(10,4))
            plt.plot(combined2, label="RPS")
            plt.scatter(anomalies_global2, [combined2[i] for i in anomalies_global2], color="red", s=30, label="anomalie globalne")
            plt.axhline(upper2, color="red", linestyle="--", label=f"górna granica CI ({confidence2:.1f}%)")
            plt.xlabel("Czas (sekundy)")
            plt.ylabel("RPS")
            plt.title(f"(zmienione parametry) Detekcja anomalii stat={name} alpha={alpha}")
            plt.legend()
            plt.tight_layout()
            plot_path2 = os.path.join(RESULTS_DIR, f"signal_changed_{name}_alpha{alpha}.png")
            fig.savefig(plot_path2, dpi=150)
            plt.close(fig)

            # Rolling dla zmienionych danych
            rolling_anoms2 = rolling_bootstrap_anomalies(combined2, window_size=60, step=1, stat_func=func, n_bootstrap=DEFAULT_N_BOOTSTRAP_ROLL, alpha=alpha)

            summary_rows2.append(["changed", name, alpha, f"{upper2:.2f}", len(anomalies_global2), count_attack2, len(rolling_anoms2), plot_path2])

    append_results_csv(csv_path, summary_rows2)

    print(f"Wyniki zapisane do: {csv_path}")
    print(f"Wykresy i histogramy zapisane w katalogu: {RESULTS_DIR}")


if __name__ == "__main__":
    run_experiments()

