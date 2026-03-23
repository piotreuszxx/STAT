import numpy as np
import matplotlib.pyplot as plt

# 1. Generowanie danych: normalny ruch i atak DDoS
rng = np.random.default_rng(0)
normal_rps = rng.normal(loc=500, scale=30, size=360) # 6 minut po 60 sek
ddos_rps = np.array([3000, 4500, 7200, 8000, 6000, 5500, 4000, 3500]) # atak
combined = np.concatenate([normal_rps, ddos_rps])

# 2. Funkcja bootstrap CI dla średniej
def bootstrap_ci(x, stat_func=np.mean, n_bootstrap=5000, alpha=0.05):
    x = np.asarray(x)
    n = len(x)
    rng = np.random.default_rng(1)
    stats = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = x[idx]
        stats[i] = stat_func(sample)

    lower = np.quantile(stats, alpha/2)
    upper = np.quantile(stats, 1-alpha/2)
    return lower, upper, stats
lower, upper, stats_norm = bootstrap_ci(normal_rps)
print("Górna granica 95% CI dla średniej normalnego ruchu:", upper)

# 3. Wykrywanie anomalii (wartości > górna granica CI)
for i, r in enumerate(combined):
    if r > upper:
        print(f"[ANOMALIA] obserwacja {i+1}: wartość={r}")
    else:
        print(f"OK: {r}")

# 4a. Wykres czasowy ruchu
plt.figure(figsize=(10,4))
plt.plot(combined, label="RPS")
plt.axhline(upper, color="red", linestyle="--", label="górna granica CI")
plt.xlabel("Czas (sekundy)")
plt.ylabel("RPS")
plt.title("Ruch sieciowy – normalny i atak DDoS")
plt.legend()
plt.tight_layout()
plt.show()

# 4b. Histogram rozkładu bootstrapowej średniej
plt.figure(figsize=(6,4))
plt.hist(stats_norm, bins=40, edgecolor="black")
plt.axvline(upper, color="red", linestyle="--", label="górna granica CI")
plt.xlabel("Średnia bootstrapowa")
plt.ylabel("Liczba wystąpień")
plt.title("Rozkład bootstrapowy średniej ruchu normalnego")
plt.legend()
plt.tight_layout()
plt.show()