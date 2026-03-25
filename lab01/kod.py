"""
Zadanie: Klasteryzacja w Pythonie (K-Means + ocena jakości +
wizualizacje)
Wymagania:
- dane: make_blobs lub CSV
- standaryzacja
- K-Means dla K=2..10
- miary: Silhouette, Davies–Bouldin, Calinski–Harabasz
- wybór najlepszego K (np. po Silhouette)
- wizualizacja: klastry + centroidy, Silhouette plot, Silhouette vs
K
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
# -------------------------
# 1) Dane
# -------------------------
def load_data(use_csv: bool = False, csv_path: str = "dane.csv") -> np.ndarray:
    """
    Zwraca macierz cech X o wymiarach (n_samples, n_features).
    Jeśli use_csv=True, wczytaj dane z pliku CSV (min. 2 kolumny
    numeryczne).
    Jeśli use_csv=False, wygeneruj dane make_blobs.
    """
    if use_csv:
        # TODO: wczytaj CSV (np. pandas), wybierz kolumny numeryczne i zwróć jako numpy array

        # podpowiedź:
        # import pandas as pd
        # df = pd.read_csv(csv_path)
        # X = df[["col1","col2"]].to_numpy()
        raise NotImplementedError("TODO: wczytanie danych z CSV")
    else:
        # TODO: zmień parametry tak, aby było min. 500 próbek i 3-5 centrów
        X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.2, random_state=42)
        return X
    
# -------------------------
# 2) Standaryzacja
# -------------------------
def standardize(X: np.ndarray) -> np.ndarray:
    # TODO: użyj StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# -------------------------
# 3) Ewaluacja dla wielu K
# -------------------------
def evaluate_kmeans(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10):
    """
    Dla K w [k_min..k_max] trenuje KMeans i liczy:
    - silhouette_score (im wyższy, tym lepiej)
    - davies_bouldin_score (im niższy, tym lepiej)
    - calinski_harabasz_score (im wyższy, tym lepiej)
    Zwraca listę wyników oraz najlepsze K wg silhouette.
    """
    results = []
    
    best_k = None
    best_sil = -1.0

    for k in range(k_min, k_max + 1):
        # TODO: uruchom KMeans
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        # TODO: policz miary jakości
        sil = silhouette_score(X_scaled, labels)
        dbi = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        results.append(
            {
            "K": k,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "calinski_harabasz": ch,
            "model": model,
            "labels": labels,
            }
        )
        if sil > best_sil:
            best_sil = sil
            best_k = k
    return results, best_k

# -------------------------
# 4) Wykres: Silhouette vs K
# -------------------------
def plot_silhouette_vs_k(results):
    ks = [r["K"] for r in results]
    sils = [r["silhouette"] for r in results]
    plt.figure(figsize=(7, 4))
    plt.plot(ks, sils, marker="o")
    plt.title("Silhouette vs K")
    plt.xlabel("Liczba klastrów K")
    plt.ylabel("Średni Silhouette")
    plt.grid(True, alpha=0.3)
    plt.show()

# -------------------------
# 5) Wykres: klastry + centroidy
# -------------------------
def plot_clusters_2d(X_scaled, labels, centroids, title="K-Means klasteryzacja"):
    """
    Zakładamy dane 2D (dwie cechy). Jeśli masz więcej cech, zrób PCA do 2D
    (opcjonalnie).
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200)
    plt.title(title)
    plt.xlabel("Cecha 1 (standaryzowana)")
    plt.ylabel("Cecha 2 (standaryzowana)")
    plt.show()

# -------------------------
# 6) Wykres: Silhouette plot dla najlepszego K
# -------------------------
def plot_silhouette_diagram(X_scaled, labels, k, sil_avg):
    sil_values = silhouette_samples(X_scaled, labels)

    plt.figure(figsize=(7, 5))
    y_lower = 10

    for cluster_id in range(k):
        vals = sil_values[labels == cluster_id]
        vals.sort()
        size = vals.shape[0]
        y_upper = y_lower + size
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.8)
        plt.text(-0.05, y_lower + 0.5 * size, str(cluster_id))
        y_lower = y_upper + 10

    plt.axvline(sil_avg, linestyle="--")
    plt.title(f"Wykres Silhouette dla K={k} (średnia={sil_avg:.3f})")
    plt.xlabel("Wartość silhouette s(i)")
    plt.yticks([])
    plt.show()

# -------------------------
# 7) Main
# -------------------------
def main():
    # TODO: jeśli chcesz, ustaw use_csv=True i podaj csv_path
    X = load_data(use_csv=False, csv_path="dane.csv")
    X_scaled = standardize(X)
    results, best_k = evaluate_kmeans(X_scaled, k_min=2, k_max=10)
    # wypisz tabelkę wyników
    print("Wyniki dla różnych K:")
    for r in results:
        print(
        f"K={r['K']} | Silhouette={r['silhouette']:.3f} | "
        f"DBI={r['davies_bouldin']:.3f} | CH={r['calinski_harabasz']:.1f}"
        )

    print(f"\nNajlepsze K wg silhouette: {best_k}")

    # wybierz rekord dla best_k
    best = next(r for r in results if r["K"] == best_k)
    best_labels = best["labels"]
    best_model = best["model"]
    best_sil = best["silhouette"]

    # wykresy
    plot_silhouette_vs_k(results)
    plot_clusters_2d(
        X_scaled,
        best_labels,
        best_model.cluster_centers_,
        title=f"K-Means (K={best_k}) | Silhouette={best_sil:.3f}",
    )
    plot_silhouette_diagram(X_scaled, best_labels, best_k, best_sil)

    # TODO: dopisz 3–5 zdań interpretacji na podstawie miar i wykresów
    # np. czy są wartości ujemne, czy klastry są rozdzielone itd.

if __name__ == "__main__":
    main()      