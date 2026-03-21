# Projekt: Analityka anomalii w ruchu sieciowym z użyciem bootstrap i modeli ML

## Cel projektu

Celem projektu jest implementacja systemu wykrywania anomalii w danych ruchu sieciowego z wykorzystaniem metod statystycznych oraz modeli uczenia maszynowego.

Analiza powinna bazować na metodzie bootstrap do wyznaczania przedziałów ufności oraz modelach nienadzorowanej detekcji anomalii takich jak:

* Isolation Forest
* Local Outlier Factor (LOF)

System powinien analizować dane ruchu HTTP opisane przez liczbę zapytań na sekundę (RPS) oraz inne cechy ruchu sieciowego.

Projekt powinien zawierać eksperymenty z parametrami metod, wizualizacje wyników oraz porównanie skuteczności modeli.

---

# 1. Generowanie lub wczytanie danych

Stwórz dane symulujące ruch sieciowy:

### Normalny ruch

* średnia liczba zapytań: około 500 RPS
* odchylenie standardowe: 30–50

### Ruch ataku

* wartości RPS znacznie większe (np. 3000–8000)

Połącz dane w jeden sygnał czasowy:

```
combined = [normal_rps + attack_rps]
```

Zachowaj indeksy czasowe.

---

# 2. Bootstrapowy przedział ufności

Zaimplementuj funkcję:

```
bootstrap_ci(x, stat_func=np.mean, n_bootstrap=5000, alpha=0.05)
```

Funkcja powinna:

1. losować próbki bootstrapowe
2. liczyć statystykę
3. wyznaczać percentyle
4. zwracać:

   * dolną granicę CI
   * górną granicę CI
   * rozkład statystyk bootstrap

---

# 3. Eksperyment z poziomem istotności α

Oblicz bootstrapowy przedział ufności dla wartości:

```
alpha = 0.1
alpha = 0.01
alpha = 0.001
```

Dla każdego przypadku:

* zapisz górną granicę CI
* policz liczbę wykrytych anomalii

Wydrukuj tabelę wyników.

---

# 4. Zmiana statystyki bootstrap

Przetestuj trzy statystyki:

1. średnia
2. mediana
3. własna funkcja: 90 percentyl

Porównaj:

* wartości progów anomalii
* liczbę wykrytych punktów

---

# 5. Wykrywanie anomalii w całym sygnale

Zamiast analizować tylko fragment ataku, sprawdzaj cały sygnał:

```
for i, r in enumerate(combined):
```

Oznacz wartości większe niż górna granica CI jako anomalie.

Na wykresie zaznacz:

* sygnał RPS
* linię progu
* punkty anomalii

---

# 6. Okno czasowe (rolling window)

Dodaj analizę z użyciem okna czasowego.

Dla każdego okna:

1. oblicz bootstrap CI
2. wykryj anomalie

Porównaj wyniki:

* próg globalny
* próg lokalny (rolling window)

---

# 7. Eksperyment z parametrami ruchu

Zmodyfikuj parametry ruchu:

normalny ruch:

```
loc = 800
scale = 100
```

atak:

```
1500–3000 RPS
```

Sprawdź czy metoda nadal wykrywa anomalie.

---

# 8. Isolation Forest

Zbuduj model:

```
IsolationForest()
```

Testuj różne wartości:

```
contamination = 0.01
contamination = 0.05
contamination = 0.1
contamination = 0.2
```

Dla każdej wartości:

* policz liczbę wykrytych anomalii
* zapisz indeksy

---

# 9. Trenowanie tylko na ruchu normalnym

Model powinien zostać nauczony tylko na danych normalnych:

```
iso.fit(normal_rps.reshape(-1,1))
```

Następnie:

```
predict(combined)
```

Porównaj wyniki z trenowaniem na pełnym zbiorze.

---

# 10. Parametry strukturalne Isolation Forest

Przetestuj różne parametry:

```
n_estimators = 50, 100, 200
max_samples = 0.5, 1.0
```

Sprawdź stabilność wyników.

---

# 11. Local Outlier Factor

Przetestuj różne wartości:

```
n_neighbors = 5
n_neighbors = 10
n_neighbors = 20
n_neighbors = 50
```

Porównaj:

* listy wykrytych anomalii

---

# 12. Ensemble metod

Porównaj anomalie wykryte przez:

* Isolation Forest
* LOF

Oblicz:

```
set_if ∩ set_lof
```

Wypisz wspólne anomalie.

---

# 13. Dane wielowymiarowe

Rozszerz dane o dodatkowe cechy:

* RPS
* liczba unikalnych IP
* średni rozmiar odpowiedzi
* error rate

Zbuduj macierz cech:

```
X = [rps, unique_ips, avg_bytes, error_rate]
```

---

# 14. Metryki jakości

Oblicz:

* precision
* recall
* F1

dla modeli:

* Isolation Forest
* LOF

---

# 15. Standaryzacja cech

Zastosuj:

```
StandardScaler
```

Porównaj metryki:

* ze skalowaniem
* bez skalowania

---

# 16. Wizualizacja

Stwórz wykresy:

1. ruch sieciowy z anomaliami
2. histogram bootstrap
3. porównanie modeli
4. wykres cech wielowymiarowych
