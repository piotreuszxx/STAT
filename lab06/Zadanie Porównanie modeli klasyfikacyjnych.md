## Zadanie: Porównanie modeli klasyfikacyjnych i ocena niepewności estymacji 

## **Cel ćwiczenia** 

Celem ćwiczenia jest porównanie jakości kilku modeli klasyfikacyjnych oraz ocena wpływu niepewności estymacji na wyniki predykcyjne. Użyj walidację krzyżową do obliczenia średniej dokładności oraz odchylenia standardowego wyników, a następnie wybierze najlepszy model na podstawie uzyskanych rezultatów. 

## **Ogólne omówienie** 

Dla zbioru danych z repozytorium UCI Machine Learning Repository Breast Cancer 

```
from sklearn.datasetsimport load_breast_cancer
data = load_breast_cancer()
X =data.data
y =data.target
```

Zbuduj cztery różne modele klasyfikacyjne: **Logistic Regression, Random Forest, SVM i KNN** . 

Do oceny ich jakości zastosuj  metodę walidacji krzyżowej **k-fold (k=5)** , która polega na wielokrotnym podziale danych na zbiór treningowy i testowy. Dla każdego modelu oblicz średnią dokładność (accuracy), która określa ogólną jakość predykcji, oraz odchylenie standardowe, które informuje o stabilności modelu i niepewności estymacji wyników. Model o najwyższej średniej dokładności i najmniejszej zmienności wyniku jest uznawany za najlepszy. 

## **Rozwiązanie ma zawierać** : 
1. Przedstaw kod w języku Python, 

2. Przedstaw wyniki w postaci wycinka ekranu działającego programu, 

3. Wnioski, udziel odpowiedzi: 
o Co oznacza, że zbiór danych jest „dobrze rozdzielny” w kontekście klasyfikacji?
o Czy model o wysokiej średniej accuracy, ale bardzo dużym odchyleniu standardowym można uznać za stabilny? Dlaczego tak / dlaczego nie?
o Co mierzy odchylenie standardowe wyników w cross-validation i jak interpretujesz niską wartość tej miary?
o Co rozumiemy przez „niepewność estymacji” w kontekście oceny jakości modelu?
o Podaj przykład sytuacji, w której dwa modele mają podobną średnią accuracy, ale różną niepewność estymacji. Który model wybrałbyś i dlaczego?

## **Opis zbioru danych – Breast Cancer** 

Zbiór **Breast Cancer Wisconsin (Diagnostic)** pochodzi z **UCI Machine Learning Repository** , jednego z najpopularniejszych repozytoriów danych do badań nad uczeniem maszynowym. Dane pochodzą z badań nad rozpoznawaniem nowotworu piersi na podstawie cyfrowego obrazu biopsji . 

## **Cel zbioru** 

Celem datasetu jest klasyfikacja wykrytego guza jako: 

- **benign (B)** – łagodny, 

- **malignant (M)** – złośliwy. 

Zadanie polega na stworzeniu modelu predykcyjnego, który potrafi automatycznie ocenić charakter zmiany na podstawie zmierzonych cech struktury komórek. 

|**PARAMETR**<br>**WARTOŚĆ**|**PARAMETR**<br>**WARTOŚĆ**|
|---|---|
|LICZBA PRÓBEK|569|
|LICZBA CECH (FEATURES)|30 wartości numerycznych|



|LICZBA KLAS|2 (Malignant / Benign)<br>|
|---|---|
|FORMAT DANYCH|wartości liczbowe typu foat|
|BRAKUJĄCE DANE|brak missing values|
|ŹRÓDŁO DANYCH|Szpital Uniwersytetu Wisconsin, Madison|



## **Opis cech** 

Każda próbka opisuje właściwości jądra komórkowego wykryte na obrazie mikroskopowym. Dla 10 podstawowych parametrów obliczono trzy typy statystyk: 

- mean – średnia z wielokrotnych pomiarów, 

- se – błąd standardowy (zmienność między komórkami), 

- worst – największa wartość (najbardziej skrajny pomiar). 

To właśnie te cechy umożliwiają modelom ML rozróżnienie nowotworów łagodnych od złośliwych. 

## **Podstawowe parametry morfologiczne** 

- promień ( _radius_ ) – średnia odległość od centrum do granicy komórki 

- tekstura ( _texture_ ) – zróżnicowanie intensywności pikseli 

- obwód ( _perimeter_ ) 

- powierzchnia ( _area_ ) 

- wygładzenie ( _smoothness_ ) 

- zwartość ( _compactness_ ) 

- wklęsłość ( _concavity_ ) 

- liczba wklęsłych fragmentów ( _concave points_ ) 

- symetria 

- fraktalność 

**==> picture [383 x 239] intentionally omitted <==**

**----- Start of picture text -----**<br>
PCA 2D - Breast Cancer Dataset<br>800<br>x * malignant<br>x benign<br>600<br>400<br>N  %<br>2 x<br>=Q 200 . x aexenx x 7<br>SE 2ie Bh ys oO% xx Seex at x x x<br>3V 0 IOs 2dOROe agRES8 Box xRXX__x<br>= BRxx Rex iXxxXxx = x x x<br>= -200 a or x :<br>a S x «x<br>x x<br>—400 x<br>x<br>—600<br>x<br>—1000 0 1000 2000 3000 4000<br>Principal Component 1<br>**----- End of picture text -----**<br>


