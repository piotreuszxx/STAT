import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Wczytanie danych
tips = sns.load_dataset("tips")

# Przygotowanie zmiennych
y = tips["tip"]
X = tips["total_bill"]
X = sm.add_constant(X) #

# Dopasowanie modelu
model = sm.OLS(y, X).fit()
print(model.summary())

# Wizualizacja
plt.scatter(tips["total_bill"], tips["tip"], alpha=0.5,
label="Dane")
plt.plot(tips["total_bill"], model.predict(X), color="red",
label="Linia regresji")
plt.xlabel("Total bill ($)")
plt.ylabel("Tip ($)")
plt.title("Regresja liniowa: napiwek vs rachunek")
plt.legend()
plt.show()