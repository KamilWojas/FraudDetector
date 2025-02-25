import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Pobierz pełną ścieżkę do pliku
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/creditcard.csv")

def load_data():
    """Wczytuje dane z pliku CSV."""
    df = pd.read_csv(DATA_PATH)
    return df

def explore_data(df):
    """Podstawowa eksploracja danych."""
    print("📊 Podstawowe statystyki zbioru danych:")
    print(df.describe())

    print("\n🔍 Liczba oszukańczych transakcji vs. uczciwych:")
    print(df["Class"].value_counts())

    # Wykres liczby oszukańczych vs. uczciwych transakcji
    plt.figure(figsize=(6,4))
    sns.countplot(x=df["Class"], palette="coolwarm")
    plt.title("Liczba oszukańczych (1) vs. uczciwych (0) transakcji")
    plt.xlabel("Klasa")
    plt.ylabel("Liczba transakcji")
    plt.show()

    # Histogram wartości transakcji
    plt.figure(figsize=(6,4))
    sns.histplot(df["Amount"], bins=50, kde=True)
    plt.title("Rozkład wartości transakcji")
    plt.xlabel("Kwota transakcji (Amount)")
    plt.ylabel("Liczba transakcji")
    plt.show()

    # Histogram czasu transakcji
    plt.figure(figsize=(6,4))
    sns.histplot(df["Time"], bins=50, kde=True)
    plt.title("Rozkład czasu transakcji")
    plt.xlabel("Czas")
    plt.ylabel("Liczba transakcji")
    plt.show()

    # Macierz korelacji
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Macierz korelacji cech")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    explore_data(df)

