import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Pobranie pełnej ścieżki do pliku
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/creditcard.csv")


def load_data():
    """Wczytuje dane z pliku CSV."""
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):
    """Normalizacja danych i podział na zbiór treningowy i testowy."""

    # Normalizacja kolumny 'Amount'
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # Przekształcenie 'Time' na godziny (opcjonalnie)
    df["Time"] = (df["Time"] % (24 * 3600)) / 3600  # Zamiana na godziny

    # Podział na cechy (X) i etykiety (y)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Podział na zbiór treningowy i testowy (80% - trening, 20% - test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"🔹 Rozmiar zbioru treningowego: {len(X_train)}")
    print(f"🔹 Rozmiar zbioru testowego: {len(X_test)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)


