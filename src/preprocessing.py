import pandas as pd
import os

# Pobierz pełną ścieżkę do pliku
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/raw/creditcard.csv")


def load_data():
    """Wczytuje dane z pliku CSV."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Plik nie istnieje: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Załadowano {len(df)} rekordów.")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())



