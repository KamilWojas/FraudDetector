import pandas as pd

# Ścieżka do danych
DATA_PATH = "../data/raw/creditcard.csv"

def load_data():
    """Wczytuje dane z pliku CSV."""
    df = pd.read_csv(DATA_PATH)
    print(f"Załadowano {len(df)} rekordów.")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
