import pandas as pd
import os

from imblearn.over_sampling import SMOTE
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

    # Normalizacja 'Amount'
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # Przekształcenie 'Time' na godziny (opcjonalnie)
    df["Time"] = (df["Time"] % (24 * 3600)) / 3600

    # Podział na cechy (X) i etykiety (y)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Podział na zbiór treningowy i testowy (80% - trening, 20% - test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"🔹 Przed SMOTE: Fraudów w zbiorze treningowym: {sum(y_train)} na {len(y_train)} transakcji.")

    # Oversampling (SMOTE)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)  # 50% fraudów względem normalnych transakcji
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(
        f"✅ Po SMOTE: Fraudów w zbiorze treningowym: {sum(y_train_resampled)} na {len(y_train_resampled)} transakcji.")

    return X_train_resampled, X_test, y_train_resampled, y_test


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)


