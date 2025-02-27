import joblib
import pandas as pd
import os
from fastapi import FastAPI
from pydantic import BaseModel
from data_preparation import preprocess_data

# Ścieżka do zapisanego modelu
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/random_forest.pkl")

# Wczytanie modelu
model = joblib.load(MODEL_PATH)

# Inicjalizacja FastAPI
app = FastAPI()


# Definicja formatu danych wejściowych
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def home():
    return {"message": "API do wykrywania fraudów działa!"}


@app.post("/predict/")
def predict(transaction: Transaction):
    """Przewiduje, czy transakcja jest fraudem (1) czy nie (0)."""
    data = pd.DataFrame([transaction.dict()])  # Konwersja wejścia do DataFrame
    prediction = model.predict(data)[0]  # Predykcja modelu
    probability = model.predict_proba(data)[0][1]  # Prawdopodobieństwo fraudu

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }

