import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from data_preparation import load_data, preprocess_data

# Pobranie danych
df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

# Tworzenie i trenowanie modelu
print("🔄 Trenowanie modelu Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)

# Ewaluacja modelu
print("\n📊 Wyniki modelu:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔍 Macierz błędów:")
print(confusion_matrix(y_test, y_pred))
print("\n📄 Raport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Zapisanie modelu do pliku
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/random_forest.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model zapisany jako {MODEL_PATH}")

