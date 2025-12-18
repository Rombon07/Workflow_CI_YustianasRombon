import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import os
import shutil

# --- 1. KONFIGURASI DAGSHUB ---
DAGSHUB_USERNAME = 'yustianasrombon7'
DAGSHUB_REPO_NAME = 'Eksperimen_SML_YustianasRombon'

# Cek Environment (CI/CD vs Lokal)
if not os.getenv("MLFLOW_TRACKING_URI"):
    print("Running Lokal: Menginisialisasi DagsHub secara interaktif...")
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
else:
    print("Running di CI/CD: Menggunakan Environment Variables.")

def main():
    # --- 2. Load Data ---
    print("Memuat data...")
    try:
        df = pd.read_csv('water_potability_clean.csv')
    except FileNotFoundError:
        df = pd.read_csv('./water_potability_clean.csv') 
    
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Training & Hyperparameters ---
    print("Training Random Forest...")
    
    # Simpan parameter dalam dictionary agar bisa di-log ke DagsHub
    params = {
        "n_estimators": 50,
        "max_depth": 20,
        "min_samples_split": 5,
        "random_state": 42
    }
    
    rf = RandomForestClassifier(**params)
    
    # --- LOG PARAMETERS (AGAR MUNCUL DI DASHBOARD DAGSHUB) ---
    # Ini akan mengisi kolom max_depth, n_estimators, dll di UI DagsHub
    mlflow.log_params(params)
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # --- 4. Hitung Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    
    # --- 5. Log Metrics ke MLflow ---
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })
    
    # Log model ke Tracking Server
    mlflow.sklearn.log_model(rf, "model")
    
    # --- 6. Log Artefak Gambar ---
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    # --- 7. Save Lokal untuk Docker (Kriteria 3 Advance) ---
    # Menyimpan di root repository agar bisa dibaca oleh workflow CI
    LOCAL_MODEL_PATH = "../model_output"
    
    if os.path.exists(LOCAL_MODEL_PATH):
        shutil.rmtree(LOCAL_MODEL_PATH)
        
    print(f"Menyimpan model ke folder lokal '{LOCAL_MODEL_PATH}' untuk Docker...")
    mlflow.sklearn.save_model(rf, LOCAL_MODEL_PATH)
    
    print("Selesai.")

if __name__ == "__main__":
    main()