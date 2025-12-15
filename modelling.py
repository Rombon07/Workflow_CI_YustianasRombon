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

mlflow.set_experiment("Water Potability CI Pipeline")

def main():
    # --- 2. Load Data ---
    print("Memuat data...")
    try:
        df = pd.read_csv('water_potability_clean.csv')
    except FileNotFoundError:
        print("File tidak ditemukan, mencoba path absolut...")
        df = pd.read_csv(os.path.join(os.getcwd(), 'water_potability_clean.csv'))
    
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Training ---
    print("Training Random Forest...")
    # Menggunakan parameter yang lumayan oke (bisa diganti hasil tuning nanti)
    rf = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=20, min_samples_split=5)
    
    with mlflow.start_run(run_name="Complete_CI_Run") as run:
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # --- HITUNG METRICS (DIKEMBALIKAN) ---
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {f1}")
        
        # --- LOG METRICS KE MLFLOW ---
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        mlflow.sklearn.log_model(rf, "model")
        
        # --- LOG ARTEFAK GAMBAR (DIKEMBALIKAN) ---
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # 2. Feature Importance
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

        # --- SAVE LOKAL UNTUK DOCKER (WAJIB ADA) ---
        if os.path.exists("model_output"):
            shutil.rmtree("model_output")
            
        print("Menyimpan model ke folder lokal 'model_output' untuk Docker...")
        mlflow.sklearn.save_model(rf, "model_output")
        
        print("Selesai.")

if __name__ == "__main__":
    main()