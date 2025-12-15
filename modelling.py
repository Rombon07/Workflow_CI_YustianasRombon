import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import os

# --- 1. KONFIGURASI DAGSHUB (Wajib untuk Advanced) ---
# Ganti dengan username dan nama repo DagsHub Anda
DAGSHUB_USERNAME = 'yustianasrombon7' 
DAGSHUB_REPO_NAME = 'Eksperimen_SML_YustianasRombon' 

print("Menghubungkan ke DagsHub...")
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
mlflow.set_experiment("Water Potability Experiment")

def main():
    # --- 2. Load Data ---
    print("Memuat data...")
    # Pastikan file csv sudah ada di folder yang sama
    df = pd.read_csv('water_potability_clean.csv')
    
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Hyperparameter Tuning (Syarat Skilled) ---
    print("Memulai Hyperparameter Tuning...")
    rf = RandomForestClassifier(random_state=42)
    
    # Grid sederhana untuk demo (biar tidak terlalu lama runningnya)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    
    # Mulai MLflow Run
    with mlflow.start_run(run_name="Hyperparameter_Tuning_RF"):
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best Params: {best_params}")
        
        # Prediksi
        y_pred = best_model.predict(X_test)
        
        # Hitung Metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {acc}")
        
        # --- 4. Logging ke MLflow (Manual Logging) ---
        # Log Parameter Terbaik
        mlflow.log_params(best_params)
        
        # Log Metrik
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "model")
        
        # --- 5. Custom Artifacts (Syarat Advanced - Wajib 2 Tambahan) ---
        
        # Artefak 1: Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png") # Simpan ke file dulu
        mlflow.log_artifact("confusion_matrix.png") # Upload ke DagsHub
        print("Artefak 1: Confusion Matrix terupload.")
        
        # Artefak 2: Feature Importance Plot
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X.columns
        
        plt.figure(figsize=(10,6))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), features[indices], rotation=90)
        plt.tight_layout()
        plt.savefig("feature_importance.png") # Simpan ke file dulu
        mlflow.log_artifact("feature_importance.png") # Upload ke DagsHub
        print("Artefak 2: Feature Importance terupload.")
        
        print("Selesai! Cek DagsHub Anda.")

if __name__ == "__main__":
    main()