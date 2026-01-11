# 🛡️ CI Workflow & Automation for ML

![Build Status](https://img.shields.io/github/actions/workflow/status/Rombon07/Workflow_CI_YustianasRombon/main.yml?style=for-the-badge&label=Build&logo=github)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Code Quality](https://img.shields.io/badge/Code%20Quality-A-success?style=for-the-badge)
![Testing](https://img.shields.io/badge/Testing-Pytest-orange?style=for-the-badge&logo=pytest&logoColor=white)

## 📋 Overview

Repositori ini berisi konfigurasi dan skrip **Continuous Integration (CI)** yang dirancang untuk mendukung pengembangan proyek Machine Learning. 

Tujuan utama dari workflow ini adalah menerapkan praktik **MLOps** yang baik dengan memastikan bahwa setiap perubahan kode (commit/push) pada eksperimen ML telah memenuhi standar kualitas, bebas dari *syntax error*, dan lulus pengujian logika dasar sebelum di-*merge*.

> *"Reliable Code leads to Reliable Models."*

---

## ⚙️ Architecture: The CI Pipeline

Workflow ini dibangun menggunakan **GitHub Actions** yang secara otomatis memicu serangkaian pemeriksaan setiap kali ada kode baru.

**Alur Pipeline:**
1.  **Trigger:** Event `push` atau `pull_request` ke branch `main`.
2.  **Environment Setup:** Menyiapkan lingkungan Ubuntu dengan Python versi tertentu.
3.  **Dependency Installation:** Menginstal *library* yang dibutuhkan (scikit-learn, pandas, pytest, flake8).
4.  **Linting (Quality Check):** Memeriksa gaya penulisan kode agar sesuai standar PEP8.
5.  **Unit Testing:** Menguji fungsi-fungsi krusial (seperti fungsi *preprocessing* dan *inverse transform*).

---

## 🛠️ Tech Stack & Tools

Kami menggunakan alat standar industri untuk menjaga integritas kode:

| Tools | Kategori | Kegunaan dalam Project |
| :--- | :--- | :--- |
| **GitHub Actions** | Orchestration | Menjalankan pipeline otomatis di cloud. |
| **Flake8** | Linter | Mendeteksi error sintaksis dan gaya kode yang buruk. |
| **Pytest** | Testing Framework | Menjalankan unit test untuk memvalidasi logika ML. |
| **Pip** | Package Manager | Manajemen dependensi proyek. |

---

## 🧪 Testing Strategy (MLOps Focus)

Bagian terpenting dari repositori ini adalah strategi pengujian untuk komponen Machine Learning.

### 1. Code Quality (Linting)
Menggunakan **Flake8** untuk memastikan tidak ada *unused imports*, variabel yang tidak terdefinisi, atau kode yang berantakan yang dapat menyulitkan *debugging* model di kemudian hari.

### 2. Unit Testing for ML Functions
Menggunakan **Pytest** untuk memvalidasi logika matematika. Contoh kasus uji (Test Case) yang ditangani:

* **Shape Integrity:** Memastikan output dari *train_test_split* memiliki dimensi yang benar.
* **Data Leakage Check:** Memastikan tidak ada irisan data antara Training set dan Test set.
* **Transformation Logic:** Memastikan fitur yang di-*scale* dapat dikembalikan ke nilai aslinya (**Inverse Transform Check**).

---

## 💡 Implementation Highlight: Validasi Inverse Transform

Salah satu fungsi kritis yang diuji dalam pipeline ini adalah validasi mekanisme `inverse_transform`.

**Contoh Logic Test (`tests/test_scaling.py`):**
Pipeline ini memastikan bahwa jika kita mengubah target $y$ (Rupiah/Harga) menjadi skala 0-1, kita harus bisa mengembalikannya ke Rupiah dengan akurat.

```python
def test_inverse_transform_accuracy():
    # 1. Setup Dummy Data
    original_data = np.array([[1000], [5000], [9000]])
    
    # 2. Scale Down
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(original_data)
    
    # 3. Inverse Back
    restored_data = scaler.inverse_transform(scaled_data)
    
    # 4. Assertion (Harus sama persis)
    np.testing.assert_array_almost_equal(original_data, restored_data, decimal=2)
