"""
=============================================================================
 TRAINING MODEL - ASL Sign Language Detection
=============================================================================
 Script ini digunakan untuk melatih model klasifikasi dari data landmark
 yang sudah direkam menggunakan collect_data.py.

 CARA PAKAI:
 1. Pastikan data sudah direkam di folder 'data/'
 2. Jalankan:  .\venv\Scripts\python.exe backend\train_model.py
 3. Model akan disimpan di 'models/asl_model.pkl'

 MODEL:
 - Random Forest Classifier (akurat & mudah untuk data kecil)
 - Input: 63 angka (21 landmark x 3 koordinat x,y,z)
 - Output: Salah satu dari 10 kata ASL
=============================================================================
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# KONFIGURASI
# ============================================================

# Daftar kata (harus sama urutannya dengan collect_data.py)
WORDS = ["DON'T", "HOLD", "ONTO", "WHAT", "NOT", "YOURS", "JUST", "LET", "THINGS", "BE"]

# Path folder
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.pkl")

# ============================================================
# 1. BACA SEMUA DATA
# ============================================================

def load_data():
    """Membaca semua file .npy dari folder data/ dan mengembalikan X, y."""
    X = []  # Fitur (63 angka landmark)
    y = []  # Label (nama kata)

    # Cek apakah folder data ada
    if not os.path.exists(DATA_DIR):
        print("[ERROR] Folder 'data/' tidak ditemukan!")
        print("   Jalankan collect_data.py dulu untuk merekam data.")
        return None, None

    for word in WORDS:
        word_dir = os.path.join(DATA_DIR, word)

        # Cek apakah folder kata ada
        if not os.path.exists(word_dir):
            print(f"[WARN] Folder '{word}' tidak ditemukan, dilewati.")
            continue

        # Baca semua file .npy di folder kata
        sample_files = [f for f in os.listdir(word_dir) if f.endswith(".npy")]
        for sample_file in sample_files:
            file_path = os.path.join(word_dir, sample_file)
            landmarks = np.load(file_path)

            # Validasi: pastikan data 63 angka
            if landmarks.shape == (63,):
                X.append(landmarks)
                y.append(word)
            else:
                print(f"[WARN] Skip data tidak valid: {file_path} (shape: {landmarks.shape})")

    return np.array(X), np.array(y)

# ============================================================
# 2. TRAINING
# ============================================================

def train_model(X, y):
    """Melatih Random Forest dan menampilkan evaluasi."""
    print(f"\n[INFO] Total data: {len(X)} sampel, {len(set(y))} kelas\n")

    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Training: {len(X_train)} sampel")
    print(f"   Testing : {len(X_test)} sampel")

    # Random Forest - model yang bagus untuk data landmark
    print("\n[INFO] Melatih model Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,     # Jumlah pohon keputusan
        max_depth=20,         # Kedalaman maksimal pohon
        random_state=42,
        n_jobs=-1             # Pakai semua core CPU
    )
    model.fit(X_train, y_train)

    # ============================================================
    # 3. EVALUASI
    # ============================================================

    # Prediksi data testing
    y_pred = model.predict(X_test)

    # Akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[OK] Akurasi Model: {accuracy * 100:.2f}%")

    # Laporan detail per kelas
    print("\n[INFO] Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Matriks kebingungan (confusion matrix)
    print("[INFO] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ============================================================
    # 4. SIMPAN MODEL
    # ============================================================

    # Buat folder models jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Simpan model dan daftar kata
    joblib.dump({
        'model': model,
        'words': WORDS
    }, MODEL_PATH, compress=3)

    print(f"\n[OK] Model tersimpan di: {MODEL_PATH}")
    print(f"   Akurasi: {accuracy * 100:.2f}%")

    # ============================================================
    # 5. IMPORTANCE FITUR (informasi tambahan)
    # ============================================================
    importances = model.feature_importances_
    # Fitur 0-20 = x, 21-41 = y, 42-62 = z
    x_importance = importances[0:21].sum()
    y_importance = importances[21:42].sum()
    z_importance = importances[42:63].sum()

    print("\n[INFO] Kontribusi Koordinat:")
    print(f"   X (horizontal): {x_importance * 100:.1f}%")
    print(f"   Y (vertikal)  : {y_importance * 100:.1f}%")
    print(f"   Z (kedalaman) : {z_importance * 100:.1f}%")

    return accuracy

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  [TRAIN] TRAINING MODEL - ASL Sign Language Detection")
    print("=" * 60)

    # 1. Baca data
    X, y = load_data()

    # Cek apakah ada data
    if X is None or len(X) == 0:
        print("\n[ERROR] Tidak ada data untuk dilatih!")
        print("   Jalankan backend/collect_data.py terlebih dahulu.")
        exit(1)

    # 2. Latih model
    train_model(X, y)

    print("\n[OK] Training selesai! Model siap digunakan.\n")
