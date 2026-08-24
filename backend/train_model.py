r"""
=============================================================================
 TRAINING MODEL v2 - ASL Sign Language Detection (2 Tangan + Motion)
=============================================================================
 Script ini melatih model klasifikasi dari data SEQUENCE gerakan tangan
 yang direkam menggunakan collect_data.py v2.

 PERBEDAAN DARI VERSI LAMA:
 - Input: sequence 30 frame x 126 nilai (2 tangan), bukan 1 frame statis
 - Fitur motion: mean + std + velocity -> model paham GERAKAN, bukan
   hanya pose diam

 CARA PAKAI:
 1. Pastikan data sudah direkam di folder 'data/' (format baru v2)
 2. Jalankan:  .\venv\Scripts\python.exe backend\train_model.py
 3. Model akan disimpan di 'models/asl_model.pkl'

 FITUR YANG DIEKSTRAK DARI SETIAP SEQUENCE:
 - Mean   : rata-rata posisi landmark selama gerakan (pose utama)
 - Std    : variasi posisi (seberapa stabil/berubah)
 - Velocity: kecepatan rata-rata antar frame (dinamika gerakan)
 Total: 378 angka per sampel
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

# Konfigurasi sequence (harus sama dengan collect_data.py)
SEQ_LEN = 30      # jumlah frame per sampel
FRAME_SIZE = 126  # 2 tangan x 21 titik x 3 koordinat

# ============================================================
# 1. BACA SEMUA DATA SEQUENCE
# ============================================================

def load_data():
    """
    Membaca semua file .npy dari folder data/.
    Setiap file berisi sequence bentuk (SEQ_LEN, FRAME_SIZE).

    Returns:
        X: array bentuk (jumlah_sampel, SEQ_LEN, FRAME_SIZE)
        y: array label kata
    """
    X = []
    y = []

    if not os.path.exists(DATA_DIR):
        print("[ERROR] Folder 'data/' tidak ditemukan!")
        print("   Jalankan collect_data.py dulu untuk merekam data.")
        return None, None

    for word in WORDS:
        word_dir = os.path.join(DATA_DIR, word)

        if not os.path.exists(word_dir):
            print(f"[WARN] Folder '{word}' tidak ditemukan, dilewati.")
            continue

        sample_files = [f for f in os.listdir(word_dir) if f.endswith(".npy")]
        valid_count = 0
        for sample_file in sample_files:
            file_path = os.path.join(word_dir, sample_file)
            seq = np.load(file_path)

            # Validasi format baru: (SEQ_LEN, FRAME_SIZE)
            if seq.shape == (SEQ_LEN, FRAME_SIZE):
                X.append(seq)
                y.append(word)
                valid_count += 1
            elif seq.shape == (63,):
                # Data lama (versi 1) - skip dengan pesan sekali saja
                print(f"[WARN] '{word}' punya data lama (format v1). "
                      f"Hapus folder 'data/{word}' dan rekam ulang!")
                break
            else:
                print(f"[WARN] Skip data tidak valid: {file_path} (shape: {seq.shape})")

        if valid_count > 0:
            print(f"  {word:8s}: {valid_count} sampel")

    if len(X) == 0:
        return None, None
    return np.array(X), np.array(y)

# ============================================================
# 2. EKSTRAKSI FITUR MOTION
# ============================================================

def extract_features(sequences):
    """
    Ubah sequence (N, SEQ_LEN, FRAME_SIZE) menjadi fitur (N, 378).

    Fitur per sequence:
    - mean     : rata-rata posisi tiap landmark selama gerakan
    - std      : standar deviasi posisi (variasi gerakan)
    - velocity : rata-rata perubahan antar frame (kecepatan gerakan)
    """
    # Mean & std sepanjang sumbu waktu
    feat_mean = sequences.mean(axis=1)          # (N, 126)
    feat_std = sequences.std(axis=1)            # (N, 126)

    # Velocity: selisih antar frame berturut-turut, lalu dirata-rata
    velocities = np.diff(sequences, axis=1)     # (N, SEQ_LEN-1, 126)
    feat_vel = velocities.mean(axis=1)          # (N, 126)

    # Gabungkan semua fitur
    features = np.concatenate([feat_mean, feat_std, feat_vel], axis=1)  # (N, 378)

    return features

# ============================================================
# 3. TRAINING
# ============================================================

def train_model(X_seq, y):
    """Melatih Random Forest dari fitur motion dan menampilkan evaluasi."""
    print(f"\n[INFO] Total data: {len(X_seq)} sampel, {len(set(y))} kelas")

    # Ekstrak fitur motion
    print("[INFO] Ekstraksi fitur motion (mean + std + velocity)...")
    X = extract_features(X_seq)
    print(f"[INFO] Bentuk fitur: {X.shape}")

    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Training: {len(X_train)} sampel")
    print(f"   Testing : {len(X_test)} sampel")

    # Random Forest
    print("\n[INFO] Melatih model Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,     # Lebih banyak pohon untuk fitur lebih kompleks
        max_depth=25,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ============================================================
    # 4. EVALUASI
    # ============================================================

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[OK] Akurasi Model: {accuracy * 100:.2f}%")

    print("\n[INFO] Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("[INFO] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ============================================================
    # 5. SIMPAN MODEL
    # ============================================================

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump({
        'model': model,
        'words': WORDS,
        'seq_len': SEQ_LEN,
        'frame_size': FRAME_SIZE,
        'version': 2
    }, MODEL_PATH, compress=3)

    print(f"\n[OK] Model tersimpan di: {MODEL_PATH}")
    print(f"   Akurasi: {accuracy * 100:.2f}%")

    return accuracy

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  [TRAIN] TRAINING MODEL v2 - ASL Detection (2 Tangan + Motion)")
    print("=" * 60)

    # 1. Baca data
    X_seq, y = load_data()

    if X_seq is None or len(X_seq) == 0:
        print("\n[ERROR] Tidak ada data untuk dilatih!")
        print("   Jalankan backend/collect_data.py terlebih dahulu.")
        exit(1)

    # 2. Latih model
    train_model(X_seq, y)

    print("\n[OK] Training selesai! Model siap digunakan.\n")