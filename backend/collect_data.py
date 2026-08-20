"""
=============================================================================
 ALAT REKAM DATA LATIH - ASL Sign Language Detection
=============================================================================
 Script ini digunakan untuk merekam data latih dari gerakan tangan (ASL).

 CARA PAKAI:
 1. Jalankan:  .\venv\Scripts\python.exe backend\collect_data.py
 2. Pilih kata yang ingin direkam dengan menekan tombol angka 0-9
 3. Lakukan isyarat tangan di depan kamera
 4. Tekan SPACE untuk menyimpan 1 sampel data
 5. Tekan Q untuk keluar

 DAFTAR KATA & TOMBOL:
   0 = DON'T      1 = HOLD       2 = ONTO
   3 = WHAT       4 = NOT        5 = YOURS
   6 = JUST       7 = LET        8 = THINGS
   9 = BE

 Setiap sampel adalah 63 angka (21 titik landmark x 3 koordinat x,y,z).
 Data disimpan di folder 'data/'.
=============================================================================
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# ============================================================
# KONFIGURASI
# ============================================================

# Daftar kata yang akan dilatih (urutannya harus konsisten dengan train_model.py)
WORDS = ["DON'T", "HOLD", "ONTO", "WHAT", "NOT", "YOURS", "JUST", "LET", "THINGS", "BE"]

# Folder penyimpanan data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Jumlah sampel per kata yang diinginkan
TARGET_SAMPLES = 150

# ============================================================
# INISIALISASI MEDIAPIPE
# ============================================================

# MediaPipe Hands - mendeteksi 21 titik di tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,      # Mode video (real-time)
    max_num_hands=1,              # Deteksi 1 tangan saja (biar simple)
    min_detection_confidence=0.7, # Ambang deteksi
    min_tracking_confidence=0.5   # Ambang pelacakan
)

# ============================================================
# PERSIAPAN FOLDER
# ============================================================

# Pastikan folder data/word_X ada
for word in WORDS:
    word_dir = os.path.join(DATA_DIR, word)
    os.makedirs(word_dir, exist_ok=True)

def count_samples(word):
    """Menghitung jumlah sampel yang sudah direkam untuk suatu kata."""
    word_dir = os.path.join(DATA_DIR, word)
    return len([f for f in os.listdir(word_dir) if f.endswith(".npy")])

def extract_landmarks(hand_landmarks):
    """
    Mengubah 21 titik landmark menjadi array 63 angka.
    Format: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)

# ============================================================
# LOOP REKAM
# ============================================================

def main():
    # Buka webcam (0 = kamera bawaan laptop)
    cap = cv2.VideoCapture(0)

    # Cek webcam
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka webcam! Pastikan kamera tersedia.")
        return

    # Kata yang sedang direkam (default: kata ke-0)
    current_index = 0

    print("\n" + "=" * 60)
    print("  [HAND] ALAT REKAM DATA ASL - Sign Language")
    print("=" * 60)
    print("\nDaftar kata:")
    for i, word in enumerate(WORDS):
        status = f"({count_samples(word)}/{TARGET_SAMPLES})"
        print(f"  Tombol {i} = {word:8s} {status}")

    print("\n[KONTROL]")
    print("  [0-9]  = Ganti kata yang direkam")
    print("  SPACE  = Simpan 1 sampel (tahan pose tangan)")
    print("  Q      = Keluar")
    print("=" * 60 + "\n")

    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror tampilan (biar natural)
        frame = cv2.flip(frame, 1)

        # Konversi BGR -> RGB (MediaPipe butuh RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deteksi tangan
        result = hands.process(rgb_frame)

        # Gambar landmark jika tangan terdeteksi
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # ---- TAMPILAN INFO DI LAYAR ----

        # Info kata yang sedang direkam
        current_word = WORDS[current_index]
        current_count = count_samples(current_word)

        cv2.putText(frame, f"Kata: {current_word}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Sampel: {current_count}/{TARGET_SAMPLES}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Status tangan
        if result.multi_hand_landmarks:
            cv2.putText(frame, "Tangan Terdeteksi", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tangan Tidak Terdeteksi", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Tampilkan frame
        cv2.imshow("Rekam Data ASL - Tekan SPACE untuk menyimpan", frame)

        # ============================================================
        # BACA INPUT KEYBOARD
        # ============================================================
        key = cv2.waitKey(1) & 0xFF

        # Tombol Q -> keluar
        if key == ord('q') or key == 27:  # 27 = ESC
            break

        # Tombol 0-9 -> pilih kata
        if 48 <= key <= 57:  # ASCII '0' = 48, '9' = 57
            index = key - 48
            if index < len(WORDS):
                current_index = index
                print(f"[INFO] Ganti ke kata: {WORDS[current_index]}")

        # Tombol SPACE -> simpan sampel
        if key == 32:  # ASCII SPACE = 32
            if result.multi_hand_landmarks:
                # Ambil landmark tangan pertama
                landmarks = extract_landmarks(result.multi_hand_landmarks[0])

                # Cek apakah tangan masih terdeteksi (valid)
                if landmarks.shape == (63,):
                    # Nomor urut sampel berikutnya
                    next_num = count_samples(current_word)

                    # Simpan ke file
                    save_path = os.path.join(DATA_DIR, current_word, f"sample_{next_num}.npy")
                    np.save(save_path, landmarks)

                    print(f"[OK] Disimpan: {current_word}/sample_{next_num}.npy "
                          f"({next_num + 1}/{TARGET_SAMPLES})")

                    # Jika sudah mencapai target, beri tahu
                    if next_num + 1 >= TARGET_SAMPLES:
                        print(f"[OK] Kata '{current_word}' sudah mencapai {TARGET_SAMPLES}!")
                else:
                    print("[WARN] Landmark tidak valid, coba lagi.")
            else:
                print("[WARN] Tangan tidak terdeteksi! Lakukan isyarat tangan dulu.")

    # Bersihkan resource
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Ringkasan akhir
    print("\n" + "=" * 60)
    print("  RINGKASAN DATA")
    print("=" * 60)
    total = 0
    for word in WORDS:
        n = count_samples(word)
        total += n
        status = "[OK] CUKUP" if n >= TARGET_SAMPLES else "[WARN] KURANG"
        print(f"  {word:8s}: {n:4d} sampel  {status}")
    print(f"  {'TOTAL':8s}: {total:4d} sampel")
    print("=" * 60 + "\n")

# ============================================================
# JALANKAN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()