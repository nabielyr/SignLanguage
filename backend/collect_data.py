r"""
=============================================================================
 ALAT REKAM DATA LATIH v2 - ASL Sign Language Detection
=============================================================================
 Script ini merekam data latih dari GERAKAN tangan (2 tangan + motion).

 PERBEDAAN DARI VERSI LAMA:
 - Mendeteksi 2 tangan sekaligus (bukan hanya 1)
 - Merekam SEQUENCE 30 frame (~1 detik gerakan), bukan 1 frame statis
 - Model bisa membedakan pose statis vs gerakan dinamis

 CARA PAKAI:
 1. Jalankan:  .\venv\Scripts\python.exe backend\collect_data.py
 2. Pilih kata yang ingin direkam dengan menekan tombol angka 0-9
 3. Tekan SPACE untuk mulai merekam 1 sampel:
    -> Sistem otomatis merekam 30 frame (~1 detik)
    -> Lakukan gerakan ASL selama rekaman berjalan!
 4. Ulangi sampel kata yang sama minimal 100 kali
    (variasikan kecepatan & sudut gerakan agar model pintar)
 5. Tekan Q untuk keluar

 DAFTAR KATA & TOMBOL:
   0 = DON'T      1 = HOLD       2 = ONTO
   3 = WHAT       4 = NOT        5 = YOURS
   6 = JUST       7 = LET        8 = THINGS
   9 = BE

 FORMAT DATA:
 Setiap sampel = array numpy bentuk (SEQ_LEN, 126)
 - SEQ_LEN = 30 frame
 - 126 = 2 tangan x 21 titik landmark x 3 koordinat (x,y,z)
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

# Panjang sequence (jumlah frame per sampel)
# 30 frame @ ~15-20 fps proses = sekitar 1.5-2 detik gerakan
SEQ_LEN = 30

# Jumlah nilai per frame: 2 tangan x 21 titik x 3 koordinat
FRAME_SIZE = 126

# ============================================================
# INISIALISASI MEDIAPIPE
# ============================================================

# MediaPipe Hands - mendeteksi hingga 2 tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # DETEKSI 2 TANGAN
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ============================================================
# PERSIAPAN FOLDER
# ============================================================

for word in WORDS:
    word_dir = os.path.join(DATA_DIR, word)
    os.makedirs(word_dir, exist_ok=True)

def count_samples(word):
    """Menghitung jumlah sampel yang sudah direkam untuk suatu kata."""
    word_dir = os.path.join(DATA_DIR, word)
    return len([f for f in os.listdir(word_dir) if f.endswith(".npy")])

def extract_frame_landmarks(result):
    """
    Ekstrak landmark KEDUA tangan dari satu frame.

    Returns:
        Array numpy bentuk (126,):
        - Index 0-62   : tangan pertama (diurutkan konsisten)
        - Index 63-125 : tangan kedua (nol jika tidak terdeteksi)

    Urutan tangan dibuat konsisten berdasarkan handedness
    (tangan "Left" selalu di depan, "Right" di belakang),
    sehingga model tidak bingung karena posisi tangan tertukar.
    """
    frame_data = np.zeros(FRAME_SIZE)

    if not result.multi_hand_landmarks:
        return frame_data

    hands_list = []
    for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
        lm = []
        for point in hand_landmarks.landmark:
            lm.extend([point.x, point.y, point.z])

        # Ambil label kiri/kanan untuk urutan konsisten
        label = "Right"  # default
        if result.multi_handedness and i < len(result.multi_handedness):
            label = result.multi_handedness[i].classification[0].label

        hands_list.append((label, np.array(lm)))

    # Urutkan: "Left" dulu (index 0), lalu "Right" (index 1)
    hands_list.sort(key=lambda x: 0 if x[0] == "Left" else 1)

    # Isi data frame
    for i, (_, lm) in enumerate(hands_list[:2]):
        frame_data[i * 63:(i + 1) * 63] = lm

    return frame_data

# ============================================================
# LOOP REKAM
# ============================================================

def main():
    # Buka webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka webcam! Pastikan kamera tersedia.")
        return

    current_index = 0

    # State perekaman sequence
    recording = False      # sedang merekam sequence?
    seq_buffer = []        # buffer frame untuk sequence saat ini

    print("\n" + "=" * 60)
    print("  [HAND] ALAT REKAM DATA ASL v2 - 2 Tangan + Motion")
    print("=" * 60)
    print("\nDaftar kata:")
    for i, word in enumerate(WORDS):
        status = f"({count_samples(word)}/{TARGET_SAMPLES})"
        print(f"  Tombol {i} = {word:8s} {status}")

    print("\n[KONTROL]")
    print(f"  [0-9]  = Ganti kata yang direkam")
    print(f"  SPACE  = Rekam 1 sampel ({SEQ_LEN} frame gerakan otomatis)")
    print("  Q      = Keluar")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror tampilan (biar natural)
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Gambar landmark kedua tangan
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # ---- PROSES REKAMAN SEQUENCE ----
        if recording:
            # Ekstrak data frame ini (kedua tangan)
            frame_data = extract_frame_landmarks(result)
            seq_buffer.append(frame_data)

            # Gambar progress bar rekaman di layar
            progress = len(seq_buffer) / SEQ_LEN
            bar_width = int(progress * 400)
            cv2.rectangle(frame, (10, 120), (410, 140), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 120), (10 + bar_width, 140), (0, 200, 255), -1)
            cv2.putText(frame, f"MEREKAM {len(seq_buffer)}/{SEQ_LEN}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Sequence selesai? Simpan!
            if len(seq_buffer) >= SEQ_LEN:
                recording = False
                sequence = np.array(seq_buffer)  # bentuk (SEQ_LEN, 126)

                current_word = WORDS[current_index]
                next_num = count_samples(current_word)
                save_path = os.path.join(DATA_DIR, current_word, f"sample_{next_num}.npy")
                np.save(save_path, sequence)

                print(f"[OK] Disimpan: {current_word}/sample_{next_num}.npy "
                      f"(bentuk {sequence.shape}) "
                      f"({next_num + 1}/{TARGET_SAMPLES})")

                if next_num + 1 >= TARGET_SAMPLES:
                    print(f"[OK] Kata '{current_word}' sudah mencapai target!")

        # ---- TAMPILAN INFO DI LAYAR ----
        current_word = WORDS[current_index]
        current_count = count_samples(current_word)

        cv2.putText(frame, f"Kata: {current_word}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Sampel: {current_count}/{TARGET_SAMPLES}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        num_hands = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        cv2.putText(frame, f"Tangan terdeteksi: {num_hands}/2", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if num_hands > 0 else (0, 0, 255), 2)

        if not recording:
            cv2.putText(frame, "Tekan SPACE untuk mulai rekam", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Rekam Data ASL v2 - SPACE=rekam gerakan 1 detik", frame)

        # ============================================================
        # BACA INPUT KEYBOARD
        # ============================================================
        key = cv2.waitKey(1) & 0xFF

        # Q / ESC -> keluar
        if key == ord('q') or key == 27:
            break

        # 0-9 -> ganti kata
        if 48 <= key <= 57:
            index = key - 48
            if index < len(WORDS):
                current_index = index
                print(f"[INFO] Ganti ke kata: {WORDS[current_index]}")

        # SPACE -> mulai rekam sequence baru
        if key == 32 and not recording:
            recording = True
            seq_buffer = []
            print(f"[INFO] Mulai merekam {SEQ_LEN} frame... lakukan gerakannya!")

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

if __name__ == "__main__":
    main()