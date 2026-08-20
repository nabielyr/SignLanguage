# 🖐️ SignSpeak — ASL Sign Language Detection

![Project Status](https://img.shields.io/badge/status-Development-blue)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green)

**SignSpeak** adalah aplikasi web yang membaca **American Sign Language (ASL)** secara real-time menggunakan webcam. Aplikasi ini mengumpulkan kata-kata yang kamu isyaratkan, menyusunnya menjadi kalimat alami, lalu **membacakannya** menggunakan Text-to-Speech.

---

## 🎯 Fitur

- 📹 **Real-time hand tracking** — Deteksi 21 titik landmark tangan dengan MediaPipe
- 🧠 **Klasifikasi kata ASL** — Model Random Forest yang dilatih dari data landmark
- 📝 **Pengumpulan kata** — Kamu bisa mengumpulkan kata-kata yang diisyaratkan
- ✨ **Pembuatan kalimat** — Kata-kata disusun menjadi kalimat natural
- 🔊 **Text-to-Speech** — Kalimat dibacakan dengan suara
- 🎨 **UI modern** — Dark theme dengan glassmorphism, animasi halus

---

## 📋 Persiapan

Sebelum mulai, pastikan kamu sudah punya:

- **Python 3.12+** → [Download](https://www.python.org/downloads/)
- **Webcam** (bawaan laptop atau eksternal)

---

## 🚀 Cara Install

Buka **PowerShell** di folder proyek ini, lalu jalankan:

```powershell
# 1. Buat virtual environment (hanya sekali)
python -m venv venv

# 2. Install semua dependency (hanya sekali)
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 🎬 Cara Pakai (3 Langkah)

### Langkah 1: Rekam Data Latih

Jalankan alat rekam data:

```powershell
.\venv\Scripts\python.exe backend\collect_data.py
```

Jendela webcam akan terbuka. **Cara merekam:**

1. Tekan tombol **0-9** untuk memilih kata yang ingin direkam
2. Lakukan isyarat ASL di depan kamera
3. Tekan **SPACE** untuk menyimpan 1 sampel
4. Rekam minimal **150 sampel per kata** (tahan pose, tekan SPACE berulang kali)
5. Tekan **Q** untuk keluar

**Daftar kata & tombol:**

| Tombol | Kata     | Tombol | Kata    |
|--------|----------|--------|---------|
| 0      | DON'T    | 5      | YOURS   |
| 1      | HOLD     | 6      | JUST    |
| 2      | ONTO     | 7      | LET     |
| 3      | WHAT     | 8      | THINGS  |
| 4      | NOT      | 9      | BE      |

### Langkah 2: Latih Model

Setelah data cukup, latih model:

```powershell
.\venv\Scripts\python.exe backend\train_model.py
```

Model akan disimpan di `models/asl_model.pkl` dengan laporan akurasi.

### Langkah 3: Jalankan Aplikasi

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.app:app --reload --port 8000
```

Buka browser ke: **http://localhost:8000**

---

## 🎮 Cara Menggunakan Aplikasi

1. Klik tombol **kamera** (ikon ▶) untuk mulai
2. Izinkan akses webcam di browser
3. Tunjukkan isyarat tangan di depan kamera
4. Saat kata terdeteksi dengan benar, klik **"Kumpulkan Kata"**
5. Ulangi untuk kata-kata berikutnya
6. Klik **"Buat Kalimat"** untuk menyusun kalimat
7. Klik **"Bacakan"** untuk mendengar hasilnya

### Kontrol Tambahan
- **Undo** — Hapus kata terakhir yang dikumpulkan
- **Reset** — Hapus semua kata

---

## 🏗️ Struktur Proyek

```
SignLanguage/
├── backend/
│   ├── app.py              # Server FastAPI (API + WebSocket)
│   ├── collect_data.py     # Alat rekam data latih
│   ├── train_model.py      # Latih model klasifikasi
│   └── __init__.py
├── frontend/
│   ├── index.html          # Halaman utama
│   ├── style.css           # Desain modern
│   └── script.js           # Logika frontend
├── data/                   # Data latih (dibuat otomatis)
├── models/                 # Model hasil training
├── audio/                  # File audio TTS
├── requirements.txt        # Dependency Python
└── README.md
```

---

## 🧠 Teknologi

| Teknologi | Fungsi |
|-----------|--------|
| **MediaPipe Hands** | Deteksi 21 titik landmark tangan |
| **scikit-learn** | Random Forest classifier |
| **FastAPI** | Backend API & WebSocket |
| **OpenCV** | Olah gambar webcam |
| **gTTS** | Text-to-Speech |
| **HTML/CSS/JS** | Frontend modern |

---

## 🔧 Troubleshooting

### "Tangan tidak terdeteksi"
- Pastikan pencahayaan cukup
- Tangan harus terlihat jelas di frame webcam
- Jarak tangan 30-60 cm dari kamera

### Akurasi rendah
- Tambah lebih banyak data latih (target 200+ sampel per kata)
- Variasikan posisi dan sudut tangan saat merekam
- Latih ulang model dengan data baru

### "CORS" error di browser
- Pastikan buka aplikasi dari `http://localhost:8000` (bukan file HTML langsung)

---

## 📝 Catatan

- Project ini **untuk belajar** — model dilatih dari isyarat kamu sendiri
- Semakin banyak data, semakin akurat modelnya
- Pastikan kamera tidak dipakai aplikasi lain

---

Dibuat dengan ❤️ untuk belajar ASL.