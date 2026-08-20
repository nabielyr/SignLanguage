"""
=============================================================================
 BACKEND API - ASL Sign Language Detection
=============================================================================
 Server FastAPI yang mengelola:
 1. WebSocket untuk prediksi real-time dari webcam
 2. Deteksi tangan dengan MediaPipe
 3. Klasifikasi kata dengan Random Forest model
 4. Penyusunan kalimat & Text-to-Speech

 CARA JALANKAN:
 .\venv\Scripts\python.exe -m uvicorn backend.app:app --reload --port 8000

 Frontend akan otomatis tersedia di: http://localhost:8000
=============================================================================
"""

import os
import io
import base64
import json
import threading
import numpy as np
import cv2
import mediapipe as mp
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from gtts import gTTS

# ============================================================
# KONFIGURASI PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_model.pkl")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

os.makedirs(AUDIO_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL = None
WORDS = []

def load_model():
    """Memuat model dari disk. Jika belum ada, tampilkan pesan."""
    global MODEL, WORDS
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        MODEL = data['model']
        WORDS = data['words']
        print(f"[OK] Model dimuat: {len(WORDS)} kata dikenali")
        return True
    else:
        print("[WARN] Model belum ada. Jalankan train_model.py dulu.")
        return False

load_model()

# ============================================================
# MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_landmarks(hand_landmarks):
    """Mengubah 21 landmark menjadi array 63 angka."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)

def predict_word(landmarks):
    """Prediksi kata dari array landmark."""
    if MODEL is None:
        return None, 0.0

    # Reshape ke format yang diminta model
    features = landmarks.reshape(1, -1)

    # Prediksi probabilitas
    probabilities = MODEL.predict_proba(features)[0]
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]

    # Hanya kembalikan prediksi jika confidence cukup tinggi
    if confidence >= 0.6:
        return WORDS[predicted_idx], float(confidence)
    return None, float(confidence)

# ============================================================
# TEXT-TO-SPEECH
# ============================================================

def generate_speech(text):
    """Mengubah teks menjadi file audio MP3."""
    mp3_path = os.path.join(AUDIO_DIR, "speech.mp3")
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(mp3_path)
        return mp3_path
    except Exception as e:
        print(f"[ERROR] Gagal generate TTS: {e}")
        return None

# ============================================================
# SMOOTHING (menghaluskan prediksi)
# ============================================================

# Buffer untuk prediksi berurutan (mengurangi prediksi yang berkedip)
PREDICTION_BUFFER_SIZE = 5
prediction_buffer = []

def get_smoothed_prediction(current_prediction):
    """
    Menghaluskan prediksi dengan menyimpan N prediksi terakhir
    dan mengambil yang paling sering muncul.
    """
    global prediction_buffer
    prediction_buffer.append(current_prediction)

    # Jaga ukuran buffer
    if len(prediction_buffer) > PREDICTION_BUFFER_SIZE:
        prediction_buffer.pop(0)

    # Ambil prediksi yang paling sering muncul di buffer
    if prediction_buffer:
        from collections import Counter
        counter = Counter(prediction_buffer)
        most_common, count = counter.most_common(1)[0]

        # Konfirmasi hanya jika muncul minimal 3 dari 5 kali
        if count >= 3:
            return most_common
    return None

# ============================================================
# SERVER STATE (sesi pengguna)
# ============================================================

class SessionState:
    """Menyimpan status pengumpulan kata untuk satu sesi."""
    def __init__(self):
        self.words = []          # Daftar kata yang sudah dikumpulkan
        self.last_word = None    # Kata terakhir yang dideteksi (untuk cegah duplikat)
        self.last_word_count = 0 # Seberapa lama kata terakhir terdeteksi berturut-turut

    def reset(self):
        self.words = []
        self.last_word = None
        self.last_word_count = 0

    def add_word(self, word):
        """Menambahkan kata jika beda dari kata sebelumnya (anti spam)."""
        if word == self.last_word:
            self.last_word_count += 1
        else:
            self.last_word = word
            self.last_word_count = 1

        # Tambahkan hanya jika kata bertahan minimal 3 frame berturut-turut
        if self.last_word_count == 3:
            self.words.append(word)
            return True
        return False

session = SessionState()

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="ASL Sign Language Detection")

# ============================================================
# ROUTES (API)
# ============================================================

@app.get("/")
async def index():
    """Halaman utama frontend."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/api/words")
async def get_words():
    """Mengembalikan daftar kata yang dikenali model."""
    return {"words": WORDS, "model_loaded": MODEL is not None}

@app.post("/api/speak")
async def speak(request: dict):
    """Generate TTS dari kalimat yang dikirim."""
    text = request.get("text", "")
    if not text:
        return {"success": False, "error": "Teks kosong"}

    mp3_path = generate_speech(text)
    if mp3_path:
        return {"success": True, "audio_url": "/audio/speech.mp3"}
    return {"success": False, "error": "Gagal membuat audio"}

# Serve folder audio
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Serve static files frontend
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ============================================================
# WEBSOCKET (komunikasi real-time dengan frontend)
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Menerima frame video dari frontend, mengirim prediksi kembali."""
    await websocket.accept()

    # Reset sesi untuk koneksi baru
    session.reset()

    try:
        while True:
            # Terima pesan dari frontend
            message = await websocket.receive_text()
            data = json.loads(message)

            # === AKSI: PROSES FRAME VIDEO ===
            if data.get("type") == "frame":
                # Baca base64 image
                image_base64 = data["image"].split(",")[1]  # buang prefix "data:image/jpeg;base64,"
                image_bytes = base64.b64decode(image_base64)

                # Konversi ke array numpy untuk OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Proses deteksi
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                # Variabel hasil
                current_word = None
                confidence = 0.0
                landmarks_detected = False
                landmark_points = []

                if result.multi_hand_landmarks:
                    landmarks_detected = True
                    hand_landmarks = result.multi_hand_landmarks[0]

                    # Kumpulkan titik landmark untuk digambar di frontend
                    for lm in hand_landmarks.landmark:
                        landmark_points.append({
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z
                        })

                    # Prediksi kata
                    if MODEL is not None:
                        landmarks = extract_landmarks(hand_landmarks)
                        word, conf = predict_word(landmarks)
                        if word:
                            # Smoothed prediction
                            smoothed = get_smoothed_prediction(word)
                            if smoothed:
                                current_word = smoothed
                                confidence = conf

                # Kirim hasil prediksi ke frontend
                await websocket.send_json({
                    "type": "prediction",
                    "word": current_word,
                    "confidence": round(confidence * 100, 1),
                    "landmarks_detected": landmarks_detected,
                    "landmarks": landmark_points
                })

            # === AKSI: TAMBAH KATA KE KOLEKSI ===
            elif data.get("type") == "add_word":
                word = data.get("word")
                if word:
                    added = session.add_word(word)
                    await websocket.send_json({
                        "type": "words_update",
                        "words": session.words,
                        "added": added,
                        "last_added": word if added else None
                    })

            # === AKSI: RESET KOLEKSI KATA ===
            elif data.get("type") == "reset":
                session.reset()
                await websocket.send_json({
                    "type": "words_update",
                    "words": session.words,
                    "added": False,
                    "last_added": None
                })

            # === AKSI: HAPUS KATA TERAKHIR ===
            elif data.get("type") == "undo":
                if session.words:
                    session.words.pop()
                await websocket.send_json({
                    "type": "words_update",
                    "words": session.words,
                    "added": False,
                    "last_added": None
                })

            # === AKSI: BUAT KALIMAT ===
            elif data.get("type") == "generate_sentence":
                if session.words:
                    # Susun kalimat alami dari kata-kata
                    sentence = build_sentence(session.words)
                    await websocket.send_json({
                        "type": "sentence",
                        "sentence": sentence
                    })
                else:
                    await websocket.send_json({
                        "type": "sentence",
                        "sentence": "",
                        "error": "Belum ada kata yang dikumpulkan"
                    })

    except WebSocketDisconnect:
        print("[INFO] Koneksi WebSocket terputus")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
    finally:
        prediction_buffer.clear()

# ============================================================
# PEMBANGUN KALIMAT
# ============================================================

def build_sentence(words):
    """
    Menyusun kata-kata yang dikumpulkan menjadi kalimat alami.
    Kata-kata dari ASL diurutkan dan digabung dengan aturan bahasa Inggris.
    """
    if not words:
        return ""

    # Gabung kata dengan spasi
    raw = " ".join(words)

    # Aturan penyusunan natural sentence:
    # 1. Kapital huruf pertama
    # 2. Tambahkan apostrof jika ada "DONT" -> "don't"

    # Perbaiki kata-kata khusus
    replacements = {
        "DON'T": "don't",
        "HOLD": "hold",
        "ONTO": "onto",
        "WHAT": "what",
        "NOT": "not",
        "YOURS": "yours",
        "JUST": "just",
        "LET": "let",
        "THINGS": "things",
        "BE": "be"
    }

    # Ubah semua kata ke huruf kecil dengan pemetaan
    sentence_words = []
    for w in words:
        sentence_words.append(replacements.get(w, w.lower()))

    # Gabung menjadi kalimat
    sentence = " ".join(sentence_words)

    # Tambahkan tanda baca dan kapitalisasi
    sentence = sentence.capitalize()

    # Cek apakah ini kalimat perintah/permintaan (biasanya tidak pakai tanda tanya)
    if sentence:
        sentence += "."

    return sentence

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n[START] ASL Sign Language Detection")
    print("=" * 50)
    print("Server berjalan di: http://localhost:8000")
    print("Tekan Ctrl+C untuk berhenti\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)