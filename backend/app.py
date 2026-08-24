r"""
=============================================================================
 BACKEND API v2 - ASL Sign Language Detection (2 Tangan + Motion)
=============================================================================
 Server FastAPI yang mengelola:
 1. WebSocket untuk prediksi real-time dari webcam
 2. Deteksi DUA tangan dengan MediaPipe
 3. Klasifikasi kata berbasis SEQUENCE gerakan (rolling buffer 30 frame)
    dengan fitur motion: mean + std + velocity
 4. Penyusunan kalimat & Text-to-Speech

 CARA JALANKAN:
 .\venv\Scripts\python.exe -m uvicorn backend.app:app --port 8000

 Frontend akan otomatis tersedia di: http://localhost:8000
=============================================================================
"""

import os
import base64
import json
from collections import Counter, deque

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

# Konfigurasi sequence (harus sama dengan collect_data.py & train_model.py)
SEQ_LEN = 30      # jumlah frame dalam rolling buffer
FRAME_SIZE = 126  # 2 tangan x 21 titik x 3 koordinat

# ============================================================
# LOAD MODEL
# ============================================================

MODEL = None
WORDS = []
MODEL_VERSION = 1

def load_model():
    """Memuat model dari disk. Jika belum ada, tampilkan pesan."""
    global MODEL, WORDS, MODEL_VERSION
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        MODEL = data['model']
        WORDS = data['words']
        MODEL_VERSION = data.get('version', 1)
        print(f"[OK] Model dimuat: {len(WORDS)} kata dikenali (v{MODEL_VERSION})")
        return True
    else:
        print("[WARN] Model belum ada. Jalankan train_model.py dulu.")
        return False

load_model()

# ============================================================
# MEDIAPIPE HANDS (2 TANGAN)
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # DETEKSI 2 TANGAN
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_frame_landmarks(result):
    """
    Ekstrak landmark KEDUA tangan dari satu frame hasil MediaPipe.
    Urutan tangan konsisten: "Left" dulu, lalu "Right".
    Returns array bentuk (126,).
    """
    frame_data = np.zeros(FRAME_SIZE)

    if not result.multi_hand_landmarks:
        return frame_data

    hands_list = []
    for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
        lm = []
        for point in hand_landmarks.landmark:
            lm.extend([point.x, point.y, point.z])

        label = "Right"
        if result.multi_handedness and i < len(result.multi_handedness):
            label = result.multi_handedness[i].classification[0].label

        hands_list.append((label, np.array(lm)))

    hands_list.sort(key=lambda x: 0 if x[0] == "Left" else 1)

    for i, (_, lm) in enumerate(hands_list[:2]):
        frame_data[i * 63:(i + 1) * 63] = lm

    return frame_data

def extract_features_from_sequence(sequence):
    """
    Ekstrak fitur motion dari sequence (SEQ_LEN, FRAME_SIZE).
    Harus sama persis dengan train_model.py!

    Returns array bentuk (378,): [mean(126), std(126), velocity(126)]
    """
    seq = np.array(sequence)                    # (SEQ_LEN, 126)
    feat_mean = seq.mean(axis=0)                # (126,)
    feat_std = seq.std(axis=0)                  # (126,)
    velocities = np.diff(seq, axis=0)           # (SEQ_LEN-1, 126)
    feat_vel = velocities.mean(axis=0)          # (126,)
    return np.concatenate([feat_mean, feat_std, feat_vel])  # (378,)

def predict_word(features):
    """Prediksi kata dari vektor fitur motion."""
    if MODEL is None:
        return None, 0.0

    features_reshaped = features.reshape(1, -1)

    probabilities = MODEL.predict_proba(features_reshaped)[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])

    # Hanya kembalikan prediksi jika confidence cukup tinggi
    if confidence >= 0.6:
        return WORDS[predicted_idx], confidence
    return None, confidence

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

PREDICTION_BUFFER_SIZE = 5
prediction_buffer = []

def get_smoothed_prediction(current_prediction):
    """Ambil prediksi yang paling sering muncul di N prediksi terakhir."""
    global prediction_buffer
    prediction_buffer.append(current_prediction)

    if len(prediction_buffer) > PREDICTION_BUFFER_SIZE:
        prediction_buffer.pop(0)

    if prediction_buffer:
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
    """
    Menyimpan status per sesi WebSocket:
    - words: daftar kata yang dikumpulkan
    - frame_buffer: rolling buffer 30 frame terakhir untuk deteksi motion
    - last_word / last_word_count: anti-spam pengumpulan kata
    """
    def __init__(self):
        self.words = []
        self.last_word = None
        self.last_word_count = 0
        self.frame_buffer = deque(maxlen=SEQ_LEN)

    def reset(self):
        self.words = []
        self.last_word = None
        self.last_word_count = 0
        self.frame_buffer.clear()

    def add_word(self, word):
        """Tambahkan kata jika bertahan minimal 3 prediksi berturut-turut."""
        if word == self.last_word:
            self.last_word_count += 1
        else:
            self.last_word = word
            self.last_word_count = 1

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
    return {
        "words": WORDS,
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION
    }

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
    session.reset()

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            # === AKSI: PROSES FRAME VIDEO ===
            if data.get("type") == "frame":
                image_base64 = data["image"].split(",")[1]
                image_bytes = base64.b64decode(image_base64)

                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                # Variabel hasil
                current_word = None
                confidence = 0.0
                num_hands = 0
                hands_points = []   # landmark per tangan: [[21 titik], [21 titik]]

                if result.multi_hand_landmarks:
                    num_hands = len(result.multi_hand_landmarks)

                    # Kirim landmark per tangan ke frontend
                    for hand_landmarks in result.multi_hand_landmarks:
                        points = []
                        for lm in hand_landmarks.landmark:
                            points.append({"x": lm.x, "y": lm.y, "z": lm.z})
                        hands_points.append(points)

                    # ---- PREDIKSI BERBASIS SEQUENCE (motion) ----
                    if MODEL is not None and MODEL_VERSION >= 2:
                        # Simpan frame ini ke rolling buffer
                        frame_data = extract_frame_landmarks(result)
                        session.frame_buffer.append(frame_data)

                        # Prediksi hanya saat buffer penuh (30 frame terkumpul)
                        if len(session.frame_buffer) >= SEQ_LEN:
                            features = extract_features_from_sequence(
                                list(session.frame_buffer)
                            )
                            word, conf = predict_word(features)
                            if word:
                                smoothed = get_smoothed_prediction(word)
                                if smoothed:
                                    current_word = smoothed
                                    confidence = conf

                # Kirim hasil ke frontend
                await websocket.send_json({
                    "type": "prediction",
                    "word": current_word,
                    "confidence": round(confidence * 100, 1),
                    "num_hands": num_hands,
                    "hands": hands_points
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
    """Menyusun kata-kata yang dikumpulkan menjadi kalimat alami."""
    if not words:
        return ""

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

    sentence_words = [replacements.get(w, w.lower()) for w in words]
    sentence = " ".join(sentence_words).capitalize()

    if sentence:
        sentence += "."

    return sentence

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("")
    print("[START] ASL Sign Language Detection v2 (2 Tangan + Motion)")
    print("=" * 50)
    print("Server berjalan di: http://localhost:8000")
    print("Tekan Ctrl+C untuk berhenti")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8000)