/* ============================================================
   SIGNSPEAK - FRONTEND LOGIC
   ============================================================ */

// ---------- DOM Elements ----------
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');
const videoContainer = document.getElementById('video-container');
const videoPlaceholder = document.getElementById('video-placeholder');
const detectionWord = document.getElementById('detection-word');
const detectionConfidence = document.getElementById('detection-confidence');
const liveIndicator = document.getElementById('live-indicator');
const handStatus = document.getElementById('hand-status');
const handStatusText = document.getElementById('hand-status-text');
const modelStatus = document.getElementById('model-status');
const modelStatusText = document.getElementById('model-status-text');

const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnAddWord = document.getElementById('btn-add-word');
const btnUndo = document.getElementById('btn-undo');
const btnReset = document.getElementById('btn-reset');
const btnGenerate = document.getElementById('btn-generate');
const btnSpeak = document.getElementById('btn-speak');

const wordsList = document.getElementById('words-list');
const sentenceDisplay = document.getElementById('sentence-display');
const vocabGrid = document.getElementById('vocab-grid');
const vocabCount = document.getElementById('vocab-count');
const toast = document.getElementById('toast');

// ---------- State ----------
let ws = null;
let cameraStream = null;
let isCameraOn = false;
let isProcessing = false;
let currentDetectedWord = null;
let currentConfidence = 0;
let collectedWords = [];
let currentSentence = '';
let wsConnected = false;

const WORDS = ["DON'T", "HOLD", "ONTO", "WHAT", "NOT", "YOURS", "JUST", "LET", "THINGS", "BE"];

// Simbol emoji untuk setiap kata (sekadar referensi visual)
const WORD_SYMBOLS = {
    "DON'T": "🚫",
    "HOLD": "✋",
    "ONTO": "👉",
    "WHAT": "❓",
    "NOT": "🙅",
    "YOURS": "🤲",
    "JUST": "✅",
    "LET": "🔓",
    "THINGS": "📦",
    "BE": "✨"
};

// ---------- Toast Notification ----------
function showToast(message, type = 'info', duration = 3000) {
    toast.textContent = message;
    toast.className = `toast show ${type}`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, duration);
}

// ---------- WebSocket Connection ----------
async function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

    ws.onopen = () => {
        console.log('✅ WebSocket terhubung');
        wsConnected = true;
        updateModelStatus(true);
    };

    ws.onclose = () => {
        console.log('❌ WebSocket terputus');
        wsConnected = false;
        updateModelStatus(false);
        // Coba reconnect setelah 2 detik
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        wsConnected = false;
        updateModelStatus(false);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'prediction':
            handlePrediction(data);
            break;
        case 'words_update':
            handleWordsUpdate(data);
            break;
        case 'sentence':
            handleSentence(data);
            break;
    }
}

function handlePrediction(data) {
    currentDetectedWord = data.word;
    currentConfidence = data.confidence;

    // Update tampilan kata terdeteksi
    if (data.word) {
        detectionWord.textContent = data.word;
        detectionWord.classList.add('highlight');
        detectionConfidence.textContent = `${data.confidence}%`;
        detectionConfidence.style.color = getComputedStyle(document.documentElement).getPropertyValue('--success');

        // Enable tombol kumpulkan
        btnAddWord.disabled = false;
    } else {
        detectionWord.textContent = '—';
        detectionWord.classList.remove('highlight');
        detectionConfidence.textContent = '0%';
        detectionConfidence.style.color = '';

        // Disable tombol jika sudah pernah dikumpulkan
        btnAddWord.disabled = !wsConnected || isCameraOn === false;
    }

    // Update status tangan (deteksi hingga 2 tangan)
    if (data.num_hands > 0) {
        setHandStatus(`Tangan terdeteksi: ${data.num_hands}/2`, '#10b981');
        drawLandmarks(data.hands);
    } else {
        setHandStatus('Tangan tidak terdeteksi', '#64748b');
        clearOverlay();
    }
}

function handleWordsUpdate(data) {
    collectedWords = data.words;
    renderWords();

    if (data.added && data.last_added) {
        showToast(`✅ "${data.last_added}" ditambahkan!`, 'success');
        playAddSound();
    }

    // Update tombol
    btnGenerate.disabled = collectedWords.length === 0;
}

function handleSentence(data) {
    if (data.error) {
        showToast(data.error, 'error');
        return;
    }

    currentSentence = data.sentence;
    renderSentence(currentSentence);

    if (currentSentence) {
        showToast('✨ Kalimat berhasil dibuat!', 'success');
        // Auto-speak kalimat
        speakSentence();
    }
}

// ---------- WebSocket Send Helpers ----------
function sendMessage(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
        return true;
    }
    return false;
}

// ---------- Camera Control ----------
async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        isCameraOn = true;
        btnStart.disabled = true;
        btnStop.disabled = false;
        liveIndicator.classList.add('on');
        videoPlaceholder.style.display = 'none';
        setHandStatus('Menunggu tangan...', '#94a3b8');

        // Mulai loop pengiriman frame
        requestAnimationFrame(sendFrame);
        showToast('🎥 Kamera aktif!', 'success');

    } catch (error) {
        console.error('Gagal mengakses kamera:', error);
        showToast('❌ Gagal mengakses kamera. Pastikan webcam tersedia.', 'error', 5000);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;
    isCameraOn = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
    liveIndicator.classList.remove('on');
    videoPlaceholder.style.display = 'flex';
    setHandStatus('Kamera dimatikan', '#64748b');
    clearOverlay();
    detectionWord.textContent = '—';
    detectionConfidence.textContent = '0%';
    btnAddWord.disabled = true;
}

// Kirim frame ke server untuk diproses
async function sendFrame() {
    if (!isCameraOn || video.readyState !== video.HAVE_ENOUGH_DATA || isProcessing) {
        if (isCameraOn) {
            requestAnimationFrame(sendFrame);
        }
        return;
    }

    isProcessing = true;

    // Buat canvas kecil untuk menghemat bandwidth
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 640;
    tempCanvas.height = 360;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, 640, 360);

    // Kirim sebagai base64 JPEG
    const imageData = tempCanvas.toDataURL('image/jpeg', 0.7);

    if (sendMessage({ type: 'frame', image: imageData })) {
        // Lanjutkan setelah server merespon
        setTimeout(() => {
            isProcessing = false;
            requestAnimationFrame(sendFrame);
        }, 50);
    } else {
        isProcessing = false;
        requestAnimationFrame(sendFrame);
    }
}

// ---------- Landmark Drawing (Multi-hand) ----------

// Koneksi antar landmark (indeks pasangan tulang tangan)
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],          // Ibu jari
    [0, 5], [5, 6], [6, 7], [7, 8],          // Telunjuk
    [5, 9], [9, 10], [10, 11], [11, 12],     // Jari tengah
    [9, 13], [13, 14], [14, 15], [15, 16],   // Jari manis
    [13, 17], [17, 18], [18, 19], [19, 20],  // Kelingking
    [0, 17]                                  // Pangkal telapak
];

// Warna berbeda untuk tiap tangan: ungu (tangan 1), cyan (tangan 2)
const HAND_COLORS = [
    { line: 'rgba(139, 92, 246, 0.9)', dot: '#a78bfa', glow: 'rgba(139, 92, 246, 0.3)' },
    { line: 'rgba(6, 182, 212, 0.9)',  dot: '#67e8f9', glow: 'rgba(6, 182, 212, 0.3)' }
];

function drawLandmarks(hands) {
    if (!hands || hands.length === 0) {
        clearOverlay();
        return;
    }

    // Ukuran video container
    const rect = videoContainer.getBoundingClientRect();
    const canvas = overlay;
    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = overlayCtx;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Gambar setiap tangan dengan warnanya masing-masing
    hands.forEach((landmarks, handIdx) => {
        if (!landmarks || landmarks.length === 0) return;

        const color = HAND_COLORS[handIdx % HAND_COLORS.length];

        // Gambar koneksi (garis antar titik)
        ctx.strokeStyle = color.line;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';

        for (const [a, b] of HAND_CONNECTIONS) {
            if (landmarks[a] && landmarks[b]) {
                ctx.beginPath();
                ctx.moveTo(landmarks[a].x * canvas.width, landmarks[a].y * canvas.height);
                ctx.lineTo(landmarks[b].x * canvas.width, landmarks[b].y * canvas.height);
                ctx.stroke();
            }
        }

        // Gambar titik landmark
        for (let i = 0; i < landmarks.length; i++) {
            const lm = landmarks[i];
            const x = lm.x * canvas.width;
            const y = lm.y * canvas.height;

            // Lingkaran luar (glow)
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, 2 * Math.PI);
            ctx.fillStyle = color.glow;
            ctx.fill();

            // Titik dalam
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = color.dot;
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    });
}

function clearOverlay() {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

// ---------- Words Rendering ----------
function renderWords() {
    wordsList.innerHTML = '';

    if (collectedWords.length === 0) {
        wordsList.innerHTML = `
            <div class="empty-words">
                <span>Belum ada kata. Deteksi isyarat lalu klik "Kumpulkan".</span>
            </div>
        `;
        return;
    }

    collectedWords.forEach((word, index) => {
        const chip = document.createElement('div');
        chip.className = 'word-chip';
        chip.innerHTML = `
            <span class="word">${word}</span>
            <span class="chip-index">${index + 1}</span>
        `;
        wordsList.appendChild(chip);
    });

    // Scroll ke bawah
    wordsList.scrollTop = wordsList.scrollHeight;
}

function renderSentence(sentence) {
    if (sentence) {
        sentenceDisplay.innerHTML = `
            <div class="sentence-text">${escapeHtml(sentence)}</div>
        `;
        btnSpeak.disabled = false;
    } else {
        sentenceDisplay.innerHTML = `
            <div class="sentence-placeholder">
                <span>Kumpulkan kata-kata untuk membuat kalimat...</span>
            </div>
        `;
        btnSpeak.disabled = true;
    }
}

// ---------- Vocabulary Rendering ----------
function renderVocabulary() {
    vocabGrid.innerHTML = '';
    WORDS.forEach(word => {
        const item = document.createElement('div');
        item.className = 'vocab-item';
        item.innerHTML = `
            <span class="vocab-symbol">${WORD_SYMBOLS[word] || '🖐️'}</span>
            <span class="vocab-word">${word}</span>
        `;
        vocabGrid.appendChild(item);
    });
    vocabCount.textContent = `${WORDS.length} kata`;
}

// ---------- UI Helpers ----------
function setHandStatus(text, color) {
    handStatusText.textContent = text;
    const dot = handStatus.querySelector('.status-dot');
    dot.style.background = color;
    if (color === '#10b981') {
        dot.style.boxShadow = `0 0 8px ${color}`;
    } else {
        dot.style.boxShadow = 'none';
    }
}

function updateModelStatus(connected) {
    const dot = modelStatus.querySelector('.status-dot');
    if (connected) {
        modelStatus.classList.add('active');
        modelStatusText.textContent = 'Terhubung';
    } else {
        modelStatus.classList.remove('active');
        modelStatusText.textContent = 'Menghubungkan...';
    }
}

function playAddSound() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        oscillator.frequency.setValueAtTime(523.25, audioCtx.currentTime); // C5
        oscillator.frequency.setValueAtTime(659.25, audioCtx.currentTime + 0.1); // E5

        gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);

        oscillator.start(audioCtx.currentTime);
        oscillator.stop(audioCtx.currentTime + 0.3);
    } catch (e) {
        // Audio tidak tersedia, abaikan
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ---------- Text-to-Speech ----------
async function speakSentence() {
    if (!currentSentence) return;

    try {
        btnSpeak.disabled = true;
        const response = await fetch('/api/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: currentSentence })
        });

        const data = await response.json();

        if (data.success) {
            const audio = new Audio(data.audio_url);
            audio.play();
            audio.onended = () => {
                btnSpeak.disabled = false;
            };
            showToast('🔊 Membacakan kalimat...', 'info', 2000);
        } else {
            btnSpeak.disabled = false;
            showToast('❌ Gagal membuat audio', 'error');
        }
    } catch (error) {
        console.error('TTS Error:', error);
        btnSpeak.disabled = false;
        showToast('❌ Gagal memuat audio', 'error');
    }
}

// ---------- Event Listeners ----------
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);

btnAddWord.addEventListener('click', () => {
    if (currentDetectedWord) {
        sendMessage({ type: 'add_word', word: currentDetectedWord });
        // Nonaktifkan sementara untuk mencegah spam
        btnAddWord.disabled = true;
    } else {
        showToast('⚠️ Tidak ada kata yang terdeteksi', 'error');
    }
});

btnUndo.addEventListener('click', () => {
    sendMessage({ type: 'undo' });
});

btnReset.addEventListener('click', () => {
    sendMessage({ type: 'reset' });
    currentSentence = '';
    renderSentence('');
    showToast('🔄 Semua kata direset', 'info');
});

btnGenerate.addEventListener('click', () => {
    sendMessage({ type: 'generate_sentence' });
});

btnSpeak.addEventListener('click', speakSentence);

// ---------- Initialize ----------
async function init() {
    renderVocabulary();

    // Fetch info model dari server
    try {
        const response = await fetch('/api/words');
        const data = await response.json();
        if (data.model_loaded) {
            modelStatusText.textContent = `Model siap · ${data.words.length} kata`;
            modelStatus.classList.add('active');
        } else {
            modelStatusText.textContent = 'Model belum dilatih';
            modelStatus.classList.remove('active');
        }
    } catch (error) {
        console.error('Gagal memuat info model:', error);
        modelStatusText.textContent = 'Server tidak merespons';
    }

    // Koneksi WebSocket
    await connectWebSocket();

    // Center overlay canvas
    function resizeOverlay() {
        const rect = videoContainer.getBoundingClientRect();
        overlay.width = rect.width;
        overlay.height = rect.height;
    }
    resizeOverlay();
    window.addEventListener('resize', resizeOverlay);
}

// Mulai aplikasi
init();