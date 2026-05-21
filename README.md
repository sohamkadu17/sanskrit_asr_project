# Sanskrit ASR Project

End-to-end Sanskrit speech recognition with shloka identification and pronunciation feedback. The backend runs a fine-tuned Wav2Vec2 model, matches transcriptions to shlokas with FAISS, and serves a single-page UI for recording or uploading audio.

## Features
- ASR inference with a fine-tuned Wav2Vec2 model.
- Auto-identify shlokas via FAISS similarity search.
- Optional external/LLM fallback for unknown shlokas.
- Practice mode with pronunciation feedback and scoring.
- Web UI with mic recording and drag-and-drop uploads.

## Project Layout
```
app.py                      # FastAPI server + API routes
transcribe.py               # ASR inference utilities
shloka_search.py            # FAISS shloka matching + optional external/LLM
ui/                         # Static frontend (index.html, script.js, style.css)
train_sanskrit_asr.py       # Training pipeline
evaluate_asr.py             # ASR evaluation (if used)
train_output.txt            # Training logs
shloka_index.faiss          # FAISS index (auto-generated)
```

## Quick Start
### 1) Create a virtual environment (optional but recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install runtime dependencies
```bash
pip install fastapi uvicorn transformers torch soundfile imageio-ffmpeg numpy scipy
pip install sentence-transformers faiss-cpu requests python-dotenv google-genai
```

### 3) Run the server
```bash
python app.py
```
Open http://localhost:8000 in your browser.

## API Endpoints
- POST `/api/transcribe` - Transcribe audio only.
- POST `/api/identify_shloka` - Transcribe + auto-identify shloka.
- POST `/api/practice_shloka` - Transcribe + compare with a selected shloka.
- GET `/api/shlokas` - List available shlokas.

## Environment Variables
Create a `.env` file if you want external or LLM search:
```
# Enable external Wikipedia/Wikisource lookup
SHLOKA_EXTERNAL_SEARCH=1

# Enable Gemini LLM fallback
SHLOKA_LLM_SEARCH=1
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Optional debug flags:
```
SHLOKA_EXTERNAL_DEBUG=1
SHLOKA_LLM_DEBUG=1
```

## Audio Formats
- Supported: wav, mp3, mp4, m4a, webm, ogg, flac
- The server converts container formats to 16kHz mono WAV when needed.

## Training (Optional)
The training pipeline is in `train_sanskrit_asr.py` and uses the `ai4bharat/Kathbath` Sanskrit dataset. It also builds a vocabulary and optional n-gram LM artifacts.

Install training dependencies:
```bash
pip install datasets jiwer huggingface-hub pyctcdecode kenlm
```

Run training:
```bash
python train_sanskrit_asr.py
```

## Notes
- `transcribe.py` currently points to a local checkpoint path. Update `CHECKPOINT` if your model lives elsewhere.
- FAISS index is created automatically on first run if it does not exist.

## License
Add your license information here.
