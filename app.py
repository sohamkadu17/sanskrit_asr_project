import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re
from difflib import SequenceMatcher

# Import the existing transcribe function from your script
from transcribe import transcribe_file
from shloka_search import get_shloka_identifier

app = FastAPI()

# Allow CORS so the frontend can easily communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize_for_comparison(text):
    # Keep only Devanagari characters (U+0900 to U+097F) and spaces
    text = re.sub(r"[^\u0900-\u097F ]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def get_pronunciation_feedback(expected, actual):
    norm_expected = normalize_for_comparison(expected)
    norm_actual = normalize_for_comparison(actual)
    
    matcher = SequenceMatcher(None, norm_expected, norm_actual)
    score = round(matcher.ratio() * 100, 1)
    
    # Generate highlighted HTML string showing mistakes
    # Green = Correct, Red underline = Replaced, Red strikethrough = Missed, Yellow = Extra sound
    feedback_html = ""
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            feedback_html += f'<span style="color: #4cd137;">{norm_expected[i1:i2]}</span>'
        elif tag == 'replace':
            feedback_html += f'<span style="color: #e84118; text-decoration: underline;" title="You said: {norm_actual[j1:j2]}">{norm_expected[i1:i2]}</span>'
        elif tag == 'delete':
            feedback_html += f'<span style="color: #e84118; text-decoration: line-through;" title="Missed">{norm_expected[i1:i2]}</span>'
        elif tag == 'insert':
            feedback_html += f'<span style="color: #fbc531;" title="Extra sound inserted">[{norm_actual[j1:j2]}]</span>'
            
    return {"score": score, "html": feedback_html}

@app.post("/api/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    try:
        # Get original file extension (.wav, .mp3, .webm, etc.)
        suffix = os.path.splitext(audio.filename)[1]
        if not suffix:
            suffix = ".webm" # default for browser recording
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        
        # Call your existing PyTorch model script!
        print(f"Transcribing {audio.filename}...")
        result = transcribe_file(tmp_path)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return JSONResponse({"transcription": result})
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/identify_shloka")
async def api_identify_shloka(audio: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(audio.filename)[1]
        if not suffix: suffix = ".webm"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        
        # 1. Transcribe
        print(f"Transcribing for Identification...")
        transcription = transcribe_file(tmp_path)
        os.unlink(tmp_path)
        
        # 2. Identify
        identifier = get_shloka_identifier()
        match = identifier.identify(transcription)
        
        pronunciation = None
        if match:
            # Compare what the user spoke (transcription) with the actual Sanskrit Shloka
            pronunciation = get_pronunciation_feedback(match["sanskrit"], transcription)
            match["pronunciation"] = pronunciation
        
        return JSONResponse({
            "transcription": transcription,
            "match": match
        })
    except Exception as e:
        print(f"Error in identification: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/shlokas")
def get_shlokas():
    from shloka_search import SHLOKA_DB
    return JSONResponse(SHLOKA_DB)

@app.post("/api/practice_shloka")
async def api_practice_shloka(audio: UploadFile = File(...), target_id: str = Form(...)):
    try:
        from shloka_search import SHLOKA_DB
        target_shloka = next((s for s in SHLOKA_DB if s["id"] == target_id), None)
        if not target_shloka:
            return JSONResponse({"error": "Shloka not found"}, status_code=404)

        suffix = os.path.splitext(audio.filename)[1]
        if not suffix: suffix = ".webm"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        
        print(f"Transcribing Practice Audio...")
        transcription = transcribe_file(tmp_path)
        os.unlink(tmp_path)
        
        pronunciation = get_pronunciation_feedback(target_shloka["sanskrit"], transcription)
        
        return JSONResponse({
            "transcription": transcription,
            "target": target_shloka,
            "pronunciation": pronunciation
        })
    except Exception as e:
        print(f"Error in practice: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Serve the static UI files AFTER API routes
os.makedirs("ui", exist_ok=True)
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    print("Starting Sanskrit ASR Web UI...")
    print("Open your browser and navigate to: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
