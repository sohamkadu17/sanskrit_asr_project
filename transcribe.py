"""
Sanskrit ASR — Quick Inference Script
Uses checkpoint-8000 (41% WER, ~2.65 epochs trained).
Usage:
    python transcribe.py audio1.wav audio2.wav ...
    python transcribe.py --mic          # record from microphone (requires sounddevice)
"""

import os, re, sys, unicodedata
import torch
import numpy as np

os.environ["HF_HOME"]               = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"]    = "D:/hf_cache/hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

CHECKPOINT = "D:/sanskrit_asr_project/sanskrit_asr/checkpoint-15100"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ──────────────────────────────────────────────────────────────
print(f"Loading model from {CHECKPOINT} on {DEVICE}…")
processor = Wav2Vec2Processor.from_pretrained(CHECKPOINT)
model     = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT).to(DEVICE).eval()
print("Model ready.\n")


def transcribe(audio_array: np.ndarray, sampling_rate: int = 16000) -> str:
    """Transcribe a 1-D float32 numpy audio array sampled at `sampling_rate` Hz."""
    if sampling_rate != 16000:
        # Simple resample via scipy if needed
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(16000, sampling_rate)
        audio_array = resample_poly(audio_array, 16000 // g, sampling_rate // g)

    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription


def transcribe_file(path: str) -> str:
    """Load a wav/mp3/flac/mp4 file and return its transcription."""
    import soundfile as sf
    import subprocess
    import tempfile
    import shutil
    
    # Handle MP4 and other container formats by converting to wav first
    if path.lower().endswith(('.mp4', '.m4a', '.aac', '.webm', '.ogg')):
        try:
            print(f"  Converting to WAV with ffmpeg…")
            
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Use the isolated imageio-ffmpeg binary
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            result = subprocess.run([
                ffmpeg_path, '-i', path, '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', tmp_path, '-y'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                import os as os_module
                os_module.unlink(tmp_path)
                print(f"\n  ⚠️  FFmpeg DLL conflict detected (error {result.returncode})")
                print(f"  \n  QUICK FIX: Convert manually outside the Conda environment:")
                print(f"  1. Open Command Prompt (not PowerShell)")
                print(f"  2. Run this command:")
                print(f"     ffmpeg -i \"{path}\" -acodec pcm_s16le -ar 16000 -ac 1 audio_converted.wav")
                print(f"  3. Then run:")
                print(f"     python transcribe.py audio_converted.wav")
                raise RuntimeError(f"FFmpeg DLL issue - use manual conversion above")
            
            # Load the temp wav file
            audio_array, sr = sf.read(tmp_path, dtype="float32")
            
            # Clean up temp file
            import os as os_module
            os_module.unlink(tmp_path)
            
        except Exception as e:
            print(f"  Error: {e}")
            raise
    else:
        audio_array, sr = sf.read(path, dtype="float32")
        if audio_array.ndim > 1:          # stereo → mono
            audio_array = audio_array.mean(axis=1)
    
    return transcribe(audio_array, sr)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py file1.wav [file2.wav ...]")
        sys.exit(0)

    if sys.argv[1] == "--mic":
        import sounddevice as sd
        DURATION = 5   # seconds
        SR       = 16000
        print(f"Recording {DURATION}s from microphone… (speak now)")
        audio = sd.rec(int(DURATION * SR), samplerate=SR,
                       channels=1, dtype="float32")
        sd.wait()
        result = transcribe(audio.squeeze(), SR)
        print(f"\nTranscription: {result}")
    else:
        for path in sys.argv[1:]:
            print(f"File: {path}")
            result = transcribe_file(path)
            print(f"  → {result}\n")
