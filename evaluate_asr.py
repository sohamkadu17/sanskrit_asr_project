import argparse
import os
import re
import sys
import site
import unicodedata
from typing import List
import csv
import json

import torch
import numpy as np
import jiwer
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Match training-time cache settings (avoid C: drive)
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/hf_cache/hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Devanagari digit (0-9) to spoken form
_DEVA_DIGIT_WORDS = {
    "\u0966": "\u0936\u0942\u0928\u094d\u092f",  # 0 -> shunya
    "\u0967": "\u090f\u0915",                    # 1 -> ek
    "\u0968": "\u0926\u094b",                    # 2 -> do
    "\u0969": "\u0924\u0940\u0928",              # 3 -> teen
    "\u096a": "\u091a\u093e\u0930",              # 4 -> chaar
    "\u096b": "\u092a\u093e\u0901\u091a",        # 5 -> paanch
    "\u096c": "\u091b\u0939",                    # 6 -> chhah
    "\u096d": "\u0938\u093e\u0924",              # 7 -> saat
    "\u096e": "\u0906\u0920",                    # 8 -> aath
    "\u096f": "\u0928\u094c",                    # 9 -> nau
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)

    for digit, word in _DEVA_DIGIT_WORDS.items():
        text = text.replace(digit, f" {word} ")

    text = re.sub(r"[\u0964\u0965\u0951-\u0954\u0970]", " ", text)
    text = re.sub(r"[^\u0900-\u097F ]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def decode_greedy(logits: np.ndarray, processor: Wav2Vec2Processor) -> List[str]:
    pred_ids = np.argmax(logits, axis=-1)
    return processor.batch_decode(pred_ids)


def _disable_user_site_packages():
    # Prevent mixed user-site packages from crashing native deps (pyarrow/pandas).
    user_site = site.getusersitepackages()
    if user_site in sys.path:
        sys.path.remove(user_site)
    os.environ["PYTHONNOUSERSITE"] = "1"


def load_dataset_sanskrit(hf_token: str):
    # Import datasets only when needed to avoid pyarrow import during module load.
    from datasets import load_dataset, Audio, DatasetDict

    dataset = load_dataset("ai4bharat/Kathbath", "sanskrit", token=hf_token)

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    dataset = dataset.rename_column("audio_filepath", "audio")
    # Disable built-in decoding to avoid torchcodec dependency.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    return dataset


def load_audio_from_path(path: str, target_sr: int = 16000) -> np.ndarray:
    audio_array, sr = sf.read(path, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    if sr != target_sr:
        # Resample only when needed to keep evaluation consistent.
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, sr)
        audio_array = resample_poly(audio_array, target_sr // g, sr // g)

    return audio_array


def load_manifest(path: str):
    items = []
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".tsv"]:
        delimiter = "," if ext == ".csv" else "\t"
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                audio_path = row.get("audio_path") or row.get("path") or row.get("audio")
                text = row.get("text") or row.get("transcript") or row.get("transcription")
                if audio_path and text is not None:
                    items.append({"audio": {"path": audio_path}, "text": text})
    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                audio_path = row.get("audio_path") or row.get("path") or row.get("audio")
                text = row.get("text") or row.get("transcript") or row.get("transcription")
                if isinstance(audio_path, dict):
                    audio_path = audio_path.get("path")
                if audio_path and text is not None:
                    items.append({"audio": {"path": audio_path}, "text": text})
    else:
        raise ValueError("Manifest must be .csv, .tsv, or .jsonl")

    if not items:
        raise ValueError("Manifest is empty or missing required columns")

    return items


def get_batch(split, start: int, batch_size: int):
    batch = split[start : start + batch_size]
    if isinstance(batch, dict):
        audio_items = batch["audio"]
        texts = batch["text"]
    else:
        audio_items = [item["audio"] for item in batch]
        texts = [item["text"] for item in batch]
    return audio_items, texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sanskrit ASR model.")
    parser.add_argument(
        "--model",
        default="D:/sanskrit_asr_project/sanskrit_asr/checkpoint-15100",
        help="Path to model checkpoint or model dir",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of samples (0 = all)",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional CSV/TSV/JSONL with columns: audio_path,text",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Force CUDA if available",
    )
    args = parser.parse_args()

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {args.model}")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device).eval()

    if args.manifest:
        split = load_manifest(args.manifest)
    else:
        from huggingface_hub import get_token
        hf_token = get_token()
        if hf_token is None:
            raise RuntimeError(
                "No Hugging Face token found. Run: huggingface-cli login"
            )
        _disable_user_site_packages()
        dataset = load_dataset_sanskrit(hf_token)
        split = dataset[args.split]

    if args.max_samples and args.max_samples > 0:
        limit = min(args.max_samples, len(split))
        if hasattr(split, "select"):
            split = split.select(range(limit))
        else:
            split = split[:limit]

    references = []
    hypotheses = []

    print(f"Evaluating split: {args.split} | samples: {len(split)}")

    for i in range(0, len(split), args.batch_size):
        audio_items, texts = get_batch(split, i, args.batch_size)
        audio_arrays = [load_audio_from_path(a["path"]) for a in audio_items]
        texts = [normalize(t) for t in texts]

        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits.detach().cpu().numpy()

        pred_str = decode_greedy(logits, processor)
        hypotheses.extend([normalize(p) for p in pred_str])
        references.extend(texts)

        if (i // args.batch_size) % 50 == 0:
            print(f"Processed {min(i + args.batch_size, len(split))}/{len(split)}")

    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)

    print("\nEvaluation Results")
    print("------------------")
    print(f"WER: {wer * 100:.2f}%")
    print(f"CER: {cer * 100:.2f}%")


if __name__ == "__main__":
    main()
