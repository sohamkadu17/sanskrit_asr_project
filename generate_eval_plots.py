import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import colors as mcolors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

try:
    from datasets import load_dataset, Audio, DatasetDict
except ImportError as exc:
    raise ImportError("Please install datasets: pip install datasets") from exc

try:
    import soundfile as sf
except ImportError as exc:
    raise ImportError("Please install soundfile: pip install soundfile") from exc

try:
    from sklearn.metrics import confusion_matrix
except ImportError as exc:
    raise ImportError("Please install scikit-learn: pip install scikit-learn") from exc


def latest_checkpoint(base_dir: str) -> str | None:
    paths = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not paths:
        return None

    def step(path: str) -> int:
        try:
            return int(path.rsplit("-", 1)[-1])
        except Exception:
            return -1

    return sorted(paths, key=step)[-1]


def normalize_devanagari(text: str) -> str:
    import re as _re
    text = _re.sub(r"[^\u0900-\u097F ]", "", text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text


def add_noise_snr(audio: np.ndarray, snr_db):
    audio = np.asarray(audio, dtype="float32")
    if snr_db == "clean":
        return audio
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    noise = np.random.normal(0.0, 1.0, size=audio.shape)
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-8
    desired_noise_rms = rms / (10 ** (float(snr_db) / 20.0))
    noise = noise * (desired_noise_rms / noise_rms)
    return audio + noise


def align_tokens(ref_tokens, hyp_tokens):
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )

    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                op = "equal" if cost == 0 else "substitute"
                aligned.append((ref_tokens[i - 1], hyp_tokens[j - 1], op))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            aligned.append((ref_tokens[i - 1], None, "delete"))
            i -= 1
        else:
            aligned.append((None, hyp_tokens[j - 1], "insert"))
            j -= 1
    return list(reversed(aligned)), dp


def infer_single(processor, model, audio_array, sampling_rate, device):
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_text = processor.batch_decode(pred_ids)[0]
    return pred_text, logits.squeeze(0).cpu().numpy()


def parse_train_output(path: str):
    if not os.path.exists(path):
        return []
    logs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "loss" in line or "eval_" in line:
                try:
                    if line.startswith("{") and line.endswith("}"):
                        logs.append(json.loads(line))
                except Exception:
                    continue
    return logs


_AUDIO_PATH_CACHE: dict[str, str] = {}


def _resolve_audio_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if path in _AUDIO_PATH_CACHE:
        return _AUDIO_PATH_CACHE[path]

    candidates = []
    env_cache = os.environ.get("HF_DATASETS_CACHE")
    if env_cache:
        candidates.append(env_cache)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(os.path.join(hf_home, "datasets"))

    candidates.append(os.path.expanduser("~/.cache/huggingface/datasets"))

    basename = os.path.basename(path)
    for base in candidates:
        if not base or not os.path.isdir(base):
            continue
        matches = glob.glob(os.path.join(base, "**", basename), recursive=True)
        if matches:
            _AUDIO_PATH_CACHE[path] = matches[0]
            return matches[0]

    return path


def _load_audio_file(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    resolved = _resolve_audio_path(path)
    audio, sr = sf.read(resolved, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise ImportError("Please install scipy for resampling: pip install scipy") from exc
        from math import gcd
        g = gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
        sr = target_sr
    return audio, sr


def _load_audio_source(audio_meta, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    if isinstance(audio_meta, dict):
        audio_bytes = audio_meta.get("bytes")
        if audio_bytes:
            import io
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        else:
            audio_path = audio_meta.get("path")
            if not audio_path:
                raise RuntimeError("Audio metadata missing path/bytes.")
            audio, sr = _load_audio_file(audio_path, target_sr=target_sr)
    else:
        audio, sr = _load_audio_file(str(audio_meta), target_sr=target_sr)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise ImportError("Please install scipy for resampling: pip install scipy") from exc
        from math import gcd
        g = gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
        sr = target_sr
    return audio, sr


def main():
    parser = argparse.ArgumentParser(description="Generate ASR evaluation plots.")
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--top-words", type=int, default=15)
    parser.add_argument("--top-chars", type=int, default=30)
    parser.add_argument("--snr-levels", nargs="+", default=["clean", "20", "10", "0"])
    parser.add_argument("--model-dir", default="./sanskrit_asr_model")
    parser.add_argument("--checkpoint-dir", default="./sanskrit_asr")
    parser.add_argument("--train-log", default="train_output.txt")
    parser.add_argument("--dataset-name", default="ai4bharat/Kathbath")
    parser.add_argument("--dataset-config", default="sanskrit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-vector", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
    })

    def save_figure(path_base: str):
        plt.tight_layout()
        plt.savefig(f"{path_base}.png", bbox_inches="tight", dpi=300)
        if args.save_vector:
            plt.savefig(f"{path_base}.pdf", bbox_inches="tight")

    if os.path.isdir(args.model_dir):
        processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    else:
        ckpt = latest_checkpoint(args.checkpoint_dir)
        if ckpt is None:
            raise RuntimeError("Model checkpoint not found.")
        processor = Wav2Vec2Processor.from_pretrained(ckpt)
        model = Wav2Vec2ForCTC.from_pretrained(ckpt)

    model = model.to(device).eval()

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    dataset = dataset.rename_column("audio_filepath", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    def prepare_audio(batch):
        audio_meta = batch["audio"]
        audio, sr = _load_audio_source(audio_meta, target_sr=16000)
        batch["speech"] = audio
        batch["sampling_rate"] = sr
        return batch

    columns_to_remove = [
        col for col in dataset["train"].column_names if col not in ("text", "speech", "sampling_rate")
    ]
    dataset = dataset.map(prepare_audio, remove_columns=columns_to_remove)
    dataset = dataset.map(lambda x: {"text": normalize_devanagari(x["text"])})

    val_ds = dataset["validation"]
    subset_size = min(args.subset_size, len(val_ds))
    val_ds = val_ds.select(range(subset_size))

    records = []
    for idx in range(len(val_ds)):
        row = val_ds[idx]
        audio = row["speech"]
        sr = row["sampling_rate"]
        ref = row["text"]
        start = time.perf_counter()
        hyp, logits = infer_single(processor, model, audio, sr, device)
        duration = len(audio) / float(sr)
        elapsed = time.perf_counter() - start
        rtf = elapsed / max(duration, 1e-6)
        records.append({
            "idx": idx,
            "ref": ref,
            "hyp": hyp,
            "wer": wer(ref, hyp),
            "rtf": rtf,
            "logits": logits,
            "duration": duration,
        })
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(val_ds)}")

    df = pd.DataFrame(records)
    print(df[["wer", "rtf"]].describe())

    worst = df.sort_values("wer", ascending=False).iloc[0]
    ref_words = worst["ref"].split()
    hyp_words = worst["hyp"].split()
    _, dp_matrix = align_tokens(ref_words, hyp_words)
    plt.figure(figsize=(8, 8))
    sns.heatmap(dp_matrix, cmap="viridis", cbar=True, square=True)
    plt.title("Normalized Word-Level Alignment Matrix")
    plt.xlabel("Hypothesis tokens")
    plt.ylabel("Reference tokens")
    save_figure(os.path.join(args.output_dir, "levenshtein_matrix_words"))
    plt.close()

    word_pairs = []
    char_pairs = []
    for _, row in df.iterrows():
        ref_norm = normalize_devanagari(row["ref"])
        hyp_norm = normalize_devanagari(row["hyp"])
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        aligned_words, _ = align_tokens(ref_words, hyp_words)
        for r, h, _op in aligned_words:
            word_pairs.append((r if r is not None else "<eps>", h if h is not None else "<eps>"))
        ref_chars = list(ref_norm.replace(" ", ""))
        hyp_chars = list(hyp_norm.replace(" ", ""))
        aligned_chars, _ = align_tokens(ref_chars, hyp_chars)
        for r, h, _op in aligned_chars:
            char_pairs.append((r if r is not None else "<eps>", h if h is not None else "<eps>"))

    word_counts = pd.Series([r for r, _ in word_pairs]).value_counts()
    top_words = list(word_counts.head(args.top_words).index)
    labels_words = top_words + ["<eps>", "<other>"]

    def ascii_label(text: str, prefix: str, idx: int, max_len: int = 10) -> str:
        try:
            cleaned = text.encode("ascii", "ignore").decode("ascii")
        except Exception:
            cleaned = ""
        cleaned = cleaned.strip()
        if not cleaned:
            return f"{prefix}{idx}"
        return cleaned[:max_len]

    word_label_map = {w: ascii_label(w, "w", i) for i, w in enumerate(labels_words)}

    def map_word(w):
        if w in top_words or w == "<eps>":
            return w
        return "<other>"

    word_y = [map_word(r) for r, _ in word_pairs]
    word_pred = [map_word(h) for _, h in word_pairs]
    cm_words = confusion_matrix(word_y, word_pred, labels=labels_words, normalize="true")
    word_tick_labels = [word_label_map[w] for w in labels_words]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_words,
        cmap="magma",
        xticklabels=word_tick_labels,
        yticklabels=word_tick_labels,
        norm=mcolors.PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0),
        square=True,
        linewidths=0.3,
        linecolor="gray",
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Normalized Word-Level Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Reference")
    plt.xticks(rotation=45, ha="right")
    save_figure(os.path.join(args.output_dir, "confusion_words"))
    plt.close()

    char_labels = sorted(set([c for c, _ in char_pairs] + [c for _, c in char_pairs]))
    if len(char_labels) > args.top_chars:
        char_counts = pd.Series([r for r, _ in char_pairs]).value_counts()
        char_labels = list(char_counts.head(args.top_chars).index) + ["<eps>", "<other>"]

        def map_char(c):
            if c in char_labels or c == "<eps>":
                return c
            return "<other>"

        char_y = [map_char(r) for r, _ in char_pairs]
        char_pred = [map_char(h) for _, h in char_pairs]
        cm_chars = confusion_matrix(char_y, char_pred, labels=char_labels, normalize="true")
    else:
        char_y = [r for r, _ in char_pairs]
        char_pred = [h for _, h in char_pairs]
        cm_chars = confusion_matrix(char_y, char_pred, labels=char_labels, normalize="true")

    char_label_map = {c: ascii_label(c, "c", i) for i, c in enumerate(char_labels)}
    char_tick_labels = [char_label_map[c] for c in char_labels]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_chars,
        cmap="magma",
        xticklabels=char_tick_labels,
        yticklabels=char_tick_labels,
        norm=mcolors.PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0),
        square=True,
        linewidths=0.3,
        linecolor="gray",
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Normalized Character-Level Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Reference")
    plt.xticks(rotation=45, ha="right")
    save_figure(os.path.join(args.output_dir, "confusion_chars"))
    plt.close()

    label_map_path = os.path.join(args.output_dir, "labels_mapping.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "word_labels": word_label_map,
                "char_labels": char_label_map,
                "labels_words": labels_words,
                "labels_chars": char_labels,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    history = parse_train_output(args.train_log)
    if history:
        hist_df = pd.DataFrame(history)
        fig, ax1 = plt.subplots(figsize=(9, 5))
        if "loss" in hist_df:
            ax1.plot(hist_df["loss"].dropna().reset_index(drop=True), label="train_loss")
        if "eval_loss" in hist_df:
            ax1.plot(hist_df["eval_loss"].dropna().reset_index(drop=True), label="eval_loss")
        ax1.set_xlabel("Log step")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax2 = ax1.twinx()
        if "eval_wer" in hist_df:
            ax2.plot(hist_df["eval_wer"].dropna().reset_index(drop=True), color="red", label="eval_wer")
            ax2.set_ylabel("WER")
        plt.title("Loss and WER Curves")
        save_figure(os.path.join(args.output_dir, "loss_wer_curves"))
        plt.close()
    else:
        print("No training log history found for loss/WER curves.")

    plt.figure(figsize=(9, 5))
    rtf_values = np.clip(df["rtf"].to_numpy(), 1e-4, None)
    plt.scatter(rtf_values, df["wer"] * 100.0, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Real-Time Factor (RTF, log scale)")
    plt.ylabel("WER (%)")
    plt.title("RTF vs Accuracy")
    save_figure(os.path.join(args.output_dir, "rtf_vs_wer"))
    plt.close()

    sample = df.iloc[0]
    logits = sample["logits"]
    ref_text = normalize_devanagari(sample["ref"])
    ref_ids = processor.tokenizer(ref_text).input_ids
    if ref_ids:
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        token_probs = probs[:, ref_ids]
        plt.figure(figsize=(14, 5))
        vmax = float(np.percentile(token_probs, 99.0))
        sns.heatmap(
            token_probs.T,
            cmap="inferno",
            cbar=True,
            vmin=0.0,
            vmax=max(vmax, 1e-6),
            cbar_kws={"shrink": 0.8},
        )
        plt.title("CTC Logit Alignment Heatmap")
        plt.xlabel("Time frames")
        plt.ylabel("Reference token index")
        save_figure(os.path.join(args.output_dir, "ctc_alignment_heatmap"))
        plt.close()
    else:
        print("Skipping alignment heatmap: empty reference tokens.")

    robust_records = []
    for snr in args.snr_levels:
        snr_value = "clean" if str(snr).lower() == "clean" else float(snr)
        wers = []
        for idx in range(len(val_ds)):
            row = val_ds[idx]
            audio = np.asarray(row["speech"], dtype="float32")
            sr = row["sampling_rate"]
            noisy = add_noise_snr(audio, snr_value)
            hyp, _ = infer_single(processor, model, noisy, sr, device)
            wers.append(wer(row["text"], hyp))
        robust_records.append({"snr": snr_value, "wer": float(np.mean(wers))})
    robust_df = pd.DataFrame(robust_records)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=robust_df, x="snr", y="wer")
    plt.title("Noise Robustness (WER vs SNR)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("WER")
    save_figure(os.path.join(args.output_dir, "robustness_wer_snr"))
    plt.close()

    print("Saved figures to:", args.output_dir)


if __name__ == "__main__":
    main()
