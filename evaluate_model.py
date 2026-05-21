import os
import re
import json
import unicodedata
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import jiwer
from datasets import load_dataset, Audio, DatasetDict
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from huggingface_hub import get_token

# Redirect HF caches to D drive (match training setup)
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/hf_cache/hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_PATH = "D:/sanskrit_asr_project/sanskrit_asr/checkpoint-15100"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DEVA_DIGIT_WORDS = {
    "\u0966": "\u0936\u0942\u0928\u094D\u092F",   # ० -> शून्य
    "\u0967": "\u090F\u0915",                     # १ -> एक
    "\u0968": "\u0926\u094B",                     # २ -> दो
    "\u0969": "\u0924\u0940\u0928",             # ३ -> तीन
    "\u096A": "\u091A\u093E\u0930",             # ४ -> चार
    "\u096B": "\u092A\u093E\u0901\u091A",       # ५ -> पाँच
    "\u096C": "\u091B\u0939",                     # ६ -> छह
    "\u096D": "\u0938\u093E\u0924",             # ७ -> सात
    "\u096E": "\u0906\u0920",                     # ८ -> आठ
    "\u096F": "\u0928\u094C",                     # ९ -> नौ
}


def normalize(text: str) -> str:
    # Normalize to NFC for consistent Devanagari representation.
    text = unicodedata.normalize("NFC", text)

    # Expand Devanagari digits to spoken words.
    for digit, word in _DEVA_DIGIT_WORDS.items():
        text = text.replace(digit, f" {word} ")

    # Remove written-only symbols and accents.
    text = re.sub(r"[\u0964\u0965\u0951-\u0954\u0970]", " ", text)

    # Keep only Devanagari block + spaces.
    text = re.sub(r"[^\u0900-\u097F ]", "", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    audio_arrays = [a["array"] for a in batch["audio"]]

    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors=None,
        padding="do_not_pad",
    )
    batch["input_values"] = inputs.input_values

    batch["labels"] = processor(
        text=[normalize(t) for t in batch["text"]],
        return_tensors=None,
    ).input_ids

    return batch


def load_dataset_sanskrit():
    hf_token = get_token()
    if hf_token is None:
        raise RuntimeError(
            "No Hugging Face token found. Run: huggingface-cli login"
        )

    dataset = load_dataset("ai4bharat/Kathbath", "sanskrit", token=hf_token)
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    dataset = dataset.rename_column("audio_filepath", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def decode_batch(logits, processor):
    # Greedy decoding (fast). Replace with beam search if needed.
    pred_ids = np.argmax(logits, axis=-1)
    return processor.batch_decode(pred_ids)


def evaluate(model, processor, dataset, split="validation", batch_size=4):
    model.eval()
    device = next(model.parameters()).device

    ds = dataset[split]
    ds = ds.map(
        lambda b: prepare_dataset(b, processor),
        batched=True,
        batch_size=8,
        remove_columns=ds.column_names,
        load_from_cache_file=True,
    )

    data_collator = DataCollatorCTCWithPadding(processor)

    # Manual batching to avoid Trainer dependency.
    total_refs = []
    total_hyps = []

    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]
        collated = data_collator(batch)

        input_values = collated["input_values"].to(device)
        with torch.no_grad():
            logits = model(input_values).logits.cpu().numpy()

        pred_str = decode_batch(logits, processor)

        label_ids = collated["labels"].cpu().numpy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        total_hyps.extend(pred_str)
        total_refs.extend(label_str)

        if (i // batch_size) % 50 == 0:
            print(f"Processed {min(i + batch_size, len(ds))}/{len(ds)}")

    wer = jiwer.wer(total_refs, total_hyps)
    return wer


def main():
    print(f"Loading processor and model from {MODEL_PATH}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(DEVICE)

    dataset = load_dataset_sanskrit()

    wer = evaluate(model, processor, dataset, split="validation", batch_size=4)
    print(f"Validation WER: {wer:.4f}")


if __name__ == "__main__":
    main()
