import os
import re
import json
import torch
import jiwer
import unicodedata
import numpy as np

# Redirect ALL Hugging Face caches to D drive (prevents C drive from filling up)
os.environ["HF_HOME"]              = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"]    = "D:/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"]   = "D:/hf_cache/hub"
os.environ["HUGGINGFACE_HUB_CACHE"]= "D:/hf_cache/hub"

# Suppress Windows symlink warning in HuggingFace cache (cosmetic only, cache still works)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset, Audio, DatasetDict
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)

from huggingface_hub import get_token
from dataclasses import dataclass
from typing import Union

# ════════════════════════════════════════════════════════════════════════════
# PyTorch 2.6+ Compatibility: Force weights_only=False for old checkpoints
# ════════════════════════════════════════════════════════════════════════════
# Our checkpoints were created with PyTorch < 2.6 and contain pickle objects that
# fail with weights_only=True. The Transformers Trainer library calls torch.load
# with weights_only=True explicitly, which fails. We force it to False globally
# since we created these checkpoints ourselves and they're trusted.

_torch_load_original = torch.load

def _torch_load_safe(f, *args, **kwargs):
    """Load torch files with weights_only=False (force override for old checkpoints)."""
    # ALWAYS use weights_only=False for our checkpoints, override any caller setting
    kwargs['weights_only'] = False
    return _torch_load_original(f, *args, **kwargs)

torch.load = _torch_load_safe

MODEL_NAME   = "facebook/wav2vec2-large-xlsr-53"
_decoder     = None   # set in __main__ once tokenizer is ready
_reorder_ids = None   # vocab index permutation: blank at 0 (pyctcdecode requirement)

# Devanagari digit (०–९) → Sanskrit spoken word form
# Digits appear in text but audio says the word, so map them before training.
_DEVA_DIGIT_WORDS = {
    "\u0966": "\u0936\u0942\u0928\u094D\u092F",   # ० → शून्य
    "\u0967": "\u090F\u0915",                       # १ → एक
    "\u0968": "\u0926\u094B",                       # २ → दो
    "\u0969": "\u0924\u0940\u0928",                 # ३ → तीन
    "\u096A": "\u091A\u093E\u0930",                 # ४ → चार
    "\u096B": "\u092A\u093E\u0901\u091A",           # ५ → पाँच
    "\u096C": "\u091B\u0939",                       # ६ → छह
    "\u096D": "\u0938\u093E\u0924",                 # ७ → सात
    "\u096E": "\u0906\u0920",                       # ८ → आठ
    "\u096F": "\u0928\u094C",                       # ९ → नौ
}


def normalize(text):
    # 1. NFC: collapse multi-codepoint sequences into single composed codepoints.
    #    Critical for Sanskrit — the same akshara can be encoded multiple ways.
    text = unicodedata.normalize("NFC", text)

    # 2. Expand Devanagari digits to spoken word forms.
    #    Audio says "ek", not the glyph "१", so the label must match.
    for digit, word in _DEVA_DIGIT_WORDS.items():
        text = text.replace(digit, f" {word} ")

    # 3. Remove written-only symbols that are never vocalised:
    #    - Danda ।  (U+0964) and double-danda ॥ (U+0965) — sentence punctuation
    #    - Vedic accent marks (U+0951–U+0954) — prosodic annotations
    #    - Abbreviation sign ॰ (U+0970)
    text = re.sub(r"[\u0964\u0965\u0951-\u0954\u0970]", " ", text)

    # 4. Strip everything outside Devanagari Unicode block + ASCII space.
    #    Sandhi-fused forms are valid Devanagari and kept as-is — the model
    #    learns sandhi from audio without needing external splitting.
    text = re.sub(r"[^\u0900-\u097F ]", "", text)

    # 5. Collapse runs of whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def prepare_dataset(batch):

    audio_arrays = [a["array"] for a in batch["audio"]]

    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors=None,
        padding="do_not_pad"
    )
    batch["input_values"] = inputs.input_values

    batch["labels"] = processor(
        text=[normalize(t) for t in batch["text"]],
        return_tensors=None
    ).input_ids

    return batch



@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):

        input_features = [
            {"input_values": f["input_values"]}
            for f in features
        ]

        label_features = [
            {"input_ids": f["labels"]}
            for f in features
        ]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        batch["labels"] = labels

        return batch



def compute_metrics(pred):
    logits = pred.predictions          # shape: (batch, time, vocab)

    if _decoder is not None:
        # Beam-search CTC decoding (+ n-gram LM if loaded).
        # Reorder the vocab dim so [PAD]/blank sits at index 0 — required by
        # pyctcdecode, which always assumes labels[0] is the CTC blank token.
        reordered = logits[:, :, _reorder_ids]
        pred_str  = [_decoder.decode(logit) for logit in reordered]
    else:
        # Fallback: greedy argmax decoding (used if pyctcdecode isn't installed).
        pred_ids = np.argmax(logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}



if __name__ == '__main__':
    # 1 LOAD DATASET
    ########################################

    hf_token = get_token()

    if hf_token is None:
        raise RuntimeError(
            "No Hugging Face token found.\n"
            "Run this command in your terminal and paste your token:\n\n"
            "    huggingface-cli login\n\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )

    dataset = load_dataset(
        "ai4bharat/Kathbath",
        "sanskrit",
        token=hf_token
    )

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)

        dataset = DatasetDict({
            "train": split["train"],
            "validation": split["test"]
        })

    print("Train size:", len(dataset["train"]))
    print("Validation size:", len(dataset["validation"]))

    ########################################
    # 2 AUDIO PROCESSING
    ########################################

    dataset = dataset.rename_column("audio_filepath", "audio")

    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000)
    )

    ########################################
    # 3 TEXT NORMALIZATION
    ########################################

    ########################################
    # 4 BUILD VOCABULARY
    ########################################

    all_chars = set()
    for text in dataset["train"]["text"] + dataset["validation"]["text"]:
        all_chars.update(normalize(text))

    # Ensure space is present so word delimiter mapping doesn't KeyError
    all_chars.add(" ")

    vocab_list = sorted(all_chars)

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open("vocab.json", "w") as f:
        json.dump(vocab_dict, f)

    ########################################
    # 4b BUILD N-GRAM LM CORPUS
    ########################################
    # Requires:  pip install kenlm pyctcdecode
    # kenlm bundles lmplz + build_binary binaries; on Windows you may also need
    # Visual C++ Build Tools so it can compile from source.

    import pathlib, subprocess

    _LM_DIR    = pathlib.Path("D:/sanskrit_asr_project")
    _LM_CORPUS = _LM_DIR / "lm_corpus.txt"
    _LM_ARPA   = _LM_DIR / "lm.arpa"
    _LM_BIN    = _LM_DIR / "lm.binary"
    _LM_ORDER  = 3   # trigram

    _corpus_texts = dataset["train"]["text"] + dataset["validation"]["text"]
    _LM_CORPUS.write_text(
        "\n".join(normalize(t) for t in _corpus_texts),
        encoding="utf-8"
    )
    print(f"LM corpus: {len(_corpus_texts)} sentences -> {_LM_CORPUS}")

    _lm_model = None
    try:
        import kenlm as _kenlm
        
        # Note: On Windows, KenLM binaries (lmplz, build_binary) are not distributed
        # with the pip wheel. Training will proceed with beam search without LM.
        # For production LM integration, consider:
        # - Training on Linux/WSL2 where KenLM compiles with binaries
        # - Or using a pre-trained .binary file
        
        if not _LM_BIN.exists():
            print(f"Skipping LM training (KenLM binaries unavailable on Windows)")
            print(f"  Model will use beam search without n-gram LM")
        else:
            _lm_model = _kenlm.Model(str(_LM_BIN))
            print(f"Loaded {_LM_ORDER}-gram LM ({_LM_BIN.stat().st_size // 1024:,} KB)")
    except Exception as _e:
        print(f"KenLM not available: {type(_e).__name__}")

    ########################################
    # 5 PROCESSOR
    ########################################

    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    ########################################
    # 5b CTC BEAM-SEARCH DECODER
    ########################################

    try:
        from pyctcdecode import build_ctcdecoder
        import inspect

        # Build vocab list ordered by token-id so _labels[i] ↔ logits[:, :, i].
        _blank_id  = processor.tokenizer.pad_token_id
        _vocab_inv = {idx: tok for tok, idx in processor.tokenizer.get_vocab().items()}
        _n_vocab   = len(_vocab_inv)

        # pyctcdecode always reads _labels[0] as the CTC blank token, but our
        # tokenizer places [PAD] at the last index.  Fix: rotate the index list
        # so blank comes first, and apply the same rotation in compute_metrics.
        _other_ids   = [i for i in range(_n_vocab) if i != _blank_id]
        _reorder_ids = [_blank_id] + _other_ids   # module-level, used in compute_metrics

        _labels = [
            ""  if _vocab_inv[_reorder_ids[i]] == "[PAD]" else   # blank → ""
            " " if _vocab_inv[_reorder_ids[i]] == "|"     else   # word delimiter → space
            _vocab_inv[_reorder_ids[i]]
            for i in range(_n_vocab)
        ]

        # Check if this version of pyctcdecode supports kenlm_model parameter
        _build_sig = inspect.signature(build_ctcdecoder)
        _supports_kenlm = 'kenlm_model' in _build_sig.parameters
        
        if _supports_kenlm and _lm_model:
            _decoder = build_ctcdecoder(
                _labels,
                kenlm_model=_lm_model,   # None → pure beam search; Model → beam + LM
                alpha=0.5,               # LM interpolation weight
                beta=0.5,                # word insertion bonus
                beam_width=32,           # Reduce from default 100 for faster evaluation
            )
        else:
            # Older pyctcdecode or no LM: pure beam search only
            # beam_width=32 is much faster than default 100 with minimal accuracy loss
            _decoder = build_ctcdecoder(_labels, beam_width=32)
        
        print(f"CTC decoder ready (beam search, LM={'yes' if (_supports_kenlm and _lm_model) else 'no'})")

    except Exception as _dec_err:
        print(f"Beam-search decoder unavailable ({type(_dec_err).__name__}: {_dec_err})")
        print("Falling back to greedy decoding.")

    ########################################
    # 6 PREPARE DATA
    ########################################

    train_columns = dataset["train"].column_names
    dataset = dataset.map(
        prepare_dataset,
        batched=True,
        batch_size=8,
        remove_columns=train_columns,
        load_from_cache_file=True
    )

    ########################################
    # 7 MODEL
    ########################################

    # Load weights from the latest local checkpoint (or fall back to base model).
    import glob, os as _os
    _ckpt_dirs = sorted(
        glob.glob("D:/sanskrit_asr_project/sanskrit_asr/checkpoint-*"),
        key=lambda p: int(p.rsplit("-", 1)[-1])
    )
    CHECKPOINT = _ckpt_dirs[-1] if _ckpt_dirs else "facebook/wav2vec2-large-xlsr-53"
    print(f"Loading model from: {CHECKPOINT}")

    model = Wav2Vec2ForCTC.from_pretrained(
        CHECKPOINT,
        ignore_mismatched_sizes=True,
        local_files_only=bool(_ckpt_dirs),
    )

    # SpecAugment
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_time_length = 10

    # Regularization
    model.config.layerdrop = 0.05

    # Stability
    model.config.ctc_zero_infinity = True

    # Required when gradient_checkpointing=True to avoid cache conflict
    model.config.use_cache = False

    # Freeze CNN feature encoder — only fine-tune transformer layers
    # This saves memory and stabilizes training on small datasets
    model.freeze_feature_encoder()

    ########################################
    # 8 DATA COLLATOR
    ########################################

    data_collator = DataCollatorCTCWithPadding(processor)

    ########################################
    # 9 METRICS
    ########################################

    ########################################
    # 10 TRAINING CONFIG
    ########################################

    training_args = TrainingArguments(

        output_dir="D:/sanskrit_asr_project/sanskrit_asr",

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,    # keep eval from OOMing on large XLSR model
        gradient_accumulation_steps=8,   # keep effective batch = 8

        eval_accumulation_steps=4,       # offload logits to CPU every 4 steps to save GPU memory

        eval_strategy="steps",

        # Continuation run: 5 more epochs × ceil(24156/8) = 15100 steps.
        # Fresh cosine cycle starting from checkpoint-8000 weights.
        # Lower peak LR (1e-4) because the model is already partially fine-tuned.
        max_steps=15100,

        learning_rate=1e-4,

        warmup_steps=300,

        lr_scheduler_type="cosine",

        fp16=True,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_grad_norm=1.0,

        logging_steps=100,

        save_steps=500,

        eval_steps=2000,

        save_total_limit=2,

        dataloader_num_workers=2,

        report_to="none"
    )

    ########################################
    # 11 TRAINER
    ########################################

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        processing_class=processor,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    ########################################
    # START TRAINING
    ########################################

    # Resume from the latest checkpoint: restores model weights, step counter,
    # and optimizer/scheduler state where compatible. Modern Transformers handles
    # batch size mismatches gracefully by reinitializing the optimizer if needed.
    
    _resume_checkpoint = CHECKPOINT if _ckpt_dirs else None
    
    trainer.train(resume_from_checkpoint=_resume_checkpoint)

    ########################################
    # SAVE MODEL
    ########################################

    model.save_pretrained("D:/sanskrit_asr_project/sanskrit_asr_model")

    processor.save_pretrained("D:/sanskrit_asr_project/sanskrit_asr_model")
