**IMPLEMENTION & CODING**

**4.1 DATABASE SCHEMA (IF APPLICABLE)**
- Storage model: In-memory JSON list plus FAISS vector index file; no relational DB.
- Shloka record schema:
```json
{
  "id": "string",
  "source": "string",
  "chapter": "number",
  "verse": "number",
  "sanskrit": "string",
  "transliteration": "string",
  "meaning": "string"
}
```
- Search index: `shloka_index.faiss` stores embeddings for `sanskrit` field.
- Optional match payload: `confidence`, `distance`, and `url` when external/LLM match is used.

**4.2 GUI DESIGN**
- Layout: Single-page UI with glassmorphism panel, animated gradient orbs, and clear input/output zones.
- Key components:
  - Mode selector: Auto-identify or practice a specific shloka.
  - Record button with pulse animation and status text.
  - Drag-and-drop upload area with browse fallback.
  - Loader with spinner during transcription.
  - Output panel with transcription, matched shloka, transliteration, meaning, and pronunciation highlights.
- Typography: Inter for UI text; Yantramanav for Devanagari output.
- Responsive: Input panels stack vertically on narrow screens.

**4.3 ALGORITHMS / FLOW CHARTS**

A) ASR + Shloka Identification Flow
```mermaid
flowchart TD
A[Audio Input: Mic/File] --> B[Backend API Upload]
B --> C[Transcribe Audio (Wav2Vec2)]
C --> D{Mode?}
D -->|Auto Identify| E[Shloka Identifier]
D -->|Practice| F[Lookup Target Shloka]
E --> G[FAISS Similarity + Optional External/LLM]
G --> H[Best Match + Confidence]
F --> H
H --> I[Pronunciation Feedback (SequenceMatcher)]
I --> J[JSON Response to UI]
J --> K[Render Output + Highlights]
```

B) Pronunciation Feedback Algorithm
- Normalize text (keep only Devanagari + spaces).
- Compute sequence alignment between expected and actual.
- Highlight rules:
  - Green = correct
  - Red underline = replaced
  - Red strike = missed
  - Yellow = extra insertion
- Score = alignment ratio x 100

**4.4 OPERATIONAL DETAILS (MODULE-WISE DESCRIPTION AND SAMPLE CODE) / SITE MAPS**

A) Module: ASR Inference
- Purpose: Load fine-tuned Wav2Vec2 model and return transcription.
- Sample code:
```python
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
logits = model(inputs.input_values.to(DEVICE)).logits
predicted_ids = torch.argmax(logits, dim=-1)
text = processor.batch_decode(predicted_ids)[0]
```

B) Module: Shloka Identification
- Purpose: Find closest shloka using FAISS embeddings; fallback to external/LLM if enabled.
- Sample code:
```python
query_embedding = self.model.encode([query_text], convert_to_numpy=True)
distances, indices = self.index.search(query_embedding, top_k)
```

C) Module: Pronunciation Feedback
- Purpose: Character-level alignment and colored feedback HTML.
- Sample code:
```python
matcher = SequenceMatcher(None, norm_expected, norm_actual)
score = round(matcher.ratio() * 100, 1)
```

D) Module: Backend API
- Purpose: Accept audio, transcribe, identify/practice, and return JSON.
- Endpoints (site map for API):
  - POST /api/transcribe
  - POST /api/identify_shloka
  - POST /api/practice_shloka
  - GET /api/shlokas

E) Module: Frontend UI
- Purpose: Record/upload audio, call APIs, render results.
- Main UI sections (site map for UI):
  - Header (title + instructions)
  - Mode selector + practice preview
  - Record button panel
  - Upload drop zone
  - Loader
  - Output panel

**TESTING**

**5.1 ACCEPTANCE TESTING**
| ID | Test Case | Steps | Expected Result |
|---|---|---|---|
| AT-01 | Auto-identify from mic | Select Auto -> Record 5s -> Stop | Transcription shown, matched shloka card shown |
| AT-02 | Practice mode feedback | Select a shloka -> Record -> Stop | Transcription + pronunciation score + highlights |
| AT-03 | File upload flow | Drag .wav -> Upload | Loader shows then transcription appears |
| AT-04 | Non-audio file | Upload .txt file | Alert: Please upload an audio file |
| AT-05 | Backend down | Upload audio | UI shows connection error |
| AT-06 | Empty speech | Record silence | No crash; empty/minimal transcription |
| AT-07 | Unknown shloka | Speak random text | No matching shloka found |

**5.2 UNIT TESTING (MODULE WISE)**
| Module | Test Case | Input | Expected Output |
|---|---|---|---|
| ASR transcribe | 16kHz audio array | Known sample | Non-empty Devanagari string |
| ASR transcribe_file | .webm file | Valid audio | Converts to WAV, returns transcription |
| Normalize | Text with punctuation | "karmanyeva..." | Cleaned text, punctuation removed |
| Shloka search | Exact shloka | Known shloka text | Top match is same ID |
| Pronunciation feedback | Expected vs actual | Small mismatch | Score < 100 and HTML highlights |

**5.3 INTEGRATION TESTING**
| ID | Test Case | Steps | Expected Result |
|---|---|---|---|
| IT-01 | UI -> Backend -> ASR | Upload audio | UI renders transcription |
| IT-02 | UI -> Identify -> FAISS | Speak known shloka | Correct shloka card returned |
| IT-03 | UI -> Practice -> Feedback | Select shloka -> record | Feedback highlights shown |
| IT-04 | External/LLM disabled | Disable flags | FAISS match still returned |
| IT-05 | Multi-format audio | Upload .mp3/.webm | Transcription completes |

**RESULTS & DISCUSSION**

**6.1 MAIN GUI SNAPSHOTS (AS APPLICABLE)**
- Snapshot 1: Home UI with mode selector and record/upload panels.
- Snapshot 2: Transcription output after auto-identify (shloka card + meaning).
- Snapshot 3: Practice mode with pronunciation feedback highlights and score.
- Snapshot 4: Loader state during transcription.

Results Discussion (concise)
- The ASR pipeline transcribes Sanskrit audio using a fine-tuned Wav2Vec2 model with reported WER around 41% at checkpoint-15100.
- FAISS similarity search provides reliable identification for known shlokas in the local database.
- The pronunciation feedback module gives immediate, interpretable guidance with highlighted errors.
- The web UI offers an end-to-end workflow (record -> transcribe -> identify -> feedback) that is responsive and user-friendly.
