import os
import json
import re
import difflib
import numpy as np
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
try:
    import requests
except ImportError:
    requests = None
try:
    from google import genai
except ImportError:
    genai = None

_DEPENDENCIES_OK = None
_SentenceTransformer = None
_faiss = None
_DEPENDENCY_ERROR = None

_LLM_READY = False
_LLM_ERROR = None
_LLM_MODEL = None
_LLM_CLIENT = None
_LLM_ENABLED = None

if load_dotenv is not None:
    load_dotenv()

def _lazy_import_deps():
    global _DEPENDENCIES_OK, _SentenceTransformer, _faiss, _DEPENDENCY_ERROR
    if _DEPENDENCIES_OK is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer as _ST
        import faiss as _faiss_mod
        _SentenceTransformer = _ST
        _faiss = _faiss_mod
        _DEPENDENCIES_OK = True
    except Exception as exc:
        _DEPENDENCIES_OK = False
        _DEPENDENCY_ERROR = exc

# Mock Dataset of famous Shlokas (You can expand this later by loading a full JSON)
SHLOKA_DB = [
    {
        "id": "Custom-1",
        "source": "Evening Prayer",
        "chapter": 0,
        "verse": 0,
        "sanskrit": "शुभं करोति कल्याणमारोग्यं धनसम्पदा ।\nशत्रुबुद्धिविनाशाय दीपज्योतिर्नमोऽस्तु ते ॥",
        "transliteration": "śhubhaṁ karoti kalyāṇam ārogyaṁ dhana-sampadā\nśhatru-buddhi-vināśhāya dīpa-jyotir namo 'stu te",
        "meaning": "I salute the light of the lamp, which brings auspiciousness, prosperity, good health, abundance of wealth, and the destruction of the intellect's enemy (ignorance)."
    },
    {
        "id": "BG-2.47",
        "source": "Bhagavad Gita",
        "chapter": 2,
        "verse": 47,
        "sanskrit": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।\nमा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
        "transliteration": "karmaṇy-evādhikāras te mā phaleṣhu kadāchana\nmā karma-phala-hetur bhūr mā te saṅgo ’stvakarmani",
        "meaning": "You have a right to perform your prescribed duty, but you are not entitled to the fruits of action. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty."
    },
    {
        "id": "BG-4.7",
        "source": "Bhagavad Gita",
        "chapter": 4,
        "verse": 7,
        "sanskrit": "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत ।\nअभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥",
        "transliteration": "yadā yadā hi dharmasya glānir bhavati bhārata\nabhyutthānam adharmasya tadātmānaṁ sṛijāmyaham",
        "meaning": "Whenever and wherever there is a decline in religious practice, O descendant of Bharata, and a predominant rise of irreligion—at that time I descend Myself."
    },
    {
        "id": "BG-4.8",
        "source": "Bhagavad Gita",
        "chapter": 4,
        "verse": 8,
        "sanskrit": "परित्राणाय साधूनां विनाशाय च दुष्कृताम् ।\nधर्मसंस्थापनार्थाय सम्भवामि युगे युगे ॥",
        "transliteration": "paritrāṇāya sādhūnāṁ vināśhāya cha duṣhkṛitām\ndharma-sansthāpanārthāya sambhavāmi yuge yuge",
        "meaning": "To deliver the pious and to annihilate the miscreants, as well as to reestablish the principles of religion, I Myself appear, millennium after millennium."
    },
    {
        "id": "BG-2.23",
        "source": "Bhagavad Gita",
        "chapter": 2,
        "verse": 23,
        "sanskrit": "नैनं छिन्दन्ति शस्त्राणि नैनं दहति पावकः ।\nन चैनं क्लेदयन्त्यापो न शोषयति मारुतः ॥",
        "transliteration": "nainaṁ chhindanti śhastrāṇi nainaṁ dahati pāvakaḥ\nna chainaṁ kledayantyāpo na śhoṣhayati mārutaḥ",
        "meaning": "The soul can never be cut to pieces by any weapon, nor burned by fire, nor moistened by water, nor withered by the wind."
    },
    {
        "id": "BG-18.66",
        "source": "Bhagavad Gita",
        "chapter": 18,
        "verse": 66,
        "sanskrit": "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज ।\nअहं त्वां सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः ॥",
        "transliteration": "sarva-dharmān parityajya mām ekaṁ śharaṇaṁ vraja\nahaṁ tvāṁ sarva-pāpebhyo mokṣhayiṣhyāmi mā śhuchaḥ",
        "meaning": "Abandon all varieties of religion and just surrender unto Me. I shall deliver you from all sinful reactions. Do not fear."
    }
]

INDEX_FILE = "shloka_index.faiss"
WIKI_TIMEOUT_SECONDS = 8
WIKI_SITES = ("sa.wikisource", "sa.wikipedia", "en.wikisource", "en.wikipedia")
WIKI_QUERY_MAX_CHARS = 80
DEBUG_EXTERNAL = os.getenv("SHLOKA_EXTERNAL_DEBUG", "").strip().lower() in {"1", "true", "yes"}
DEBUG_LLM = os.getenv("SHLOKA_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes"}
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def _normalize_devanagari(text: str) -> str:
    text = re.sub(r"[^\u0900-\u097F ]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def _extract_devanagari(text: str) -> str:
    if not text:
        return ""
    # Keep the longest Devanagari-rich chunk.
    chunks = re.findall(r"[\u0900-\u097F\s।॥]+", text)
    chunks = [re.sub(r"\s+", " ", c).strip() for c in chunks]
    chunks = [c for c in chunks if len(c) >= 10]
    if not chunks:
        return ""
    return max(chunks, key=len)

def _extract_json(text: str):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        if cleaned.startswith("{") and "}" not in cleaned:
            # Attempt to repair truncated JSON.
            cleaned = cleaned + "}"
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if not match:
                return None
        else:
            return None
    payload = match.group(0).strip()
    payload = payload.replace("\r", " ").replace("\n", " ")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        payload = re.sub(r",\s*}", "}", payload)
        payload = re.sub(r",\s*]", "]", payload)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

def _llm_ready() -> bool:
    global _LLM_READY, _LLM_ERROR, _LLM_MODEL, _LLM_CLIENT, _LLM_ENABLED
    if _LLM_ENABLED is not None:
        return _LLM_READY
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    enabled = os.getenv("SHLOKA_LLM_SEARCH", "1").strip().lower() in {"1", "true", "yes"}
    _LLM_ENABLED = enabled and bool(api_key)
    if not _LLM_ENABLED:
        return False
    if genai is None:
        _LLM_ERROR = RuntimeError("google-genai not installed")
        return False
    try:
        _LLM_CLIENT = genai.Client(api_key=api_key)
        _LLM_MODEL = DEFAULT_GEMINI_MODEL
        _LLM_READY = True
    except Exception as exc:
        _LLM_ERROR = exc
        _LLM_READY = False
    return _LLM_READY

def _llm_identify(query_text: str):
    if not _llm_ready():
        if DEBUG_LLM and _LLM_ERROR:
            print(f"[llm] not ready: {_LLM_ERROR}")
        return None
    prompt = (
        "You are an expert Sanskrit shloka identifier. "
        "Given a noisy Sanskrit transcription, identify the most likely shloka or verse. "
        "Return ONLY a single-line JSON object with keys: source, title, sanskrit, transliteration, meaning, "
        "confidence (0-100), url. No markdown, no code fences. If unsure, return {}.\n\n"
        f"Transcription:\n{query_text}\n"
    )
    try:
        response = _LLM_CLIENT.models.generate_content(
            model=_LLM_MODEL,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 512,
                "response_mime_type": "application/json",
            },
        )
        text = response.text if hasattr(response, "text") else ""
        if DEBUG_LLM:
            print(f"[llm] raw={text}")
        data = _extract_json(text)
        if not data:
            if DEBUG_LLM:
                print("[llm] no json parsed")
            return None
        if not data.get("sanskrit"):
            return None
        return {
            "id": f"gemini:{data.get('title', 'unknown')}",
            "source": data.get("source", "Gemini"),
            "chapter": 0,
            "verse": 0,
            "sanskrit": data.get("sanskrit", ""),
            "transliteration": data.get("transliteration", ""),
            "meaning": data.get("meaning", ""),
            "confidence": float(data.get("confidence", 0.0)),
            "distance": None,
            "url": data.get("url", ""),
        }
    except Exception as exc:
        if DEBUG_LLM:
            print(f"[llm] failed: {exc}")
        return None

def _build_search_queries(text: str) -> list:
    # Build multiple shorter queries to improve wiki search recall.
    queries = []
    normalized = _normalize_devanagari(text)
    if normalized:
        words = normalized.split()
        if words:
            queries.append(" ".join(words[:6]))
            queries.append(" ".join(words[:12]))
            queries.append(" ".join(words[-6:]))
            mid_start = max(0, (len(words) // 2) - 3)
            queries.append(" ".join(words[mid_start:mid_start + 6]))
        queries.append(normalized[:WIKI_QUERY_MAX_CHARS])
    if text:
        queries.append(text[:WIKI_QUERY_MAX_CHARS])
    # Deduplicate and keep non-trivial queries.
    deduped = []
    for q in queries:
        q = q.strip()
        if len(q) < 6 or q in deduped:
            continue
        deduped.append(q)
    return deduped

def _wiki_search(site: str, query_text: str, limit: int = 3):
    base_url = f"https://{site}.org/w/api.php"
    headers = {"User-Agent": "sanskrit-asr/1.0"}
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query_text,
        "format": "json",
        "srlimit": limit,
    }
    response = requests.get(base_url, params=params, headers=headers, timeout=WIKI_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    return data.get("query", {}).get("search", [])

def _wiki_extract(site: str, pageid: int):
    base_url = f"https://{site}.org/w/api.php"
    headers = {"User-Agent": "sanskrit-asr/1.0"}
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "pageids": pageid,
        "format": "json",
    }
    response = requests.get(base_url, params=params, headers=headers, timeout=WIKI_TIMEOUT_SECONDS)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    page = pages.get(str(pageid), {})
    return page.get("extract", "")

def _external_identify(query_text: str):
    if requests is None:
        return None
    if not os.getenv("SHLOKA_EXTERNAL_SEARCH", "1").strip().lower() in {"1", "true", "yes"}:
        return None
    candidates = []
    queries = _build_search_queries(query_text)
    if DEBUG_EXTERNAL:
        print(f"[external-search] queries={queries}")
    for site in WIKI_SITES:
        for search_query in queries:
            try:
                results = _wiki_search(site, search_query, limit=3)
                if DEBUG_EXTERNAL:
                    print(f"[external-search] site={site} query='{search_query}' results={len(results)}")
                for item in results:
                    pageid = item.get("pageid")
                    title = item.get("title", "")
                    snippet = re.sub(r"<.*?>", "", item.get("snippet", ""))
                    extract = _wiki_extract(site, pageid) if pageid else ""
                    devanagari = _extract_devanagari(extract) or _extract_devanagari(snippet)
                    url = f"https://{site}.org/wiki/{title.replace(' ', '_')}"
                    if DEBUG_EXTERNAL:
                        dev_len = len(devanagari) if devanagari else 0
                        print(f"[external-search] hit title='{title}' dev_len={dev_len} url={url}")
                    candidates.append({
                        "title": title,
                        "snippet": snippet,
                        "extract": extract,
                        "devanagari": devanagari,
                        "url": url,
                        "site": site,
                    })
            except Exception:
                if DEBUG_EXTERNAL:
                    print(f"[external-search] site={site} query='{search_query}' failed")
                continue

    if not candidates:
        if DEBUG_EXTERNAL:
            print("[external-search] no candidates")
        return None

    normalized_query = _normalize_devanagari(query_text)
    best = None
    best_score = 0.0
    for cand in candidates:
        if cand["devanagari"]:
            candidate_text = _normalize_devanagari(cand["devanagari"])
        else:
            candidate_text = _normalize_devanagari(cand["snippet"])
        if not candidate_text:
            continue
        score = difflib.SequenceMatcher(None, normalized_query, candidate_text).ratio()
        if DEBUG_EXTERNAL:
            print(f"[external-search] score={score:.4f} title='{cand['title']}'")
        if score > best_score:
            best_score = score
            best = cand

    if not best:
        if DEBUG_EXTERNAL:
            print("[external-search] no best match")
        return None

    return {
        "id": f"{best['site']}:{best['title']}",
        "source": "Wikipedia" if "wikipedia" in best["site"] else "Wikisource",
        "chapter": 0,
        "verse": 0,
        "sanskrit": best["devanagari"] or best["snippet"],
        "transliteration": "",
        "meaning": best["extract"] or best["snippet"],
        "confidence": round(best_score * 100, 1),
        "distance": None,
        "url": best["url"],
    }

class ShlokaIdentifier:
    def __init__(self):
        _lazy_import_deps()
        if not _DEPENDENCIES_OK:
            raise RuntimeError(
                "Shloka identification requires: pip install sentence-transformers faiss-cpu"
            ) from _DEPENDENCY_ERROR
        # We use a multilingual model that understands Hindi/Sanskrit embeddings quite well
        print("Loading SentenceTransformer model (this may take a moment)...")
        self.model = _SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self._load_or_build_index()

    def _load_or_build_index(self):
        if os.path.exists(INDEX_FILE):
            print("Loading existing FAISS index...")
            self.index = _faiss.read_index(INDEX_FILE)
        else:
            print("Building FAISS index for the first time...")
            self.index = _faiss.IndexFlatL2(self.dimension)
            
            # Extract just the Sanskrit text for embedding
            texts = [shloka["sanskrit"].replace("\n", " ") for shloka in SHLOKA_DB]
            
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Add to FAISS index
            self.index.add(embeddings)
            _faiss.write_index(self.index, INDEX_FILE)
            print("FAISS index built and saved!")

    def identify(self, query_text: str, top_k: int = 1):
        """Search the FAISS index for the closest matching Shloka."""
        # Clean up query
        query_text = query_text.strip()
        if not query_text:
            return None

        llm_match = _llm_identify(query_text)
        if llm_match:
            return llm_match

        external_match = _external_identify(query_text)
        if external_match:
            return external_match
            
        # Encode query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx < len(SHLOKA_DB):
                match = SHLOKA_DB[idx].copy()
                
                # Convert L2 distance to a pseudo-confidence score (0 to 100%)
                # L2 distance for normalized vectors typically ranges 0 to 4. 
                # Smaller distance = higher confidence
                confidence = max(0, 100 - (dist * 10)) 
                
                match["confidence"] = round(confidence, 1)
                match["distance"] = float(dist)
                results.append(match)
                
        return results[0] if results else None

# Singleton instance to be used by FastAPI
_identifier_instance = None

def get_shloka_identifier():
    global _identifier_instance
    if _identifier_instance is None:
        _identifier_instance = ShlokaIdentifier()
    return _identifier_instance
