import streamlit as st
import pymupdf
import numpy as np
import re
import json
import time
import io
import os
import hashlib
from PIL import Image
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# =========================================================
# SETTINGS & CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Samruk Republic | Document Intelligence",
    page_icon="🇰🇿",
    layout="wide"
)

# Fixed, small model list — NOT dynamically pulled from the API.
# Reproducibility matters for research: you need to be able to say
# in your methodology exactly which model(s) produced your results.
# Order = priority. If the first hits a rate limit / is unavailable,
# the wrapper below falls back to the next one.
#
# NOTE (checked Aug 2026): gemini-2.0-flash was shut down June 1, 2026.
# gemini-2.5-flash is scheduled to shut down Oct 16, 2026 — do not build
# on it. Current stable GA models are gemini-3.6-flash (primary workhorse)
# and gemini-3.5-flash-lite (cheaper/faster fallback, also GA).
MODEL_PRIORITY = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


# =========================================================
# TITLE & BRANDING
# =========================================================

st.title("🇰🇿 Samruk Republic — Document Analyzer")
st.subheader("Evidence-grounded document QA with claim-level validation")

st.write(
    "Upload an official PDF document and ask a question about it. "
    "Every answer is broken into claims, and each claim is independently "
    "checked against the source document."
)

st.divider()


# =========================================================
# GEMINI CLIENT
# =========================================================

# [FIX] Request timeout so a hung network call cannot freeze the live demo.
API_TIMEOUT_MS = 30_000

# [FIX] "Sticky" fallback: if a model just failed with overload / timeout
# (503, 504, 429), skip it for a few minutes instead of waiting for it to
# fail again on every call. On Nov 16 2026 gemini-3.6-flash answered
# 503/504 for most requests; without this, every question took ~3 minutes
# because each call (answer + every claim validation) first waited for the
# primary model to time out.
MODEL_COOLDOWN_SECONDS = 300
_model_cooldown_until = {}


def _read_api_key():
    """
    [FIX] Reads the key from .streamlit/secrets.toml (GOOGLE_API_KEY or
    GEMINI_API_KEY) and falls back to environment variables. The original
    st.secrets.get(...) raises an exception (instead of returning None)
    when secrets.toml does not exist at all, which crashed the app before
    the friendly error message could be shown.
    """
    api_key = None
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        try:
            api_key = st.secrets.get(name, None)
        except Exception:
            api_key = None
        if api_key:
            break
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        api_key = str(api_key).strip().strip('"').strip("'").strip()
    return api_key or None


@st.cache_resource
def get_genai_client():
    api_key = _read_api_key()

    if not api_key:
        st.error(
            "GOOGLE_API_KEY not found. Add it to "
            ".streamlit/secrets.toml as:\n\n"
            'GOOGLE_API_KEY = "your_key_here"'
        )
        st.stop()

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=API_TIMEOUT_MS)
    )


client = get_genai_client()

st.sidebar.caption(f"🤖 Models (priority order): `{', '.join(MODEL_PRIORITY)}`")


# =========================================================
# CALL WITH FALLBACK ACROSS THE FIXED MODEL LIST + RETRY
# =========================================================

def call_gemini_with_fallback(
    contents,
    response_mime_type=None,
    temperature=0.1,
    max_retries_per_model=2,
    base_delay=3
):
    """
    Tries MODEL_PRIORITY in order. For each model, retries on rate-limit
    errors (429 / RESOURCE_EXHAUSTED) with a short backoff before moving
    to the next model. Returns (text, model_name_used, error_message).

    error_message is None on success, or the last error seen across all
    attempts if every model failed — callers can show this to the user
    or log it, instead of failing silently.
    """

    last_error = None

    now = time.time()
    available = [m for m in MODEL_PRIORITY if _model_cooldown_until.get(m, 0) <= now]
    # Models that recently failed are not dropped, only moved to the end of
    # the list, so they are still tried as a last resort.
    models_to_try = available + [m for m in MODEL_PRIORITY if m not in available]

    for model_index, model_name in enumerate(models_to_try):
        is_last_model = model_index == len(models_to_try) - 1

        for attempt in range(max_retries_per_model):
            try:
                # [FIX] temperature is NOT sent any more: for gemini-3.6-flash
                # and gemini-3.5-flash-lite Google lists temperature/top_p/top_k
                # as deprecated and ignored (and announces HTTP 400 for future
                # models). The argument is kept in the signature so callers
                # do not change; behaviour on the current models is identical.
                config = types.GenerateContentConfig()
                if response_mime_type:
                    config.response_mime_type = response_mime_type

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                # [FIX] response.text can be None (safety block / empty
                # candidate). Previously that was returned as "success" with
                # no error message; now it is treated as a failed attempt so
                # the next model in the list is tried.
                text = response.text
                if text is None or not str(text).strip():
                    raise ValueError("Empty response (no text in candidate)")
                return text, model_name, None

            except Exception as e:
                # Catch broadly, not just ClientError/APIError — network
                # errors, auth errors, SDK-internal errors, etc. should
                # all be visible instead of silently killing the pipeline.
                err_msg = f"{type(e).__name__}: {e}"
                last_error = err_msg
                print(f"[Samruk] Gemini call failed on {model_name} "
                      f"(attempt {attempt + 1}/{max_retries_per_model}): {err_msg}")

                is_rate_limit = (
                    "429" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                    or "quota" in err_msg.lower()
                )
                # NOTE: 503 UNAVAILABLE ("high demand") and 504
                # DEADLINE_EXCEEDED are deliberately NOT retried on the same
                # model: during testing (Nov 16) gemini-3.6-flash returned
                # these repeatedly, and switching straight to the fallback
                # model is much faster for a live demo.
                is_not_found = "404" in err_msg or "NOT_FOUND" in err_msg
                is_overloaded = (
                    is_rate_limit
                    or "503" in err_msg or "UNAVAILABLE" in err_msg
                    or "504" in err_msg or "DEADLINE_EXCEEDED" in err_msg
                    or "timed out" in err_msg.lower() or "timeout" in err_msg.lower()
                )
                if is_overloaded and attempt == max_retries_per_model - 1 or (
                        is_overloaded and not is_rate_limit):
                    _model_cooldown_until[model_name] = time.time() + MODEL_COOLDOWN_SECONDS

                if (is_overloaded and not is_rate_limit and is_last_model
                        and attempt < max_retries_per_model - 1):
                    # last model in the list: one short retry on 503/504
                    time.sleep(2)
                    continue

                if is_rate_limit and ("per day" in err_msg.lower() or "PerDay" in err_msg):
                    # Daily free-tier quota is used up (the free tier allowed
                    # only 20 requests/day for gemini-3.6-flash on Nov 16):
                    # retrying is pointless, go straight to the next model.
                    _model_cooldown_until[model_name] = time.time() + 3600
                    break

                if is_rate_limit and attempt < max_retries_per_model - 1:
                    time.sleep(base_delay * (attempt + 1))
                    continue

                if is_rate_limit or is_not_found:
                    # move on to the next model in the priority list
                    break

                # Any other error: don't keep retrying this model, but
                # still try the next model in the list before giving up.
                break

    return None, None, last_error


# =========================================================
# AI EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


with st.spinner("Loading semantic embedding model..."):
    embed_model = load_model()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_pdf_text(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    bullet_chars = ["\x07", "\x08", "▪", "◦", "●", "■"]
    for char in bullet_chars:
        text = text.replace(char, "•")

    text = re.sub(r"(?<=\w)\n(?=\w)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


# =========================================================
# OCR: Gemini Vision primary, local Tesseract best-effort fallback
# =========================================================
#
# Honest description of what this actually is: two OCR paths, tried
# in order. It is NOT a guaranteed-uptime failover system — if Gemini
# Vision fails AND Tesseract isn't installed on the machine running
# this app, the page is simply skipped with an empty string. Tesseract
# requires the binary to be installed on the host (not just pip
# installed), so don't rely on it for a live demo on an unknown machine.

def extract_text_with_tesseract(page):
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="kaz+rus+eng").strip()
    except Exception:
        return ""


def extract_text_with_ocr(page):
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")

    text, model_used, gemini_error = call_gemini_with_fallback(
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            (
                "Extract all text from this document page exactly as "
                "written. Preserve paragraph structure. Return only the "
                "extracted text, no commentary, no markdown formatting."
            )
        ],
        temperature=0.0
    )

    if text and text.strip():
        return text.strip(), f"Gemini Vision ({model_used})"

    tesseract_text = extract_text_with_tesseract(page)
    if tesseract_text:
        return tesseract_text, "Local Tesseract (fallback)"

    if not TESSERACT_AVAILABLE:
        tesseract_note = "pytesseract not installed"
    else:
        tesseract_note = "Tesseract returned no text (binary likely not installed/configured)"

    reason = gemini_error or "Gemini returned empty text"
    return "", f"Failed — Gemini: {reason} | Tesseract: {tesseract_note}"


# =========================================================
# SENTENCE SPLITTING & CHUNKING (with overlap)
# =========================================================

def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def create_chunks(text, page_number, max_chars=1200, overlap=250):
    """
    Creates text chunks, forcing a chunk boundary at every paragraph break.

    Official Kazakh documents often start pages with a letterhead block
    (organization name, address, phone, reference number) immediately
    followed by the substantive content, with no sentence-ending
    punctuation between them. Without a forced paragraph boundary, that
    boilerplate gets packed into the same chunk as the actual content,
    diluting its embedding and hurting retrieval for exactly the kind
    of dates/numbers questions this project cares about most.
    """
    text = clean_pdf_text(text)

    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        sentences = split_into_sentences(paragraph)

        for sentence in sentences:
            if len(sentence) > max_chars:
                if current_chunk:
                    chunks.append({"page": page_number, "text": current_chunk.strip()})
                    current_chunk = ""
                chunks.append({"page": page_number, "text": sentence})
                continue

            if not current_chunk:
                current_chunk = sentence
            elif len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += " " + sentence
            else:
                chunks.append({"page": page_number, "text": current_chunk.strip()})
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence

        # Force a boundary at the end of each paragraph instead of letting
        # the next paragraph's text merge into this chunk.
        if current_chunk:
            chunks.append({"page": page_number, "text": current_chunk.strip()})
            current_chunk = ""

    return chunks


# =========================================================
# CACHED PDF PROCESSING (extraction + OCR + chunking + embeddings)
# =========================================================

# [FIX] Upper bound on how many scanned pages are sent to Gemini Vision in
# one document. Each OCR page is one API call (~5-15 s), so a 100-page scan
# would block the demo for many minutes and burn the free-tier quota.
MAX_OCR_PAGES = 40


@st.cache_data(show_spinner="Extracting, cleaning, chunking and embedding document...")
def process_pdf(pdf_bytes, _embed_model):

    # [FIX] A corrupted or password-protected PDF used to crash the app
    # with a traceback. Now it is reported as a readable error.
    try:
        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return [], [], None, [f"Cannot open PDF: {type(e).__name__}: {e}"], True
    if pdf.needs_pass:
        pdf.close()
        return [], [], None, ["PDF is password-protected — remove the password and upload again."], True

    total_pages = pdf.page_count

    pages = []
    ocr_log = []
    ocr_pages_used = 0
    ocr_failed = False

    progress_bar = st.progress(0, text="Reading document pages...")

    for page_number, page in enumerate(pdf):
        raw_text = page.get_text()

        if len(raw_text.strip()) < 20:
            if ocr_pages_used >= MAX_OCR_PAGES:
                ocr_log.append(
                    f"Page {page_number + 1}: skipped — OCR limit of "
                    f"{MAX_OCR_PAGES} scanned pages per document reached"
                )
                continue
            ocr_pages_used += 1
            progress_bar.progress(
                page_number / total_pages,
                text=f"Page {page_number + 1}/{total_pages} looks like a scan — running OCR..."
            )
            extracted_text, method_used = extract_text_with_ocr(page)
            raw_text = extracted_text
            if not extracted_text:
                ocr_failed = True
            ocr_log.append(f"Page {page_number + 1}: {method_used}")
        else:
            progress_bar.progress(
                (page_number + 1) / total_pages,
                text=f"Reading page {page_number + 1}/{total_pages}..."
            )

        cleaned_text = clean_pdf_text(raw_text)

        if cleaned_text:
            pages.append({"page": page_number + 1, "text": cleaned_text})

    progress_bar.empty()
    pdf.close()

    chunks = []
    for page in pages:
        chunks.extend(
            create_chunks(text=page["text"], page_number=page["page"])
        )

    embeddings = None
    if chunks:
        chunk_texts = [c["text"] for c in chunks]
        embeddings = _embed_model.encode(
            chunk_texts, normalize_embeddings=True, show_progress_bar=False
        )

    return pages, chunks, embeddings, ocr_log, ocr_failed


# =========================================================
# LLM: STRUCTURED ANSWER GENERATION (answer + claims, NO self-graded status)
# =========================================================

ANSWER_PROMPT_TEMPLATE = """You are Samruk Republic, an assistant that answers questions
about official documents using ONLY the information provided in CONTEXT.

Rules:
- Answer only using information found in CONTEXT below.
- If CONTEXT does not contain enough information to answer, say so explicitly
  in the "answer" field and set "insufficient_evidence" to true.
- Break your answer down into short atomic claims (one factual statement each).
- For every claim, list the page numbers from CONTEXT that support it.
- Never include a claim that is not directly supported by CONTEXT.
- Do NOT judge or label your own claims as supported/unsupported — that is
  done by a separate, independent verification step. Just state the claims.
- Respond ONLY with valid JSON matching exactly this schema, no markdown fences,
  no extra commentary:

{{
  "answer": "natural language answer to the question",
  "insufficient_evidence": false,
  "claims": [
    {{"claim": "short factual statement", "source_pages": [1, 2]}}
  ]
}}

QUESTION:
{question}

CONTEXT:
{context}
"""


def build_context(results):
    parts = []
    for r in results:
        parts.append(f"[Page {r['page']}]\n{r['text']}")
    return "\n\n".join(parts)


def generate_structured_answer(question, context):
    prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, context=context)

    raw_text, model_used, error = call_gemini_with_fallback(
        contents=prompt,
        response_mime_type="application/json",
        temperature=0.1
    )

    if not raw_text:
        st.caption(f"Debug — answer generation failed: {error}")
        return {
            "answer": "⚠️ All available models failed (see debug caption above — "
                      "likely a rate limit / daily quota). Wait a bit and try again.",
            "insufficient_evidence": True,
            "claims": []
        }, None

    data = _parse_answer_json(raw_text)
    return data, model_used


def _to_page_list(value):
    """
    [FIX] The model sometimes returns page numbers as strings ("3",
    "стр. 3", "3-4") or a single number instead of a list. The original
    code compared them directly with integer chunk pages, so such claims
    silently found no evidence and were marked "unsupported".
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    pages = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            pages.append(int(item))
            continue
        for m in re.finditer(r"(\d+)\s*[-–]\s*(\d+)|(\d+)", str(item)):
            if m.group(1):
                lo, hi = int(m.group(1)), int(m.group(2))
                if 0 < hi - lo <= 20:
                    pages.extend(range(lo, hi + 1))
                else:
                    pages.extend([lo, hi])
            else:
                pages.append(int(m.group(3)))
    seen = []
    for p in pages:
        if p not in seen:
            seen.append(p)
    return seen


def _parse_answer_json(raw_text):
    """
    [FIX] Tolerates ```json fences and non-dict JSON. The original code
    crashed with AttributeError (answer_data.get) if the model returned a
    JSON list, and showed raw JSON text if it wrapped it in fences.
    """
    text = (raw_text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.S)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, flags=re.S)
        data = None
        if match:
            try:
                data = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                data = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        return {
            "answer": raw_text or "The model returned an unparseable response.",
            "insufficient_evidence": None,
            "claims": []
        }

    claims = []
    raw_claims = data.get("claims")
    if not isinstance(raw_claims, list):
        raw_claims = []
    for c in raw_claims:
        if isinstance(c, dict) and str(c.get("claim", "")).strip():
            claims.append({
                "claim": str(c.get("claim", "")).strip(),
                "source_pages": _to_page_list(c.get("source_pages"))
            })
        elif isinstance(c, str) and c.strip():
            claims.append({"claim": c.strip(), "source_pages": []})
    data["claims"] = claims
    data["answer"] = str(data.get("answer", "") or "")
    return data


# =========================================================
# LLM: INDEPENDENT CLAIM-LEVEL EVIDENCE VALIDATION
# =========================================================
#
# This is the core research component. It is a SEPARATE call from
# answer generation — the model that validates a claim does not see
# that it was the one who produced it. This is what makes the
# supported/unsupported label meaningful evidence, rather than the
# generating model grading its own homework.

VALIDATION_PROMPT_TEMPLATE = """You are a precision fact-checking module.

You are given a CLAIM and an EVIDENCE passage taken from the source document.
Decide whether EVIDENCE supports CLAIM.

A claim is SUPPORTED even if it uses different wording than the evidence,
as long as the meaning matches (e.g. a bullet point "Robotics: Advanced"
supports the claim "lists robotics skill level as Advanced"). Minor OCR
typos in EVIDENCE should not by themselves cause a claim to be marked
unsupported.

Respond with exactly one word: supported, unsupported, or contradicted.

CLAIM:
{claim}

EVIDENCE:
{evidence}
"""


def validate_claim(claim_text, source_pages, chunks):
    evidence_texts = [c["text"] for c in chunks if c["page"] in source_pages]
    evidence = "\n\n".join(evidence_texts)

    if not evidence:
        return "unsupported"

    prompt = VALIDATION_PROMPT_TEMPLATE.format(claim=claim_text, evidence=evidence)

    raw_text, _, error = call_gemini_with_fallback(contents=prompt, temperature=0.0)

    time.sleep(1)  # stay well under free-tier RPM across several claims

    if not raw_text:
        print(f"[Samruk] Claim validation failed: {error}")
        return "unsupported"

    return _parse_validation_label(raw_text)


def _parse_validation_label(raw_text):
    """
    [FIX] Fail-closed label parsing. The original code returned
    "supported" for ANY reply that did not contain the substrings
    "unsupported" or "contradict" — e.g. "Not supported", "NO", an empty
    string or an explanation. That is the opposite of what the paper
    describes (a claim is shown as supported only if the validator says
    so). Now only an explicit "supported" counts as supported.
    """
    label = (raw_text or "").strip().lower()
    first = re.sub(r"[^a-z]", "", label.split()[0]) if label.split() else ""

    if "contradict" in label:
        return "contradicted"
    if ("unsupported" in label or "not supported" in label
            or "not_supported" in label or first == "no"):
        return "unsupported"
    if first == "supported" or label.startswith("supported"):
        return "supported"
    if re.search(r"\bsupported\b", label):
        return "supported"
    return "unsupported"


STATUS_BADGE = {
    "supported": "✅ Supported",
    "unsupported": "⚠️ Unsupported",
    "contradicted": "❌ Contradicted"
}


# =========================================================
# PDF UPLOAD & INPUTS
# =========================================================

# [DEPLOY] Built-in public sample document + example questions, so phone
# visitors (QR code) can try the app without uploading a PDF.
SAMPLE_PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "samples", "bolashak_rules_573.pdf")
SAMPLE_PDF_NAME = "bolashak_rules_573.pdf"
EXAMPLE_QUESTIONS = [
    "В течение какого срока победитель должен заключить договор с администратором?",
    "Какая общая денежная сумма выплачивается по стипендии Болашак?",
    "Из скольких туров состоит конкурсный отбор на стипендию Болашак?",
]


class _SampleFile:
    """[DEPLOY] Mimics st.file_uploader's object for the bundled sample PDF."""
    name = SAMPLE_PDF_NAME

    def getvalue(self):
        with open(SAMPLE_PDF_PATH, "rb") as f:
            return f.read()


def _use_sample():
    st.session_state["use_sample"] = True


def _ask_example(q):
    st.session_state["use_sample"] = True
    st.session_state["question_input"] = q


sample_available = os.path.exists(SAMPLE_PDF_PATH)
if sample_available:
    st.button("📘 Использовать пример: правила стипендии «Болашак»",
              on_click=_use_sample)
    st.caption("Примеры вопросов (нажмите, чтобы задать):")
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        st.button(f"❓ {example}", key=f"example_q_{i}",
                  on_click=_ask_example, args=(example,))

uploaded_file = st.file_uploader("📄 Upload an official PDF document", type=["pdf"])
question = st.text_input("🔎 What do you want to know?", key="question_input")

# [DEPLOY] an uploaded file always wins; otherwise use the sample if chosen
if uploaded_file is None and sample_available and st.session_state.get("use_sample"):
    uploaded_file = _SampleFile()
question = (question or "").strip()  # [FIX] ignore whitespace-only questions

# [FIX] Answers are remembered per (document, question) for this browser
# session, so re-running the same question during the demo (or any rerun of
# the script) does not call Gemini again and cannot hit the rate limit twice.
if "answer_cache" not in st.session_state:
    st.session_state["answer_cache"] = {}


# [DEPLOY] Quota guard for the public (QR-code) version. Only NEW questions
# that call Gemini are counted; repeated questions served from the cache are
# free. Limits are read from secrets / environment; 0 disables a limit.
def _read_int_setting(name, default):
    value = None
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    if value is None:
        value = os.environ.get(name)
    try:
        return max(0, int(value)) if value is not None else default
    except (TypeError, ValueError):
        return default


MAX_QUESTIONS_PER_SESSION = _read_int_setting("MAX_QUESTIONS_PER_SESSION", 3)
MAX_QUESTIONS_PER_DAY = _read_int_setting("MAX_QUESTIONS_PER_DAY", 30)


@st.cache_resource
def _daily_question_counter():
    # [DEPLOY] shared by all visitors of this server process
    return {"day": None, "count": 0}


def _quota_block_message():
    """[DEPLOY] Returns a Russian message if a limit is reached, else None."""
    used = st.session_state.get("questions_asked", 0)
    if MAX_QUESTIONS_PER_SESSION and used >= MAX_QUESTIONS_PER_SESSION:
        return (
            f"Спасибо за интерес к Samruk Republic! В демо-версии можно задать "
            f"не больше {MAX_QUESTIONS_PER_SESSION} новых вопросов за один сеанс, "
            "чтобы бесплатный лимит Gemini хватил всем. Уже заданные вопросы "
            "можно открыть снова. Полную версию покажет автор у стенда."
        )
    counter = _daily_question_counter()
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if counter["day"] != today:
        counter["day"], counter["count"] = today, 0
    if MAX_QUESTIONS_PER_DAY and counter["count"] >= MAX_QUESTIONS_PER_DAY:
        return (
            "На сегодня лимит вопросов демо-версии исчерпан. Спасибо за интерес! "
            "Живую демонстрацию покажет автор у стенда."
        )
    return None


def _count_question():
    """[DEPLOY] Called once per new (non-cached) question."""
    st.session_state["questions_asked"] = st.session_state.get("questions_asked", 0) + 1
    counter = _daily_question_counter()
    counter["count"] += 1


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file:

    st.success(f"Document uploaded: {uploaded_file.name}")

    pdf_bytes = uploaded_file.getvalue()  # [FIX] getvalue() is safe on reruns; read() can return b"" after the first read
    doc_hash = hashlib.sha1(pdf_bytes).hexdigest()
    pages, chunks, embeddings, ocr_log, ocr_failed = process_pdf(pdf_bytes, embed_model)

    if ocr_failed:
        # [FIX] Do not keep a half-processed document in the cache: if OCR
        # failed (e.g. rate limit), the next rerun will try again instead of
        # serving the broken result forever.
        process_pdf.clear()
        st.warning(
            "Some pages could not be read (see OCR details). They will be "
            "retried automatically on the next run (press R or re-upload)."
        )

    st.divider()
    st.header("📄 Document Parsing")

    if pages:
        st.write(f"Successfully extracted text from **{len(pages)} pages**.")
        if ocr_log:
            with st.expander("🛠️ OCR processing details"):
                for entry in ocr_log:
                    st.caption(f"- {entry}")
    else:
        st.error("No text was found in this PDF, even after OCR.")
        if ocr_log:
            st.markdown("**Debug — why OCR failed on each page:**")
            for entry in ocr_log:
                st.caption(entry)

    if chunks:
        st.info(f"Document segmented into **{len(chunks)} semantic text chunks**.")
        with st.expander("🔍 View document chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"### Chunk {i + 1} — Page {chunk['page']}")
                st.write(chunk["text"])
                st.divider()

    if question and not chunks:
        # [FIX] previously the question was silently ignored
        st.error("There is no extracted text to search in, so the question cannot be answered.")

    if question and chunks and embeddings is not None:

        st.divider()
        st.header("🧠 Semantic Retrieval")
        st.write(f"Searching index for: **{question}**")

        with st.spinner("Computing cosine similarities..."):
            question_embedding = embed_model.encode([question], normalize_embeddings=True)[0]
            scores = np.dot(embeddings, question_embedding)
            ranked_indices = np.argsort(scores)[::-1]
            top_k = min(8, len(chunks))

            results = []
            for index in ranked_indices[:top_k]:
                results.append({
                    "page": chunks[index]["page"],
                    "text": chunks[index]["text"],
                    "score": float(scores[index])
                })

        st.divider()
        st.header("🤖 Samruk Answer & Independent Verification")

        cache_key = (doc_hash, question)
        cached = st.session_state["answer_cache"].get(cache_key)

        if cached:
            answer_data, model_used, verified_claims, generation_failed = cached
        else:
            # [DEPLOY] stop before any Gemini call if a limit is reached
            block_message = _quota_block_message()
            if block_message:
                st.info(block_message)
                st.stop()
            _count_question()

            with st.spinner("Generating grounded response..."):
                context = build_context(results)
                answer_data, model_used = generate_structured_answer(question, context)
            generation_failed = model_used is None
            if generation_failed:
                # [DEPLOY] a failed call (no internet / quota) is not counted
                st.session_state["questions_asked"] = max(0, st.session_state.get("questions_asked", 1) - 1)
                _daily_question_counter()["count"] = max(0, _daily_question_counter()["count"] - 1)

            claims = answer_data.get("claims", [])
            verified_claims = []
            if claims:
                # [FIX] Validation now runs BEFORE the answer is displayed, so
                # the final verdict below can depend on it (paper, section
                # "Claim Verification": an unconfirmed claim must not be
                # presented as an established fact).
                progress = st.progress(0, text="Validating each claim against the source document...")
                for n, claim in enumerate(claims):
                    status = validate_claim(
                        claim_text=claim.get("claim", ""),
                        source_pages=claim.get("source_pages", []),
                        chunks=chunks
                    )
                    verified_claims.append({**claim, "status": status})
                    progress.progress((n + 1) / len(claims),
                                      text=f"Validated claim {n + 1}/{len(claims)}")
                progress.empty()

            if not generation_failed:
                st.session_state["answer_cache"][cache_key] = (
                    answer_data, model_used, verified_claims, generation_failed
                )

        if model_used:
            st.caption(f"Answer generated by: `{model_used}`")

        n_supported = sum(1 for c in verified_claims if c["status"] == "supported")
        n_problem = len(verified_claims) - n_supported

        if generation_failed:
            st.error(
                "⚠️ The language model could not be reached (internet / quota). "
                "The retrieved passages below still come from the document — "
                "check the page numbers manually."
            )
            st.write(answer_data.get("answer", ""))

        elif n_supported == 0:
            # Paper: "If the required information is missing, the user is told so."
            st.warning(
                "🤷 **Не знаю / I don't know.** The document does not provide "
                "sufficient confirmed evidence to answer this question."
            )
            if answer_data.get("answer"):
                with st.expander("Model's explanation (not confirmed as a fact)"):
                    st.write(answer_data.get("answer", ""))

        else:
            if answer_data.get("insufficient_evidence"):
                st.warning(
                    "⚠️ The document does not provide sufficient evidence "
                    "to fully answer this question."
                )
            st.markdown("#### Answer")
            st.write(answer_data.get("answer", ""))
            if n_problem:
                st.warning(
                    f"{n_problem} of {len(verified_claims)} statements could NOT be "
                    "confirmed by the document (⚠️/❌ below). Rely only on the ✅ statements."
                )

        if verified_claims:
            st.markdown("#### Claims & Independent Verification")
            for claim in verified_claims:
                badge = STATUS_BADGE.get(claim["status"], claim["status"])
                pages_str = ", ".join(str(p) for p in claim.get("source_pages", [])) or "—"
                st.markdown(f"- {badge} — {claim.get('claim', '')} *(page {pages_str})*")

        st.divider()
        st.header("📌 Retrieved Context Passages")
        st.write(f"Found **{len(results)} relevant context passages**.")

        for result_number, result in enumerate(results):
            score_percent = result["score"] * 100
            st.markdown(f"### 📄 Context {result_number + 1}")
            st.write(f"**Page:** {result['page']}")
            st.write(f"**Similarity Score:** {score_percent:.1f}%")
            st.progress(min(max(result["score"], 0.0), 1.0))
            st.markdown("#### 📖 Source Extract")
            st.write(result["text"])
            st.divider()

        best_score = results[0]["score"]
        if best_score >= 0.45:
            st.success("High-confidence semantic alignment found.")
        elif best_score >= 0.30:
            st.warning("Moderate alignment found. Verify details against source text.")
        else:
            st.error("Low similarity score. Relevant details may be missing from the document.")
