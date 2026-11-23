"""English and Indian-language translator with offline speech support.

The Flask API serves the frontend from static/ and uses NLLB, NLTK, and MMS-TTS.
"""
import io
import re
import string
import threading
import time
import wave

import nltk
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

MODEL_NAME = "facebook/nllb-200-distilled-600M"
MAX_CHARS = 5000
BEAMS = 4
QUANTIZE = False

# Dropdown label -> NLLB language code
LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
    "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Marathi": "mar_Deva",
    "Bhojpuri": "bho_Deva", "Bengali": "ben_Beng", "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
}

# Bhojpuri has no facebook/mms-tts-bho checkpoint, so it is not offered for voice.
VOICE_LANGUAGES = {
    "English": "eng",
    "Hindi": "hin", "Tamil": "tam", "Telugu": "tel", "Kannada": "kan",
    "Malayalam": "mal", "Marathi": "mar", "Bengali": "ben",
    "Gujarati": "guj", "Punjabi": "pan",
}

# ---------- Setup (runs once at startup) ----------
try:  # NLTK sentence-splitter data; downloaded only if missing
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# Encoder reads the English tokens, decoder generates the target-language tokens
# Load the already downloaded model from the local Hugging Face cache. This
# avoids a Hub request (and its unauthenticated-request warning) at startup.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, src_lang="eng_Latn", local_files_only=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, local_files_only=True
)
model.eval()
if QUANTIZE:
    # Dynamic quantization makes CPU inference smaller and faster, with possible small quality loss.
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

# One lock protects shared model and tokenizer state during translations and speech.
model_lock = threading.Lock()
voice_lock = threading.Lock()
voice_models = {}

# Serving only static/ keeps Python files and evaluation data outside the web root.
app = Flask(__name__, static_folder="static", static_url_path="")


# ---------- NLP functions ----------
def preprocess(text):
    """Clean text, then return sentences while keeping line breaks separate."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    slang = {
        "u": "you", "r": "are", "ur": "your", "pls": "please",
        "thx": "thanks", "wt": "what", "wat": "what", "hw": "how",
        "whr": "where", "abt": "about", "bcz": "because",
    }
    sentences = []

    # Normalize spaces inside each line before splitting it into sentences.
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue

        for sentence in sent_tokenize(line):
            # Match lowercase slang only when whitespace or a line edge surrounds it.
            sentence = re.sub(
                r"(?<!\S)(?:ur|u|r|pls|thx|wt|wat|hw|whr|abt|bcz)(?!\S)",
                lambda match: slang[match.group(0).lower()],
                sentence,
            )
            sentence = sentence.strip()
            if not sentence:
                continue

            # Start each sentence with a capital letter.
            sentence = sentence[0].upper() + sentence[1:]

            # Use a question mark for common question openings, otherwise a period.
            if not sentence.endswith((".", "!", "?", "\u2026")):
                first_word = re.match(r"[A-Za-z]+", sentence)
                question_words = {
                    "what", "how", "where", "when", "why", "who", "which",
                    "do", "does", "did", "is", "are", "can", "could",
                    "will", "would",
                }
                ending = "?" if first_word and first_word.group(0).lower() in question_words else "."
                sentence += ending

            sentences.append(sentence)

    return sentences


def split_indic(text):
    """Split Indic text on lines and common Indic sentence-ending marks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            parts = re.split(r"(?<=[।.?!])\s+", line)
            sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


def translate(text, src_code, tgt_code, roundtrip=False):
    """Translate in either direction and optionally check the round trip."""
    source_language = next(
        (name for name, code in LANGUAGES.items() if code == src_code), None
    )
    sentences = preprocess(text) if source_language == "English" else split_indic(text)
    if not sentences:
        return "", 0.0, "", None

    with model_lock:
        tokenizer.src_lang = src_code
        try:
            inputs = tokenizer(
                sentences, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            )
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                num_beams=BEAMS,
                max_length=256,
                return_dict_in_generate=True,
                output_scores=True,
            )
            translated_sentences = tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )

            # This is the model's own confidence, not a correctness guarantee.
            confidence = torch.exp(outputs.sequences_scores).mean().item() * 100
            confidence = min(100.0, max(0.0, confidence))

            back_text = ""
            roundtrip_score = None
            if roundtrip:
                # Feed target text directly; English preprocessing would alter it.
                tokenizer.src_lang = tgt_code
                target_inputs = tokenizer(
                    translated_sentences, return_tensors="pt", padding=True,
                    truncation=True, max_length=256,
                )
                back_outputs = model.generate(
                    **target_inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(src_code),
                    num_beams=BEAMS,
                    max_length=256,
                )
                back_sentences = tokenizer.batch_decode(
                    back_outputs, skip_special_tokens=True
                )
                back_text = " ".join(back_sentences)
                original_text = " ".join(sentences)
                roundtrip_score = roundtrip_bleu_score(
                    original_text, back_text, indic=source_language != "English"
                )
        finally:
            # Restore the default so later requests do not inherit another language.
            tokenizer.src_lang = "eng_Latn"

    return " ".join(translated_sentences), confidence, back_text, roundtrip_score


def bleu_score(reference, hypothesis):
    """Sentence BLEU (0-100) with smoothing, since sentences are short."""
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    n = min(4, len(reference_tokens))
    if n == 0:
        return 0.0

    smooth = SmoothingFunction().method4
    weights = (1 / n,) * n
    return 100 * sentence_bleu(
        [reference_tokens], hypothesis_tokens,
        weights=weights, smoothing_function=smooth,
    )


def chrf_score(reference, hypothesis):
    """Return the sentence chrF score on a 0-100 scale."""
    return 100 * sentence_chrf(reference, hypothesis)


def roundtrip_bleu_score(reference, hypothesis, indic=False):
    """Score cleaned round-trip text without deleting Indic vowel marks."""
    if indic:
        punctuation = "[" + re.escape(string.punctuation + "।") + "]"
        clean_reference = re.sub(punctuation, "", reference.lower()).split()
        clean_hypothesis = re.sub(punctuation, "", hypothesis.lower()).split()
    else:
        clean_reference = re.sub(r"[^\w\s]", "", reference.lower()).split()
        clean_hypothesis = re.sub(r"[^\w\s]", "", hypothesis.lower()).split()
    if not clean_reference or not clean_hypothesis:
        return 0.0

    smooth = SmoothingFunction().method4
    n = min(4, len(clean_reference))
    weights = (1 / n,) * n
    score = sentence_bleu(
        [clean_reference], clean_hypothesis, weights=weights,
        smoothing_function=smooth,
    )
    return 100 * score


def interpret_bleu(score):
    """Return a short description for a BLEU-style score."""
    if score > 60:
        return "Excellent"
    if score > 30:
        return "Good"
    if score > 10:
        return "Basic"
    return "Weak"


def get_voice_model(language):
    """Load one cached MMS voice model the first time its language is used."""
    language_code = VOICE_LANGUAGES.get(language)
    if not language_code:
        raise ValueError(f"Voice is not supported for {language}.")

    checkpoint = f"facebook/mms-tts-{language_code}"
    with voice_lock:
        if checkpoint not in voice_models:
            # Local-only loading keeps speech synthesis usable without internet.
            voice_tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, local_files_only=True
            )
            if getattr(voice_tokenizer, "is_uroman", False):
                raise ValueError(
                    f"Voice for {language} requires uroman, which is not installed."
                )
            voice_model = VitsModel.from_pretrained(
                checkpoint, local_files_only=True
            )
            voice_model.eval()
            voice_models[checkpoint] = (voice_tokenizer, voice_model)

    return voice_models[checkpoint]


def make_wav(text, language):
    """Synthesize speech and return it as a WAV byte stream."""
    voice_tokenizer, voice_model = get_voice_model(language)
    voice_inputs = voice_tokenizer(text, return_tensors="pt")

    with model_lock, torch.inference_mode():
        speech = voice_model(**voice_inputs).waveform[0].cpu().numpy()

    # WAV PCM needs signed 16-bit samples instead of floating-point amplitudes.
    pcm_audio = (np.clip(speech, -1, 1) * 32767).astype(np.int16)
    audio_file = io.BytesIO()
    with wave.open(audio_file, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice_model.config.sampling_rate)
        wav_file.writeframes(pcm_audio.tobytes())

    audio_file.seek(0)
    return audio_file


# ---------- Routes ----------
@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/api/languages")
def languages():
    return jsonify(list(LANGUAGES))


@app.route("/api/voices")
def voices():
    return jsonify(list(VOICE_LANGUAGES))


@app.route("/api/translate", methods=["POST"])
def translate_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    source = data.get("source")
    target = data.get("target")
    reference = (data.get("reference") or "").strip()
    roundtrip_requested = data.get("roundtrip", False) is True

    # Error handling: empty input, unsupported language, oversized input
    if not text:
        return jsonify(error="Please enter some English text."), 400
    if not isinstance(source, str) or source not in LANGUAGES:
        return jsonify(error=f"Unsupported source language: {source}"), 400
    if not isinstance(target, str) or target not in LANGUAGES:
        return jsonify(error=f"Unsupported target language: {target}"), 400
    if source == target:
        return jsonify(error="Source and target languages must be different."), 400
    if (source == "English") == (target == "English"):
        return jsonify(
            error="Exactly one language must be English; Indic-to-Indic translation is not supported."
        ), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text too long (max {MAX_CHARS} characters)."), 400

    started_at = time.perf_counter()
    try:
        translated, confidence, back_text, roundtrip_score = translate(
            text, LANGUAGES[source], LANGUAGES[target],
            roundtrip=roundtrip_requested,
        )
    except Exception as exc:  # model/runtime failure
        return jsonify(error=f"Translation failed: {exc}"), 500

    result = {
        "translation": translated,
        "confidence": round(confidence, 2),
        "seconds": round(time.perf_counter() - started_at, 2),
    }
    if back_text is not None and roundtrip_score is not None:
        result["roundtrip"] = {
            "label": "Round-trip consistency check (not reference BLEU)",
            "text": back_text,
            "bleu": round(roundtrip_score, 2),
            "interpretation": interpret_bleu(roundtrip_score),
        }
    if reference:  # These scores compare output with the supplied reference.
        score = bleu_score(reference, translated)
        result["bleu"] = round(score, 2)
        result["chrf"] = round(chrf_score(reference, translated), 2)
        result["interpretation"] = interpret_bleu(score)
    return jsonify(result)


@app.route("/api/speak", methods=["POST"])
def speak_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    language = data.get("language")

    if not text:
        return jsonify(error="Please enter text to speak."), 400
    if len(text) > 1000:
        return jsonify(error="Text is too long for speech (max 1,000 characters)."), 400
    if not isinstance(language, str) or language not in LANGUAGES:
        return jsonify(error=f"Unsupported language: {language}"), 400
    if language not in VOICE_LANGUAGES:
        return jsonify(error=f"Offline voice is not available for {language}."), 400

    try:
        audio_file = make_wav(text, language)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except OSError:
        return jsonify(
            error=(
                f"The {language} voice model is not cached locally. "
                "Download it once before using speech offline."
            )
        ), 503
    except Exception as exc:
        return jsonify(error=f"Speech synthesis failed: {exc}"), 500

    return send_file(audio_file, mimetype="audio/wav")


if __name__ == "__main__":
    # 0.0.0.0 lets a phone on the same Wi-Fi/hotspot reach this server
    app.run(host="0.0.0.0", port=5000)
