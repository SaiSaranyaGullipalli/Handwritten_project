import os
import re
import threading
import warnings

import cv2
import numpy as np
import pyttsx3
import torch
import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk
from nltk.corpus import wordnet
from spellchecker import SpellChecker
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, logging as hf_logging

import nltk
from tkinter import filedialog, messagebox, Text, Frame, Label, Button, Listbox, Toplevel


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

MODEL_CANDIDATES = [
    "microsoft/trocr-base-handwritten",
    "microsoft/trocr-large-handwritten",
]
MODEL_DIR = "./models/trocr_cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.type == "cuda"
FAST_BEAMS = 1
RETRY_BEAMS = 2 if DEVICE.type == "cuda" else 1
BATCH_SIZE = 8 if DEVICE.type == "cuda" else 3
MAX_LENGTH = 128
SHORT_WORD_BEAMS = 3 if DEVICE.type == "cuda" else 2

spell = SpellChecker()

try:
    wordnet.synsets("test")
except Exception:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def apply_case_style(original, replacement):
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement


def grammar_correct_text(text):
    rules = {
        r"\bthey axe\b": "they are",
        r"\bthey is\b": "they are",
        r"\bhe are\b": "he is",
        r"\bshe are\b": "she is",
        r"\bi am\b": "I am",
        r"\bdont\b": "don't",
        r"\bdoesnt\b": "doesn't",
        r"\bcant\b": "can't",
        r"\bteh\b": "the",
        r"\bform\b": "from",
        r"\bwirh\b": "with",
        r"\bgoinq\b": "going",
        r"\baxe\b": "are",
        r"\bi\b": "I",
        r"\bim\b": "I'm",
        r"\bthier\b": "their",
        r"\brecieve\b": "receive",
        r"\bseperate\b": "separate",
    }

    corrected = text
    details = []
    for pattern, replacement in rules.items():
        def repl(match):
            original = match.group(0)
            updated = apply_case_style(original, replacement)
            if original != updated:
                details.append(("Grammar", original, updated))
            return updated

        corrected = re.sub(pattern, repl, corrected, flags=re.IGNORECASE)

    lines = []
    sentence_ended = True
    for line in corrected.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            lines.append("")
            continue

        updated = line
        if sentence_ended and line[:1].islower():
            updated = line[0].upper() + line[1:]
            details.append(("Grammar", line[:1], updated[:1]))

        lines.append(updated)
        sentence_ended = bool(re.search(r"[.!?][\"')\]]*$", updated))

    corrected = "\n".join(lines)
    return corrected, details


def spelling_correct_text(text):
    token_pattern = re.compile(r"\b[a-zA-Z]+\b")
    words = token_pattern.findall(text)
    misspelled = spell.unknown(word.lower() for word in words)
    details = []
    replacements = {}

    for word in words:
        lower_word = word.lower()
        if lower_word not in misspelled:
            continue
        if len(word) <= 2:
            continue
        if lower_word in replacements:
            continue

        correction = spell.correction(lower_word)
        if not correction or correction == lower_word:
            continue
        if abs(len(correction) - len(lower_word)) > 2:
            continue
        mismatch_count = sum(
            a != b for a, b in zip(lower_word, correction)
        ) + abs(len(lower_word) - len(correction))
        if mismatch_count > max(2, len(lower_word) // 2):
            continue

        replacement = apply_case_style(word, correction)
        replacements[lower_word] = replacement
        details.append(("Spelling", word, replacement))

    def replace_token(match):
        token = match.group(0)
        return replacements.get(token.lower(), token)

    corrected = token_pattern.sub(replace_token, text)
    return corrected, details


def highlight_corrections(text_widget, details):
    text_widget.tag_delete("spelling_fix")
    text_widget.tag_delete("grammar_fix")
    text_widget.tag_configure("spelling_fix", background="#fff3a3")
    text_widget.tag_configure("grammar_fix", background="#c8f7c5")

    used_ranges = set()
    for kind, _, replacement in details:
        if not replacement:
            continue
        start = "1.0"
        found_start = None
        found_end = None
        while True:
            candidate = text_widget.search(replacement, start, stopindex="end")
            if not candidate:
                break
            candidate_end = f"{candidate}+{len(replacement)}c"
            marker = (candidate, candidate_end, replacement)
            if marker not in used_ranges:
                used_ranges.add(marker)
                found_start = candidate
                found_end = candidate_end
                break
            start = candidate_end
        if not found_start:
            continue
        tag = "spelling_fix" if kind == "Spelling" else "grammar_fix"
        text_widget.tag_add(tag, found_start, found_end)


def build_correction_summary(details, limit=20):
    lines = []
    for index, (kind, original, replacement) in enumerate(details[:limit], start=1):
        lines.append(f"{index}. {kind}: {original} -> {replacement}")
    if len(details) > limit:
        lines.append(f"... and {len(details) - limit} more changes")
    return "\n".join(lines)


class HandwrittenOCR:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model_name = None
        for model_name in MODEL_CANDIDATES:
            try:
                self.processor = TrOCRProcessor.from_pretrained(
                    model_name,
                    cache_dir=MODEL_DIR,
                    local_files_only=True,
                    use_fast=True,
                )
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    model_name,
                    cache_dir=MODEL_DIR,
                    local_files_only=True,
                )
                self.model_name = model_name
                break
            except Exception:
                continue

        if self.model_name is None:
            messagebox.showerror(
                "Model Missing",
                "TrOCR model not found locally. Connect to the internet once to download it.",
            )
            raise SystemExit

        self.model.to(DEVICE)
        if USE_FP16:
            self.model.half()

        if DEVICE.type == "cuda":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        self.model.eval()
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id

    def extract_ink_gray(self, image):
        bgr = image.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        b_channel, g_channel, r_channel = cv2.split(bgr)
        darkness = 255 - gray
        color_spread = np.maximum.reduce([r_channel, g_channel, b_channel]) - np.minimum.reduce([r_channel, g_channel, b_channel])

        # Combine dark strokes and colored strokes so black, blue, red, green pen all remain visible.
        ink_strength = np.maximum(darkness, cv2.addWeighted(color_spread, 0.7, saturation, 0.3, 0))
        ink_strength = cv2.normalize(ink_strength, None, 0, 255, cv2.NORM_MINMAX)

        paper_suppressed = cv2.addWeighted(255 - value, 0.35, ink_strength, 0.65, 0)
        paper_suppressed = cv2.normalize(paper_suppressed, None, 0, 255, cv2.NORM_MINMAX)
        return 255 - paper_suppressed

    def preprocess_document(self, image):
        gray = self.extract_ink_gray(image)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def build_binary(self, gray):
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            11,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def detect_text_bounds(self, gray, binary):
        coords = cv2.findNonZero(binary)
        if coords is None:
            return gray

        x, y, w, h = cv2.boundingRect(coords)
        pad_x = max(16, int(w * 0.03))
        pad_top = max(18, int(h * 0.04))
        pad_bottom = max(36, int(h * 0.10))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_top)
        x2 = min(gray.shape[1], x + w + pad_x)
        y2 = min(gray.shape[0], y + h + pad_bottom)
        return gray[y1:y2, x1:x2]

    def is_meaningful_line_region(self, gray):
        binary = self.build_binary(gray)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return False

        x, y, w, h = cv2.boundingRect(coords)
        foreground_pixels = int(np.count_nonzero(binary))
        box_area = max(1, w * h)
        image_area = max(1, gray.shape[0] * gray.shape[1])
        fill_ratio = foreground_pixels / float(box_area)
        coverage_ratio = foreground_pixels / float(image_area)

        if w < 60 or h < 14:
            return False
        if foreground_pixels < 120:
            return False
        if fill_ratio < 0.015:
            return False
        if coverage_ratio < 0.006:
            return False
        return True

    def detect_lines(self, gray):
        binary = self.build_binary(gray)
        cropped_gray = self.detect_text_bounds(gray, binary)
        cropped_binary = self.build_binary(cropped_gray)

        row_profile = np.sum(cropped_binary > 0, axis=1).astype(np.float32)
        if row_profile.size == 0 or row_profile.max() == 0:
            return [cropped_gray]

        smooth_window = max(5, (cropped_gray.shape[0] // 80) * 2 + 1)
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        smooth = np.convolve(row_profile, kernel, mode="same")
        threshold = max(2.0, smooth.max() * 0.12)

        spans = []
        start = None
        for i, value in enumerate(smooth):
            if value > threshold and start is None:
                start = i
            elif value <= threshold and start is not None:
                if i - start >= 14:
                    spans.append((start, i))
                start = None
        if start is not None and len(smooth) - start >= 14:
            spans.append((start, len(smooth)))

        if not spans:
            return [cropped_gray]

        line_regions = []
        gap_pad = max(10, cropped_gray.shape[0] // 100)
        for top, bottom in spans:
            y1 = max(0, top - gap_pad)
            y2 = min(cropped_gray.shape[0], bottom + gap_pad)
            line = cropped_gray[y1:y2, :]
            if line.shape[1] < 40 or line.shape[0] < 12:
                continue
            if not self.is_meaningful_line_region(line):
                continue
            line_regions.append(line)

        if not line_regions:
            return [cropped_gray]

        heights = [line.shape[0] for line in line_regions]
        median_height = float(np.median(heights)) if heights else 0.0
        refined_lines = []
        for line in line_regions:
            if median_height and line.shape[0] > median_height * 1.65:
                refined_lines.extend(self.split_large_line_region(line, median_height))
            else:
                refined_lines.append(line)

        return refined_lines or [cropped_gray]

    def build_paragraph_chunks(self, gray):
        height = gray.shape[0]
        if height < 220:
            return [gray]

        chunk_height = max(220, int(height * 0.40))
        overlap = max(60, int(chunk_height * 0.30))
        step = max(80, chunk_height - overlap)

        chunks = []
        start = 0
        while start < height:
            end = min(height, start + chunk_height)
            y1 = max(0, start - 10)
            y2 = min(height, end + 10)
            chunk = gray[y1:y2, :]
            if chunk.shape[0] >= 80:
                chunks.append(chunk)
            if end >= height:
                break
            start += step

        return chunks or [gray]

    def split_large_line_region(self, gray, reference_height):
        binary = self.build_binary(gray)
        row_profile = np.sum(binary > 0, axis=1).astype(np.float32)
        if row_profile.size == 0 or row_profile.max() == 0:
            return [gray]

        smooth_window = max(3, int(reference_height // 3) | 1)
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        smooth = np.convolve(row_profile, kernel, mode="same")
        threshold = max(2.0, smooth.max() * 0.18)

        spans = []
        start = None
        for i, value in enumerate(smooth):
            if value > threshold and start is None:
                start = i
            elif value <= threshold and start is not None:
                if i - start >= max(10, int(reference_height * 0.45)):
                    spans.append((start, i))
                start = None

        if start is not None and len(smooth) - start >= max(10, int(reference_height * 0.45)):
            spans.append((start, len(smooth)))

        if len(spans) <= 1:
            return [gray]

        pieces = []
        pad = max(4, int(reference_height * 0.12))
        for top, bottom in spans:
            y1 = max(0, top - pad)
            y2 = min(gray.shape[0], bottom + pad)
            piece = gray[y1:y2, :]
            if piece.shape[0] >= 10:
                if not self.is_meaningful_line_region(piece):
                    continue
                pieces.append(piece)
        return pieces or [gray]

    def prepare_line_image(self, gray, stronger=False, fast_mode=False):
        if stronger:
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            sharpened = cv2.GaussianBlur(gray, (0, 0), 1.0)
            gray = cv2.addWeighted(gray, 1.25, sharpened, -0.25, 0)

        h, w = gray.shape
        scale = 64 / float(max(h, 1))
        max_width = 1400 if fast_mode else 2000
        min_width = 256 if fast_mode else 320
        new_w = max(min_width, min(max_width, int(w * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(gray, (new_w, 64), interpolation)
        padded = cv2.copyMakeBorder(
            resized,
            12,
            12,
            20,
            20,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        return Image.fromarray(padded).convert("RGB")

    def decode_images(self, images, beams):
        pixel_values = self.processor(images, return_tensors="pt", padding=True).pixel_values.to(DEVICE)
        if USE_FP16:
            pixel_values = pixel_values.half()

        with torch.inference_mode():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=MAX_LENGTH,
                num_beams=beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                use_cache=True,
            )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def decode_image_candidates(self, image, beams, num_candidates):
        pixel_values = self.processor([image], return_tensors="pt", padding=True).pixel_values.to(DEVICE)
        if USE_FP16:
            pixel_values = pixel_values.half()

        candidate_count = max(1, min(num_candidates, beams))
        with torch.inference_mode():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=MAX_LENGTH,
                num_beams=max(beams, candidate_count),
                num_return_sequences=candidate_count,
                early_stopping=True,
                no_repeat_ngram_size=2,
                use_cache=True,
            )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def build_short_input_images(self, gray):
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25,
            7,
        )
        otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thickened = cv2.dilate(
            otsu,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )

        return [
            self.prepare_line_image(gray, fast_mode=False),
            self.prepare_line_image(gray, stronger=True, fast_mode=False),
            self.prepare_line_image(adaptive, fast_mode=False),
            self.prepare_line_image(thickened, fast_mode=False),
        ]

    def weak_text(self, text):
        text = text.strip()
        if not text:
            return True
        letters = sum(ch.isalpha() for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        compact_alnum = "".join(ch for ch in text if ch.isalnum())

        if letters + digits == 0:
            return True

        # Keep numeric and mixed alphanumeric OCR results such as IDs, room numbers,
        # serials, and codes. Only discard obvious repeated-letter noise.
        if letters > 0 and digits == 0 and len(compact_alnum) >= 4:
            if len(set(compact_alnum.lower())) == 1:
                return True

        if re.search(r"([A-Za-z])\1\1\1", text):
            return True
        return False

    def score_text(self, text):
        stripped = text.strip()
        if not stripped:
            return -1000

        letters = sum(ch.isalpha() for ch in stripped)
        digits = sum(ch.isdigit() for ch in stripped)
        spaces = sum(ch.isspace() for ch in stripped)
        punctuation = sum(ch in ".,!?;:'\"-()" for ch in stripped)
        score = letters * 3 + digits * 3 + spaces + punctuation
        if re.fullmatch(r"[A-Za-z0-9]+(?:[-/:#&][A-Za-z0-9]+)*[.]?", stripped):
            score += 12
        if digits and letters:
            score += 8
        elif digits:
            score += 6
        return score

    def choose_best_candidate(self, candidates):
        valid = [self.cleanup(text) for text in candidates if text and not self.weak_text(text)]
        if not valid:
            return ""

        deduped = []
        seen = set()
        for text in valid:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)

        return max(deduped, key=self.score_text)

    def cleanup(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        return re.sub(r"\s+([,.;:!?])", r"\1", text)

    def is_similar_text(self, first, second):
        first_key = re.sub(r"[^a-z0-9]+", " ", first.lower()).strip()
        second_key = re.sub(r"[^a-z0-9]+", " ", second.lower()).strip()
        if not first_key or not second_key:
            return False
        if first_key == second_key:
            return True

        first_words = set(first_key.split())
        second_words = set(second_key.split())
        if not first_words or not second_words:
            return False

        overlap = len(first_words & second_words)
        return overlap >= min(len(first_words), len(second_words))

    def ocr_batch(self, line_images, fast_mode=False):
        prepared = [self.prepare_line_image(img, fast_mode=fast_mode) for img in line_images]
        outputs = []

        for start in range(0, len(prepared), BATCH_SIZE):
            batch = prepared[start:start + BATCH_SIZE]
            outputs.extend(self.decode_images(batch, FAST_BEAMS))

        if fast_mode:
            return [self.cleanup(text) for text in outputs if text.strip()]

        retry_indices = [i for i, text in enumerate(outputs) if self.weak_text(text)]
        if retry_indices:
            retry_images = [
                self.prepare_line_image(line_images[i], stronger=True, fast_mode=fast_mode)
                for i in retry_indices
            ]
            retry_outputs = []
            for start in range(0, len(retry_images), BATCH_SIZE):
                batch = retry_images[start:start + BATCH_SIZE]
                retry_outputs.extend(self.decode_images(batch, RETRY_BEAMS))

            for idx, retry_text in zip(retry_indices, retry_outputs):
                if retry_text.strip() and not self.weak_text(retry_text):
                    outputs[idx] = retry_text
                else:
                    outputs[idx] = ""

        return [self.cleanup(text) for text in outputs]

    def ocr_precise_short_input(self, line_images):
        results = []
        for line in line_images:
            candidates = []
            for variant_image in self.build_short_input_images(line):
                candidates.extend(
                    self.decode_image_candidates(
                        variant_image,
                        SHORT_WORD_BEAMS,
                        num_candidates=min(3, SHORT_WORD_BEAMS),
                    )
                )
            best = self.choose_best_candidate(candidates)
            if best:
                results.append(best)
        return results

    def recognize(self, path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError("Unable to read the selected image.")

        gray = self.preprocess_document(image)
        lines = self.detect_lines(gray)
        short_input = len(lines) <= 3
        if short_input:
            line_texts = self.ocr_precise_short_input(lines)
            if not line_texts:
                line_texts = self.ocr_batch(lines, fast_mode=True)
        else:
            line_texts = self.ocr_batch(lines, fast_mode=False)
        valid_line_texts = [text for text in line_texts if text and not self.weak_text(text)]
        final_text = "\n".join(valid_line_texts)

        if len(lines) <= 2 and len(valid_line_texts) >= 2:
            return final_text

        if len(lines) <= 2:
            candidates = []
            candidates.extend(valid_line_texts)
            fallback_candidates = self.ocr_precise_short_input([gray])
            candidates.extend(text for text in fallback_candidates if text)
            if candidates:
                best_text = self.choose_best_candidate(candidates)
                return self.cleanup(best_text)

        if short_input and valid_line_texts:
            return final_text

        if len(lines) <= 5:
            if valid_line_texts:
                return final_text
            fallback_candidates = self.ocr_batch(lines, fast_mode=True)
            fallback_candidates = [
                text for text in fallback_candidates if text and not self.weak_text(text)
            ]
            if fallback_candidates:
                return "\n".join(fallback_candidates)

        if len(valid_line_texts) >= 6:
            return final_text

        paragraph_chunks = self.build_paragraph_chunks(gray)
        chunk_lines = []
        for chunk in paragraph_chunks:
            chunk_detected_lines = self.detect_lines(chunk)
            chunk_texts = self.ocr_batch(chunk_detected_lines, fast_mode=False)
            chunk_lines.extend(text for text in chunk_texts if text)

        combined_lines = []
        seen = set()
        for text in valid_line_texts + chunk_lines:
            key = text.strip().lower()
            if not key or key in seen or self.weak_text(text):
                continue
            if text in chunk_lines and valid_line_texts:
                if not any(self.is_similar_text(text, line_text) for line_text in valid_line_texts):
                    continue
            seen.add(key)
            combined_lines.append(text)

        if combined_lines:
            return "\n".join(combined_lines)

        fallback_candidates = self.ocr_batch([gray], fast_mode=True)
        if fallback_candidates:
            fallback_text = fallback_candidates[0]
            if fallback_text.strip():
                return fallback_text

        raise ValueError("No text was recognized. Try a clearer image with better lighting.")


def load_main_app(root):
    for widget in root.winfo_children():
        widget.destroy()

    ocr = HandwrittenOCR()
    root.geometry("1600x950")
    root.configure(bg="#CAC1FF")

    btn_font = font.Font(family="Arial", size=12, weight="bold")

    left = Frame(root, bg="#CAC1FF")
    left.place(relx=0, rely=0, relwidth=0.80, relheight=1)

    right = Frame(root, bg="#ECECEC")
    right.place(relx=0.80, rely=0, relwidth=0.20, relheight=1)

    preview_frame = Frame(left, bg="#CAC1FF")
    preview_frame.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=0.6)

    output_frame = Frame(left, bg="#CAC1FF")
    output_frame.place(relx=0.0, rely=0.6, relwidth=1.0, relheight=0.4)

    text_output = Text(
        output_frame,
        font=("Consolas", 14),
        wrap="word",
        height=24,
        width=100,
    )
    text_output.pack(fill="both", expand=True, padx=10, pady=10)

    status_var = tk.StringVar(value="Ready")
    Label(
        right,
        textvariable=status_var,
        font=("Arial", 11, "bold"),
        bg="#ECECEC",
        fg="#2c3e50",
        wraplength=260,
        justify="left",
    ).pack(pady=(10, 5), padx=10, fill="x")

    img_label = {"widget": None}
    img_path = {"path": None}
    convert_button = {"widget": None}
    speech_state = {
        "engine": None,
        "thread": None,
        "segments": [],
        "segment_index": 0,
        "source_text": "",
        "is_speaking": False,
        "stop_requested": False,
    }

    def set_status(message):
        status_var.set(message)

    def upload():
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp *.tif *.tiff")]
        )
        if not path:
            return

        img_path["path"] = path
        if img_label["widget"]:
            img_label["widget"].destroy()

        img = Image.open(path)
        img.thumbnail((1100, 520))
        tk_img = ImageTk.PhotoImage(img)
        label = Label(preview_frame, image=tk_img, bg="#CAC1FF")
        label.image = tk_img
        label.pack(expand=True)
        img_label["widget"] = label

        text_output.delete("1.0", "end")
        text_output.tag_remove("spelling_fix", "1.0", "end")
        text_output.tag_remove("grammar_fix", "1.0", "end")
        set_status("Image loaded. Ready for OCR.")

    def replace():
        upload()

    def delete_image():
        if img_label["widget"]:
            img_label["widget"].destroy()
            img_label["widget"] = None
        img_path["path"] = None
        text_output.delete("1.0", "end")
        text_output.tag_remove("spelling_fix", "1.0", "end")
        text_output.tag_remove("grammar_fix", "1.0", "end")
        set_status("Image removed.")

    def convert():
        if not img_path["path"]:
            messagebox.showwarning("Warning", "Upload image first.")
            return

        convert_button["widget"].config(state="disabled")
        set_status("Recognizing handwriting. Please wait...")

        def worker():
            try:
                text = ocr.recognize(img_path["path"])
            except Exception as exc:
                root.after(
                    0,
                    lambda: (
                        convert_button["widget"].config(state="normal"),
                        set_status("OCR failed."),
                        messagebox.showerror("OCR Error", str(exc)),
                    ),
                )
                return

            root.after(
                0,
                lambda: (
                    text_output.delete("1.0", "end"),
                    text_output.insert("1.0", text),
                    text_output.tag_remove("spelling_fix", "1.0", "end"),
                    text_output.tag_remove("grammar_fix", "1.0", "end"),
                    convert_button["widget"].config(state="normal"),
                    set_status(f"Recognition complete. Using {ocr.model_name.split('/')[-1]}."),
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def grammar_and_spelling():
        text = text_output.get("1.0", "end-1c")
        text, spelling_details = spelling_correct_text(text)
        text, grammar_details = grammar_correct_text(text)
        all_details = spelling_details + grammar_details

        if not all_details:
            messagebox.showinfo("Grammar & Spelling", "No errors found.")
            return

        text_output.delete("1.0", "end")
        text_output.insert("1.0", text)
        highlight_corrections(text_output, all_details)
        set_status(
            f"Corrections applied: {len(spelling_details)} spelling, {len(grammar_details)} grammar."
        )
        messagebox.showinfo(
            "Grammar & Spelling",
            "Corrections applied:\n\n"
            f"Spelling fixes: {len(spelling_details)}\n"
            f"Grammar fixes: {len(grammar_details)}\n\n"
            f"{build_correction_summary(all_details)}",
        )

    def synonyms():
        try:
            original_word = text_output.selection_get()
            start = text_output.index("sel.first")
            end = text_output.index("sel.last")

            entries = []
            seen = set()
            for synset in wordnet.synsets(original_word):
                definition = synset.definition().strip()
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    key = synonym.lower()
                    if key == original_word.lower() or key in seen:
                        continue
                    seen.add(key)
                    entries.append((synonym, definition))

            entries.sort(key=lambda item: item[0].lower())

            if not entries:
                messagebox.showinfo("Synonyms", "No synonyms found.")
                return

            win = Toplevel(root)
            win.title("Select Synonym")
            win.geometry("420x460")

            lb = Listbox(win, font=("Arial", 12))
            lb.pack(fill="both", expand=True, padx=10, pady=10)

            meaning_var = tk.StringVar(value="Select a synonym to see its meaning.")
            Label(
                win,
                textvariable=meaning_var,
                font=("Arial", 11),
                wraplength=380,
                justify="left",
                anchor="w",
                bg="#f5f5f5",
                padx=10,
                pady=10,
            ).pack(fill="x", padx=10, pady=(0, 10))

            for synonym, _ in entries:
                lb.insert("end", synonym)

            replaced = {"done": False}

            def update_meaning(event=None):
                try:
                    index = lb.curselection()[0]
                    synonym, definition = entries[index]
                    meaning_var.set(f"{synonym}: {definition}")
                except Exception:
                    meaning_var.set("Select a synonym to see its meaning.")

            def replace_word(event=None):
                try:
                    index = lb.curselection()[0]
                    new_word = entries[index][0]
                    text_output.delete(start, end)
                    text_output.insert(start, new_word)
                    replaced["done"] = True
                except Exception:
                    pass

            def undo_word():
                if replaced["done"]:
                    text_output.delete(start, f"{start} wordend")
                    text_output.insert(start, original_word)
                    replaced["done"] = False

            lb.bind("<<ListboxSelect>>", update_meaning)
            lb.bind("<Double-Button-1>", replace_word)
            Button(win, text="Replace", command=replace_word).pack(pady=3)
            Button(win, text="Undo", command=undo_word).pack(pady=3)

        except Exception:
            messagebox.showinfo("Synonyms", "Select a word first.")

    def to_pdf():
        content = text_output.get("1.0", "end-1c")
        if not content.strip():
            return

        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            with open(path, "w", encoding="utf-8") as file:
                file.write(content)
            set_status("Text exported.")

    def phonate():
        text = text_output.get("1.0", "end-1c").strip()
        if not text:
            return

        if speech_state["is_speaking"]:
            speech_state["stop_requested"] = True
            engine = speech_state["engine"]
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            speech_state["is_speaking"] = False
            set_status("Phonation paused.")
            return

        normalized_text = re.sub(r"\n+", "\n", text).strip()
        if speech_state["source_text"] != normalized_text or not speech_state["segments"]:
            segments = [normalized_text] if normalized_text else []

            speech_state["segments"] = segments
            speech_state["segment_index"] = 0
            speech_state["source_text"] = normalized_text

        speech_state["stop_requested"] = False

        def speak():
            engine = pyttsx3.init()
            speech_state["engine"] = engine
            speech_state["is_speaking"] = True

            finished = False
            try:
                while speech_state["segment_index"] < len(speech_state["segments"]):
                    if speech_state["stop_requested"]:
                        break

                    current_segment = speech_state["segments"][speech_state["segment_index"]]
                    engine.say(current_segment)
                    engine.runAndWait()

                    if speech_state["stop_requested"]:
                        break
                    speech_state["segment_index"] += 1

                finished = speech_state["segment_index"] >= len(speech_state["segments"])
            finally:
                try:
                    engine.stop()
                except Exception:
                    pass
                speech_state["engine"] = None
                speech_state["is_speaking"] = False
                speech_state["stop_requested"] = False
                if finished:
                    speech_state["segment_index"] = 0
                    root.after(0, lambda: set_status("Phonation complete."))
                else:
                    root.after(0, lambda: set_status("Phonation paused."))

        speech_state["thread"] = threading.Thread(target=speak, daemon=True)
        speech_state["thread"].start()
        if speech_state["segment_index"] == 0:
            set_status("Speaking recognized text.")
        else:
            set_status("Resuming phonation.")

    def add_button(label, command, bg="#FFFFFF", fg="#000000"):
        button = Button(
            right,
            text=label,
            command=command,
            font=btn_font,
            bg=bg,
            fg=fg,
            padx=20,
            pady=10,
        )
        button.pack(pady=10, fill="x")
        return button

    Label(right, text="Actions", font=("Arial", 14, "bold"), bg="#ECECEC").pack(pady=15)

    add_button("Upload", upload)
    add_button("Replace", replace)
    convert_button["widget"] = add_button("Convert", convert, "#2ecc71", "white")
    add_button("Grammar & Spell", grammar_and_spelling, "#9b59b6", "white")
    add_button("Synonyms", synonyms)
    add_button("To PDF", to_pdf, "#3498db", "white")
    add_button("Phonate", phonate)
    add_button("Delete Image", delete_image, "#e74c3c", "white")


def start_app():
    root = tk.Tk()
    root.title("Smart Handwriting Recognition")
    root.geometry("760x460")
    root.configure(bg="#DDC1FF")

    title_font = font.Font(family="Arial", size=28, weight="bold")
    btn_font = font.Font(family="Arial", size=16, weight="bold")

    Label(
        root,
        text="Smart Handwriting Recognition",
        font=title_font,
        bg="#DDC1FF",
        fg="#2c3e50",
        pady=40,
    ).pack()

    Button(
        root,
        text="Get Started",
        font=btn_font,
        bg="#2ecc71",
        fg="white",
        padx=30,
        pady=15,
        command=lambda: load_main_app(root),
    ).pack(pady=40)

    root.mainloop()


if __name__ == "__main__":
    start_app()
