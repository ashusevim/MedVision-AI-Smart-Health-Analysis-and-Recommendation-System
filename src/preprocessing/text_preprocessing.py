"""
Text Preprocessing Module for MedVision-AI.

Provides medical-domain text preprocessing including tokenization,
stopword removal, abbreviation expansion, medical entity extraction,
and clinical text normalisation.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Medical stopwords & abbreviation dictionaries
# ---------------------------------------------------------------------------

MEDICAL_STOPWORDS: frozenset[str] = frozenset({
    # General English stopwords commonly found in clinical notes
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its",
    "this", "that", "these", "those", "not", "no", "nor", "so", "if",
    # Clinical filler words
    "patient", "pt", "history", "noted", "noted.", "reports", "states",
    "per", "also", "without", "w/", "w/o", "s/p", "status", "post",
    "otherwise", "however", "therefore", "additionally", "furthermore",
})

# Common medical abbreviations -> expanded forms
MEDICAL_ABBREVIATIONS: dict[str, str] = {
    "pt": "patient",
    "pts": "patients",
    "hx": "history",
    "dx": "diagnosis",
    "tx": "treatment",
    "rx": "prescription",
    "sx": "symptoms",
    "fx": "fracture",
    "cx": "culture",
    "fxn": "function",
    "w/": "with",
    "w/o": "without",
    "s/p": "status post",
    "c/o": "complains of",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "n/v": "nausea and vomiting",
    "n/v/d": "nausea vomiting diarrhea",
    "abd": "abdominal",
    "bibas": "bibasilar",
    "bilat": "bilateral",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "dm": "diabetes mellitus",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "mi": "myocardial infarction",
    "tia": "transient ischemic attack",
    "cva": "cerebrovascular accident",
    "uti": "urinary tract infection",
    "afib": "atrial fibrillation",
    "ckd": "chronic kidney disease",
    "esrd": "end-stage renal disease",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "bp": "blood pressure",
    "temp": "temperature",
    "map": "mean arterial pressure",
    "icu": "intensive care unit",
    "oru": "observation unit",
    "er": "emergency room",
    "ed": "emergency department",
    "or": "operating room",
    "post-op": "post-operative",
    "pre-op": "pre-operative",
    "intra-op": "intra-operative",
    "prn": "as needed",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qd": "once daily",
    "qhs": "at bedtime",
    "po": "by mouth",
    "iv": "intravenous",
    "im": "intramuscular",
    "sq": "subcutaneous",
    "ns": "normal saline",
    "lr": "lactated ringer",
    "nsaid": "non-steroidal anti-inflammatory drug",
    "asa": "aspirin",
    "acei": "ACE inhibitor",
    "arb": "angiotensin receptor blocker",
    "ppi": "proton pump inhibitor",
    "stat": "immediately",
    "dc": "discontinue",
    "d/c": "discharge",
    "f/u": "follow-up",
    "f/u": "follow up",
    "wbc": "white blood cell",
    "rbc": "red blood cell",
    "hgb": "hemoglobin",
    "hct": "hematocrit",
    "plt": "platelet",
    "cr": "creatinine",
    "bun": "blood urea nitrogen",
    "cmp": "comprehensive metabolic panel",
    "bmp": "basic metabolic panel",
    "cbc": "complete blood count",
    "ua": "urinalysis",
    "ekg": "electrocardiogram",
    "ecg": "electrocardiogram",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "us": "ultrasound",
    "xray": "x-ray",
    "x-ray": "x-ray",
    "cxr": "chest x-ray",
    "pa": "posteroanterior",
    "ap": "anteroposterior",
    "lat": "lateral",
    "r/o": "rule out",
    "nr": "no response",
    "nkda": "no known drug allergies",
    "nkfa": "no known food allergies",
    "nka": "no known allergies",
    "y/o": "year old",
    "yo": "year old",
    "mo": "month old",
}

# Medical entity patterns for extraction
_ENTITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("dosage", re.compile(
        r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|cc|units?|meq|%|mmol|kg|lb|oz)\b",
        re.IGNORECASE,
    )),
    ("vital_sign", re.compile(
        r"\b(heart rate|blood pressure|respiratory rate|temperature|oxygen sat|spo2|bp|hr|rr|temp)\s*[:=]?\s*\d+",
        re.IGNORECASE,
    )),
    ("lab_value", re.compile(
        r"\b(wbc|rbc|hgb|hct|plt|cr|bun|glucose|sodium|potassium|chloride|co2|calcium|albumin)\s*[:=]?\s*\d+(\.\d+)?\b",
        re.IGNORECASE,
    )),
    ("procedure", re.compile(
        r"\b(ct scan|mri|ultrasound|x-ray|xray|ecg|ekg|echo|biopsy|endoscopy|colonoscopy|angiogram|angiography|thoracentesis|paracentesis|lumbar puncture|intubation|extubation|catheterization)\b",
        re.IGNORECASE,
    )),
    ("medication", re.compile(
        r"\b(metformin|lisinopril|amlodipine|omeprazole|atorvastatin|metoprolol|losartan|gabapentin|hydrochlorothiazide|sertraline|simvastatin|montelukast|escitalopram|rosuvastatin|bupropion|furosemide|pantoprazole|duloxetine|prednisone|tamsulosin|amoxicillin|ciprofloxacin|azithromycin|levofloxacin|doxycycline|vancomycin|piperacillin|meropenem|ceftriaxone|cefepime)\b",
        re.IGNORECASE,
    )),
    ("anatomy", re.compile(
        r"\b(heart|lung|liver|kidney|brain|spine|chest|abdomen|pelvis|extremity|head|neck|back|arm|leg|hand|foot|shoulder|knee|hip|elbow|wrist|ankle|thorax|mediastinum|diaphragm|pleura|pericardium|aorta|ventricle|atrium)\b",
        re.IGNORECASE,
    )),
    ("diagnosis_code", re.compile(
        r"\b[A-Z]\d{2}(\.\d{1,2})?\b"  # ICD-10 pattern
    )),
    ("date", re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    )),
    ("negation", re.compile(
        r"\b(no|not|denies|without|negative|absent|unremarkable|cleared|ruled out|unlikely)\b",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TextPreprocessConfig:
    """Configuration for the TextPreprocessor.

    Attributes:
        lowercase: Convert text to lowercase.
        expand_abbreviations: Expand medical abbreviations.
        remove_stopwords: Remove stopwords during preprocessing.
        normalize_unicode: Normalise unicode characters.
        remove_punctuation: Strip punctuation.
        remove_digits: Strip standalone digits.
        strip_extra_whitespace: Collapse multiple spaces/newlines.
        extract_entities: Whether to extract entities during preprocessing.
        custom_abbreviations: Additional abbreviation mappings.
        custom_stopwords: Additional stopwords.
        max_token_length: Maximum token length; longer tokens are split.
        min_token_length: Minimum token length; shorter tokens are dropped.
    """

    lowercase: bool = True
    expand_abbreviations: bool = True
    remove_stopwords: bool = False
    normalize_unicode: bool = True
    remove_punctuation: bool = False
    remove_digits: bool = False
    strip_extra_whitespace: bool = True
    extract_entities: bool = False
    custom_abbreviations: dict[str, str] = field(default_factory=dict)
    custom_stopwords: set[str] = field(default_factory=set)
    max_token_length: int = 50
    min_token_length: int = 1


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """Medical text preprocessor for clinical NLP pipelines.

    Provides tokenization, abbreviation expansion, stopword removal,
    medical entity extraction, and clinical text normalisation.

    Args:
        config: A :class:`TextPreprocessConfig` instance.

    Example::

        preprocessor = TextPreprocessor(TextPreprocessConfig(lowercase=True))
        cleaned = preprocessor.preprocess("Pt c/o SOB x2 days. Dx: COPD.")
        # -> "patient complains of shortness of breath x2 days. diagnosis: chronic obstructive pulmonary disease."
    """

    def __init__(self, config: Optional[TextPreprocessConfig] = None) -> None:
        self._config = config or TextPreprocessConfig()
        self._abbreviations = {**MEDICAL_ABBREVIATIONS, **self._config.custom_abbreviations}
        self._stopwords = MEDICAL_STOPWORDS | self._config.custom_stopwords
        self._abbrev_pattern = self._build_abbrev_pattern()
        logger.info(
            "TextPreprocessor initialised (abbrevs=%d, stopwords=%d)",
            len(self._abbreviations), len(self._stopwords),
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """Execute the full preprocessing pipeline on *text*.

        Pipeline order:
        1. Unicode normalisation
        2. Lowercasing
        3. Abbreviation expansion
        4. Clinical text normalisation
        5. Punctuation removal (optional)
        6. Digit removal (optional)
        7. Extra whitespace stripping

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        result = text

        # 1. Unicode normalisation
        if self._config.normalize_unicode:
            result = unicodedata.normalize("NFKC", result)

        # 2. Lowercase
        if self._config.lowercase:
            result = result.lower()

        # 3. Abbreviation expansion
        if self._config.expand_abbreviations:
            result = self.handle_abbreviations(result)

        # 4. Clinical text normalisation
        result = self.normalize_clinical_text(result)

        # 5. Punctuation removal
        if self._config.remove_punctuation:
            result = re.sub(r"[^\w\s]", " ", result)

        # 6. Digit removal
        if self._config.remove_digits:
            result = re.sub(r"\b\d+\b", "", result)

        # 7. Strip extra whitespace
        if self._config.strip_extra_whitespace:
            result = re.sub(r"\s+", " ", result).strip()

        return result

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    def tokenize(
        self,
        text: str,
        return_offsets: bool = False,
    ) -> Union[list[str], list[tuple[str, tuple[int, int]]]]:
        """Tokenize *text* into a list of tokens.

        Uses a medical-aware tokenizer that preserves:
        - Decimal numbers (e.g., "3.5")
        - Hyphenated terms (e.g., "post-operative")
        - Slash-separated terms (e.g., "w/wo")
        - ICD-10 codes (e.g., "J44.1")

        Args:
            text: Input text.
            return_offsets: If ``True``, return (token, (start, end)) tuples.

        Returns:
            List of tokens or (token, offset) tuples.
        """
        if not text:
            return []

        # Preserve special medical patterns before splitting
        # Pattern matches: words with hyphens, slashes, decimals, codes
        token_pattern = re.compile(
            r"[A-Za-z]\d{2}(?:\.\d{1,2})?"  # ICD-10 codes like J44.1
            r"|\d+\.\d+"                       # decimal numbers
            r"|\w+(?:[-/]\w+)*"               # words with hyphens/slashes
            r"|[^\s\w]"                         # single punctuation
        )

        if return_offsets:
            return [(m.group(), (m.start(), m.end())) for m in token_pattern.finditer(text)]

        return [m.group() for m in token_pattern.finditer(text)]

    def remove_stopwords(
        self,
        tokens: Union[list[str], str],
    ) -> list[str]:
        """Remove medical and general stopwords from *tokens*.

        Args:
            tokens: A list of tokens or a raw text string (which will
                be tokenized first).

        Returns:
            List of tokens with stopwords removed.
        """
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)

        filtered = [
            t for t in tokens
            if t.lower() not in self._stopwords
            and len(t) >= self._config.min_token_length
            and len(t) <= self._config.max_token_length
        ]
        return filtered

    def handle_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations in *text*.

        Replaces known abbreviations with their expanded forms, preserving
        word boundaries to avoid partial replacements.

        Args:
            text: Input text.

        Returns:
            Text with abbreviations expanded.
        """
        if not self._abbrev_pattern:
            return text

        def _replace_match(match: re.Match[str]) -> str:
            word = match.group(0)
            lookup = word.lower() if self._config.lowercase else word
            replacement = self._abbreviations.get(lookup)
            if replacement:
                return replacement
            return word

        return self._abbrev_pattern.sub(_replace_match, text)

    def extract_medical_entities(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """Extract medical entities from *text*.

        Identifies dosages, vital signs, lab values, procedures,
        medications, anatomical terms, ICD-10 codes, dates, and
        negation cues.

        Args:
            text: Input clinical text.

        Returns:
            A list of entity dictionaries, each with keys
            ``"entity_type"``, ``"text"``, ``"start"``, and ``"end"``.
        """
        entities: list[dict[str, Any]] = []

        for entity_type, pattern in _ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                entities.append({
                    "entity_type": entity_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })

        # Sort by position
        entities.sort(key=lambda e: e["start"])
        logger.debug("Extracted %d medical entities", len(entities))
        return entities

    def normalize_clinical_text(self, text: str) -> str:
        """Normalise clinical text formatting and common patterns.

        Handles:
        - Unit normalisation (``mg.`` -> ``mg``)
        - Range normalisation (``10-20`` -> ``10 to 20``)
        - Bullet/heading markers (``1.``, ``-``, ``*``)
        - Fragment cleanup

        Args:
            text: Input clinical text.

        Returns:
            Normalised text.
        """
        result = text

        # Normalise unit abbreviations (remove trailing period)
        result = re.sub(r"\b(mg|mcg|ml|cc|meq|mmol|kg|lb|oz|mm|cm)\.\b", r"\1", result, flags=re.IGNORECASE)

        # Normalise numeric ranges (10-20 -> 10 to 20) but preserve words-with-hyphens
        result = re.sub(r"\b(\d+)\s*-\s*(\d+)\b", r"\1 to \2", result)

        # Normalise vital sign separators
        result = re.sub(r"[:=]\s*", ": ", result)

        # Remove numbered list markers
        result = re.sub(r"^\s*\d+\.\s+", " ", result, flags=re.MULTILINE)

        # Remove bullet markers
        result = re.sub(r"^\s*[-*]\s+", " ", result, flags=re.MULTILINE)

        # Normalise section headers (ALL CAPS followed by colon)
        result = re.sub(r"\b([A-Z]{2,}):\s", lambda m: m.group(1).title() + ": ", result)

        # Normalise line breaks to spaces (unless sentence boundary)
        result = re.sub(r"(?<=[a-z,;])\n", " ", result)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_abbrev_pattern(self) -> re.Pattern[str]:
        """Build a regex pattern that matches any known abbreviation."""
        if not self._abbreviations:
            return re.compile(r"(?!)")  # never matches

        # Sort by length (longest first) to prefer longer matches
        sorted_abbrs = sorted(self._abbreviations.keys(), key=len, reverse=True)
        escaped = [re.escape(a) for a in sorted_abbrs]
        # Use word boundaries, but allow hyphens and slashes
        pattern_str = r"\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern_str, re.IGNORECASE)

    @property
    def abbreviations(self) -> dict[str, str]:
        """Return the current abbreviation dictionary (read-only copy)."""
        return dict(self._abbreviations)

    def add_abbreviation(self, abbreviation: str, expansion: str) -> None:
        """Register a custom abbreviation at runtime.

        Args:
            abbreviation: The short form.
            expansion: The expanded form.
        """
        self._abbreviations[abbreviation.lower()] = expansion
        self._abbrev_pattern = self._build_abbrev_pattern()
        logger.debug("Added abbreviation: %s -> %s", abbreviation, expansion)

    def add_stopword(self, word: str) -> None:
        """Register a custom stopword at runtime.

        Args:
            word: The stopword to add.
        """
        self._stopwords.add(word.lower())
