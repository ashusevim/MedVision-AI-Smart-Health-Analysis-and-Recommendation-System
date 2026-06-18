"""
ClinicalNLP - Clinical text processing with NER, relation extraction, and negation detection.

This module provides a comprehensive pipeline for processing unstructured clinical
text (e.g. discharge summaries, progress notes, radiology reports). It supports
named-entity recognition, negation detection (identifying affirmed vs negated
findings), and relation extraction between clinical entities.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for structured outputs
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    """Supported clinical entity types."""

    CONDITION = "CONDITION"
    MEDICATION = "MEDICATION"
    PROCEDURE = "PROCEDURE"
    ANATOMY = "ANATOMY"
    LAB_VALUE = "LAB_VALUE"
    SYMPTOM = "SYMPTOM"
    TEMPORAL = "TEMPORAL"


class AssertionStatus(str, Enum):
    """Assertion status for an entity."""

    AFFIRMED = "AFFIRMED"
    NEGATED = "NEGATED"
    HYPOTHETICAL = "HYPOTHETICAL"
    HISTORICAL = "HISTORICAL"


class RelationType(str, Enum):
    """Types of relations between clinical entities."""

    TREATS = "TREATS"
    CAUSES = "CAUSES"
    DIAGNOSES = "DIAGNOSES"
    LOCATED_AT = "LOCATED_AT"
    INDICATES = "INDICATES"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"


@dataclass
class Entity:
    """A recognised clinical entity."""

    text: str
    entity_type: EntityType
    start: int
    end: int
    assertion: AssertionStatus = AssertionStatus.AFFIRMED
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """A relation between two clinical entities."""

    head: Entity
    tail: Entity
    relation_type: RelationType
    confidence: float = 1.0


@dataclass
class ClinicalNoteResult:
    """Full processing result for a clinical note."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    negated_spans: list[tuple[int, int, str]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Negation triggers and relation patterns
# ---------------------------------------------------------------------------

_NEGATION_TRIGGERS: list[str] = [
    "no evidence of",
    "denies",
    "denied",
    "negative for",
    "no signs of",
    "no history of",
    "without evidence of",
    "without",
    "not found",
    "not detected",
    "absent",
    "unremarkable",
    "no",
    "rule out",
    "ruled out",
    "free of",
]

_HYPOTHETICAL_TRIGGERS: list[str] = [
    "possible",
    "suspected",
    "may have",
    "might be",
    "suggestive of",
    "cannot rule out",
    "consider",
    "likely",
    "probable",
    "suggests",
]

_HISTORICAL_TRIGGERS: list[str] = [
    "history of",
    "prior",
    "previous",
    "past",
    "formerly",
]

# Simple regex patterns for entity recognition
_ENTITY_PATTERNS: dict[EntityType, list[str]] = {
    EntityType.MEDICATION: [
        r"\b\d+\s?mg\b",
        r"\b\d+\s?ml\b",
        r"\b\d+\s?mcg\b",
        r"\b\d+\s?units?\b",
        r"\b(?:tab|capsule|tablet)s?\b",
        r"\b(?:IV|PO|IM|SC|SQ|PR|SL|topical|inhaled)\b",
    ],
    EntityType.LAB_VALUE: [
        r"\b\d+\.?\d*\s*(?:mg/dL|mmol/L|mEq/L|g/dL|ng/mL|pg/mL|U/L|%)\b",
        r"\b(?:WBC|RBC|Hgb|Hct|Platelets|Creatinine|BUN|Glucose|Na|K|Cl)\b",
    ],
    EntityType.TEMPORAL: [
        r"\b\d+\s?(?:days?|weeks?|months?|years?)\s?(?:ago|prior|before)\b",
        r"\b(?:yesterday|today|last|this)\s+(?:night|morning|week|month)\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    ],
}

# Relation pattern templates
_RELATION_PATTERNS: list[tuple[re.Pattern[str], RelationType]] = [
    (re.compile(r"(?P<head>\w+)\s+treats?\s+(?P<tail>\w+)", re.I), RelationType.TREATS),
    (re.compile(r"(?P<head>\w+)\s+causes?\s+(?P<tail>\w+)", re.I), RelationType.CAUSES),
    (re.compile(r"(?P<head>\w+)\s+diagnos\w+\s+(?P<tail>\w+)", re.I), RelationType.DIAGNOSES),
    (re.compile(r"(?P<head>\w+)\s+(?:located|found)\s+(?:in|at|on)\s+(?P<tail>\w+)", re.I), RelationType.LOCATED_AT),
    (re.compile(r"(?P<head>\w+)\s+indicates?\s+(?P<tail>\w+)", re.I), RelationType.INDICATES),
    (re.compile(r"(?P<head>\w+)\s+(?:associated|correlated)\s+with\s+(?P<tail>\w+)", re.I), RelationType.ASSOCIATED_WITH),
]


# ---------------------------------------------------------------------------
# ClinicalNLP class
# ---------------------------------------------------------------------------


class ClinicalNLP:
    """Process clinical text with NER, negation detection, and relation extraction.

    The pipeline operates in a rule-based / pattern-matching mode by default
    and can be extended with model-based components (e.g. spaCy or MedSpaCy
    models) via the *config* dictionary.

    Args:
        config: Optional configuration dictionary.  Supported keys:

            * ``ner_model`` — Path or name of an NER model (reserved for
              future model-based NER).
            * ``negation_model`` — Path or name of a negation detection model.
            * ``entity_types`` — List of ``EntityType`` values to detect.
            * ``custom_negation_triggers`` — Additional negation trigger
              phrases.
            * ``custom_entity_patterns`` — Dict mapping ``EntityType`` to
              regex pattern lists.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}

        # Merge custom negation triggers
        self._negation_triggers: list[str] = list(_NEGATION_TRIGGERS)
        for trigger in self.config.get("custom_negation_triggers", []):
            self._negation_triggers.append(trigger.lower())

        # Merge custom entity patterns
        self._entity_patterns: dict[EntityType, list[str]] = {
            k: list(v) for k, v in _ENTITY_PATTERNS.items()
        }
        for etype, patterns in self.config.get("custom_entity_patterns", {}).items():
            etype = EntityType(etype)
            self._entity_patterns.setdefault(etype, []).extend(patterns)

        # Compile patterns
        self._compiled_patterns: dict[EntityType, list[re.Pattern[str]]] = {}
        for etype, patterns in self._entity_patterns.items():
            self._compiled_patterns[etype] = [re.compile(p, re.I) for p in patterns]

        self._active_entity_types: list[EntityType] = [
            EntityType(e) for e in self.config.get(
                "entity_types", [e.value for e in EntityType]
            )
        ]

        logger.info(
            "ClinicalNLP initialised — entity_types=%s, negation_triggers=%d",
            [e.value for e in self._active_entity_types],
            len(self._negation_triggers),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract clinical named entities from text.

        Uses pattern matching to identify entities of the configured types.
        Each entity is annotated with its assertion status (affirmed, negated,
        hypothetical, historical).

        Args:
            text: Clinical text (e.g. a discharge summary paragraph).

        Returns:
            List of ``Entity`` objects with positions and types.
        """
        entities: list[Entity] = []

        for etype in self._active_entity_types:
            patterns = self._compiled_patterns.get(etype, [])
            for pattern in patterns:
                for match in pattern.finditer(text):
                    span_text = match.group()
                    start, end = match.span()

                    # Determine assertion status based on preceding context
                    assertion = self._determine_assertion(text, start)

                    entities.append(
                        Entity(
                            text=span_text,
                            entity_type=etype,
                            start=start,
                            end=end,
                            assertion=assertion,
                            confidence=0.85,  # rule-based confidence estimate
                        )
                    )

        # Sort by position
        entities.sort(key=lambda e: e.start)
        return entities

    def detect_negation(self, text: str) -> list[tuple[int, int, str]]:
        """Detect negated spans in the text.

        Each negation trigger found in the text defines a scope that extends
        to the end of the current sentence (or the next trigger).  All text
        within that scope is reported as a negated span.

        Args:
            text: Clinical text.

        Returns:
            List of ``(start, end, trigger)`` tuples for each negated span.
        """
        negated_spans: list[tuple[int, int, str]] = []
        text_lower = text.lower()

        for trigger in self._negation_triggers:
            idx = text_lower.find(trigger)
            while idx != -1:
                # Scope extends to the next sentence boundary
                scope_end = self._find_sentence_end(text, idx + len(trigger))
                negated_spans.append((idx, scope_end, trigger))
                idx = text_lower.find(trigger, idx + len(trigger))

        negated_spans.sort(key=lambda s: s[0])
        return negated_spans

    def extract_relations(self, text: str) -> list[Relation]:
        """Extract relations between clinical entities.

        Uses pattern matching to identify semantic relations (e.g. "treats",
        "causes") between entities in the text.

        Args:
            text: Clinical text.

        Returns:
            List of ``Relation`` objects.
        """
        entities = self.extract_entities(text)
        entity_map: dict[str, Entity] = {e.text.lower(): e for e in entities}
        relations: list[Relation] = []

        for pattern, rel_type in _RELATION_PATTERNS:
            for match in pattern.finditer(text):
                head_text = match.group("head").lower()
                tail_text = match.group("tail").lower()

                # Look up or create placeholder entities
                head_ent = entity_map.get(head_text) or Entity(
                    text=head_text,
                    entity_type=EntityType.CONDITION,
                    start=match.start("head"),
                    end=match.end("head"),
                )
                tail_ent = entity_map.get(tail_text) or Entity(
                    text=tail_text,
                    entity_type=EntityType.CONDITION,
                    start=match.start("tail"),
                    end=match.end("tail"),
                )

                relations.append(
                    Relation(
                        head=head_ent,
                        tail=tail_ent,
                        relation_type=rel_type,
                        confidence=0.75,
                    )
                )

        return relations

    def process_clinical_note(self, text: str) -> dict[str, Any]:
        """Run the full clinical NLP pipeline on a note.

        Performs entity extraction, negation detection, and relation
        extraction, then assembles a structured summary.

        Args:
            text: Raw clinical note text.

        Returns:
            Dictionary with keys ``entities``, ``relations``,
            ``negated_spans``, and ``summary``.
        """
        entities = self.extract_entities(text)
        negated_spans = self.detect_negation(text)
        relations = self.extract_relations(text)

        # Build summary statistics
        entity_type_counts: dict[str, int] = {}
        for ent in entities:
            key = ent.entity_type.value
            entity_type_counts[key] = entity_type_counts.get(key, 0) + 1

        negated_entity_count = sum(
            1 for e in entities if e.assertion == AssertionStatus.NEGATED
        )

        summary = {
            "total_entities": len(entities),
            "entity_type_counts": entity_type_counts,
            "negated_entities": negated_entity_count,
            "total_relations": len(relations),
            "total_negated_spans": len(negated_spans),
            "assertion_distribution": {
                status.value: sum(1 for e in entities if e.assertion == status)
                for status in AssertionStatus
            },
        }

        result = ClinicalNoteResult(
            entities=entities,
            relations=relations,
            negated_spans=negated_spans,
            summary=summary,
        )

        logger.info(
            "Clinical note processed — %d entities, %d relations, %d negated spans",
            len(entities),
            len(relations),
            len(negated_spans),
        )

        return {
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type.value,
                    "start": e.start,
                    "end": e.end,
                    "assertion": e.assertion.value,
                    "confidence": round(e.confidence, 3),
                }
                for e in result.entities
            ],
            "relations": [
                {
                    "head": r.head.text,
                    "tail": r.tail.text,
                    "type": r.relation_type.value,
                    "confidence": round(r.confidence, 3),
                }
                for r in result.relations
            ],
            "negated_spans": [
                {"start": s, "end": e, "trigger": t}
                for s, e, t in result.negated_spans
            ],
            "summary": result.summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_assertion(self, text: str, entity_start: int) -> AssertionStatus:
        """Determine the assertion status of an entity based on preceding context.

        Inspects the text before *entity_start* for negation, hypothetical,
        or historical triggers.

        Args:
            text: Full text.
            entity_start: Start character offset of the entity.

        Returns:
            An ``AssertionStatus`` value.
        """
        # Look at the preceding sentence for triggers
        context_start = max(0, entity_start - 80)
        context = text[context_start:entity_start].lower()

        # Check for negation triggers
        for trigger in self._negation_triggers:
            if trigger.lower() in context:
                return AssertionStatus.NEGATED

        # Check for hypothetical triggers
        for trigger in _HYPOTHETICAL_TRIGGERS:
            if trigger.lower() in context:
                return AssertionStatus.HYPOTHETICAL

        # Check for historical triggers
        for trigger in _HISTORICAL_TRIGGERS:
            if trigger.lower() in context:
                return AssertionStatus.HISTORICAL

        return AssertionStatus.AFFIRMED

    @staticmethod
    def _find_sentence_end(text: str, start: int) -> int:
        """Find the end of the sentence starting at *start*.

        Searches for the next sentence-ending punctuation mark (.!?)
        after *start*.

        Args:
            text: Full text.
            start: Character offset to start searching from.

        Returns:
            Character offset of the end of the sentence (inclusive).
        """
        for i in range(start, len(text)):
            if text[i] in ".!?":
                return i + 1
        return len(text)
