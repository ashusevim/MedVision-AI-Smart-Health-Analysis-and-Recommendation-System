"""
TreatmentSuggester - Suggest and rank treatments based on diagnosis.

This module provides a rule-based and scoring-driven treatment recommendation
engine. It matches diagnosed conditions to treatment protocols, ranks
candidates using configurable criteria (efficacy, cost, side-effect profile),
and generates human-readable explanations for each recommendation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TreatmentCategory(str, Enum):
    """Categories of medical treatments."""

    PHARMACOLOGICAL = "pharmacological"
    SURGICAL = "surgical"
    PROCEDURAL = "procedural"
    LIFESTYLE = "lifestyle"
    THERAPEUTIC = "therapeutic"
    PREVENTIVE = "preventive"
    PALLIATIVE = "palliative"


@dataclass
class Treatment:
    """Represents a single treatment option."""

    name: str
    category: TreatmentCategory
    description: str
    efficacy_score: float = 0.5      # 0–1, higher = more effective
    cost_score: float = 0.5          # 0–1, higher = more expensive
    side_effect_score: float = 0.5   # 0–1, higher = more side effects
    evidence_level: str = "C"        # A (strong) / B (moderate) / C (limited)
    contraindications: list[str] = field(default_factory=list)
    dosage_info: str = ""
    duration: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in treatment knowledge base (illustrative)
# ---------------------------------------------------------------------------

_TREATMENT_KB: dict[str, list[Treatment]] = {
    "hypertension": [
        Treatment(
            name="Lisinopril",
            category=TreatmentCategory.PHARMACOLOGICAL,
            description="ACE inhibitor for blood pressure control",
            efficacy_score=0.85,
            cost_score=0.2,
            side_effect_score=0.25,
            evidence_level="A",
            contraindications=["pregnancy", "angioedema", "bilateral_renal_stenosis"],
            dosage_info="10–40 mg PO daily",
            duration="Ongoing",
        ),
        Treatment(
            name="Amlodipine",
            category=TreatmentCategory.PHARMACOLOGICAL,
            description="Calcium channel blocker for blood pressure control",
            efficacy_score=0.82,
            cost_score=0.2,
            side_effect_score=0.3,
            evidence_level="A",
            contraindications=["severe_aortic_stenosis", "cardiogenic_shock"],
            dosage_info="5–10 mg PO daily",
            duration="Ongoing",
        ),
        Treatment(
            name="DASH Diet",
            category=TreatmentCategory.LIFESTYLE,
            description="Dietary approach to stop hypertension — rich in fruits, vegetables, low-fat dairy",
            efficacy_score=0.55,
            cost_score=0.1,
            side_effect_score=0.0,
            evidence_level="A",
            contraindications=[],
            duration="Ongoing",
        ),
        Treatment(
            name="Hydrochlorothiazide",
            category=TreatmentCategory.PHARMACOLOGICAL,
            description="Thiazide diuretic for blood pressure reduction",
            efficacy_score=0.78,
            cost_score=0.15,
            side_effect_score=0.35,
            evidence_level="A",
            contraindications=["anuria", "sulfa_allergy"],
            dosage_info="12.5–25 mg PO daily",
            duration="Ongoing",
        ),
    ],
    "type_2_diabetes": [
        Treatment(
            name="Metformin",
            category=TreatmentCategory.PHARMACOLOGICAL,
            description="Biguanide — first-line oral hypoglycaemic agent",
            efficacy_score=0.80,
            cost_score=0.15,
            side_effect_score=0.3,
            evidence_level="A",
            contraindications=["severe_renal_impairment", "metabolic_acidosis"],
            dosage_info="500–2000 mg PO daily",
            duration="Ongoing",
        ),
        Treatment(
            name="Lifestyle Modification",
            category=TreatmentCategory.LIFESTYLE,
            description="Diet, exercise, and weight management programme",
            efficacy_score=0.60,
            cost_score=0.1,
            side_effect_score=0.0,
            evidence_level="A",
            contraindications=[],
            duration="Ongoing",
        ),
        Treatment(
            name="Glipizide",
            category=TreatmentCategory.PHARMACOLOGICAL,
            description="Sulfonylurea — stimulates insulin secretion",
            efficacy_score=0.72,
            cost_score=0.15,
            side_effect_score=0.45,
            evidence_level="A",
            contraindications=["type_1_diabetes", "diabetic_ketoacidosis"],
            dosage_info="5–20 mg PO daily",
            duration="Ongoing",
        ),
    ],
    "pneumonia": [
        Treatment(
            name="Amoxicillin",
            category=TreatmentCategory.PHARMACOLOGICAL,
            description="First-line antibiotic for community-acquired pneumonia",
            efficacy_score=0.80,
            cost_score=0.15,
            side_effect_score=0.2,
            evidence_level="A",
            contraindications=["penicillin_allergy"],
            dosage_info="500 mg PO TID",
            duration="5–7 days",
        ),
        Treatment(
            name="Azithromycin",
            category=TreatmentCategory.PHARMACOLOGICAL,
            description="Macrolide antibiotic — covers atypical pathogens",
            efficacy_score=0.78,
            cost_score=0.25,
            side_effect_score=0.25,
            evidence_level="A",
            contraindications=["qt_prolongation", "macrolide_allergy"],
            dosage_info="500 mg day 1, then 250 mg days 2–5 PO daily",
            duration="5 days",
        ),
    ],
    "migraine": [
        Treatment(
            name="Sumatriptan",
            category=TreatmentCategory.PHARMACOLOGICAL,
            description="5-HT1 agonist for acute migraine attacks",
            efficacy_score=0.82,
            cost_score=0.4,
            side_effect_score=0.35,
            evidence_level="A",
            contraindications=["coronary_artery_disease", "uncontrolled_hypertension"],
            dosage_info="50–100 mg PO at onset; may repeat after 2 h",
            duration="As needed",
        ),
        Treatment(
            name="Topiramate",
            category=TreatmentCategory.PHARMACOLOGICAL,
            description="Anticonvulsant used for migraine prophylaxis",
            efficacy_score=0.65,
            cost_score=0.3,
            side_effect_score=0.45,
            evidence_level="A",
            contraindications=["glaucoma", "metabolic_acidosis"],
            dosage_info="25–100 mg PO daily",
            duration="Ongoing (prophylaxis)",
        ),
        Treatment(
            name="Lifestyle & Trigger Management",
            category=TreatmentCategory.LIFESTYLE,
            description="Identify and avoid migraine triggers; maintain regular sleep and meals",
            efficacy_score=0.45,
            cost_score=0.05,
            side_effect_score=0.0,
            evidence_level="B",
            contraindications=[],
            duration="Ongoing",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Ranking criteria
# ---------------------------------------------------------------------------

class RankingCriteria:
    """Weights for multi-criteria treatment ranking.

    Args:
        efficacy_weight: Importance of treatment efficacy.
        cost_weight: Importance of low cost (inverted so lower cost is better).
        side_effect_weight: Importance of low side-effect burden.
        evidence_weight: Importance of high evidence level.
    """

    def __init__(
        self,
        efficacy_weight: float = 0.4,
        cost_weight: float = 0.2,
        side_effect_weight: float = 0.25,
        evidence_weight: float = 0.15,
    ) -> None:
        self.efficacy_weight = efficacy_weight
        self.cost_weight = cost_weight
        self.side_effect_weight = side_effect_weight
        self.evidence_weight = evidence_weight

    def compute_score(self, treatment: Treatment) -> float:
        """Compute a composite score for a treatment.

        Higher score = more favourable.

        Args:
            treatment: A ``Treatment`` instance.

        Returns:
            Float score.
        """
        evidence_map = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}
        evidence_val = evidence_map.get(treatment.evidence_level, 0.5)

        score = (
            self.efficacy_weight * treatment.efficacy_score
            + self.cost_weight * (1.0 - treatment.cost_score)      # lower cost is better
            + self.side_effect_weight * (1.0 - treatment.side_effect_score)  # fewer SE is better
            + self.evidence_weight * evidence_val
        )
        return score


# ---------------------------------------------------------------------------
# TreatmentSuggester
# ---------------------------------------------------------------------------


class TreatmentSuggester:
    """Suggest and rank treatments based on diagnosis and patient context.

    Uses a built-in treatment knowledge base and configurable ranking criteria
    to generate personalised treatment recommendations.  Contraindications are
    checked against patient information and incompatible treatments are flagged.

    Args:
        config: Optional configuration dictionary.  Supported keys:

            * ``ranking_criteria`` — A ``RankingCriteria`` instance or dict
              of weight overrides.
            * ``custom_treatments`` — Dict mapping condition names to lists
              of ``Treatment`` objects to extend the knowledge base.
            * ``max_suggestions`` — Maximum number of treatments to return
              (default 5).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}

        # Ranking criteria
        rc = self.config.get("ranking_criteria", {})
        if isinstance(rc, RankingCriteria):
            self.criteria = rc
        else:
            self.criteria = RankingCriteria(**rc)

        # Treatment knowledge base (mutable copy)
        self._kb: dict[str, list[Treatment]] = {
            k: list(v) for k, v in _TREATMENT_KB.items()
        }

        # Extend with custom treatments
        for condition, treatments in self.config.get("custom_treatments", {}).items():
            key = condition.lower().replace(" ", "_")
            self._kb.setdefault(key, []).extend(treatments)

        self._max_suggestions = self.config.get("max_suggestions", 5)

        logger.info(
            "TreatmentSuggester initialised — %d conditions in KB, "
            "max_suggestions=%d",
            len(self._kb),
            self._max_suggestions,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(
        self,
        diagnosis: str,
        patient_info: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Suggest treatments for a given diagnosis.

        Treatments are filtered (contraindications checked against
        *patient_info*) and ranked by the configured criteria.

        Args:
            diagnosis: Condition name (case-insensitive, underscores/spaces
                interchangeable).
            patient_info: Optional dictionary of patient attributes.  Keys
                such as ``"allergies"``, ``"conditions"``, ``"medications"``
                are used to check contraindications.

        Returns:
            List of treatment dictionaries sorted by descending composite
            score.  Each dict includes ``name``, ``category``, ``description``,
            ``score``, ``contraindicated``, and ``contraindication_reasons``.
        """
        patient_info = patient_info or {}
        key = diagnosis.lower().replace(" ", "_")
        treatments = self._kb.get(key, [])

        if not treatments:
            logger.warning("No treatments found for diagnosis '%s'", diagnosis)
            return []

        results: list[dict[str, Any]] = []
        patient_conditions = set(
            c.lower() for c in patient_info.get("conditions", [])
        )
        patient_allergies = set(
            a.lower() for a in patient_info.get("allergies", [])
        )

        for treatment in treatments:
            # Check contraindications
            ci_reasons = []
            for ci in treatment.contraindications:
                ci_lower = ci.lower()
                if ci_lower in patient_conditions or ci_lower in patient_allergies:
                    ci_reasons.append(ci)

            contraindicated = len(ci_reasons) > 0
            score = self.criteria.compute_score(treatment)
            if contraindicated:
                score *= 0.1  # heavily penalise contraindicated treatments

            results.append({
                "name": treatment.name,
                "category": treatment.category.value,
                "description": treatment.description,
                "efficacy_score": round(treatment.efficacy_score, 2),
                "cost_score": round(treatment.cost_score, 2),
                "side_effect_score": round(treatment.side_effect_score, 2),
                "evidence_level": treatment.evidence_level,
                "dosage_info": treatment.dosage_info,
                "duration": treatment.duration,
                "composite_score": round(score, 4),
                "contraindicated": contraindicated,
                "contraindication_reasons": ci_reasons,
            })

        # Sort by composite score descending
        results.sort(key=lambda r: r["composite_score"], reverse=True)
        return results[: self._max_suggestions]

    def rank_treatments(
        self,
        treatments: list[dict[str, Any]],
        criteria: Optional[dict[str, float]] = None,
    ) -> list[dict[str, Any]]:
        """Re-rank a list of treatment dictionaries using custom criteria.

        Args:
            treatments: List of treatment dicts (as returned by ``suggest``).
            criteria: Optional dict of weight overrides for efficacy, cost,
                side_effect, and evidence.

        Returns:
            Re-ranked list with updated ``composite_score`` values.
        """
        if criteria:
            ranker = RankingCriteria(**criteria)
        else:
            ranker = self.criteria

        ranked = []
        for t in treatments:
            treatment_obj = Treatment(
                name=t["name"],
                category=TreatmentCategory(t["category"]),
                description=t["description"],
                efficacy_score=t.get("efficacy_score", 0.5),
                cost_score=t.get("cost_score", 0.5),
                side_effect_score=t.get("side_effect_score", 0.5),
                evidence_level=t.get("evidence_level", "C"),
                contraindications=t.get("contraindication_reasons", []),
            )
            score = ranker.compute_score(treatment_obj)
            entry = {**t, "composite_score": round(score, 4)}
            ranked.append(entry)

        ranked.sort(key=lambda r: r["composite_score"], reverse=True)
        return ranked

    def explain_recommendation(self, treatment: dict[str, Any]) -> str:
        """Generate a human-readable explanation for a treatment recommendation.

        Args:
            treatment: Treatment dictionary (as returned by ``suggest``).

        Returns:
            Explanatory string.
        """
        name = treatment.get("name", "Unknown")
        category = treatment.get("category", "unknown")
        efficacy = treatment.get("efficacy_score", 0)
        evidence = treatment.get("evidence_level", "C")
        contraindicated = treatment.get("contraindicated", False)
        ci_reasons = treatment.get("contraindication_reasons", [])

        parts = [
            f"Treatment: {name} ({category})",
            f"  Efficacy: {efficacy:.0%}",
            f"  Evidence level: {evidence}",
        ]

        if contraindicated:
            parts.append(
                f"  ⚠ CONTRAINDICATED — reasons: {', '.join(ci_reasons)}"
            )

        side_effects = treatment.get("side_effect_score", 0)
        if side_effects > 0.3:
            parts.append(
                f"  Note: Moderate-to-high side-effect burden ({side_effects:.0%})"
            )

        cost = treatment.get("cost_score", 0)
        if cost > 0.5:
            parts.append(f"  Note: Higher relative cost ({cost:.0%})")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Knowledge base management
    # ------------------------------------------------------------------

    def add_treatment(self, condition: str, treatment: Treatment) -> None:
        """Add a treatment to the knowledge base for a condition.

        Args:
            condition: Condition name.
            treatment: ``Treatment`` instance.
        """
        key = condition.lower().replace(" ", "_")
        self._kb.setdefault(key, []).append(treatment)

    def list_conditions(self) -> list[str]:
        """Return all condition keys in the knowledge base."""
        return sorted(self._kb.keys())
