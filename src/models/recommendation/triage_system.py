"""
TriageSystem - Emergency triage assessment and prioritisation.

This module implements an emergency department triage system based on the
5-level Emergency Severity Index (ESI). It assesses patient data to determine
a triage level, estimate expected wait times, and prioritise a queue of
patients for treatment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triage levels (ESI-based)
# ---------------------------------------------------------------------------


class TriageLevel(IntEnum):
    """Emergency Severity Index triage levels.

    Lower values indicate higher acuity (more urgent).

    * **1** — Resuscitation (immediate life-saving intervention required)
    * **2** — Emergent (high-risk situation, confused/lethargic/disoriented,
      severe pain/distress)
    * **3** — Urgent (multiple resources needed, stable vitals)
    * **4** — Less urgent (one resource needed, stable vitals)
    * **5** — Non-urgent (no resources needed, stable vitals)
    """

    RESUSCITATION = 1
    EMERGENT = 2
    URGENT = 3
    LESS_URGENT = 4
    NON_URGENT = 5


# Human-readable descriptions
_TRIAGE_DESCRIPTIONS: dict[TriageLevel, str] = {
    TriageLevel.RESUSCITATION: "Immediate life-saving intervention required",
    TriageLevel.EMERGENT: "High-risk situation; severe pain or distress",
    TriageLevel.URGENT: "Stable but requires multiple resources",
    TriageLevel.LESS_URGENT: "Stable; requires one resource",
    TriageLevel.NON_URGENT: "Stable; no resources required",
}

# Expected wait-time ranges per triage level (in minutes)
_WAIT_TIME_RANGES: dict[TriageLevel, tuple[int, int]] = {
    TriageLevel.RESUSCITATION: (0, 0),
    TriageLevel.EMERGENT: (0, 15),
    TriageLevel.URGENT: (15, 60),
    TriageLevel.LESS_URGENT: (60, 120),
    TriageLevel.NON_URGENT: (120, 240),
}


# ---------------------------------------------------------------------------
# Vital-sign thresholds
# ---------------------------------------------------------------------------

@dataclass
class VitalThresholds:
    """Thresholds for vital-sign-based acuity determination.

    Default values follow standard clinical guidelines.
    """

    heart_rate_critical_low: int = 40
    heart_rate_critical_high: int = 150
    heart_rate_warning_low: int = 50
    heart_rate_warning_high: int = 120

    systolic_bp_critical_low: int = 70
    systolic_bp_critical_high: int = 220
    systolic_bp_warning_low: int = 90
    systolic_bp_warning_high: int = 180

    respiratory_rate_critical_low: int = 8
    respiratory_rate_critical_high: int = 36
    respiratory_rate_warning_low: int = 10
    respiratory_rate_warning_high: int = 28

    oxygen_saturation_critical: float = 90.0
    oxygen_saturation_warning: float = 94.0

    temperature_critical_high: float = 41.0
    temperature_warning_high: float = 39.5
    temperature_critical_low: float = 34.0
    temperature_warning_low: float = 35.0

    gcs_critical: int = 8      # Glasgow Coma Scale ≤ 8 = critical
    gcs_warning: int = 13      # GCS ≤ 13 = warning


# ---------------------------------------------------------------------------
# Triage assessment result
# ---------------------------------------------------------------------------

@dataclass
class TriageAssessment:
    """Result of a triage assessment."""

    triage_level: TriageLevel
    description: str
    estimated_wait_minutes: int
    reasoning: list[str] = field(default_factory=list)
    critical_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TriageSystem
# ---------------------------------------------------------------------------


class TriageSystem:
    """Emergency triage assessment and patient prioritisation.

    Implements a rule-based triage algorithm inspired by the Emergency
    Severity Index (ESI).  The system evaluates patient vitals, chief
    complaint, and consciousness level to assign one of five triage
    levels, then provides estimated wait times and queue prioritisation.

    Args:
        config: Optional configuration dictionary.  Supported keys:

            * ``vital_thresholds`` — A ``VitalThresholds`` instance or dict
              of threshold overrides.
            * ``custom_complaint_mapping`` — Dict mapping chief-complaint
              keywords to ``TriageLevel`` values.
            * ``resource_time_multiplier`` — Float multiplier applied to
              wait-time estimates when the department is busier than usual.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}

        # Vital thresholds
        vt = self.config.get("vital_thresholds", {})
        if isinstance(vt, VitalThresholds):
            self.thresholds = vt
        else:
            self.thresholds = VitalThresholds(**vt)

        # Custom complaint → triage level mapping
        self._complaint_mapping: dict[str, TriageLevel] = {
            "cardiac arrest": TriageLevel.RESUSCITATION,
            "respiratory arrest": TriageLevel.RESUSCITATION,
            "unresponsive": TriageLevel.RESUSCITATION,
            "anaphylaxis": TriageLevel.RESUSCITATION,
            "chest pain": TriageLevel.EMERGENT,
            "stroke symptoms": TriageLevel.EMERGENT,
            "severe bleeding": TriageLevel.EMERGENT,
            "overdose": TriageLevel.EMERGENT,
            "severe trauma": TriageLevel.EMERGENT,
            "seizure": TriageLevel.EMERGENT,
            "difficulty breathing": TriageLevel.EMERGENT,
            "abdominal pain": TriageLevel.URGENT,
            "vomiting blood": TriageLevel.URGENT,
            "high fever": TriageLevel.URGENT,
            "fracture": TriageLevel.URGENT,
            "laceration": TriageLevel.LESS_URGENT,
            "sprain": TriageLevel.LESS_URGENT,
            "mild allergic reaction": TriageLevel.LESS_URGENT,
            "headache": TriageLevel.NON_URGENT,
            "cold symptoms": TriageLevel.NON_URGENT,
            "minor rash": TriageLevel.NON_URGENT,
            "prescription refill": TriageLevel.NON_URGENT,
        }
        for complaint, level in self.config.get("custom_complaint_mapping", {}).items():
            self._complaint_mapping[complaint.lower()] = TriageLevel(level)

        self._resource_multiplier = self.config.get("resource_time_multiplier", 1.0)

        logger.info("TriageSystem initialised — %d complaint mappings", len(self._complaint_mapping))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, patient_data: dict[str, Any]) -> TriageAssessment:
        """Assess a patient and assign a triage level.

        The algorithm evaluates:

        1. **Life-threatening signs** (→ Level 1)
        2. **Critical vitals or high-risk chief complaint** (→ Level 2)
        3. **Warning vitals or moderate-risk complaint** (→ Level 3)
        4. **Mild complaint, one resource needed** (→ Level 4)
        5. **No acute issues** (→ Level 5)

        Args:
            patient_data: Dictionary with patient information.  Expected keys:

                * ``chief_complaint`` (str)
                * ``heart_rate`` (int, optional)
                * ``systolic_bp`` (int, optional)
                * ``respiratory_rate`` (int, optional)
                * ``oxygen_saturation`` (float, optional, 0–100)
                * ``temperature`` (float, optional, °C)
                * ``gcs`` (int, optional, Glasgow Coma Scale 3–15)
                * ``conscious`` (bool, optional)
                * ``severe_pain`` (bool, optional)
                * ``age`` (int, optional)

        Returns:
            A ``TriageAssessment`` with triage level, description, wait time,
            and reasoning.
        """
        reasoning: list[str] = []
        critical_flags: list[str] = []
        level = TriageLevel.NON_URGENT  # default

        # ---- Step 1: Check for life-threatening conditions (Level 1) ----
        if self._is_resuscitation(patient_data, critical_flags, reasoning):
            level = TriageLevel.RESUSCITATION
        # ---- Step 2: Check for emergent conditions (Level 2) ----
        elif self._is_emergent(patient_data, critical_flags, reasoning):
            level = TriageLevel.EMERGENT
        # ---- Step 3: Check for urgent conditions (Level 3) ----
        elif self._is_urgent(patient_data, reasoning):
            level = TriageLevel.URGENT
        # ---- Step 4: Check for less-urgent conditions (Level 4) ----
        elif self._is_less_urgent(patient_data, reasoning):
            level = TriageLevel.LESS_URGENT
        else:
            reasoning.append("No acute findings; patient appears stable")

        # Override with chief-complaint mapping if it suggests higher acuity
        complaint_level = self._complaint_from_data(patient_data)
        if complaint_level is not None and complaint_level < level:
            reasoning.append(
                f"Chief complaint suggests {complaint_level.name} acuity; overriding"
            )
            level = complaint_level

        # Estimate wait time
        wait = self.get_wait_time(level)

        return TriageAssessment(
            triage_level=level,
            description=_TRIAGE_DESCRIPTIONS[level],
            estimated_wait_minutes=wait,
            reasoning=reasoning,
            critical_flags=critical_flags,
        )

    def get_wait_time(self, triage_level: TriageLevel) -> int:
        """Estimate wait time in minutes for a given triage level.

        Returns the midpoint of the expected range, adjusted by the
        resource-time multiplier.

        Args:
            triage_level: The patient's triage level.

        Returns:
            Estimated wait time in minutes.
        """
        low, high = _WAIT_TIME_RANGES[triage_level]
        wait = (low + high) // 2
        wait = int(wait * self._resource_multiplier)
        return max(0, wait)

    def prioritize(self, patients: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prioritise a list of patients for treatment ordering.

        Each patient dict is assessed and sorted by triage level (most
        urgent first).  Patients at the same level are sub-sorted by
        arrival time (earliest first) when available.

        Args:
            patients: List of patient data dictionaries (same format as
                ``assess()``).

        Returns:
            Sorted list of patient dicts with added ``triage_level`` and
            ``estimated_wait_minutes`` keys.
        """
        assessed: list[dict[str, Any]] = []

        for patient in patients:
            result = self.assess(patient)
            assessed.append({
                **patient,
                "triage_level": result.triage_level.value,
                "triage_label": result.triage_level.name,
                "estimated_wait_minutes": result.estimated_wait_minutes,
                "triage_reasoning": result.reasoning,
                "critical_flags": result.critical_flags,
            })

        # Sort: lower triage level first (more urgent), then earlier arrival
        assessed.sort(
            key=lambda p: (
                p["triage_level"],
                p.get("arrival_time", float("inf")),
            )
        )

        logger.info(
            "Prioritised %d patients — level distribution: %s",
            len(assessed),
            {level.name: sum(1 for p in assessed if p["triage_level"] == level.value)
             for level in TriageLevel},
        )

        return assessed

    # ------------------------------------------------------------------
    # Internal assessment helpers
    # ------------------------------------------------------------------

    def _is_resuscitation(
        self,
        data: dict[str, Any],
        flags: list[str],
        reasoning: list[str],
    ) -> bool:
        """Check for Level 1 (resuscitation) criteria."""
        # Unconscious
        if data.get("conscious") is False or data.get("gcs", 15) <= self.thresholds.gcs_critical:
            flags.append("unconscious_or_gcs_critical")
            reasoning.append("Patient is unconscious or GCS ≤ 8")
            return True

        # Respiratory rate critically low or absent
        rr = data.get("respiratory_rate")
        if rr is not None and rr <= self.thresholds.respiratory_rate_critical_low:
            flags.append("respiratory_failure")
            reasoning.append(f"Respiratory rate critically low ({rr})")
            return True

        # Systolic BP critically low (shock)
        sbp = data.get("systolic_bp")
        if sbp is not None and sbp <= self.thresholds.systolic_bp_critical_low:
            flags.append("shock")
            reasoning.append(f"Systolic BP critically low ({sbp})")
            return True

        # Heart rate critically abnormal
        hr = data.get("heart_rate")
        if hr is not None and (
            hr <= self.thresholds.heart_rate_critical_low
            or hr >= self.thresholds.heart_rate_critical_high
        ):
            flags.append("critical_heart_rate")
            reasoning.append(f"Heart rate critical ({hr})")
            return True

        return False

    def _is_emergent(
        self,
        data: dict[str, Any],
        flags: list[str],
        reasoning: list[str],
    ) -> bool:
        """Check for Level 2 (emergent) criteria."""
        # Severe pain
        if data.get("severe_pain"):
            reasoning.append("Patient reports severe pain")
            return True

        # GCS warning
        gcs = data.get("gcs", 15)
        if gcs <= self.thresholds.gcs_warning:
            reasoning.append(f"Altered consciousness (GCS={gcs})")
            return True

        # Warning-range vitals
        hr = data.get("heart_rate")
        if hr is not None and (
            hr <= self.thresholds.heart_rate_warning_low
            or hr >= self.thresholds.heart_rate_warning_high
        ):
            reasoning.append(f"Heart rate abnormal ({hr})")
            return True

        sbp = data.get("systolic_bp")
        if sbp is not None and (
            sbp <= self.thresholds.systolic_bp_warning_low
            or sbp >= self.thresholds.systolic_bp_warning_high
        ):
            reasoning.append(f"Systolic BP abnormal ({sbp})")
            return True

        rr = data.get("respiratory_rate")
        if rr is not None and (
            rr <= self.thresholds.respiratory_rate_warning_low
            or rr >= self.thresholds.respiratory_rate_warning_high
        ):
            reasoning.append(f"Respiratory rate abnormal ({rr})")
            return True

        spo2 = data.get("oxygen_saturation")
        if spo2 is not None and spo2 <= self.thresholds.oxygen_saturation_critical:
            reasoning.append(f"Oxygen saturation critical ({spo2}%)")
            return True

        # High-risk chief complaint
        complaint = (data.get("chief_complaint") or "").lower()
        for keyword, level in self._complaint_mapping.items():
            if keyword in complaint and level <= TriageLevel.EMERGENT:
                reasoning.append(f"High-risk chief complaint: '{keyword}'")
                return True

        return False

    def _is_urgent(
        self,
        data: dict[str, Any],
        reasoning: list[str],
    ) -> bool:
        """Check for Level 3 (urgent) criteria."""
        spo2 = data.get("oxygen_saturation")
        if spo2 is not None and spo2 <= self.thresholds.oxygen_saturation_warning:
            reasoning.append(f"Oxygen saturation low ({spo2}%)")
            return True

        temp = data.get("temperature")
        if temp is not None and (
            temp >= self.thresholds.temperature_warning_high
            or temp <= self.thresholds.temperature_warning_low
        ):
            reasoning.append(f"Temperature abnormal ({temp}°C)")
            return True

        # Moderate-risk chief complaint
        complaint = (data.get("chief_complaint") or "").lower()
        for keyword, level in self._complaint_mapping.items():
            if keyword in complaint and level == TriageLevel.URGENT:
                reasoning.append(f"Moderate-risk chief complaint: '{keyword}'")
                return True

        return False

    def _is_less_urgent(
        self,
        data: dict[str, Any],
        reasoning: list[str],
    ) -> bool:
        """Check for Level 4 (less-urgent) criteria."""
        complaint = (data.get("chief_complaint") or "").lower()
        for keyword, level in self._complaint_mapping.items():
            if keyword in complaint and level == TriageLevel.LESS_URGENT:
                reasoning.append(f"Minor complaint: '{keyword}'")
                return True

        return False

    def _complaint_from_data(self, data: dict[str, Any]) -> Optional[TriageLevel]:
        """Look up triage level from chief complaint mapping."""
        complaint = (data.get("chief_complaint") or "").lower()
        for keyword, level in self._complaint_mapping.items():
            if keyword in complaint:
                return level
        return None
