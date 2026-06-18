"""
HospitalRecommender - Recommend hospitals based on condition and location.

This module provides a hospital recommendation engine that matches patients
to suitable healthcare facilities based on their medical condition, geographic
location, and hospital specialities.  Ranking is performed using configurable
criteria including speciality match, distance, capacity, and quality ratings.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HospitalTier(str, Enum):
    """Hospital accreditation tier."""

    TIER_1 = "tier_1"  # Major academic / research hospital
    TIER_2 = "tier_2"  # Regional hospital with specialty centres
    TIER_3 = "tier_3"  # Community hospital
    TIER_4 = "tier_4"  # Rural / primary-care facility


@dataclass
class Hospital:
    """Represents a healthcare facility."""

    id: str
    name: str
    tier: HospitalTier
    specialties: list[str]
    latitude: float
    longitude: float
    overall_rating: float = 3.0     # 1–5
    capacity_available: float = 0.5  # 0–1, proportion of beds available
    average_wait_hours: float = 4.0  # hours
    emergency_capacity: bool = True
    icu_beds_available: int = 10
    accepts_insurance: list[str] = field(default_factory=list)
    address: str = ""
    phone: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in hospital database (illustrative)
# ---------------------------------------------------------------------------

_HOSPITAL_DB: list[Hospital] = [
    Hospital(
        id="H001",
        name="Metropolitan General Hospital",
        tier=HospitalTier.TIER_1,
        specialties=[
            "cardiology", "oncology", "neurosurgery", "trauma",
            "orthopedics", "pediatrics", "obstetrics", "pulmonology",
        ],
        latitude=40.7128,
        longitude=-74.0060,
        overall_rating=4.5,
        capacity_available=0.35,
        average_wait_hours=6.0,
        emergency_capacity=True,
        icu_beds_available=15,
        accepts_insurance=["blue_cross", "aetna", "united", "cigna"],
        address="100 Medical Center Blvd, New York, NY",
        phone="+1-212-555-0100",
    ),
    Hospital(
        id="H002",
        name="Lakeside Regional Medical Center",
        tier=HospitalTier.TIER_2,
        specialties=[
            "cardiology", "orthopedics", "general_surgery",
            "pediatrics", "obstetrics", "neurology",
        ],
        latitude=40.7580,
        longitude=-73.9855,
        overall_rating=4.1,
        capacity_available=0.55,
        average_wait_hours=3.5,
        emergency_capacity=True,
        icu_beds_available=8,
        accepts_insurance=["blue_cross", "aetna", "united"],
        address="250 Lake Ave, New York, NY",
        phone="+1-212-555-0200",
    ),
    Hospital(
        id="H003",
        name="Westside Community Hospital",
        tier=HospitalTier.TIER_3,
        specialties=["general_surgery", "pediatrics", "obstetrics", "internal_medicine"],
        latitude=40.7282,
        longitude=-74.0776,
        overall_rating=3.8,
        capacity_available=0.70,
        average_wait_hours=2.0,
        emergency_capacity=True,
        icu_beds_available=4,
        accepts_insurance=["blue_cross", "aetna"],
        address="80 West St, Jersey City, NJ",
        phone="+1-201-555-0300",
    ),
    Hospital(
        id="H004",
        name="Children's Specialized Center",
        tier=HospitalTier.TIER_1,
        specialties=["pediatrics", "pediatric_oncology", "neonatology", "pediatric_cardiology"],
        latitude=40.7484,
        longitude=-73.9857,
        overall_rating=4.8,
        capacity_available=0.40,
        average_wait_hours=5.0,
        emergency_capacity=True,
        icu_beds_available=20,
        accepts_insurance=["blue_cross", "aetna", "united", "cigna"],
        address="350 Child Way, New York, NY",
        phone="+1-212-555-0400",
    ),
    Hospital(
        id="H005",
        name="Heart & Vascular Institute",
        tier=HospitalTier.TIER_1,
        specialties=["cardiology", "cardiac_surgery", "interventional_cardiology", "vascular_surgery"],
        latitude=40.7614,
        longitude=-73.9776,
        overall_rating=4.7,
        capacity_available=0.30,
        average_wait_hours=7.0,
        emergency_capacity=True,
        icu_beds_available=25,
        accepts_insurance=["blue_cross", "united", "cigna"],
        address="500 Cardiac Way, New York, NY",
        phone="+1-212-555-0500",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute great-circle distance in kilometres using the Haversine formula.

    Args:
        lat1, lon1: Coordinates of point 1 (degrees).
        lat2, lon2: Coordinates of point 2 (degrees).

    Returns:
        Distance in kilometres.
    """
    R = 6371.0  # Earth radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
# HospitalRecommender
# ---------------------------------------------------------------------------


class HospitalRecommender:
    """Recommend hospitals based on medical condition and patient location.

    Ranks hospitals by a composite score that considers:

    * **Specialty match** — whether the hospital treats the specified condition.
    * **Proximity** — geographic distance from the patient.
    * **Quality** — overall hospital rating.
    * **Availability** — current bed capacity and wait times.

    Args:
        config: Optional configuration dictionary.  Supported keys:

            * ``max_distance_km`` — Maximum distance to consider (default 100).
            * ``specialty_weight`` — Weight for specialty match (default 0.35).
            * ``distance_weight`` — Weight for proximity (default 0.25).
            * ``quality_weight`` — Weight for hospital rating (default 0.25).
            * ``availability_weight`` — Weight for capacity / wait time (default 0.15).
            * ``custom_hospitals`` — List of ``Hospital`` objects to add.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}

        self.max_distance_km: float = self.config.get("max_distance_km", 100.0)
        self.specialty_weight: float = self.config.get("specialty_weight", 0.35)
        self.distance_weight: float = self.config.get("distance_weight", 0.25)
        self.quality_weight: float = self.config.get("quality_weight", 0.25)
        self.availability_weight: float = self.config.get("availability_weight", 0.15)

        # Hospital database (mutable copy)
        self._hospitals: list[Hospital] = list(_HOSPITAL_DB)
        for h in self.config.get("custom_hospitals", []):
            self._hospitals.append(h)

        # Map conditions to likely relevant specialties
        self._condition_specialty_map: dict[str, list[str]] = {
            "heart_attack": ["cardiology", "cardiac_surgery", "interventional_cardiology"],
            "cardiac_arrest": ["cardiology", "cardiac_surgery", "trauma"],
            "stroke": ["neurosurgery", "neurology"],
            "pneumonia": ["pulmonology", "internal_medicine"],
            "cancer": ["oncology", "pediatric_oncology"],
            "fracture": ["orthopedics", "trauma"],
            "pregnancy": ["obstetrics"],
            "child_illness": ["pediatrics", "pediatric_oncology", "pediatric_cardiology"],
            "cardiovascular_disease": ["cardiology", "cardiac_surgery", "vascular_surgery"],
        }

        logger.info(
            "HospitalRecommender initialised — %d hospitals in DB",
            len(self._hospitals),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        condition: str,
        location: tuple[float, float],
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Recommend hospitals for a condition near a location.

        Args:
            condition: Medical condition name (case-insensitive).
            location: ``(latitude, longitude)`` of the patient.
            max_results: Maximum number of results.

        Returns:
            List of hospital recommendation dicts sorted by descending
            composite score.
        """
        lat, lon = location
        relevant_specialties = self._condition_specialty_map.get(
            condition.lower().replace(" ", "_"), []
        )

        results: list[dict[str, Any]] = []
        for hospital in self._hospitals:
            distance = _haversine_km(lat, lon, hospital.latitude, hospital.longitude)

            if distance > self.max_distance_km:
                continue

            # Specialty match score
            if relevant_specialties:
                match_count = sum(
                    1 for s in relevant_specialties if s in hospital.specialties
                )
                specialty_score = match_count / len(relevant_specialties)
            else:
                # No specialty mapping — give neutral score
                specialty_score = 0.5

            # Distance score (closer = better, normalised by max_distance)
            distance_score = 1.0 - (distance / self.max_distance_km)

            # Quality score (normalise 1–5 rating to 0–1)
            quality_score = (hospital.overall_rating - 1.0) / 4.0

            # Availability score (capacity and wait time)
            capacity_score = hospital.capacity_available
            wait_score = max(0.0, 1.0 - hospital.average_wait_hours / 24.0)
            availability_score = 0.6 * capacity_score + 0.4 * wait_score

            # Composite
            composite = (
                self.specialty_weight * specialty_score
                + self.distance_weight * distance_score
                + self.quality_weight * quality_score
                + self.availability_weight * availability_score
            )

            results.append({
                "hospital_id": hospital.id,
                "name": hospital.name,
                "tier": hospital.tier.value,
                "address": hospital.address,
                "phone": hospital.phone,
                "distance_km": round(distance, 1),
                "specialty_match": round(specialty_score, 2),
                "quality_score": round(quality_score, 2),
                "availability_score": round(availability_score, 2),
                "composite_score": round(composite, 4),
                "overall_rating": hospital.overall_rating,
                "average_wait_hours": hospital.average_wait_hours,
                "emergency_capacity": hospital.emergency_capacity,
                "icu_beds_available": hospital.icu_beds_available,
            })

        results.sort(key=lambda r: r["composite_score"], reverse=True)
        return results[:max_results]

    def rank_hospitals(
        self,
        hospitals: list[dict[str, Any]],
        criteria: Optional[dict[str, float]] = None,
    ) -> list[dict[str, Any]]:
        """Re-rank a list of hospital dicts using custom criteria weights.

        Args:
            hospitals: List of hospital dicts (as returned by ``recommend``).
            criteria: Optional weight overrides (specialty, distance, quality,
                availability).

        Returns:
            Re-ranked list with updated ``composite_score``.
        """
        sw = criteria.get("specialty_weight", self.specialty_weight) if criteria else self.specialty_weight
        dw = criteria.get("distance_weight", self.distance_weight) if criteria else self.distance_weight
        qw = criteria.get("quality_weight", self.quality_weight) if criteria else self.quality_weight
        aw = criteria.get("availability_weight", self.availability_weight) if criteria else self.availability_weight

        ranked = []
        for h in hospitals:
            composite = (
                sw * h.get("specialty_match", 0)
                + dw * (1.0 - h.get("distance_km", 50) / self.max_distance_km)
                + qw * h.get("quality_score", 0)
                + aw * h.get("availability_score", 0)
            )
            ranked.append({**h, "composite_score": round(composite, 4)})

        ranked.sort(key=lambda r: r["composite_score"], reverse=True)
        return ranked

    def get_hospital_details(self, hospital_id: str) -> dict[str, Any]:
        """Retrieve full details for a hospital by its ID.

        Args:
            hospital_id: Unique hospital identifier.

        Returns:
            Dictionary with all hospital attributes.

        Raises:
            KeyError: If the hospital ID is not found.
        """
        for h in self._hospitals:
            if h.id == hospital_id:
                return {
                    "id": h.id,
                    "name": h.name,
                    "tier": h.tier.value,
                    "specialties": h.specialties,
                    "latitude": h.latitude,
                    "longitude": h.longitude,
                    "overall_rating": h.overall_rating,
                    "capacity_available": h.capacity_available,
                    "average_wait_hours": h.average_wait_hours,
                    "emergency_capacity": h.emergency_capacity,
                    "icu_beds_available": h.icu_beds_available,
                    "accepts_insurance": h.accepts_insurance,
                    "address": h.address,
                    "phone": h.phone,
                }

        raise KeyError(f"Hospital '{hospital_id}' not found in database")

    # ------------------------------------------------------------------
    # Knowledge base management
    # ------------------------------------------------------------------

    def add_hospital(self, hospital: Hospital) -> None:
        """Add a hospital to the database.

        Args:
            hospital: ``Hospital`` instance.
        """
        self._hospitals.append(hospital)

    def list_hospitals(self) -> list[str]:
        """Return IDs of all hospitals in the database."""
        return [h.id for h in self._hospitals]
