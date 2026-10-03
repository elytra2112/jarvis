"""
Astro JARVIS - current transit engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .chart_engine import (
    PLANETS,
    normalize_degree,
    sign_from_longitude,
    format_position,
)

import swisseph as swe


def calculate_current_transits(
    when_utc: datetime | None = None,
) -> dict[str, Any]:
    if when_utc is None:
        when_utc = datetime.now(timezone.utc)

    if when_utc.tzinfo is None:
        raise ValueError("when_utc must be timezone-aware")

    when_utc = when_utc.astimezone(timezone.utc)

    hour = (
        when_utc.hour
        + when_utc.minute / 60
        + when_utc.second / 3600
        + when_utc.microsecond / 3_600_000_000
    )
    jd = swe.julday(
        when_utc.year,
        when_utc.month,
        when_utc.day,
        hour,
        swe.GREG_CAL,
    )

    swe.set_sid_mode(swe.SIDM_LAHIRI)
    flags = swe.FLG_SWIEPH | swe.FLG_SPEED | swe.FLG_SIDEREAL

    result = {}
    for name, body in PLANETS.items():
        values, _ = swe.calc_ut(jd, body, flags)
        longitude = normalize_degree(values[0])
        sign, sign_index, degree = sign_from_longitude(longitude)

        result[name] = {
            "longitude": longitude,
            "sign": sign,
            "sign_index": sign_index,
            "degree": degree,
            "formatted": format_position(longitude),
            "retrograde": values[3] < 0,
            "speed_longitude": values[3],
        }

    rahu = result["Rahu"]["longitude"]
    ketu = normalize_degree(rahu + 180.0)
    k_sign, k_index, k_degree = sign_from_longitude(ketu)
    result["Ketu"] = {
        "longitude": ketu,
        "sign": k_sign,
        "sign_index": k_index,
        "degree": k_degree,
        "formatted": format_position(ketu),
        "retrograde": result["Rahu"]["retrograde"],
        "speed_longitude": result["Rahu"]["speed_longitude"],
    }

    return {
        "datetime_utc": when_utc.isoformat(),
        "system": "Sidereal/Lahiri",
        "planets": result,
    }


def transit_to_natal_contacts(
    natal_chart: dict[str, Any],
    transits: dict[str, Any],
    orb_degrees: float = 3.0,
) -> list[dict[str, Any]]:
    """
    Return close angular contacts between transiting and natal planets.

    This is deliberately a small, transparent aspect engine. Interpretation
    is left to the RAG + LLM layer.
    """
    contacts = []
    natal_planets = natal_chart.get("planets", {})
    transit_planets = transits.get("planets", {})

    for t_name, t in transit_planets.items():
        for n_name, n in natal_planets.items():
            if t_name == n_name:
                continue

            raw = abs(t["longitude"] - n["longitude"]) % 360.0
            distance = min(raw, 360.0 - raw)

            aspect = None
            for angle, label in [
                (0, "conjunction"),
                (60, "sextile"),
                (90, "square"),
                (120, "trine"),
                (180, "opposition"),
            ]:
                if abs(distance - angle) <= orb_degrees:
                    aspect = label
                    break

            if aspect:
                contacts.append({
                    "transit_planet": t_name,
                    "natal_planet": n_name,
                    "aspect": aspect,
                    "orb": round(abs(
                        distance - {
                            "conjunction": 0,
                            "sextile": 60,
                            "square": 90,
                            "trine": 120,
                            "opposition": 180,
                        }[aspect]
                    ), 2),
                    "transit_sign": t["sign"],
                    "natal_sign": n["sign"],
                })

    return contacts


def transit_to_prompt_context(
    natal_chart: dict[str, Any] | None = None,
    when_utc: datetime | None = None,
) -> str:
    transits = calculate_current_transits(when_utc)

    lines = [
        f"Transit time UTC: {transits['datetime_utc']}",
        f"Transit system: {transits['system']}",
    ]

    for name, p in transits["planets"].items():
        retro = " retrograde" if p.get("retrograde") else ""
        lines.append(f"Transit {name}: {p['formatted']}{retro}")

    if natal_chart:
        contacts = transit_to_natal_contacts(natal_chart, transits)
        if contacts:
            lines.append("Close transit-to-natal contacts:")
            for c in contacts[:20]:
                lines.append(
                    f"- {c['transit_planet']} {c['aspect']} "
                    f"natal {c['natal_planet']} "
                    f"(orb {c['orb']}°, "
                    f"transit {c['transit_sign']}, natal {c['natal_sign']})"
                )

    return "\n".join(lines)
