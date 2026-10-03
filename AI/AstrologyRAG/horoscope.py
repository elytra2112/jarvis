"""
Astro JARVIS - horoscope/forecast prompt builder.

This module does not pretend astrology is scientifically established.
It prepares calculated astrology data + retrieved book material for the
LLM to interpret as an astrological tradition.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .chart_engine import SIGNS, SIGN_LORDS, sign_from_longitude
from .transits import calculate_current_transits, transit_to_natal_contacts


def normalize_sign(sign: str) -> str | None:
    cleaned = sign.strip().lower()
    aliases = {
        "aries": "Aries", "taurus": "Taurus", "gemini": "Gemini",
        "cancer": "Cancer", "leo": "Leo", "virgo": "Virgo",
        "libra": "Libra", "scorpio": "Scorpio", "scorpion": "Scorpio",
        "sagittarius": "Sagittarius", "sag": "Sagittarius",
        "capricorn": "Capricorn", "aquarius": "Aquarius",
        "pisces": "Pisces",
    }
    return aliases.get(cleaned)


def get_sun_sign(longitude: float) -> str:
    return sign_from_longitude(longitude)[0]


def build_daily_horoscope_context(
    sign: str,
    retrieved_book_context: str | None = None,
    when_utc: datetime | None = None,
) -> str:
    canonical = normalize_sign(sign)
    if not canonical:
        raise ValueError(f"Unknown zodiac sign: {sign}")

    transits = calculate_current_transits(when_utc)

    lines = [
        f"Requested zodiac sign: {canonical}",
        f"Sign lord in this traditional mapping: {SIGN_LORDS[canonical]}",
        f"Forecast date/time UTC: {transits['datetime_utc']}",
        "",
        "CURRENT SIDEREAL TRANSITS:",
    ]

    for name, p in transits["planets"].items():
        retro = " retrograde" if p.get("retrograde") else ""
        lines.append(f"- {name}: {p['formatted']}{retro}")

    if retrieved_book_context:
        lines.extend([
            "",
            "RETRIEVED ASTROLOGY BOOK MATERIAL:",
            retrieved_book_context,
        ])

    return "\n".join(lines)


def build_personal_forecast_context(
    natal_chart: dict[str, Any],
    retrieved_book_context: str | None = None,
    when_utc: datetime | None = None,
) -> str:
    transits = calculate_current_transits(when_utc)
    contacts = transit_to_natal_contacts(natal_chart, transits)

    lines = [
        "PERSONAL ASTROLOGY FORECAST DATA",
        "",
        "NATAL CHART:",
        f"Ascendant: {natal_chart['ascendant']['formatted']}",
        (
            f"Moon nakshatra: {natal_chart['moon_nakshatra']['name']}, "
            f"Pada {natal_chart['moon_nakshatra']['pada']}"
        ),
    ]

    for name, p in natal_chart["planets"].items():
        retro = " retrograde" if p.get("retrograde") else ""
        lines.append(
            f"- Natal {name}: {p['formatted']}, House {p['house']}{retro}"
        )

    active = natal_chart["vimshottari"].get("active_mahadasha")
    if active:
        lines.extend([
            "",
            f"Active Mahadasha: {active['lord']}",
            f"Mahadasha period: {active['start']} to {active['end']}",
        ])

    lines.extend([
        "",
        "CURRENT TRANSITS:",
    ])
    for name, p in transits["planets"].items():
        retro = " retrograde" if p.get("retrograde") else ""
        lines.append(f"- Transit {name}: {p['formatted']}{retro}")

    if contacts:
        lines.extend(["", "CLOSE TRANSIT-TO-NATAL CONTACTS:"])
        for c in contacts[:20]:
            lines.append(
                f"- Transit {c['transit_planet']} {c['aspect']} "
                f"natal {c['natal_planet']} (orb {c['orb']}°)"
            )

    if retrieved_book_context:
        lines.extend([
            "",
            "RETRIEVED ASTROLOGY BOOK MATERIAL:",
            retrieved_book_context,
        ])

    return "\n".join(lines)


def make_horoscope_prompt(
    user_question: str,
    astrology_context: str,
    personal: bool = False,
) -> str:
    mode = "personal chart" if personal else "zodiac horoscope"

    return f"""
The user is asking for an astrology {mode}.

Use the supplied calculated astrology data and retrieved astrology-book
material as the basis of the interpretation.

ASTROLOGY DATA AND BOOK CONTEXT:
{astrology_context}

USER QUESTION:
{user_question}

Response rules:
- Present this as an interpretation within astrology, not as scientifically
  established fact.
- Do not claim certainty about future events.
- Do not invent chart placements, transits, dashas, or book teachings.
- Distinguish calculated placements from interpretive statements.
- If the supplied book context does not support a specific interpretation,
  say that the available material is insufficient.
- Keep the answer natural for a voice assistant.
- For a daily horoscope, organize briefly around themes such as work/study,
  relationships, energy, and practical focus only when the source material
  supports them.
- Avoid alarming or deterministic predictions.
"""
