"""
Astro JARVIS - intent router.

This module only detects astrology intent and extracts lightweight entities.
It does not calculate charts itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .horoscope import normalize_sign


ASTRO_KEYWORDS = [
    "astrology", "astrological", "horoscope", "kundli", "kundali",
    "birth chart", "natal chart", "vedic astrology", "jyotish",
    "zodiac", "ascendant", "rising sign", "lagna", "nakshatra",
    "dasha", "mahadasha", "antardasha", "retrograde", "transit",
    "planetary transit", "planet transit",
]

PREDICTION_WORDS = [
    "predict", "prediction", "forecast", "future", "today", "tomorrow",
    "this week", "next week", "this month", "next month", "horoscope",
    "what will happen", "what should i expect",
]


@dataclass
class AstrologyIntent:
    is_astrology: bool
    intent: str = "general"
    zodiac_sign: str | None = None
    needs_birth_data: bool = False


def detect_astrology_intent(text: str) -> AstrologyIntent:
    t = text.lower().strip()

    is_astrology = any(k in t for k in ASTRO_KEYWORDS)
    if not is_astrology:
        return AstrologyIntent(False)

    sign = None
    for candidate in [
        "aries", "taurus", "gemini", "cancer", "leo", "virgo",
        "libra", "scorpio", "sagittarius", "capricorn", "aquarius",
        "pisces",
    ]:
        if re.search(rf"\b{re.escape(candidate)}\b", t):
            sign = normalize_sign(candidate)
            break

    if any(x in t for x in ["birth chart", "natal chart", "my chart", "my horoscope"]):
        intent = "personal"
        needs_birth_data = True
    elif any(x in t for x in PREDICTION_WORDS):
        intent = "horoscope"
        needs_birth_data = "my " in t or "for me" in t or "myself" in t
    elif any(x in t for x in ["meaning", "mean", "represent", "signify", "what is"]):
        intent = "interpretation"
        needs_birth_data = False
    else:
        intent = "general"
        needs_birth_data = False

    return AstrologyIntent(
        is_astrology=True,
        intent=intent,
        zodiac_sign=sign,
        needs_birth_data=needs_birth_data,
    )


def route_astrology_question(text: str) -> dict[str, Any]:
    result = detect_astrology_intent(text)

    if not result.is_astrology:
        return {
            "route": "normal",
            "intent": "normal",
        }

    if result.intent == "horoscope":
        if result.needs_birth_data:
            return {
                "route": "personal_forecast",
                "intent": "horoscope",
                "zodiac_sign": result.zodiac_sign,
                "needs_birth_data": True,
            }

        if result.zodiac_sign:
            return {
                "route": "daily_horoscope",
                "intent": "horoscope",
                "zodiac_sign": result.zodiac_sign,
                "needs_birth_data": False,
            }

        return {
            "route": "horoscope_needs_sign",
            "intent": "horoscope",
            "needs_birth_data": False,
        }

    if result.intent == "personal":
        return {
            "route": "personal_chart",
            "intent": "personal",
            "needs_birth_data": True,
        }

    return {
        "route": "astrology_rag",
        "intent": result.intent,
        "zodiac_sign": result.zodiac_sign,
        "needs_birth_data": False,
    }
