"""
Astro JARVIS - natal chart engine.

Uses Swiss Ephemeris for planetary positions.
The calculations are astronomical coordinates; the interpretive meanings
belong to the selected astrology tradition.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

try:
    import swisseph as swe
except ImportError as exc:
    raise ImportError(
        "Swiss Ephemeris is required. Install it with: pip install pysweph"
    ) from exc


SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

SIGN_LORDS = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.MEAN_NODE,
}

DASHA_LORDS = [
    ("Ketu", 7),
    ("Venus", 20),
    ("Sun", 6),
    ("Moon", 10),
    ("Mars", 7),
    ("Rahu", 18),
    ("Jupiter", 16),
    ("Saturn", 19),
    ("Mercury", 17),
]

NAKSHATRAS = [
    ("Ashwini", "Ketu"),
    ("Bharani", "Venus"),
    ("Krittika", "Sun"),
    ("Rohini", "Moon"),
    ("Mrigashira", "Mars"),
    ("Ardra", "Rahu"),
    ("Punarvasu", "Jupiter"),
    ("Pushya", "Saturn"),
    ("Ashlesha", "Mercury"),
    ("Magha", "Ketu"),
    ("Purva Phalguni", "Venus"),
    ("Uttara Phalguni", "Sun"),
    ("Hasta", "Moon"),
    ("Chitra", "Mars"),
    ("Swati", "Rahu"),
    ("Vishakha", "Jupiter"),
    ("Anuradha", "Saturn"),
    ("Jyeshtha", "Mercury"),
    ("Mula", "Ketu"),
    ("Purva Ashadha", "Venus"),
    ("Uttara Ashadha", "Sun"),
    ("Shravana", "Moon"),
    ("Dhanishta", "Mars"),
    ("Shatabhisha", "Rahu"),
    ("Purva Bhadrapada", "Jupiter"),
    ("Uttara Bhadrapada", "Saturn"),
    ("Revati", "Mercury"),
]

DASHA_YEARS = dict(DASHA_LORDS)


@dataclass
class BirthData:
    birth_datetime_utc: datetime
    latitude: float
    longitude: float
    place_name: str = ""


def normalize_degree(value: float) -> float:
    return value % 360.0


def sign_from_longitude(longitude: float) -> tuple[str, int, float]:
    longitude = normalize_degree(longitude)
    sign_index = int(longitude // 30)
    degree = longitude - sign_index * 30
    return SIGNS[sign_index], sign_index, degree


def format_position(longitude: float) -> str:
    sign, _, degree = sign_from_longitude(longitude)
    whole = int(degree)
    minutes = int(round((degree - whole) * 60))
    if minutes == 60:
        whole += 1
        minutes = 0
    return f"{whole}°{minutes:02d}' {sign}"


def _julian_day(dt_utc: datetime) -> float:
    if dt_utc.tzinfo is None:
        raise ValueError("birth_datetime_utc must be timezone-aware")
    dt_utc = dt_utc.astimezone(timezone.utc)
    hour = (
        dt_utc.hour
        + dt_utc.minute / 60
        + dt_utc.second / 3600
        + dt_utc.microsecond / 3_600_000_000
    )
    return swe.julday(
        dt_utc.year, dt_utc.month, dt_utc.day, hour, swe.GREG_CAL
    )


def _planet_position(jd: float, body: int, sidereal: bool = True) -> dict[str, Any]:
    flags = swe.FLG_SWIEPH | swe.FLG_SPEED
    if sidereal:
        swe.set_sid_mode(swe.SIDM_LAHIRI)
        flags |= swe.FLG_SIDEREAL

    values, _ = swe.calc_ut(jd, body, flags)
    longitude = normalize_degree(values[0])
    sign, sign_index, degree = sign_from_longitude(longitude)

    return {
        "longitude": longitude,
        "sign": sign,
        "sign_index": sign_index,
        "degree": degree,
        "formatted": format_position(longitude),
        "retrograde": values[3] < 0,
        "speed_longitude": values[3],
    }


def _nakshatra(moon_longitude: float) -> dict[str, Any]:
    segment = 360.0 / 27.0
    index = int(normalize_degree(moon_longitude) // segment)
    within = normalize_degree(moon_longitude) - index * segment
    pada = int(within // (segment / 4.0)) + 1
    name, lord = NAKSHATRAS[index]
    fraction_elapsed = within / segment
    return {
        "name": name,
        "index": index,
        "pada": pada,
        "lord": lord,
        "fraction_elapsed": fraction_elapsed,
    }


def _vimshottari_dasha(
    moon_longitude: float,
    birth_dt: datetime,
    years_to_show: int = 3,
) -> dict[str, Any]:
    nak = _nakshatra(moon_longitude)
    start_lord = nak["lord"]
    lord_names = [x[0] for x in DASHA_LORDS]
    start_index = lord_names.index(start_lord)

    # Remaining fraction of the Moon's nakshatra at birth.
    remaining_fraction = 1.0 - nak["fraction_elapsed"]
    first_years = DASHA_YEARS[start_lord] * remaining_fraction

    events = []
    current = birth_dt.astimezone(timezone.utc)

    def add_period(lord: str, start: datetime, years: float):
        end = start.timestamp() + years * 365.2425 * 86400
        end_dt = datetime.fromtimestamp(end, tz=timezone.utc)
        events.append({
            "lord": lord,
            "start": start.isoformat(),
            "end": end_dt.isoformat(),
            "years": years,
        })
        return end_dt

    # Build enough Mahadasha periods to cover the requested horizon.
    cursor = current
    idx = start_index
    cursor = add_period(start_lord, cursor, first_years)
    idx = (idx + 1) % len(lord_names)

    target = current.timestamp() + years_to_show * 365.2425 * 86400
    while cursor.timestamp() < target:
        lord = lord_names[idx]
        cursor = add_period(lord, cursor, DASHA_YEARS[lord])
        idx = (idx + 1) % len(lord_names)

    active = None
    now = datetime.now(timezone.utc)
    for event in events:
        start = datetime.fromisoformat(event["start"])
        end = datetime.fromisoformat(event["end"])
        if start <= now < end:
            active = event
            break

    return {
        "birth_nakshatra": nak,
        "active_mahadasha": active,
        "periods": events,
    }


def calculate_birth_chart(
    birth_datetime_utc: datetime,
    latitude: float,
    longitude: float,
    place_name: str = "",
) -> dict[str, Any]:
    """
    Calculate a sidereal/Lahiri Vedic-style natal chart.

    Houses are represented as whole-sign houses, which avoids mixing
    house-system conventions with the Vedic sign-based interpretation.
    """
    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180")

    jd = _julian_day(birth_datetime_utc)

    # Sidereal Ascendant using Lahiri.
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
    cusps, ascmc = swe.houses_ex(
        jd,
        flags,
        latitude,
        longitude,
        b"W",
    )
    asc_longitude = normalize_degree(ascmc[0])

    asc_sign, asc_index, asc_degree = sign_from_longitude(asc_longitude)

    planets = {}
    for name, body in PLANETS.items():
        planets[name] = _planet_position(jd, body, sidereal=True)

    # Ketu is opposite Rahu.
    rahu_longitude = planets["Rahu"]["longitude"]
    ketu_longitude = normalize_degree(rahu_longitude + 180.0)
    ketu_sign, ketu_index, ketu_degree = sign_from_longitude(ketu_longitude)
    planets["Ketu"] = {
        "longitude": ketu_longitude,
        "sign": ketu_sign,
        "sign_index": ketu_index,
        "degree": ketu_degree,
        "formatted": format_position(ketu_longitude),
        "retrograde": planets["Rahu"]["retrograde"],
        "speed_longitude": planets["Rahu"]["speed_longitude"],
    }

    for data in planets.values():
        data["house"] = ((data["sign_index"] - asc_index) % 12) + 1

    nak = _nakshatra(planets["Moon"]["longitude"])
    dashas = _vimshottari_dasha(planets["Moon"]["longitude"], birth_datetime_utc)

    return {
        "system": "Sidereal Vedic-style, Lahiri ayanamsha, whole-sign houses",
        "birth": {
            "datetime_utc": birth_datetime_utc.astimezone(timezone.utc).isoformat(),
            "latitude": latitude,
            "longitude": longitude,
            "place_name": place_name,
        },
        "ascendant": {
            "longitude": asc_longitude,
            "sign": asc_sign,
            "sign_index": asc_index,
            "degree": asc_degree,
            "formatted": format_position(asc_longitude),
        },
        "planets": planets,
        "moon_nakshatra": nak,
        "vimshottari": dashas,
    }


def chart_to_prompt_context(chart: dict[str, Any]) -> str:
    lines = [
        f"Astrology system: {chart['system']}",
        f"Birth place: {chart['birth'].get('place_name') or 'Not specified'}",
        f"Birth UTC: {chart['birth']['datetime_utc']}",
        f"Ascendant: {chart['ascendant']['formatted']}",
        (
            f"Moon nakshatra: {chart['moon_nakshatra']['name']} "
            f"(Pada {chart['moon_nakshatra']['pada']}), "
            f"lord {chart['moon_nakshatra']['lord']}"
        ),
    ]

    for name, p in chart["planets"].items():
        retro = " retrograde" if p.get("retrograde") else ""
        lines.append(
            f"{name}: {p['formatted']}, House {p['house']}{retro}"
        )

    active = chart["vimshottari"].get("active_mahadasha")
    if active:
        lines.append(
            f"Active Vimshottari Mahadasha: {active['lord']} "
            f"from {active['start']} to {active['end']}"
        )

    return "\n".join(lines)
