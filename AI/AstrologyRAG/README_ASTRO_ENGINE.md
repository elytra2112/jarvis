# Astro JARVIS astrology engine

Files:
- chart_engine.py     Natal chart + Lahiri sidereal placements + whole-sign houses + Vimshottari Mahadasha
- transits.py         Current sidereal transits + simple transit-to-natal contacts
- horoscope.py        Daily/personal forecast context and Nemotron prompt builder
- astrology_router.py Astrology intent detection/routing

## Install

Use the same Python 3.12 executable as your JARVIS:

    python -m pip install pysweph

The package imports as:

    import swisseph as swe

## Important

The chart engine expects birth time in UTC plus latitude/longitude.
Your voice/UI layer should convert local birth time to UTC before calling
calculate_birth_chart().

The implementation uses:
- sidereal zodiac
- Lahiri ayanamsha
- whole-sign houses
- mean lunar node for Rahu
- Ketu exactly opposite Rahu
- Vimshottari Mahadasha

These are explicit conventions so the engine is deterministic. If your book
uses a different ayanamsha, node convention, or house system, we can change
them.

## Basic usage

    from datetime import datetime, timezone
    from chart_engine import calculate_birth_chart

    chart = calculate_birth_chart(
        datetime(2005, 1, 1, 12, 0, tzinfo=timezone.utc),
        latitude=23.0225,
        longitude=72.5714,
        place_name="Ahmedabad, India",
    )

    print(chart["ascendant"])
    print(chart["planets"])
    print(chart["vimshottari"]["active_mahadasha"])

For package imports from AI/AstrologyRAG, use:

    from AI.AstrologyRAG.chart_engine import calculate_birth_chart
