"""
JARVIS-LIKE PERSONAL AI ASSISTANT
--------------------------------
Upgraded from the uploaded version.

Voice pipeline:
    Microphone
        -> Groq Whisper Large V3 Turbo
        -> NVIDIA Nemotron 3 Super
        -> Kokoro local TTS
        -> Speaker

Existing commands from the original project are preserved:
- YouTube / YouTube Music
- Google search
- Wikipedia
- calculator
- time
- joke
- radio
- stop music
- game
- Mass video

The important change is that normal speech is now an actual ongoing
conversation with Nemotron instead of sending every sentence through
the old speech_recognition + pyttsx3 flow.

API configuration:
    This version uses your existing config.py for both nvidia_api_key
    and groq_api_key. No .env file is required for these two services.

Install:
    pip install openai groq sounddevice soundfile numpy kokoro requests pygame

"""

import os
import time
import random
import datetime
import subprocess
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import requests
import pygame
import sounddevice as sd
import soundfile as sf
from groq import Groq
from openai import OpenAI
from kokoro import KPipeline

from config import bando_radio_api_key, nvidia_api_key, groq_api_key
from AI.AstrologyRAG.astro_rag import get_astrology_context

from AI.AstrologyRAG.astrology_router import (
    route_astrology_question
)

from AI.AstrologyRAG.chart_engine import (
    calculate_birth_chart,
    chart_to_prompt_context,
)

from AI.AstrologyRAG.horoscope import (
    build_daily_horoscope_context,
    build_personal_forecast_context,
    make_horoscope_prompt,
)
# ============================================================
# CONFIG
# ============================================================

# Use your existing config.py for BOTH API keys.
NVIDIA_API_KEY = nvidia_api_key
GROQ_API_KEY = groq_api_key

if not NVIDIA_API_KEY or not NVIDIA_API_KEY.startswith("nvapi-"):
    raise RuntimeError(
        "nvidia_api_key is missing or invalid in config.py. "
        "Replace it with a valid NVIDIA API key."
    )

if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    raise RuntimeError(
        "groq_api_key is missing or invalid in config.py. "
        "Replace it with a valid Groq API key."
    )


BRAIN_MODEL = "nvidia/nemotron-3-super-120b-a12b"
STT_MODEL = "whisper-large-v3-turbo"

SAMPLE_RATE = 16000
CHANNELS = 1
MIC_DEVICE = 1  # Confirmed working Intel microphone array``

# Voice detection
# These are intentionally forgiving so normal laptop/headset microphones
# don't get ignored because their input level is relatively quiet.
START_MULTIPLIER = 1.45
ABSOLUTE_START_THRESHOLD = 0.006
SILENCE_MULTIPLIER = 0.70
SILENCE_SECONDS = 0.85
MAX_RECORD_SECONDS = 15
MIN_SPEECH_SECONDS = 0.25

# Audio settings
CALIBRATION_SECONDS = 0.45
BLOCK_DURATION = 0.08
PRE_SPEECH_BLOCKS = 5

# Kokoro
TTS_LANGUAGE = "a"
TTS_VOICE = "am_onyx"

# Set to False if you want text output without speaking.
VOICE_ENABLED = True


# ==========================================
# ASTROLOGY USER PROFILE
# ==========================================

ASTRO_PROFILE = None


# ============================================================
# AI CLIENTS
# ============================================================

brain = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

speech = Groq(api_key=GROQ_API_KEY)


# ============================================================
# AUDIO / TTS INITIALIZATION
# ============================================================

pygame.mixer.init()

print("Loading local Kokoro voice engine...")
tts_pipeline = KPipeline(lang_code=TTS_LANGUAGE)
print("Kokoro ready.")


# ============================================================
# CONVERSATION STATE
# ============================================================

conversation_active = True

messages = [
    {
        "role": "system",
        "content": """
You are the user's personal AI voice assistant.

Your personality is inspired by a sophisticated cinematic assistant:
calm, sharp, observant, lightly witty, confident and natural.

You are NOT literally JARVIS and must never claim to be Tony Stark's
system or a fictional movie system.

Speak like a real conversational assistant:
- Usually keep spoken answers to 1-4 sentences.
- Do not produce essay-length answers unless asked.
- Do not begin every response with "Certainly".
- Use natural contractions.
- You may use subtle dry humor when appropriate.
- If the user is thinking aloud, respond conversationally.
- Ask a short follow-up if the request is genuinely ambiguous.
- Never claim that you opened, changed, deleted, searched, launched or
  controlled something unless an actual tool in this program did it.
- You can discuss actions that could be performed, but do not pretend
  they already happened.
- Remember the conversation context provided to you.
"""
    }
]


# ============================================================
# SPEAK
# ============================================================

def say(text):
    """Print and speak the assistant response using local Kokoro."""

    if not text:
        return

    text = str(text).strip()
    print(f"\nJarvis: {text}")

    if not VOICE_ENABLED:
        return

    try:
        audio_parts = []

        generator = tts_pipeline(
            text,
            voice=TTS_VOICE,
        )

        for _, _, audio in generator:
            audio_parts.append(np.asarray(audio, dtype=np.float32))

        if not audio_parts:
            print("[TTS returned no audio]")
            return

        audio = np.concatenate(audio_parts)

        sd.play(audio, 24000)
        sd.wait()

    except Exception as e:
        print(f"TTS error: {e}")


# ============================================================
# MICROPHONE
# ============================================================

def rms(audio):
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def list_microphones():
    """Print available input devices so microphone problems are easy to diagnose."""
    try:
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        for i, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                print(f"  [{i}] {device['name']}")
    except Exception as e:
        print(f"Could not list microphones: {e}")


def calibrate_microphone(seconds=CALIBRATION_SECONDS):
    """Measure the current room/microphone noise level."""
    print("Calibrating microphone...", end="", flush=True)

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=MIC_DEVICE,
    )
    sd.wait()

    level = rms(audio.flatten())

    # Don't let a very quiet calibration make the detector too sensitive.
    level = max(level, 0.002)

    print(f" ready. (noise level: {level:.4f})")
    return level


def record_turn():
    """
    Diagnostic recording mode.

    Device 1 has been confirmed to capture normal speech, so this version
    records a full turn first instead of relying on voice-activity detection.
    Once Whisper is confirmed to work, this can be changed back to automatic
    start/stop detection.
    """
    print("\n🎤 Listening... Speak now.")

    try:
        audio = sd.rec(
            int(MAX_RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=MIC_DEVICE,
        )
        sd.wait()

        level = rms(audio.flatten())
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0

        print(f"🔊 Mic level — RMS: {level:.4f}, Peak: {peak:.4f}")

        if level < 0.002:
            print("⚠️ Very little audio detected.")
            return None

        audio = audio.reshape(-1)

        # Remove only a small amount of trailing silence.
        trim = int(SAMPLE_RATE * 0.12)
        if len(audio) > trim:
            audio = audio[:-trim]

        duration = len(audio) / SAMPLE_RATE
        print(f"🎙️ Recorded {duration:.1f}s")
        return audio

    except Exception as e:
        print(f"Microphone error: {e}")
        return None


# ============================================================
# SPEECH TO TEXT
# ============================================================

def transcribe(audio):
    """Transcribe the recorded turn with Groq Whisper."""

    temp_path = None

    try:
        print("🧠 Transcribing...")

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as tmp:
            temp_path = Path(tmp.name)

        sf.write(temp_path, audio, SAMPLE_RATE)

        with open(temp_path, "rb") as audio_file:
            result = speech.audio.transcriptions.create(
                file=("turn.wav", audio_file.read()),
                model=STT_MODEL,
                response_format="text",
                temperature=0,
            )

        if isinstance(result, str):
            text = result.strip()
        else:
            text = str(result).strip()

        print(f"📝 Whisper heard: {text!r}")
        return text

    except Exception as e:
        print(f"STT error: {e}")
        return ""

    finally:
        if temp_path:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

# ============================================================
# NEMOTRON + ASTROLOGY ENGINE + RAG
# ============================================================

def ask_ai(command):

    global ASTRO_PROFILE

    # --------------------------------------------------------
    # ROUTE THE USER'S QUESTION
    # --------------------------------------------------------

    try:
        intent = route_astrology_question(command)

        print(f"\n🔀 Astrology route: {intent.route}")

    except Exception as e:
        print(f"⚠️ Astrology router error: {e}")

        # If the router fails, treat it as a normal question.
        intent = None

    # --------------------------------------------------------
    # NORMAL JARVIS QUESTION
    # --------------------------------------------------------

    if intent is None or intent.route == "normal":

        user_content = command

    # --------------------------------------------------------
    # DAILY HOROSCOPE
    # --------------------------------------------------------

    elif intent.route == "daily_horoscope":

        print("\n🔮 Daily horoscope requested.")
        print("📚 Searching astrology knowledge base...")

        try:
            book_context = get_astrology_context(
                command,
                k=5
            )
        except Exception as e:
            print(f"❌ Astrology RAG error: {e}")
            book_context = None

        try:
            astrology_context = build_daily_horoscope_context(
                intent.sign,
                retrieved_book_context=book_context,
            )

            user_content = make_horoscope_prompt(
                command,
                astrology_context,
                personal=False,
            )

        except Exception as e:
            print(f"❌ Horoscope engine error: {e}")

            user_content = f"""
The user asked for a horoscope:

{command}

Give a concise horoscope-style response based on
astrology as an interpretive tradition.

Do not present astrology as scientifically established fact.
"""

    # --------------------------------------------------------
    # HOROSCOPE BUT NO ZODIAC SIGN
    # --------------------------------------------------------

    elif intent.route == "horoscope_needs_sign":

        # Don't send this to Nemotron.
        # ask_ai() returns the question and the main loop speaks it.

        return (
            "Sure. Which zodiac sign would you like the "
            "horoscope for?"
        )

    # --------------------------------------------------------
    # PERSONAL FORECAST
    # --------------------------------------------------------

    elif intent.route == "personal_forecast":

        print("\n🔮 Personal astrology forecast requested.")

        if ASTRO_PROFILE is None:

            return (
                "I can do that. I just need your birth date, "
                "birth time, and birthplace first."
            )

        print("📚 Searching astrology knowledge base...")

        try:
            book_context = get_astrology_context(
                command,
                k=5
            )
        except Exception as e:
            print(f"❌ Astrology RAG error: {e}")
            book_context = None

        try:
            astrology_context = build_personal_forecast_context(
                ASTRO_PROFILE,
                retrieved_book_context=book_context,
            )

            user_content = make_horoscope_prompt(
                command,
                astrology_context,
                personal=True,
            )

        except Exception as e:
            print(f"❌ Personal forecast error: {e}")

            user_content = f"""
The user asked for a personal astrology forecast:

{command}

Explain that a personalized forecast requires a valid
birth chart and current planetary positions.

Treat astrology as an interpretive tradition rather than
scientifically established fact.
"""

    # --------------------------------------------------------
    # PERSONAL BIRTH CHART
    # --------------------------------------------------------

    elif intent.route == "personal_chart":

        print("\n🪐 Personal birth chart requested.")

        if ASTRO_PROFILE is None:

            return (
                "I can calculate your birth chart. "
                "I'll need your birth date, exact birth time, "
                "and birthplace first."
            )

        print("📚 Searching astrology knowledge base...")

        try:
            book_context = get_astrology_context(
                command,
                k=5
            )
        except Exception as e:
            print(f"❌ Astrology RAG error: {e}")
            book_context = None

        try:

            # ASTRO_PROFILE is expected to contain the chart
            # produced by calculate_birth_chart().
            chart_context = chart_to_prompt_context(
                ASTRO_PROFILE
            )

            user_content = f"""
The user is asking about their personal birth chart.

BIRTH CHART:
{chart_context}

ASTROLOGY BOOK CONTEXT:
{book_context or "No relevant book passages were retrieved."}

USER QUESTION:
{command}

Explain the chart clearly and conversationally.

Use the supplied birth chart as the source for planetary
positions and houses.

Use the astrology book passages when they are relevant.

Do not invent planetary positions.

Treat astrology as an interpretive tradition rather than
scientifically established fact.

Keep the answer concise enough for voice unless the user
asks for more detail.
"""

        except Exception as e:
            print(f"❌ Birth chart error: {e}")

            user_content = f"""
The user asked:

{command}

There was a problem preparing their birth chart.

Explain that the chart could not be prepared yet and that
their birth date, exact birth time, and birthplace are
needed.

Do not pretend that a chart was calculated.
"""

    # --------------------------------------------------------
    # ASTROLOGY BOOK / GENERAL ASTROLOGY
    # --------------------------------------------------------

    elif intent.route == "astrology_rag":

        print("\n🔮 Astrology question detected.")
        print("📚 Searching astrology knowledge base...")

        try:
            context = get_astrology_context(
                command,
                k=5
            )

        except Exception as e:
            print(f"❌ Astrology RAG error: {e}")
            context = None

        if context:

            print("📖 Astrology passages found.")

            user_content = f"""
The user is asking an astrology-related question.

Use the retrieved passages from the user's astrology
knowledge base as the primary source.

ASTROLOGY KNOWLEDGE BASE:
{context}

USER QUESTION:
{command}

Answer naturally and conversationally.

Important:
- Base information from the book on the supplied passages.
- Do not invent information that is not supported by them.
- If the passages are insufficient, say so.
- When useful, mention the relevant book page.
- Keep the response concise and suitable for voice.
- Treat astrology as an interpretive tradition, not as
  scientifically established fact.
"""

        else:

            print("⚠️ No relevant astrology passages found.")

            user_content = f"""
The user asked an astrology-related question:

{command}

The local astrology knowledge base did not return
relevant passages.

Answer carefully and conversationally.

Do not pretend the astrology book contains information
that was not retrieved.

Treat astrology as an interpretive tradition rather than
scientifically established fact.
"""

    # --------------------------------------------------------
    # UNKNOWN ROUTE
    # --------------------------------------------------------

    else:

        print(
            f"⚠️ Unknown astrology route: "
            f"{getattr(intent, 'route', 'unknown')}"
        )

        user_content = command

    # --------------------------------------------------------
    # ADD USER MESSAGE TO CONVERSATION
    # --------------------------------------------------------

    messages.append({
        "role": "user",
        "content": user_content,
    })

    # --------------------------------------------------------
    # ASK NEMOTRON
    # --------------------------------------------------------

    try:

        response = brain.chat.completions.create(
            model=BRAIN_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )

        answer = response.choices[0].message.content.strip()

        messages.append({
            "role": "assistant",
            "content": answer,
        })

        return answer

    except Exception:

        # Remove the failed user message so the conversation
        # history doesn't contain an unanswered turn.
        messages.pop()

        raise



# ============================================================
# ASTROLOGY ROUTER
# ============================================================

# def is_astrology_question(command):

#     text = command.lower().strip()

#     astrology_phrases = [
#         "astrology",
#         "astrological",
#         "horoscope",
#         "kundli",
#         "kundali",
#         "birth chart",
#         "natal chart",
#         "vedic astrology",
#         "jyotish",
#         "zodiac sign",
#         "sun sign",
#         "moon sign",
#         "rising sign",
#         "ascendant",
#         "lagna",
#         "nakshatra",
#         "dasha",
#         "mahadasha",
#         "antardasha",
#         "retrograde",
#         "planetary transit",
#         "planet transit",
#     ]

#     if any(phrase in text for phrase in astrology_phrases):
#         return True

#     planets = [
#         "sun",
#         "moon",
#         "mars",
#         "mercury",
#         "jupiter",
#         "venus",
#         "saturn",
#         "uranus",
#         "neptune",
#         "pluto",
#         "rahu",
#         "ketu",
#     ]

#     astrology_words = [
#         "planet",
#         "planets",
#         "house",
#         "houses",
#         "zodiac",
#         "transit",
#         "karma",
#     ]

#     question_words = [
#         "mean",
#         "means",
#         "represent",
#         "represents",
#         "significance",
#         "signify",
#         "indicate",
#         "effect",
#         "effects",
#         "in my chart",
#         "in a chart",
#     ]

#     has_planet = any(word in text for word in planets)
#     has_astrology_word = any(word in text for word in astrology_words)
#     has_question_word = any(word in text for word in question_words)

#     if (has_planet or has_astrology_word) and has_question_word:
#         return True

#     return False


# ============================================================
# NEMOTRON + ASTROLOGY RAG
# ============================================================

# def ask_ai(command):

#     # --------------------------------------------------------
#     # CHECK FOR ASTROLOGY
#     # --------------------------------------------------------

#     astro_question = is_astrology_question(command)

#     if astro_question:

#         print("\n🔮 Astrology question detected.")
#         print("📚 Searching astrology knowledge base...")

#         try:
#             context = get_astrology_context(
#                 command,
#                 k=5
#             )

#         except Exception as e:
#             print(f"❌ Astrology RAG error: {e}")
#             context = None

#         if context:

#             print("📖 Astrology passages found.")

#             user_content = f"""
# The user is asking an astrology-related question.

# Use the retrieved passages from the user's astrology
# knowledge base as the primary source.

# ASTROLOGY KNOWLEDGE BASE:
# {context}

# USER QUESTION:
# {command}

# Answer naturally and conversationally.

# Important:
# - Base information from the book on the supplied passages.
# - Do not invent information that is not supported by them.
# - If the passages are insufficient, say so.
# - When useful, mention the book and page.
# - Keep the response concise and suitable for voice.
# - Treat astrology as an interpretive tradition, not as
#   scientifically established fact.
# """

#         else:

#             print("⚠️ No relevant astrology passages found.")

#             user_content = f"""
# The user asked an astrology-related question:

# {command}

# The local astrology knowledge base did not return
# relevant passages.

# Answer carefully and conversationally. Do not pretend
# the book contains information that was not retrieved.

# Treat astrology as an interpretive tradition rather
# than scientifically established fact.
# """

#     else:

#         # Normal JARVIS conversation
#         user_content = command

#     # --------------------------------------------------------
#     # ADD USER MESSAGE
#     # --------------------------------------------------------

#     messages.append({
#         "role": "user",
#         "content": user_content,
#     })

#     # --------------------------------------------------------
#     # ASK NEMOTRON
#     # --------------------------------------------------------

#     try:

#         response = brain.chat.completions.create(
#             model=BRAIN_MODEL,
#             messages=messages,
#             temperature=0.7,
#             max_tokens=500,
#         )

#         answer = response.choices[0].message.content.strip()

#         messages.append({
#             "role": "assistant",
#             "content": answer,
#         })

#         return answer

#     except Exception:

#         messages.pop()
#         raise


# ============================================================
# NEMOTRON CONVERSATION
# ============================================================

# def ask_ai(command):
#     """
#     Send the current turn plus previous conversation to Nemotron.

#     Unlike the old version, the AI gets actual conversation history.
#     """

#     messages.append({
#         "role": "user",
#         "content": command,
#     })

#     try:
#         response = brain.chat.completions.create(
#             model=BRAIN_MODEL,
#             messages=messages,
#             temperature=0.7,
#             max_tokens=500,
#         )

#         answer = response.choices[0].message.content.strip()

#         messages.append({
#             "role": "assistant",
#             "content": answer,
#         })

#         return answer

#     except Exception:
#         # Don't leave a failed user turn in the context as if it had
#         # been answered.
#         messages.pop()
#         raise


# ============================================================
# GOOGLE
# ============================================================

def search_google(query):
    if not query:
        say("What should I search for?")
        return

    say(f"Searching Google for {query}.")

    url = (
        "https://www.google.com/search?q="
        + requests.utils.quote(query)
    )

    webbrowser.open(url)


# ============================================================
# YOUTUBE MUSIC
# ============================================================

def open_youtube_music():
    say("Opening YouTube Music.")
    webbrowser.open("https://music.youtube.com")


def search_youtube_music(query):
    if not query:
        say("What would you like me to search for?")
        return

    say(f"Searching YouTube Music for {query}.")

    url = (
        "https://music.youtube.com/search?q="
        + requests.utils.quote(query)
    )

    webbrowser.open(url)


# ============================================================
# YOUTUBE
# ============================================================

def open_youtube():
    say("Opening YouTube.")
    webbrowser.open("https://www.youtube.com")


# ============================================================
# WIKIPEDIA
# ============================================================

def open_wikipedia():
    say("Opening Wikipedia.")
    webbrowser.open("https://www.wikipedia.org")


# ============================================================
# GAME
# ============================================================

def play_game():
    try:
        game_path = (
            r"C:\Users\Akshat\Desktop"
            r"\dont-touch-my-presents-main\main.py"
        )

        if not os.path.exists(game_path):
            say("I couldn't find the game at the configured path.")
            return

        say("Launching the game.")

        subprocess.Popen(["python", game_path])

    except Exception as e:
        print(f"Game error: {e}")
        say("I couldn't launch the game.")


# ============================================================
# CALCULATOR
# ============================================================

def open_calculator():
    say("Opening the calculator.")
    os.system("calc")


# ============================================================
# TIME
# ============================================================

def tell_time():
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    say(f"It's {current_time}.")


# ============================================================
# JOKE
# ============================================================

def get_chuck_norris_joke():
    url = (
        "https://matchilling-chuck-norris-jokes-v1."
        "p.rapidapi.com/jokes/random"
    )

    headers = {
        "accept": "application/json",
        "X-RapidAPI-Key": bando_radio_api_key,
        "X-RapidAPI-Host":
            "matchilling-chuck-norris-jokes-v1.p.rapidapi.com",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=10,
        )

        response.raise_for_status()

        return response.json().get(
            "content",
            "I couldn't find a joke."
        )

    except Exception as e:
        print(f"Joke error: {e}")
        return "Sorry, I couldn't get a joke right now."


def tell_joke():
    say(get_chuck_norris_joke())


# ============================================================
# RADIO
# ============================================================

def play_radio():
    url = (
        "https://bando-radio-api.p.rapidapi.com/"
        "stations/bycountry/Austria"
    )

    params = {
        "hidebroken": "true",
        "offset": "0",
        "limit": "10",
    }

    headers = {
        "X-RapidAPI-Key": bando_radio_api_key,
        "X-RapidAPI-Host":
            "bando-radio-api.p.rapidapi.com",
    }

    try:
        say("Finding a radio station.")

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        stations = (
            response.json()
            .get("data", {})
            .get("stations", [])
        )

        if not stations:
            say("I couldn't find a radio station.")
            return

        station = random.choice(stations)

        stream_url = station.get("stream_url", "")

        if not stream_url:
            say("I couldn't find a working stream.")
            return

        pygame.mixer.music.load(stream_url)
        pygame.mixer.music.play()

        name = station.get("name", "unknown station")

        say(f"Now playing {name}.")

    except Exception as e:
        print(f"Radio error: {e}")
        say("I couldn't start the radio.")


# ============================================================
# STOP MUSIC
# ============================================================

def stop_music():
    pygame.mixer.music.stop()
    say("Music stopped.")


# ============================================================
# MASS VIDEO
# ============================================================

def play_mass():
    say("Playing Mass.")
    webbrowser.open("https://youtu.be/X6PLAysaevA")


# ============================================================
# DIRECT COMMANDS
# ============================================================

def handle_command(command):
    """
    Handle deterministic commands first.

    Returning True means the command was handled locally and should
    NOT be sent to the LLM.
    """

    global conversation_active

    normalized = " ".join(command.lower().strip().split())

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    exit_phrases = {
        "exit",
        "quit",
        "goodbye",
        "go to sleep",
        "shutdown jarvis",
        "stop listening",
    }

    if normalized in exit_phrases:
        say("Understood. Going quiet. I'll be here when you need me.")
        conversation_active = False
        return True

    # --------------------------------------------------------
    # YOUTUBE MUSIC
    # --------------------------------------------------------

    if (
        "open youtube music" in normalized
        or "open music player" in normalized
    ):
        open_youtube_music()
        return True

    if "search youtube music" in normalized:
        query = normalized.replace(
            "search youtube music",
            "",
        ).strip()

        search_youtube_music(query)
        return True

    # --------------------------------------------------------
    # YOUTUBE
    # --------------------------------------------------------

    if normalized == "open youtube":
        open_youtube()
        return True

    # --------------------------------------------------------
    # WIKIPEDIA
    # --------------------------------------------------------

    if normalized == "open wikipedia":
        open_wikipedia()
        return True

    # --------------------------------------------------------
    # GOOGLE
    # --------------------------------------------------------

    if "search google for" in normalized:
        query = normalized.replace(
            "search google for",
            "",
        ).strip()

        search_google(query)
        return True

    if "search on google" in normalized:
        query = normalized.replace(
            "search on google",
            "",
        ).strip()

        search_google(query)
        return True

    # --------------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------------

    if (
        normalized == "open calculator"
        or normalized == "open the calculator"
    ):
        open_calculator()
        return True

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if normalized in {
        "what time is it",
        "what's the time",
        "tell me the time",
    }:
        tell_time()
        return True

    # --------------------------------------------------------
    # JOKE
    # --------------------------------------------------------

    if (
        normalized == "tell me a joke"
        or normalized == "make me laugh"
    ):
        tell_joke()
        return True

    # --------------------------------------------------------
    # GAME
    # --------------------------------------------------------

    if (
        normalized == "play a game"
        or normalized == "launch my game"
    ):
        play_game()
        return True

    # --------------------------------------------------------
    # RADIO
    # --------------------------------------------------------

    if (
        normalized == "play radio"
        or normalized == "start radio"
    ):
        play_radio()
        return True

    # --------------------------------------------------------
    # STOP MUSIC
    # --------------------------------------------------------

    if (
        normalized == "stop music"
        or normalized == "stop the music"
    ):
        stop_music()
        return True

    # --------------------------------------------------------
    # MASS
    # --------------------------------------------------------

    if normalized == "play mass":
        play_mass()
        return True

    return False


# ============================================================
# MAIN CONVERSATION LOOP
# ============================================================

def get_user_input():
    """
    Choose how to provide the next message:
    V = microphone / voice
    T = keyboard / text
    Q = quit
    """
    while conversation_active:
        try:
            choice = input("\n[V] Voice  [T] Text  [Q] Quit: ").strip().lower()

            if choice in {"q", "quit", "exit"}:
                return None

            if choice in {"t", "text"}:
                command = input("⌨️ You: ").strip()
                if command:
                    return command
                print("Please type something.")
                continue

            if choice in {"v", "voice", ""}:
                audio = record_turn()

                if audio is None:
                    continue

                command = transcribe(audio)

                if not command:
                    print("I couldn't make that out.")
                    continue

                return command

            print("Please choose V for voice, T for text, or Q to quit.")

        except KeyboardInterrupt:
            return None
        except EOFError:
            return None


def process_command(command):
    """Run one user command through local commands or the AI."""
    global conversation_active

    if not command:
        return

    print(f"\nYou: {command}")

    handled = handle_command(command)

    if handled:
        time.sleep(0.25)
        return

    try:
        response = ask_ai(command)

        if response:
            say(response)

    except Exception as e:
        print(f"AI error: {e}")
        say(
            "I hit a problem reaching my reasoning core. "
            "Give me another moment."
        )


def main():

    global conversation_active

    print()
    print("=" * 58)
    print("                 JARVIS A.I.")
    print("=" * 58)
    print("Voice : Kokoro")
    print("STT   : Whisper Large V3 Turbo")
    print("Brain : Nemotron 3 Super")
    print("Input : Voice + Text")
    print()
    print("You can choose Voice or Text for every message.")
    print("V = microphone")
    print("T = keyboard")
    print("Q = quit")
    print("=" * 58)

    try:
        mic_info = sd.query_devices(MIC_DEVICE)
        print(
            f"🎙️ Input microphone: {mic_info['name']} "
            f"(device {MIC_DEVICE})"
        )
    except Exception as e:
        print(f"⚠️ Could not identify the default microphone: {e}")
        list_microphones()

    say(
        "Hello. I'm online. "
        "You can speak to me or type your message."
    )

    while conversation_active:

        try:
            command = get_user_input()

            if command is None:
                conversation_active = False
                break

            process_command(command)

            # Prevent the microphone from immediately recording
            # the final sound coming from the speakers.
            time.sleep(0.35)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt.")
            conversation_active = False

        except Exception as e:
            print(f"\nMain loop error: {e}")
            time.sleep(1)

    print("\nJarvis offline.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
