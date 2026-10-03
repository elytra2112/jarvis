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

# Voice detection
START_MULTIPLIER = 2.4
SILENCE_SECONDS = 1.0
MAX_RECORD_SECONDS = 20
MIN_SPEECH_SECONDS = 0.35

# Kokoro
TTS_LANGUAGE = "a"
TTS_VOICE = "af_heart"

# Set to False if you want text output without speaking.
VOICE_ENABLED = True


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


def calibrate_microphone(seconds=0.7):
    """Measure room noise so the assistant can detect speech."""

    print("Calibrating microphone...", end="", flush=True)

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()

    level = max(rms(audio.flatten()), 0.002)

    print(" ready.")
    return level


def record_turn():
    """
    Wait for the user to speak, record the turn, then stop after silence.

    This replaces the old speech_recognition.listen() loop and gives
    the assistant a much more natural conversation cycle.
    """

    ambient = calibrate_microphone()
    start_threshold = max(0.008, ambient * START_MULTIPLIER)

    block_duration = 0.10
    block_size = int(SAMPLE_RATE * block_duration)

    chunks = []
    speaking = False
    silence_time = 0.0
    total_time = 0.0
    speech_time = 0.0

    print("\n🎤 Listening...")

    while total_time < MAX_RECORD_SECONDS:

        block = sd.rec(
            block_size,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()

        block = block.flatten()
        level = rms(block)

        total_time += block_duration

        # Wait for speech to begin.
        if not speaking:
            if level >= start_threshold:
                speaking = True
                chunks.append(block)
                speech_time += block_duration
                print("You: ", end="", flush=True)
            continue

        chunks.append(block)

        if level >= start_threshold * 0.65:
            speech_time += block_duration
            silence_time = 0.0
        else:
            silence_time += block_duration

        if silence_time >= SILENCE_SECONDS:
            break

    if not chunks or speech_time < MIN_SPEECH_SECONDS:
        return None

    audio = np.concatenate(chunks)

    # Remove a small amount of trailing silence.
    trim = int(SAMPLE_RATE * 0.15)
    if len(audio) > trim:
        audio = audio[:-trim]

    return audio


# ============================================================
# SPEECH TO TEXT
# ============================================================

def transcribe(audio):
    """Transcribe the recorded turn with Groq Whisper."""

    temp_path = None

    try:
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
            return result.strip()

        return str(result).strip()

    finally:
        if temp_path:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


# ============================================================
# NEMOTRON CONVERSATION
# ============================================================

def ask_ai(command):
    """
    Send the current turn plus previous conversation to Nemotron.

    Unlike the old version, the AI gets actual conversation history.
    """

    messages.append({
        "role": "user",
        "content": command,
    })

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
        # Don't leave a failed user turn in the context as if it had
        # been answered.
        messages.pop()
        raise


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

def main():

    global conversation_active

    print()
    print("=" * 58)
    print("                 JARVIS A.I.")
    print("=" * 58)
    print("Voice : Kokoro")
    print("STT   : Whisper Large V3 Turbo")
    print("Brain : Nemotron 3 Super")
    print("Mode  : Continuous conversation")
    print()
    print("Talk naturally.")
    print("Say 'go to sleep' or 'goodbye' to exit.")
    print("=" * 58)

    say(
        "Hello. I'm online. "
        "We can talk normally. What is on your mind?"
    )

    while conversation_active:

        try:
            audio = record_turn()

            if audio is None:
                continue

            command = transcribe(audio)

            if not command:
                print("I couldn't make that out.")
                continue

            print(f"\nYou: {command}")

            # First try real deterministic tools.
            handled = handle_command(command)

            if handled:
                time.sleep(0.25)
                continue

            # Everything else becomes an actual conversation.
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










# import speech_recognition as sr
# import os
# import webbrowser
# import requests
# import pyttsx3
# import pygame
# import datetime
# import random
# import subprocess
# import time

# from config import bando_radio_api_key
# from brain import ask_ai


# # ============================================================
# # INITIALIZATION
# # ============================================================

# pygame.mixer.init()

# engine = pyttsx3.init()

# # Voice settings
# engine.setProperty("rate", 175)
# engine.setProperty("volume", 1.0)

# recognizer = sr.Recognizer()

# # How long Jarvis waits for you to start speaking
# recognizer.pause_threshold = 0.8

# # Small amount of ambient noise adjustment
# recognizer.energy_threshold = 300

# # Conversation state
# conversation_active = True


# # ============================================================
# # SPEAK
# # ============================================================

# def say(text):
#     """Speak and print Jarvis's response."""

#     if not text:
#         return

#     print(f"\nJarvis: {text}")

#     engine.say(text)
#     engine.runAndWait()


# # ============================================================
# # LISTEN
# # ============================================================

# def listen():
#     """
#     Listen to the microphone and return what the user said.
#     Returns None if nothing understandable was heard.
#     """

#     with sr.Microphone() as source:

#         print("\n🎤 Listening...")

#         try:
#             audio = recognizer.listen(
#                 source,
#                 timeout=5,
#                 phrase_time_limit=15
#             )

#         except sr.WaitTimeoutError:
#             return None

#     try:

#         print("🧠 Understanding...")

#         command = recognizer.recognize_google(
#             audio,
#             language="en-in"
#         )

#         command = command.lower().strip()

#         print(f"You: {command}")

#         return command

#     except sr.UnknownValueError:

#         print("I couldn't understand that.")

#         return None

#     except sr.RequestError as e:

#         print(f"Speech recognition error: {e}")

#         say("I'm having trouble connecting to speech recognition.")

#         return None

#     except Exception as e:

#         print(f"Listening error: {e}")

#         return None


# # ============================================================
# # GOOGLE SEARCH
# # ============================================================

# def search_google(query):

#     if not query:
#         say("What should I search for?")
#         return

#     say(f"Searching Google for {query}.")

#     url = (
#         "https://www.google.com/search?q="
#         + requests.utils.quote(query)
#     )

#     webbrowser.open(url)


# # ============================================================
# # YOUTUBE MUSIC
# # ============================================================

# def open_youtube_music():

#     say("Opening YouTube Music.")

#     webbrowser.open(
#         "https://music.youtube.com"
#     )


# def search_youtube_music(query):

#     if not query:
#         say("What would you like me to search for?")
#         return

#     say(f"Searching YouTube Music for {query}.")

#     url = (
#         "https://music.youtube.com/search?q="
#         + requests.utils.quote(query)
#     )

#     webbrowser.open(url)


# # ============================================================
# # GAME
# # ============================================================

# def play_game():

#     try:

#         game_path = (
#             r"C:\Users\Akshat\Desktop"
#             r"\dont-touch-my-presents-main\main.py"
#         )

#         say("Launching the game.")

#         subprocess.Popen(
#             ["python", game_path]
#         )

#     except Exception as e:

#         print(f"Game error: {e}")

#         say("I couldn't launch the game.")


# # ============================================================
# # CALCULATOR
# # ============================================================

# def open_calculator():

#     say("Opening the calculator.")

#     os.system("calc")


# # ============================================================
# # TIME
# # ============================================================

# def tell_time():

#     current_time = datetime.datetime.now().strftime(
#         "%I:%M %p"
#     )

#     say(f"It's {current_time}.")


# # ============================================================
# # JOKE
# # ============================================================

# def get_chuck_norris_joke():

#     url = (
#         "https://matchilling-chuck-norris-jokes-v1."
#         "p.rapidapi.com/jokes/random"
#     )

#     headers = {
#         "accept": "application/json",
#         "X-RapidAPI-Key": bando_radio_api_key,
#         "X-RapidAPI-Host":
#             "matchilling-chuck-norris-jokes-v1.p.rapidapi.com"
#     }

#     try:

#         response = requests.get(
#             url,
#             headers=headers,
#             timeout=10
#         )

#         response.raise_for_status()

#         return response.json().get(
#             "content",
#             "I couldn't find a joke."
#         )

#     except Exception as e:

#         print(f"Joke error: {e}")

#         return "Sorry, I couldn't get a joke right now."


# def tell_joke():

#     say(get_chuck_norris_joke())


# # ============================================================
# # RADIO
# # ============================================================

# def play_radio():

#     url = (
#         "https://bando-radio-api.p.rapidapi.com/"
#         "stations/bycountry/Austria"
#     )

#     params = {
#         "hidebroken": "true",
#         "offset": "0",
#         "limit": "10"
#     }

#     headers = {
#         "X-RapidAPI-Key": bando_radio_api_key,
#         "X-RapidAPI-Host":
#             "bando-radio-api.p.rapidapi.com"
#     }

#     try:

#         say("Finding a radio station.")

#         response = requests.get(
#             url,
#             headers=headers,
#             params=params,
#             timeout=10
#         )

#         response.raise_for_status()

#         stations = response.json().get(
#             "data", {}
#         ).get(
#             "stations", []
#         )

#         if not stations:

#             say("I couldn't find a radio station.")
#             return

#         station = random.choice(stations)

#         stream_url = station.get(
#             "stream_url",
#             ""
#         )

#         if not stream_url:

#             say("I couldn't find a working stream.")
#             return

#         pygame.mixer.music.load(stream_url)
#         pygame.mixer.music.play()

#         name = station.get(
#             "name",
#             "unknown station"
#         )

#         say(f"Now playing {name}.")

#     except Exception as e:

#         print(f"Radio error: {e}")

#         say("I couldn't start the radio.")


# # ============================================================
# # STOP MUSIC
# # ============================================================

# def stop_music():

#     pygame.mixer.music.stop()

#     say("Music stopped.")


# # ============================================================
# # DIRECT COMMANDS
# # ============================================================

# def handle_command(command):

#     global conversation_active

#     # --------------------------------------------------------
#     # EXIT
#     # --------------------------------------------------------

#     if command in [
#         "exit",
#         "quit",
#         "goodbye",
#         "go to sleep",
#         "shutdown jarvis"
#     ]:

#         say("Alright. I'll be here when you need me.")

#         conversation_active = False

#         return True


#     # --------------------------------------------------------
#     # YOUTUBE MUSIC
#     # --------------------------------------------------------

#     if (
#         "open youtube music" in command
#         or "open music player" in command
#     ):

#         open_youtube_music()

#         return True


#     if "search youtube music" in command:

#         query = command.replace(
#             "search youtube music",
#             ""
#         ).strip()

#         search_youtube_music(query)

#         return True


#     # --------------------------------------------------------
#     # YOUTUBE
#     # --------------------------------------------------------

#     if "open youtube" in command:

#         say("Opening YouTube.")

#         webbrowser.open(
#             "https://www.youtube.com"
#         )

#         return True


#     # --------------------------------------------------------
#     # WIKIPEDIA
#     # --------------------------------------------------------

#     if "open wikipedia" in command:

#         say("Opening Wikipedia.")

#         webbrowser.open(
#             "https://www.wikipedia.org"
#         )

#         return True


#     # --------------------------------------------------------
#     # GOOGLE SEARCH
#     # --------------------------------------------------------

#     if "search google for" in command:

#         query = command.replace(
#             "search google for",
#             ""
#         ).strip()

#         search_google(query)

#         return True


#     if "search on google" in command:

#         query = command.replace(
#             "search on google",
#             ""
#         ).strip()

#         search_google(query)

#         return True


#     # --------------------------------------------------------
#     # CALCULATOR
#     # --------------------------------------------------------

#     if (
#         "open calculator" in command
#         or "open the calculator" in command
#     ):

#         open_calculator()

#         return True


#     # --------------------------------------------------------
#     # TIME
#     # --------------------------------------------------------

#     if (
#         "what time is it" in command
#         or "what's the time" in command
#         or "tell me the time" in command
#     ):

#         tell_time()

#         return True


#     # --------------------------------------------------------
#     # JOKE
#     # --------------------------------------------------------

#     if (
#         "tell me a joke" in command
#         or "make me laugh" in command
#     ):

#         tell_joke()

#         return True


#     # --------------------------------------------------------
#     # GAME
#     # --------------------------------------------------------

#     if (
#         "play a game" in command
#         or "launch my game" in command
#     ):

#         play_game()

#         return True


#     # --------------------------------------------------------
#     # RADIO
#     # --------------------------------------------------------

#     if (
#         "play radio" in command
#         or "start radio" in command
#     ):

#         play_radio()

#         return True


#     # --------------------------------------------------------
#     # STOP MUSIC
#     # --------------------------------------------------------

#     if (
#         "stop music" in command
#         or "stop the music" in command
#     ):

#         stop_music()

#         return True


#     # --------------------------------------------------------
#     # MASS VIDEO
#     # --------------------------------------------------------

#     if "play mass" in command:

#         say("Playing Mass.")

#         webbrowser.open(
#             "https://youtu.be/X6PLAysaevA"
#         )

#         return True


#     # --------------------------------------------------------
#     # NOT A COMMAND
#     # --------------------------------------------------------

#     return False


# # ============================================================
# # MAIN CONVERSATION LOOP
# # ============================================================

# def main():

#     global conversation_active

#     print()
#     print("==========================================")
#     print("              JARVIS A.I")
#     print("==========================================")
#     print("Natural conversation mode")
#     print("Say 'goodbye' to stop.")
#     print("==========================================")

#     say(
#         "Hello Akshat. I'm online. "
#         "What would you like to talk about?"
#     )

#     while conversation_active:

#         command = listen()

#         # Nothing heard
#         if not command:

#             continue

#         # Handle direct computer commands
#         handled = handle_command(command)

#         if handled:

#             continue

#         # ----------------------------------------------------
#         # NORMAL HUMAN-LIKE CONVERSATION
#         # ----------------------------------------------------

#         try:

#             response = ask_ai(command)

#             if response:

#                 say(response)

#         except Exception as e:

#             print(f"AI error: {e}")

#             say(
#                 "Sorry, I had a problem thinking "
#                 "about that."
#             )

#     print("\nJarvis offline.")


# # ============================================================
# # START
# # ============================================================

# if __name__ == "__main__":

#     main()









# # import speech_recognition as sr
# # import os
# # import webbrowser
# # import requests
# # import pyttsx3
# # import pygame
# # import datetime
# # import random
# # import subprocess
# # from config import bando_radio_api_key

# # chatStr = ""
# # pygame.mixer.init()

# # def execute_command(command):
# #     global chatStr
# #     print(chatStr)
# #     chatStr += f"Akshat: {command}\n Jarvis: "

# #     if "play music" in command:
# #         play_music_from_bando_radio()

# #     elif "play a game" in command:
# #         play_game()

# #     elif "tell me a joke" in command:
# #         joke = get_chuck_norris_joke()
# #         say(joke)

# #     elif "open youtube" in command:
# #         webbrowser.open("https://www.youtube.com")

# #     elif "open wikipedia" in command:
# #         webbrowser.open("https://www.wikipedia.com")
    
# #     elif "play mass  " in command:
# #         webbrowser.open("https://youtu.be/X6PLAysaevA?si=Nb1UAjwKNaRgUtOY")


# #     elif "open calculator" in command:
# #         os.system("calc")

# #     elif "what's the time" in command or "tell me the time" in command:
# #         current_time = datetime.datetime.now().strftime("%H:%M")
# #         say(f"The current time is {current_time}")

# #     elif "open music player" in command:
# #         open_music_player()

# #     elif "search on Google" in command:
# #         search_query = command.replace("search on Google", "")
# #         search_on_google(search_query)

# #     elif "open random website" in command:
# #         open_random_website()

# #     # Add more commands here...

# #     else:
# #         print("Command not recognized")


# # def play_game():
# #     try:
# #         # Replace the path with the actual path to your Pygame script
# #         game_path = r"C:\Users\Akshat\Desktop\dont-touch-my-presents-main\main.py"
# #         subprocess.run(["python", game_path], check=True)
# #     except Exception as e:
# #         print(f"Error launching the game: {e}")

# # def play_music_from_bando_radio():
# #     bando_radio_api_url = "https://bando-radio-api.p.rapidapi.com/stations/bycountry/Austria"
# #     querystring = {"hidebroken": "true", "offset": "0", "limit": "10"}
# #     headers = {
# #         "X-RapidAPI-Key": bando_radio_api_key,
# #         "X-RapidAPI-Host": "bando-radio-api.p.rapidapi.com"
# #     }
# #     response = requests.get(bando_radio_api_url,
# #                             headers=headers, params=querystring)
# #     stations = response.json().get("data", {}).get("stations", [])
# #     if stations:
# #         station = random.choice(stations)
# #         stream_url = station.get("stream_url", "")
# #         pygame.mixer.music.load(stream_url)
# #         pygame.mixer.music.play()
# #         say(f"Now playing {station.get('name', 'Unknown')} from Bando Radio.")
# #     else:
# #         say("Sorry, couldn't fetch stations from Bando Radio.")

# # def get_chuck_norris_joke():
# #     chuck_norris_api_url = "https://matchilling-chuck-norris-jokes-v1.p.rapidapi.com/jokes/random"
# #     headers = {
# #         "accept": "application/json",
# #         "X-RapidAPI-Key": bando_radio_api_key,  # Using the Bando Radio API key for Chuck Norris jokes
# #         "X-RapidAPI-Host": "matchilling-chuck-norris-jokes-v1.p.rapidapi.com"
# #     }
# #     response = requests.get(chuck_norris_api_url, headers=headers)
# #     return response.json().get("content", "No joke available.")

# # def say(text):
# #     engine = pyttsx3.init()
# #     engine.say(text)
# #     engine.runAndWait()

# # def take_command():
# #     r = sr.Recognizer()
# #     with sr.Microphone() as source:
# #         audio = r.listen(source)
# #         try:
# #             print("Recognizing...")
# #             command = r.recognize_google(audio, language="en-in")
# #             print(f"User said: {command}")
# #             return command.lower()
# #         except Exception as e:
# #             return "Some Error Occurred. Sorry from Jarvis"

# # if __name__ == '__main__':
# #     print('Welcome to Jarvis A.I')
# #     say("Jarvis ")
# #     while True:
# #         print("Listening...")
# #         command = take_command()
# #         execute_command(command)
