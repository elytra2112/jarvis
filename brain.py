from openai import OpenAI
from config import nvidia_api_key

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key
)

MODEL = "nvidia/nemotron-3-super-120b-a12b"

conversation = [
    {
        "role": "system",
        "content": """
You are Jarvis, a personal AI assistant.

The user's name is Akshat.

You are intelligent, friendly, concise, natural and practical.

Answer questions clearly and helpfully.

You are running inside a Python application.

Do not claim that you performed an action unless the Python
application actually performed that action.

If the user asks a normal question, answer normally.

Do not unnecessarily explain your reasoning.
"""
    }
]


def ask_ai(user_message):
    conversation.append({
        "role": "user",
        "content": user_message
    })

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=conversation,
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=False
        )

        answer = completion.choices[0].message.content

        conversation.append({
            "role": "assistant",
            "content": answer
        })

        return answer

    except Exception as e:
        print(f"NVIDIA API Error: {e}")

        conversation.pop()

        return "Sorry, I couldn't connect to my AI brain."