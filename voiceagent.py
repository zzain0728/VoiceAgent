import assemblyai as aai
import ollama
import os
import soundfile as sf
import sounddevice as sd
import replicate
import requests
import io
import sys
from typing import Optional

from groq import Groq

from dotenv import load_dotenv

load_dotenv()  # reads .env from project root
ASSEMBLYAI_API_KEY = (os.getenv("ASSEMBLYAI_API_KEY"))

REPLICATE_API_TOKEN = (os.getenv("REPLICATE_API_TOKEN"))

GROQ_API_KEY = (os.getenv("GROQ_API_KEY"))

if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY not set properly")

if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN not set properly")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set properly")


aai.settings.api_key = ASSEMBLYAI_API_KEY


print(ASSEMBLYAI_API_KEY)
print(REPLICATE_API_TOKEN)

print(GROQ_API_KEY)


class AIVoiceAgent:
    def __init__(self):
        self.replicate_token = REPLICATE_API_TOKEN

        # Will later hold AssemblyAI transcriber object for turning speech into text.

        self.transcriber = None

        # This is a conversation history in OpenAI/Replicate-style format.

        self.transcript = [{"role": "system", "content": """
                    You are an interviewer for a role in data science.
                    Can you be proactive in asking questions to see if candidate is a good fit for the role.
            
                    Please keep your answers concise, ideally under 300 characters.
                    Please generate only text and no emojis.
                    Please start by asking a welcoming question.
                    Please ask only one question at a time.
                    Instead of * please use numbered lists and use numbered list if there are 2 bullet points.
                    """}]



        def _start_transcription(self):
            print("\n 🎙️ Listening...")



client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = "You are a helpful, concise assistant."
MODEL = "llama3-70b-8192"   # try "mixtral-8x7b-32768" too

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("Chat started. Type /reset to clear, /exit to quit.\n")
while True:
    try:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() == "/exit":
            break
        if user.lower() == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(Context reset)\n")
            continue

        messages.append({"role": "user", "content": user})

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,   # tweak style/creativity
            max_tokens=512     # cap response length
        )
        reply = resp.choices[0].message.content
        print(f"Assistant: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

    except KeyboardInterrupt:
        print("\n(Interrupted)")
        break

print(messages)
'''
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
    raise RuntimeError(f"Transcription failed: {transcript.error}")


'''



