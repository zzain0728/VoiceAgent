# test_realtime.py
import os
from dotenv import load_dotenv
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions,
    StreamingEvents, StreamingParameters,
    StreamingSessionParameters, BeginEvent, TurnEvent,
    TerminationEvent, StreamingError
)

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


def on_begin(self: StreamingClient, event: BeginEvent):
    print(f"\n🎙️ Session started: {event.id}\nSay something…")

def on_turn(self: StreamingClient, event: TurnEvent):
    # print only final turns
    if event.transcript and event.end_of_turn:
        print("You said:", event.transcript)
    if event.end_of_turn and not event.turn_is_formatted:
        self.set_params(StreamingSessionParameters(format_turns=True))

def on_terminated(self: StreamingClient, event: TerminationEvent):
    print(f"\n✅ Session terminated ({event.audio_duration_seconds:.1f}s processed)")

def on_error(self: StreamingClient, err: StreamingError):
    print("❌ Error:", err)

def main():
    client = StreamingClient(StreamingClientOptions(
        api_key=aai.settings.api_key,
        api_host="streaming.assemblyai.com",
    ))

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    SAMPLE_RATE = 16000  # try 44100 if needed
    client.connect(StreamingParameters(sample_rate=SAMPLE_RATE, format_turns=True))
    try:
        client.stream(aai.extras.MicrophoneStream(sample_rate=SAMPLE_RATE))
    except KeyboardInterrupt:
        print("\n⏹ Stopping…")
    finally:
        client.disconnect(terminate=True)

if __name__ == "__main__":
    main()
