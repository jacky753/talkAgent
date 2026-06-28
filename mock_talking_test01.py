import io
import re
import tempfile

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from ollama import Client


# =========================
# 設定
# =========================

OLLAMA_MODEL = "qwen3:1.7b"#"qwen3:4b"
VOICEVOX_URL = "http://127.0.0.1:50021"
VOICEVOX_SPEAKER = 3  # 話者ID。VOICEVOX側で変更可能。

SAMPLE_RATE = 44100#16000
RECORD_SECONDS = 5


# =======================f==
# ASR: faster-whisper
# =========================

def load_whisper():
    """
    GPUが使えるならcuda、失敗したらCPU int8で起動。
    """
    try:
        return WhisperModel("small", device="cuda", compute_type="float16")
    except Exception:
        return WhisperModel("small", device="cpu", compute_type="int8")


def record_audio(seconds=RECORD_SECONDS):
    print(f"\n{seconds}秒間話してください...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio


def transcribe_audio(model, audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, SAMPLE_RATE)

        segments, info = model.transcribe(
            f.name,
            language="ja",
            beam_size=5,
            vad_filter=True,
        )

        text = "".join(seg.text for seg in segments).strip()
        return text


# =========================
# LLM: Qwen3 via Ollama
# =========================

def remove_think_tags(text):
    """
    モデルが <think>...</think> を出した場合、音声読み上げから除外。
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def ask_qwen(client, messages, user_text):
    messages.append({"role": "user", "content": user_text})

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=False,
        think=False,
        options={
            "temperature": 0.6,
            "num_predict": 180,
        },
    )

    answer = response["message"]["content"]
    answer = remove_think_tags(answer)

    messages.append({"role": "assistant", "content": answer})
    return answer


# =========================
# TTS: VOICEVOX
# =========================

def speak_voicevox(text, speaker=VOICEVOX_SPEAKER):
    if not text:
        return

    query_resp = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={
            "text": text,
            "speaker": speaker,
        },
        timeout=30,
    )
    query_resp.raise_for_status()
    audio_query = query_resp.json()

    # 話速を少し上げる
    audio_query["speedScale"] = 1.08

    synth_resp = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker},
        json=audio_query,
        timeout=60,
    )
    synth_resp.raise_for_status()

    audio_bytes = io.BytesIO(synth_resp.content)
    audio, sr = sf.read(audio_bytes, dtype="float32")

    sd.play(audio, sr)
    sd.wait()


# =========================
# Main loop
# =========================

def main():
    print("Whisperを読み込み中...")
    whisper_model = load_whisper()

    client = Client(host="http://127.0.0.1:11434")

    messages = [
        {
            "role": "system",
            "content": (
                "あなたは日本語で会話する音声アシスタントです。"
                "返答は短く、自然な会話調にしてください。"
                "長い箇条書きは避けてください。"
            ),
        }
    ]

    print("準備完了。Enterで録音、qで終了。")

    while True:
        cmd = input("\nEnter: 録音開始 / q: 終了 > ").strip().lower()
        if cmd == "q":
            break

        audio = record_audio()
        user_text = transcribe_audio(whisper_model, audio)

        if not user_text:
            print("音声を認識できませんでした。")
            continue

        print(f"\nあなた: {user_text}")

        answer = ask_qwen(client, messages, user_text)
        print(f"Qwen3: {answer}")

        speak_voicevox(answer)


if __name__ == "__main__":
    main()

    