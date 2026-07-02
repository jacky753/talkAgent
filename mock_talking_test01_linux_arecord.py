import io
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path

import requests
from faster_whisper import WhisperModel
from ollama import Client


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
VOICEVOX_SPEAKER = int(os.getenv("VOICEVOX_SPEAKER", "3"))
ENABLE_TTS = os.getenv("ENABLE_TTS", "1") != "0"

SAMPLE_RATE = int(float(os.getenv("SAMPLE_RATE", "44100")))
RECORD_SECONDS = int(float(os.getenv("RECORD_SECONDS", "5")))

# 例:
# export ALSA_DEVICE="default"
# export ALSA_DEVICE="plughw:0,0"
ALSA_DEVICE = os.getenv("ALSA_DEVICE", "default")


def list_audio_devices():
    print("=== arecord -l ===")
    subprocess.run(["arecord", "-l"], check=False)

    print("\n=== arecord -L ===")
    subprocess.run(["arecord", "-L"], check=False)


def record_audio_to_wav(seconds=RECORD_SECONDS):
    """
    sounddeviceを使わず、Linux/Ubuntuのarecordで録音する。
    sounddeviceが sd.wait() で止まる環境向け。
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    print(f"\n{seconds}秒間話してください...")
    print(f"録音デバイス: {ALSA_DEVICE}")
    sys.stdout.flush()

    cmd = [
        "arecord",
        "-D", ALSA_DEVICE,
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-d", str(seconds),
        wav_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=seconds + 10,
        )
    except subprocess.TimeoutExpired:
        Path(wav_path).unlink(missing_ok=True)
        raise RuntimeError("arecordがタイムアウトしました。マイクデバイス設定を確認してください。")

    if result.returncode != 0:
        Path(wav_path).unlink(missing_ok=True)
        raise RuntimeError(
            "arecordで録音できませんでした。\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr:\n{result.stderr}"
        )

    print("録音完了")
    return wav_path


def load_whisper():
    try:
        print("Whisper: cuda/float16で読み込みを試します")
        return WhisperModel("small", device="cuda", compute_type="float16")
    except Exception as e:
        print(f"Whisper: cudaが使えないためCPUで起動します: {e}")
        return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_wav(model, wav_path):
    segments, info = model.transcribe(
        wav_path,
        language="ja",
        beam_size=5,
        vad_filter=False,
    )
    return "".join(seg.text for seg in segments).strip()


def remove_think_tags(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def ask_qwen(client, messages, user_text):
    prompt = (
        f"{user_text}\n\n"
        "/no_think\n"
        "日本語だけで、1〜2文で短く自然に返答してください。"
        "思考過程、英語の分析文、理由説明は出力しないでください。"
    )

    messages.append({"role": "user", "content": prompt})

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=False,
        think=False,
        options={
            "temperature": 0.2,
            "num_predict": 80,
        },
    )

    try:
        answer = response["message"]["content"]
    except Exception:
        answer = response.message.content

    answer = remove_think_tags(answer)
    messages.append({"role": "assistant", "content": answer})
    return answer


def voicevox_available():
    if not ENABLE_TTS:
        return False
    try:
        r = requests.get(f"{VOICEVOX_URL}/version", timeout=3)
        return r.ok
    except Exception:
        return False


def speak_voicevox(text, speaker=VOICEVOX_SPEAKER):
    if not ENABLE_TTS or not text:
        return

    try:
        query_resp = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": text, "speaker": speaker},
            timeout=30,
        )
        query_resp.raise_for_status()
        audio_query = query_resp.json()
        audio_query["speedScale"] = 1.08

        synth_resp = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            params={"speaker": speaker},
            json=audio_query,
            timeout=60,
        )
        synth_resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(synth_resp.content)
            wav_path = f.name

        try:
            subprocess.run(["paplay", wav_path], check=False)
        finally:
            Path(wav_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"VOICEVOX読み上げをスキップしました: {e}")


def main():
    if "--devices" in sys.argv:
        list_audio_devices()
        return

    print("Whisperを読み込み中...")
    whisper_model = load_whisper()

    client = Client(host=OLLAMA_HOST)

    if ENABLE_TTS:
        if voicevox_available():
            print(f"VOICEVOX接続OK: {VOICEVOX_URL}")
        else:
            print(f"VOICEVOXに接続できません。TTSをスキップします: {VOICEVOX_URL}")

    messages = [
        {
            "role": "system",
            "content": (
                "あなたは日本語の音声対話アシスタントです。"
                "必ず日本語だけで返答してください。"
                "思考過程、分析、理由説明、英語の内部メモは出力しないでください。"
                "返答は1〜2文で短く自然にしてください。"
            ),
        }
    ]

    print("準備完了。Enterで録音、qで終了。")

    while True:
        cmd = input("\nEnter: 録音開始 / q: 終了 > ").strip().lower()
        if cmd == "q":
            break

        wav_path = None
        try:
            wav_path = record_audio_to_wav()
            user_text = transcribe_wav(whisper_model, wav_path)
        finally:
            if wav_path:
                Path(wav_path).unlink(missing_ok=True)

        if not user_text:
            print("音声を認識できませんでした。")
            continue

        print(f"\nあなた: {user_text}")

        answer = ask_qwen(client, messages, user_text)
        print(f"Qwen3: {answer}")

        speak_voicevox(answer)


if __name__ == "__main__":
    main()
