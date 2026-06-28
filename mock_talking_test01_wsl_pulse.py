import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests
from faster_whisper import WhisperModel
from ollama import Client


# =========================
# 設定
# =========================

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

VOICEVOX_URL = os.environ.get("VOICEVOX_URL")
VOICEVOX_SPEAKER = int(os.environ.get("VOICEVOX_SPEAKER", "3"))

# WSLのPulseAudio/parecord用。Whisperには16 kHz monoで十分です。
SAMPLE_RATE = int(float(os.environ.get("SAMPLE_RATE", "16000")))
RECORD_SECONDS = int(os.environ.get("RECORD_SECONDS", "5"))

# export ENABLE_TTS=0 でVOICEVOX読み上げを止められます。
ENABLE_TTS = os.environ.get("ENABLE_TTS", "1") != "0"


# =========================
# ASR: faster-whisper
# =========================

def load_whisper():
    """GPUが使えるならcuda、失敗したらCPU int8で起動。"""
    try:
        return WhisperModel("small", device="cuda", compute_type="float16")
    except Exception:
        return WhisperModel("small", device="cpu", compute_type="int8")


def ensure_pulse_env():
    """WSLgのPulseAudioソケットを使うための保険。"""
    pulse_socket = "/mnt/wslg/PulseServer"
    if not os.environ.get("PULSE_SERVER") and Path(pulse_socket).exists():
        os.environ["PULSE_SERVER"] = f"unix:{pulse_socket}"


def record_audio_file(seconds=RECORD_SECONDS):
    """
    sounddevice/PortAudioを使わず、WSLのparecordで録音する。
    PaErrorCode -9987 が出る環境ではこちらの方が安定しやすい。
    """
    ensure_pulse_env()

    if shutil.which("parecord") is None:
        raise RuntimeError(
            "parecord が見つかりません。次を実行してください: "
            "sudo apt install -y pulseaudio-utils"
        )

    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = wav_file.name
    wav_file.close()

    print(f"\n{seconds}秒間話してください...")

    cmd = [
        "timeout",
        "--signal=INT",
        f"{seconds + 1}s",
        "parecord",
        f"--rate={SAMPLE_RATE}",
        "--channels=1",
        "--format=s16le",
        "--file-format=wav",
        wav_path,
    ]

    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except Exception:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise

    if not Path(wav_path).exists() or Path(wav_path).stat().st_size < 1000:
        raise RuntimeError(
            "録音ファイルを作成できませんでした。Windows側のマイク許可、"
            "PULSE_SERVER、pactl list sources short を確認してください。"
        )

    return wav_path


def transcribe_audio_file(model, wav_path):
    segments, info = model.transcribe(
        wav_path,
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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def looks_like_thinking(text):
    markers = [
        "Okay, the user",
        "Let me think",
        "I need to respond",
        "The user said",
        "First, I should",
        "Hmm,",
        "Wait,",
    ]
    return any(m in text for m in markers)


def extract_short_japanese_fallback(text):
    text = remove_think_tags(text)
    if not looks_like_thinking(text):
        return text.strip()

    if "こんにちは" in text:
        return "こんにちは！今日はどんなことを話しましょうか？"
    if "ありがとう" in text:
        return "どういたしまして。"
    if "おはよう" in text:
        return "おはようございます！"
    if "こんばんは" in text:
        return "こんばんは！"

    candidates = re.findall(r"[ぁ-んァ-ン一-龥ー、。！？!?.0-9A-Za-z\s]{3,}[。！？!?]", text)
    jp_candidates = [c.strip() for c in candidates if re.search(r"[ぁ-んァ-ン一-龥]", c)]
    if jp_candidates:
        return jp_candidates[-1][:120]

    return "すみません、もう一度お願いします。"


def ask_qwen(client, messages, user_text):
    prompt = (
        f"{user_text}\n\n"
        "/no_think\n"
        "日本語だけで、1〜2文で短く自然に返答してください。"
        "思考過程や英語の分析文は絶対に出力しないでください。"
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

    answer = extract_short_japanese_fallback(answer)
    messages.append({"role": "assistant", "content": answer})
    return answer


# =========================
# TTS: VOICEVOX
# =========================

def get_windows_host_ip():
    try:
        out = subprocess.check_output(
            "ip route | awk '/default/ {print $3; exit}'",
            shell=True,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def candidate_voicevox_urls():
    urls = []
    if VOICEVOX_URL:
        urls.append(VOICEVOX_URL.rstrip("/"))
    urls.append("http://127.0.0.1:50021")
    win_host = get_windows_host_ip()
    if win_host:
        urls.append(f"http://{win_host}:50021")

    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def find_voicevox_url():
    for url in candidate_voicevox_urls():
        try:
            r = requests.get(f"{url}/speakers", timeout=2)
            if r.ok:
                print(f"VOICEVOX接続先: {url}")
                return url
        except requests.RequestException:
            pass
    print("VOICEVOXに接続できません。TTSをスキップします。")
    print("Windows側でVOICEVOXを起動するか、VOICEVOX_URLを設定してください。")
    return None


def play_wav_with_paplay(wav_path):
    ensure_pulse_env()
    if shutil.which("paplay") is None:
        print("paplay が見つからないため再生をスキップしました。sudo apt install -y pulseaudio-utils を実行してください。")
        return
    subprocess.run(["paplay", wav_path], check=False)


def speak_voicevox(text, base_url, speaker=VOICEVOX_SPEAKER):
    if not text or not base_url:
        return

    try:
        query_resp = requests.post(
            f"{base_url}/audio_query",
            params={"text": text, "speaker": speaker},
            timeout=30,
        )
        query_resp.raise_for_status()
        audio_query = query_resp.json()
        audio_query["speedScale"] = 1.08

        synth_resp = requests.post(
            f"{base_url}/synthesis",
            params={"speaker": speaker},
            json=audio_query,
            timeout=60,
        )
        synth_resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(synth_resp.content)
            wav_path = f.name

        try:
            play_wav_with_paplay(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    except requests.RequestException as e:
        print(f"VOICEVOX接続エラーのため読み上げをスキップしました: {e}")
    except Exception as e:
        print(f"音声再生エラーのため読み上げをスキップしました: {e}")


# =========================
# Main loop
# =========================

def main():
    ensure_pulse_env()

    print("Whisperを読み込み中...")
    whisper_model = load_whisper()

    client = Client(host=OLLAMA_HOST)
    voicevox_url = find_voicevox_url() if ENABLE_TTS else None

    messages = [
        {
            "role": "system",
            "content": (
                "/no_think\n"
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
            wav_path = record_audio_file()
            user_text = transcribe_audio_file(whisper_model, wav_path)
        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        if not user_text:
            print("音声を認識できませんでした。")
            continue

        print(f"\nあなた: {user_text}")

        answer = ask_qwen(client, messages, user_text)
        print(f"Qwen3: {answer}")

        if ENABLE_TTS:
            speak_voicevox(answer, voicevox_url)


if __name__ == "__main__":
    main()
