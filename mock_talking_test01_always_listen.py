import os
import re
import sys
import time
import wave
import tempfile
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import requests
from faster_whisper import WhisperModel
from ollama import Client


# ============================================================
# 設定
# ============================================================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
VOICEVOX_SPEAKER = int(os.getenv("VOICEVOX_SPEAKER", "3"))
ENABLE_TTS = os.getenv("ENABLE_TTS", "1") != "0"

# WSLg / PulseAudio 用
PULSE_SERVER = os.getenv("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")
PULSE_SOURCE = os.getenv("PULSE_SOURCE", "")  # 空ならPulseAudioのdefault source

# 音声入力設定
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # s16le
CHUNK_SEC = float(os.getenv("CHUNK_SEC", "0.1"))
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_SEC)
CHUNK_BYTES = CHUNK_FRAMES * CHANNELS * SAMPLE_WIDTH_BYTES

# VAD設定
CALIBRATION_SEC = float(os.getenv("CALIBRATION_SEC", "1.5"))
PRE_ROLL_SEC = float(os.getenv("PRE_ROLL_SEC", "0.4"))
MIN_SPEECH_SEC = float(os.getenv("MIN_SPEECH_SEC", "0.35"))
SILENCE_SEC = float(os.getenv("SILENCE_SEC", "0.8"))
MAX_UTTERANCE_SEC = float(os.getenv("MAX_UTTERANCE_SEC", "20"))
COOLDOWN_SEC = float(os.getenv("COOLDOWN_SEC", "0.4"))

# 0なら起動時に周囲ノイズから自動推定
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0"))

# Whisper設定
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto / cuda / cpu
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")


# ============================================================
# 共通ユーティリティ
# ============================================================

def rms_int16(pcm_bytes: bytes) -> float:
    """s16le PCM bytes のRMSを 0.0〜1.0 程度で返す。"""
    if not pcm_bytes:
        return 0.0
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)) / 32768.0)


def write_wav_from_pcm(pcm_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    return wav_path


def remove_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"(?is)^.*?(?:final answer|answer)\s*[:：]\s*", "", text).strip()
    return text.strip()


def clean_for_tts(text: str) -> str:
    """読み上げに不要なMarkdownや長すぎる空白を軽く除去。"""
    text = remove_think_tags(text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"[*#`>\-]{1,}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# PulseAudio録音: parecord常時監視 + 簡易VAD
# ============================================================

def start_parecord():
    env = os.environ.copy()
    env["PULSE_SERVER"] = PULSE_SERVER

    cmd = [
        "parecord",
        "--raw",
        "--channels=1",
        f"--rate={SAMPLE_RATE}",
        "--format=s16le",
    ]

    if PULSE_SOURCE:
        cmd.append(f"--device={PULSE_SOURCE}")

    print("音声入力を待機します。終了するには Ctrl+C を押してください。")
    print(f"PULSE_SERVER={PULSE_SERVER}")
    if PULSE_SOURCE:
        print(f"PULSE_SOURCE={PULSE_SOURCE}")
    print(f"parecord command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )
    return proc


def calibrate_noise(proc) -> float:
    """起動直後の環境音RMSからしきい値を推定。"""
    print(f"{CALIBRATION_SEC:.1f}秒間、周囲ノイズを測定します。少し静かにしてください。")
    chunks = int(CALIBRATION_SEC / CHUNK_SEC)
    values = []

    for _ in range(max(1, chunks)):
        chunk = proc.stdout.read(CHUNK_BYTES)
        if not chunk or len(chunk) < CHUNK_BYTES:
            continue
        values.append(rms_int16(chunk))

    if not values:
        threshold = 0.01
        noise = 0.0
    else:
        noise = float(np.median(values))
        threshold = max(0.008, noise * 3.0)

    print(f"推定ノイズRMS: {noise:.5f}")
    print(f"VADしきい値: {threshold:.5f}")
    return threshold


def listen_utterances():
    """
    発話単位でwavファイルパスをyieldする。
    声がしきい値を超えたら録音開始、無音が続いたら発話終了。
    """
    proc = start_parecord()
    if proc.stdout is None:
        raise RuntimeError("parecord stdoutを開けませんでした。")

    threshold = VAD_THRESHOLD if VAD_THRESHOLD > 0 else calibrate_noise(proc)

    pre_roll_max_chunks = max(1, int(PRE_ROLL_SEC / CHUNK_SEC))
    silence_max_chunks = max(1, int(SILENCE_SEC / CHUNK_SEC))
    max_chunks = max(1, int(MAX_UTTERANCE_SEC / CHUNK_SEC))
    min_speech_chunks = max(1, int(MIN_SPEECH_SEC / CHUNK_SEC))

    pre_roll = deque(maxlen=pre_roll_max_chunks)
    recording = False
    speech_chunks = []
    silence_count = 0

    print("\n待機中です。話しかけてください。")

    try:
        while True:
            chunk = proc.stdout.read(CHUNK_BYTES)

            if not chunk:
                err = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
                raise RuntimeError(f"parecordから音声を読めませんでした。\n{err}")

            level = rms_int16(chunk)
            is_voice = level >= threshold

            if not recording:
                pre_roll.append(chunk)

                if is_voice:
                    recording = True
                    speech_chunks = list(pre_roll)
                    silence_count = 0
                    print("\n発話検出...", flush=True)

            else:
                speech_chunks.append(chunk)

                if is_voice:
                    silence_count = 0
                else:
                    silence_count += 1

                too_long = len(speech_chunks) >= max_chunks
                end_by_silence = silence_count >= silence_max_chunks

                if too_long or end_by_silence:
                    total_chunks = len(speech_chunks)

                    if total_chunks >= min_speech_chunks:
                        pcm = b"".join(speech_chunks)
                        wav_path = write_wav_from_pcm(pcm)
                        print("発話終了。認識します。", flush=True)
                        yield wav_path

                    recording = False
                    speech_chunks = []
                    silence_count = 0
                    pre_roll.clear()

                    if COOLDOWN_SEC > 0:
                        time.sleep(COOLDOWN_SEC)
                        print("\n待機中です。話しかけてください。", flush=True)

    finally:
        try:
            proc.terminate()
        except Exception:
            pass


# ============================================================
# ASR: faster-whisper
# ============================================================

def load_whisper():
    if WHISPER_DEVICE == "cpu":
        return WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8" if WHISPER_COMPUTE_TYPE == "auto" else WHISPER_COMPUTE_TYPE,
        )

    if WHISPER_DEVICE == "cuda":
        return WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="float16" if WHISPER_COMPUTE_TYPE == "auto" else WHISPER_COMPUTE_TYPE,
        )

    # auto
    try:
        print("Whisper: cuda/float16で読み込みを試します")
        return WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="float16" if WHISPER_COMPUTE_TYPE == "auto" else WHISPER_COMPUTE_TYPE,
        )
    except Exception as e:
        print(f"Whisper: cudaが使えないためCPUで起動します: {e}")
        return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe_wav(model, wav_path: str) -> str:
    segments, info = model.transcribe(
        wav_path,
        language="ja",
        beam_size=5,
        vad_filter=False,
    )
    return "".join(seg.text for seg in segments).strip()


# ============================================================
# LLM: Qwen3 via Ollama
# ============================================================

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
            "num_predict": 100,
        },
    )

    try:
        answer = response["message"]["content"]
    except Exception:
        answer = response.message.content

    answer = remove_think_tags(answer)
    messages.append({"role": "assistant", "content": answer})
    return answer


# ============================================================
# TTS: VOICEVOX
# ============================================================

def voicevox_available() -> bool:
    if not ENABLE_TTS:
        return False
    try:
        r = requests.get(f"{VOICEVOX_URL}/version", timeout=3)
        return r.ok
    except Exception:
        return False


def speak_voicevox(text, speaker=VOICEVOX_SPEAKER):
    if not ENABLE_TTS:
        return

    text = clean_for_tts(text)
    if not text:
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


# ============================================================
# Main
# ============================================================

def main():
    if "--sources" in sys.argv:
        subprocess.run(["pactl", "list", "sources", "short"], check=False)
        return

    if "--sinks" in sys.argv:
        subprocess.run(["pactl", "list", "sinks", "short"], check=False)
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

    print("準備完了。常時待機モードに入ります。")

    try:
        for wav_path in listen_utterances():
            try:
                user_text = transcribe_wav(whisper_model, wav_path)
            finally:
                Path(wav_path).unlink(missing_ok=True)

            if not user_text:
                print("音声を認識できませんでした。")
                continue

            print(f"\nあなた: {user_text}")

            answer = ask_qwen(client, messages, user_text)
            print(f"Qwen3: {answer}")

            speak_voicevox(answer)

    except KeyboardInterrupt:
        print("\n終了します。")
    except Exception as e:
        print(f"\nエラー: {e}")
        print("ヒント: pactl list sources short でsource名を確認し、必要なら PULSE_SOURCE を指定してください。")
        raise


if __name__ == "__main__":
    main()
