# Transcribe local: una línea JSON por frase {id, pcm: base64 s16le mono 16 kHz} -> {id, text, ms, ...}
# large-v3-turbo en la GPU (1,1 s, 1,2 GB de VRAM); si la GPU no puede, small en CPU.
# No guarda audio ni texto, y no sale a internet (modelos en .whisper/).
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

for d in Path(sys.prefix).glob("Lib/site-packages/nvidia/*/bin"):
    os.environ["PATH"] = f"{d}{os.pathsep}{os.environ['PATH']}"  # DLLs de CUDA: antes de importar ctranslate2

import numpy as np
from faster_whisper import WhisperModel

MODELS = Path(__file__).parent.parent / ".whisper"
# Sin pista oye "Mira" donde dicen "Nira", y sin la jerga escribe "venía" por "vení". Si la cambiás, volvé a medir los falsos.
PROMPT = "Charla de amigos argentinos jugando al Wild Rift con Nira: che, boludo, dale, pará, vení, posta, pique, dragón, barón, jungla, torre, ulti."
PROMPT_MINIMAL = "Nira, che, boludo, pará, vení, posta, Wild Rift, jungla, dragón, barón, ulti."
PROMPT_CHECK = PROMPT.replace("Nira: che", "Nira: mirá, che")
DEFAULT_HOTWORDS = "Nyra, Discord, TikTok, Wild Rift"
ONLY_NAME = re.compile(r"\W*(nyra|nira|ñira|naira|lyra|neira|nila)\W*", re.I)
# Frases fantasma vistas en llamadas reales: con ruido, risas o "mmm" repite la pista o se despide
PROMPT_LEAK = re.compile(r"(^|(?<=[.!?…]))\W*charla( de amigos( argentinos.*)?)?\W*$", re.I)
GHOST = re.compile(r"\W*((muchas )?gracias( por ver.*)?|ch?au|chao|adiós|un saludo|hasta (luego|la próxima)|see you( now| next time)?|thank you|suscr[ií]b.*|subtítulos.*)\W*", re.I)
PROMPT_WORDS = " " + " ".join(re.findall(r"\w+", PROMPT.lower())) + " "

PROFILES = {
    "LEGACY": {"prompt_mode": "narrative", "hotwords": None, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"},
    "A": {"prompt_mode": "narrative", "hotwords": None, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"},
    "B": {"prompt_mode": "none", "hotwords": None, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"},
    "C": {"prompt_mode": "none", "hotwords": DEFAULT_HOTWORDS, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"},
    "D": {"prompt_mode": "none", "hotwords": DEFAULT_HOTWORDS, "vad_filter": True, "vad_threshold": 0.65, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"},
    "E": {"prompt_mode": "none", "hotwords": DEFAULT_HOTWORDS, "vad_filter": True, "vad_threshold": 0.5, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"},
    "F": {"prompt_mode": "none", "hotwords": DEFAULT_HOTWORDS, "vad_filter": True, "vad_threshold": 0.3, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"},
    "G": {"prompt_mode": "none", "hotwords": DEFAULT_HOTWORDS, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 3, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"},
    "H": {"prompt_mode": "none", "hotwords": DEFAULT_HOTWORDS, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 1, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"},
    # Perfiles de prompt / vocabulario (Fase 5)
    "P0": {"prompt_mode": "narrative", "hotwords": None, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "none", "filter_profile": "confidence_ghost"},
    "P1": {"prompt_mode": "none", "hotwords": None, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "none", "filter_profile": "confidence_ghost"},
    "P2": {"prompt_mode": "none", "hotwords": "Nyra", "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "none", "filter_profile": "confidence_ghost"},
    "P3": {"prompt_mode": "none", "hotwords": "Nyra, Wild Rift, jungla, dragón, barón, ulti", "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "none", "filter_profile": "confidence_ghost"},
    "P4": {"prompt_mode": "minimal", "hotwords": None, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "none", "filter_profile": "confidence_ghost"},
    "EXPERIMENTAL": {"prompt_mode": "minimal", "hotwords": None, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "none", "filter_profile": "confidence_ghost"},
}


def phantom_reason(text, filter_profile="legacy", kept_segments=None):
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 3 and f" {' '.join(words)} " in PROMPT_WORDS:
        return "phantom_recited"
    if filter_profile == "no_ghost":
        return None
    if GHOST.fullmatch(text):
        if filter_profile == "confidence_ghost" and kept_segments:
            # Cada segmento debe superar umbral acústico (sin promedios que oculten segmentos débiles)
            all_confident = all(
                (getattr(s, "avg_logprob", None) is not None and s.avg_logprob > -0.6)
                and (getattr(s, "no_speech_prob", None) is not None and s.no_speech_prob < 0.2)
                for s in kept_segments
            )
            if all_confident:
                return None
        return "phantom_ghost"
    return None


def on_cpu():
    return WhisperModel(str(MODELS / "small"), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True), "small en CPU", "small", "cpu", "int8"


def _aggregate_segments(segments):
    if not segments:
        return {"segment_count": 0, "avg_logprob": None, "no_speech_prob": None, "compression_ratio": None}
    avg_lp = [s.avg_logprob for s in segments if getattr(s, "avg_logprob", None) is not None]
    no_sp = [s.no_speech_prob for s in segments if getattr(s, "no_speech_prob", None) is not None]
    comp = [s.compression_ratio for s in segments if getattr(s, "compression_ratio", None) is not None]
    return {
        "segment_count": len(segments),
        "avg_logprob": round(sum(avg_lp) / len(avg_lp), 4) if avg_lp else None,
        "no_speech_prob": round(sum(no_sp) / len(no_sp), 4) if no_sp else None,
        "compression_ratio": round(sum(comp) / len(comp), 4) if comp else None,
    }


def resolve_effective_params(msg):
    profile_id = str(msg.get("profile") or "A").strip().upper()
    base = dict(PROFILES.get(profile_id, PROFILES["A"]))
    if msg.get("no_prompt"):
        base["prompt_mode"] = "none"
    if "use_prompt" in msg and msg["use_prompt"] is not None:
        base["prompt_mode"] = "narrative" if bool(msg["use_prompt"]) else "none"
    if "prompt_mode" in msg and msg["prompt_mode"] in ("narrative", "none", "minimal"):
        base["prompt_mode"] = msg["prompt_mode"]
    if "hotwords" in msg:
        hw = str(msg["hotwords"] or "").strip()
        base["hotwords"] = hw or None
    if "vad_filter" in msg and msg["vad_filter"] is not None:
        base["vad_filter"] = bool(msg["vad_filter"])
    if "vad_threshold" in msg and msg["vad_threshold"] is not None:
        base["vad_threshold"] = max(0.05, min(0.99, float(msg["vad_threshold"])))
    if "beam_size" in msg and msg["beam_size"] is not None:
        base["beam_size"] = max(1, min(10, int(msg["beam_size"])))
    if "temperature" in msg and msg["temperature"] is not None:
        base["temperature"] = max(0.0, min(1.0, float(msg["temperature"])))
    if "second_pass_mode" in msg and msg["second_pass_mode"] in ("always_only_name", "none", "confidence"):
        base["second_pass_mode"] = msg["second_pass_mode"]
    if "filter_profile" in msg and msg["filter_profile"] in ("legacy", "confidence_ghost", "no_ghost"):
        base["filter_profile"] = msg["filter_profile"]
    base["profile"] = profile_id if profile_id in PROFILES else "A"
    base["use_prompt"] = base["prompt_mode"] != "none"
    return base


def _prompt_text(prompt_mode):
    if prompt_mode == "narrative":
        return PROMPT
    if prompt_mode == "minimal":
        return PROMPT_MINIMAL
    return None


def hear(audio, prompt=PROMPT, vad_filter=True, vad_threshold=0.8, beam_size=5, temperature=0.0, hotwords=None, filter_profile="legacy"):
    t_inf0 = time.time()
    # temperature=0: sin los reintentos que llevaban una frase de 1,7 s a 4,5 s e inventaban
    kw = dict(
        language="es",
        beam_size=beam_size,
        temperature=temperature,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    if vad_filter:
        kw["vad_parameters"] = {"threshold": vad_threshold}  # las risas y los "mmm" no son habla
    if prompt:
        kw["initial_prompt"] = prompt
    if hotwords:
        kw["hotwords"] = hotwords
    segments_gen, info = model.transcribe(audio, **kw)
    segments = list(segments_gen)
    inference_ms = int((time.time() - t_inf0) * 1000)

    t_post0 = time.time()
    raw_text = " ".join(s.text.strip() for s in segments).strip()
    agg = _aggregate_segments(segments)
    post_vad = getattr(info, "duration_after_vad", None)
    post_vad_sec = round(float(post_vad), 3) if post_vad is not None else None

    if not segments:
        return {
            "text": "",
            "raw": raw_text,
            "discard_reason": "vad_no_speech",
            "inference_ms": inference_ms,
            "postprocess_ms": int((time.time() - t_post0) * 1000),
            "post_vad_duration_sec": post_vad_sec,
            **agg,
        }

    kept_segs = [s for s in segments if s.avg_logprob >= -1.05 or len(s.text.split()) > 3]
    if not kept_segs:
        return {
            "text": "",
            "raw": raw_text,
            "discard_reason": "low_confidence",
            "inference_ms": inference_ms,
            "postprocess_ms": int((time.time() - t_post0) * 1000),
            "post_vad_duration_sec": post_vad_sec,
            **agg,
        }

    joined = " ".join(s.text.strip() for s in kept_segs).strip()
    after_leak = PROMPT_LEAK.sub("", joined).strip()
    if joined and not after_leak:
        return {
            "text": "",
            "raw": raw_text,
            "discard_reason": "prompt_leak",
            "inference_ms": inference_ms,
            "postprocess_ms": int((time.time() - t_post0) * 1000),
            "post_vad_duration_sec": post_vad_sec,
            **agg,
        }

    ph = phantom_reason(after_leak, filter_profile=filter_profile, kept_segments=kept_segs)
    if ph:
        return {
            "text": "",
            "raw": raw_text,
            "discard_reason": ph,
            "inference_ms": inference_ms,
            "postprocess_ms": int((time.time() - t_post0) * 1000),
            "post_vad_duration_sec": post_vad_sec,
            **agg,
        }

    if not after_leak:
        return {
            "text": "",
            "raw": raw_text,
            "discard_reason": "empty_transcript",
            "inference_ms": inference_ms,
            "postprocess_ms": int((time.time() - t_post0) * 1000),
            "post_vad_duration_sec": post_vad_sec,
            **agg,
        }

    return {
        "text": after_leak,
        "raw": raw_text,
        "discard_reason": None,
        "inference_ms": inference_ms,
        "postprocess_ms": int((time.time() - t_post0) * 1000),
        "post_vad_duration_sec": post_vad_sec,
        **agg,
    }


def understand(audio, params):
    prompt = _prompt_text(params["prompt_mode"])
    r1 = hear(
        audio,
        prompt=prompt,
        vad_filter=params["vad_filter"],
        vad_threshold=params["vad_threshold"],
        beam_size=params["beam_size"],
        temperature=params["temperature"],
        hotwords=params["hotwords"],
        filter_profile=params["filter_profile"],
    )
    mode = params["second_pass_mode"]
    should_second_pass = (
        mode != "none"
        and params["prompt_mode"] == "narrative"
        and bool(ONLY_NAME.fullmatch(r1["text"]))
        and (mode == "always_only_name" or (r1["avg_logprob"] is not None and r1["avg_logprob"] < -0.5))
    )
    if not should_second_pass:
        r1["passes"] = 1
        r1["second_pass_triggered"] = False
        r1["second_pass_changed_text"] = False
        return r1

    # Un "Mirá." suelto salía "Nira."
    r2 = hear(
        audio,
        prompt=PROMPT_CHECK,
        vad_filter=params["vad_filter"],
        vad_threshold=params["vad_threshold"],
        beam_size=params["beam_size"],
        temperature=params["temperature"],
        hotwords=params["hotwords"],
        filter_profile=params["filter_profile"],
    )
    r2["inference_ms"] = r1["inference_ms"] + r2["inference_ms"]
    r2["postprocess_ms"] = r1["postprocess_ms"] + r2["postprocess_ms"]
    r2["passes"] = 2
    r2["second_pass_triggered"] = True
    r2["second_pass_changed_text"] = (r2["text"] != r1["text"])
    r2["first_pass_text"] = r1["text"]
    return r2


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
requested_model = "large-v3-turbo"
cuda_error = None
fallback_reason = None
t_init0 = time.time()
try:
    model, where = WhisperModel(str(MODELS / "turbo"), device="cuda", compute_type="int8_float16", local_files_only=True), "large-v3-turbo en GPU"
    active_model, active_device, active_compute = "large-v3-turbo", "cuda", "int8_float16"
    list(model.transcribe(np.zeros(16000, np.float32), language="es")[0])  # la falta de DLLs recién salta al transcribir
except Exception as e:
    cuda_error = str(e)[:160]
    fallback_reason = f"init_cuda_error: {cuda_error}"
    print(f"Whisper: la GPU no arrancó ({cuda_error})", file=sys.stderr)
    model, where, active_model, active_device, active_compute = on_cpu()
init_ms = int((time.time() - t_init0) * 1000)
print(json.dumps({
    "ready": True,
    "where": where,
    "requested_model": requested_model,
    "model": active_model,
    "device": active_device,
    "compute_type": active_compute,
    "init_ms": init_ms,
    **({"cuda_error": cuda_error, "fallback_reason": fallback_reason} if cuda_error else {}),
}), flush=True)

for line in sys.stdin:
    msg = {}
    try:
        msg = json.loads(line)
        start = time.time()
        audio = np.frombuffer(base64.b64decode(msg["pcm"]), np.int16).astype(np.float32) / 32768
        audio_duration_sec = round(len(audio) / 16000.0, 3)
        params = resolve_effective_params(msg)
        try:
            res = understand(audio, params)
        except RuntimeError as e:
            if "GPU" not in where:
                raise
            cuda_error = str(e)[:160]
            fallback_reason = f"runtime_cuda_error: {cuda_error}"
            print(f"Whisper: falló la GPU ({cuda_error}), sigo en CPU", file=sys.stderr)
            model, where, active_model, active_device, active_compute = on_cpu()
            res = understand(audio, params)
        total_ms = int((time.time() - start) * 1000)
        rtf = round(res["inference_ms"] / (audio_duration_sec * 1000.0), 3) if audio_duration_sec > 0 else 0.0
        out = {
            "id": msg["id"],
            "text": res["text"],
            "ms": total_ms,
            "inference_ms": res["inference_ms"],
            "postprocess_ms": res["postprocess_ms"],
            "rtf": rtf,
            "audio_duration_sec": audio_duration_sec,
            "post_vad_duration_sec": res["post_vad_duration_sec"],
            "requested_model": requested_model,
            "model": active_model,
            "device": active_device,
            "compute_type": active_compute,
            "fallback_reason": fallback_reason,
            "profile": params["profile"],
            "filter_profile": params["filter_profile"],
            "prompt_mode": params["prompt_mode"],
            "beam_size": params["beam_size"],
            "temperature": params["temperature"],
            "vad_filter": params["vad_filter"],
            "vad_threshold": params["vad_threshold"],
            "use_prompt": params["use_prompt"],
            "hotwords": params["hotwords"],
            "passes": res["passes"],
            "second_pass_triggered": res["second_pass_triggered"],
            "second_pass_changed_text": res["second_pass_changed_text"],
            "segment_count": res["segment_count"],
            "avg_logprob": res["avg_logprob"],
            "no_speech_prob": res["no_speech_prob"],
            "compression_ratio": res["compression_ratio"],
            "discard_reason": res["discard_reason"],
        }
        if res["raw"] != res["text"]:
            out["raw"] = res["raw"]
        if "first_pass_text" in res:
            out["first_pass_text"] = res["first_pass_text"]
    except Exception as e:
        out = {"id": msg.get("id"), "error": str(e)[:200], "status": "worker_error"}
    print(json.dumps(out, ensure_ascii=False), flush=True)
