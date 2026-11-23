# Transcribe local: una línea JSON por frase {id, pcm: base64 s16le mono 16 kHz} -> {id, text, ms}
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
PROMPT_CHECK = PROMPT.replace("Nira: che", "Nira: mirá, che")
ONLY_NAME = re.compile(r"\W*(nyra|nira|ñira|naira|lyra|neira|nila)\W*", re.I)
# Frases fantasma vistas en llamadas reales: con ruido, risas o "mmm" repite la pista o se despide
PROMPT_LEAK = re.compile(r"(^|(?<=[.!?…]))\W*charla( de amigos( argentinos.*)?)?\W*$", re.I)
GHOST = re.compile(r"\W*((muchas )?gracias( por ver.*)?|ch?au|chao|adiós|un saludo|hasta (luego|la próxima)|see you( now| next time)?|thank you|suscr[ií]b.*|subtítulos.*)\W*", re.I)
PROMPT_WORDS = " " + " ".join(re.findall(r"\w+", PROMPT.lower())) + " "


def phantom(text):
    words = re.findall(r"\w+", text.lower())
    recited = len(words) >= 3 and f" {' '.join(words)} " in PROMPT_WORDS
    return recited or bool(GHOST.fullmatch(text))


def on_cpu():
    return WhisperModel(str(MODELS / "small"), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True), "small en CPU"


LAST_RAW = ""  # lo que oyó antes de los filtros: va al log para pulirlos


def hear(audio, prompt=PROMPT):
    global LAST_RAW
    # temperature=0: sin los reintentos que llevaban una frase de 1,7 s a 4,5 s e inventaban
    segments, _ = model.transcribe(audio, language="es", beam_size=5, temperature=0,
                                   vad_filter=True, vad_parameters={"threshold": 0.8},  # las risas y los "mmm" no son habla
                                   condition_on_previous_text=False, without_timestamps=True, initial_prompt=prompt)
    # poca confianza en dos o tres palabras sueltas: es ruido
    segments = list(segments)
    LAST_RAW = " ".join(s.text.strip() for s in segments)
    kept = [s.text.strip() for s in segments if s.avg_logprob >= -1.05 or len(s.text.split()) > 3]
    text = PROMPT_LEAK.sub("", " ".join(kept)).strip()
    return "" if phantom(text) else text


def understand(audio):
    text = hear(audio)
    return hear(audio, PROMPT_CHECK) if ONLY_NAME.fullmatch(text) else text  # un "Mirá." suelto salía "Nira."


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
try:
    model, where = WhisperModel(str(MODELS / "turbo"), device="cuda", compute_type="int8_float16", local_files_only=True), "large-v3-turbo en GPU"
    list(model.transcribe(np.zeros(16000, np.float32), language="es")[0])  # la falta de DLLs recién salta al transcribir
except Exception as e:
    print(f"Whisper: la GPU no arrancó ({str(e)[:160]})", file=sys.stderr)
    model, where = on_cpu()
print(json.dumps({"ready": True, "where": where}), flush=True)

for line in sys.stdin:
    msg = {}
    try:
        msg = json.loads(line)
        start = time.time()
        audio = np.frombuffer(base64.b64decode(msg["pcm"]), np.int16).astype(np.float32) / 32768
        try:
            text = understand(audio)
        except RuntimeError as e:
            if "GPU" not in where:
                raise
            print(f"Whisper: falló la GPU ({str(e)[:160]}), sigo en CPU", file=sys.stderr)
            model, where = on_cpu()
            text = understand(audio)
        out = {"id": msg["id"], "text": text, "ms": int((time.time() - start) * 1000)}
        if LAST_RAW.strip() != text:
            out["raw"] = LAST_RAW.strip()
    except Exception as e:
        out = {"id": msg.get("id"), "error": str(e)[:200]}
    print(json.dumps(out, ensure_ascii=False), flush=True)
