# Corpus Piloto STT (40 escenarios)

Las plantillas públicas están en `docs/stt-corpus/` (versionadas en Git).
Los audios reales y el `manifest.json` privado se guardan en `voices/stt/` (ignorado por Git en `.gitignore`).

## 1. Grabar el corpus fácilmente (herramienta web local)

Ejecutá la herramienta independiente (puerto `3456`, nunca `3000`, sin iniciar NYRA/Discord/LLM):

```bash
node scripts/corpus-recorder.mjs
```

Luego abrí `http://127.0.0.1:3456` en tu navegador:
- Seleccioná tu micrófono y revisá el vúmetro.
- Grabá cada una de las 40 frases (`28 tune` + `12 eval`), escuchala, corregí la referencia si pronunciaste algo distinto y guardá.
- Podés cerrar y continuar después: el progreso se guarda automáticamente en `voices/stt/manifest.json`.
- Las grabaciones se convierten a **PCM WAV mono, 16 kHz, 16 bits** con `ffmpeg` sin reducción de ruido ni recorte automático de pausas.

## 2. Benchmark escalonado (cuando tengas `voices/stt/manifest.json`)

```bash
# Etapa A (Prompt: A vs B)
node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --stage A --split tune --json tmp/stt-stage-a.json

# Etapa B (Hotwords: B vs C)
node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --stage B --split tune --json tmp/stt-stage-b.json

# Etapa C (VAD: C=0.8, D=0.65, E=0.5, F=0.3)
node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --stage C --split tune --json tmp/stt-stage-c.json

# Etapa D (Beam size 5, 3, 1 usando el VAD ganador de Etapa C, ej. 0.65)
node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --profiles C,G,H --vad-threshold 0.65 --split tune --json tmp/stt-stage-d.json

# Etapa E (Segunda inferencia ONLY_NAME)
node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --profiles A --second-pass always_only_name --split tune
node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --profiles A --second-pass none --split tune
node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --profiles A --second-pass confidence --split tune

# Etapa F (Comparación raw vs filtros) -> revisar rawWer vs wer y discardReason en el CSV/JSON
node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --profiles A --split tune --csv tmp/stt-stage-f.csv

# Etapa G (Evaluación final congelada sobre eval: baseline A vs candidato ganador)
node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --profiles A,G --vad-threshold 0.65 --split eval --repeat 3 --json tmp/stt-eval.json

# Benchmark de segmentación Discord (400, 600, 800, 900, 1000, 1200 ms)
node scripts/benchmark-segmentation.mjs
```
