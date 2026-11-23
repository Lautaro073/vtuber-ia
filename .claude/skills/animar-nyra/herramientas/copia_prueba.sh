#!/usr/bin/env bash
# Copia aislada del server de NYRA para probar animaciones sin tocar el server del directo (puerto 3000).
#   bash copia_prueba.sh crear <carpeta_temporal>   → copia en <carpeta>/copia, puerto 3010, ganchos de prueba en el visor
#                                                     y entrada "nyra-copia" en .claude/launch.json (respaldo en launch.bak)
#   bash copia_prueba.sh restaurar-launch <carpeta_temporal> → deja launch.json como estaba (el server ya arrancado sigue)
#   bash copia_prueba.sh arrancar <carpeta_temporal> → levanta la copia en http://localhost:3010 en segundo plano
#                                                     (para agentes sin preview_start: Antigravity, Codex, etc.)
#   bash copia_prueba.sh parar <carpeta_temporal>    → apaga SOLO ese server (el PID que guardó arrancar)
#   bash copia_prueba.sh limpiar <carpeta_temporal> → restaura launch.json y borra la copia (sin tocar node_modules)
# Claude Code: crear → preview_start "nyra-copia" → restaurar-launch → abrir /obs?3d → … → preview_stop → limpiar
# Otros agentes: crear → restaurar-launch → arrancar → abrir http://localhost:3010/obs?3d en su navegador → … → parar → limpiar
# Nunca apagar ni reiniciar el server del puerto 3000: es el del directo.
# La copia no tiene .env: no se conecta a TikTok ni a Discord. No lleva PANEL_PASSWORD real (usa una de prueba).
#
# Ganchos que quedan en la página (solo en la copia):
#   __G (GESTURES), __hand, __GRIP, __scene, __cam, __aim, __THREE, y walkP para congelar la caminata.
#   Congelar un cuadro:  __G.t1 = { ...G, dur: 999, clip: () => G.clip(t, 1) }; nyra3d.gesture('t1')
#   Posición de un hueso: __scene.getObjectByName('J_Bip_R_Middle1').getWorldPosition(new __THREE.Vector3())
set -euo pipefail
cmd="${1:?crear | limpiar}"; tmp="${2:?carpeta temporal}"
root="$(cd "$(dirname "$0")/../../../.." && pwd)"
copy="$tmp/copia"

case "$cmd" in
  crear)
    [ -e "$copy" ] && { echo "ya existe $copy (corré limpiar primero)"; exit 1; }
    mkdir -p "$copy"
    (cd "$root" && cp server.js discord-voice.js research.js follow-up.js memoria.js package.json datos_curiosos.json \
      filters.json knowledge.json nyra_chat_learning.json nyra_memory.json nyra_personality.json "$copy/" && cp -r public "$copy/public")
    powershell -NoProfile -Command "New-Item -ItemType Junction -Path '$(cygpath -w "$copy/node_modules")' -Target '$(cygpath -w "$root/node_modules")' | Out-Null"
    v="$copy/public/avatar3d/vrm.js"
    python -X utf8 - "$v" <<'EOF'
import sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
for a, b in [("  camera.lookAt(aim);\n", "  camera.lookAt(aim); globalThis.__cam = camera; globalThis.__aim = aim; globalThis.__THREE = THREE;\n"),
             ("  scene.add(light);\n", "  scene.add(light); globalThis.__scene = scene;\n"),
             ("? walkOff(p)", "? walkOff(g.walkP ? g.walkP(p) : p)"),
             ("walkPose(p) : {}", "walkPose(g.walkP ? g.walkP(p) : p) : {}")]:
    if a in s:
        s = s.replace(a, b, 1)
s += "\nglobalThis.__G = GESTURES; globalThis.__hand = hand; globalThis.__GRIP = GRIP;\n"
open(p, 'w', encoding='utf-8').write(s)
EOF
    node --check "$v"
    cp "$root/.claude/launch.json" "$tmp/launch.bak"
    python -X utf8 - "$root/.claude/launch.json" "$(cygpath -m "$copy")" <<'EOF'
import json, sys
p, copy = sys.argv[1], sys.argv[2]
j = json.load(open(p, encoding='utf-8'))
j['configurations'] = [c for c in j['configurations'] if c.get('name') != 'nyra-copia']
j['configurations'].append({"name": "nyra-copia", "runtimeExecutable": "node", "runtimeArgs": [copy + "/server.js"], "port": 3010,
  "env": {"PORT": "3010", "PANEL_PASSWORD": "prueba-local-3d", "MURF_API_KEY": "", "PROMO_MINUTES": "0", "IDLE_FACT_SECONDS": "0", "NYRA_RESEARCH": "0"}})
json.dump(j, open(p, 'w', encoding='utf-8'), indent=2)
EOF
    echo "copia lista en $copy — preview_start nyra-copia y después: bash $0 restaurar-launch $tmp"
    ;;
  restaurar-launch)
    cp "$tmp/launch.bak" "$root/.claude/launch.json" && echo "launch.json restaurado"
    ;;
  arrancar)
    [ -f "$copy/server.js" ] || { echo "no hay copia en $copy (corré crear)"; exit 1; }
    if (echo > /dev/tcp/127.0.0.1/3010) 2>/dev/null; then echo "el puerto 3010 ya está ocupado"; exit 1; fi
    (cd "$copy" && PORT=3010 PANEL_PASSWORD=prueba-local-3d MURF_API_KEY= PROMO_MINUTES=0 IDLE_FACT_SECONDS=0 NYRA_RESEARCH=0 \
      nohup node server.js > "$tmp/server.log" 2>&1 & echo $! > "$tmp/server.pid")
    for _ in $(seq 1 60); do (echo > /dev/tcp/127.0.0.1/3010) 2>/dev/null && break; sleep 0.5; done
    (echo > /dev/tcp/127.0.0.1/3010) 2>/dev/null && echo "copia en http://localhost:3010/obs?3d (log: $tmp/server.log)" \
      || { echo "no arrancó; mirá $tmp/server.log"; exit 1; }
    ;;
  parar)
    # En Windows el PID que da bash no es el de node: se busca quién escucha en 3010 y se verifica antes de apagar.
    # Solo se apaga si es node.exe, escucha en 3010 y NO escucha en 3000 (el server del directo nunca se toca).
    powershell -NoProfile -Command '
      $p = (Get-NetTCPConnection -LocalPort 3010 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
      if (-not $p) { "no hay nada en 3010"; exit 0 }
      $live = (Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
      $name = (Get-Process -Id $p).ProcessName
      if ($name -ne "node" -or $p -eq $live) { "NO lo apago: PID $p ($name) no es la copia de prueba"; exit 1 }
      Stop-Process -Id $p -Force; "server de prueba apagado (PID $p)"'
    rm -f "$tmp/server.pid"
    ;;
  limpiar)
    [ -f "$tmp/launch.bak" ] && cp "$tmp/launch.bak" "$root/.claude/launch.json"
    if [ -e "$copy/node_modules" ]; then cmd //c rmdir "$(cygpath -w "$copy/node_modules")"; fi
    [ -e "$copy/node_modules" ] && { echo "no pude quitar el enlace a node_modules; no borro nada"; exit 1; }
    rm -rf "$copy"
    grep -q nyra-copia "$root/.claude/launch.json" && echo "OJO: launch.json todavía tiene nyra-copia" || echo "limpio"
    ;;
  *) echo "uso: crear | restaurar-launch | arrancar | parar | limpiar <carpeta>"; exit 1 ;;
esac
