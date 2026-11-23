# NYRA Eyes Second Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corregir diez sprites de ojos sin modificar ningún píxel no autorizado.

**Architecture:** Capturar hashes del estado actual, crear dos maestros visuales a partir de los sprites aprobados y propagarlos con máscaras oculares fijas. Validar backup, dimensiones, alfa, hashes, límites de edición e igualdad entre variantes.

**Tech Stack:** PNG RGBA, imagegen integrado, Windows PowerShell 5.1, .NET `System.Drawing`, SHA-256.

**Spec:** `docs/superpowers/specs/2026-11-21-nyra-eyes-second-correction-design.md`

## Global Constraints

- No usar Docker.
- Modificar únicamente diez PNG y crear primero `public/avatar/_backup2/`.
- Limitar toda diferencia a ojos dentro de `x=400..585, y=160..235`.
- Mantener `1024x1536`, alfa binario y nombres originales.
- Preservar todos los cambios ajenos presentes en el worktree.

## Review Focus

- El backup debe contener exactamente los diez originales de esta segunda corrección.
- Pelo y flequillo dentro del rectángulo facial no pueden cambiar: solo las dos máscaras elípticas.
- Las bocas deben permanecer idénticas byte visual por byte visual respecto del backup.
- Angry debe tener una sola línea fina por ojo e iris visibles, sin aspecto dormido.
- El parpadeo debe conservar su línea de pestañas y añadir solo sombra tramada.

---

### Task 1: Manifiesto y validador RED

**Files:**
- Create: `revision-sprites/nyra_sprite_manifest_backup2.json`
- Create: `revision-sprites/validate_corrections2.ps1`

- [x] Capturar los hashes de los 33 PNG y marcar diez objetivos.
- [x] Implementar las comprobaciones de backup, archivos protegidos, alfa, dimensiones, máscara e igualdad de ojos.
- [x] Ejecutar Postflight antes de editar y observar el fallo esperado por `_backup2` ausente.

### Task 2: Dos maestros oculares

**Files:**
- Create: `revision-sprites/_generated_masters2/eyes-angry-v2.png`
- Create: `revision-sprites/_generated_masters2/eyes-closed-shadow-v2.png`

- [x] Generar el maestro angry desde el neutral abierto.
- [x] Generar el maestro cerrado desde el cerrado actual usando el neutral abierto como referencia de sombreado.
- [x] Inspeccionar ambos maestros ampliados.

### Task 3: Backup y aplicación

**Files:**
- Create: `revision-sprites/apply_corrections2.ps1`
- Create: `public/avatar/_backup2/*.png`
- Modify: diez PNG objetivo.

- [x] Ejecutar dry-run y exigir exactamente diez objetivos.
- [x] Copiar y verificar los diez originales en `_backup2`.
- [x] Aplicar RGB solo dentro de dos elipses oculares preservando el alfa original.
- [x] Guardar con los nombres originales.

### Task 4: QA final

**Files:**
- Create: `revision-sprites/correcciones2_resultado.png`

- [x] Ejecutar Postflight y exigir PASS en todas las invariantes.
- [x] Crear e inspeccionar un contact sheet antes/después de los diez objetivos.
- [x] Confirmar con `git status --short` que no cambió ningún otro PNG.
