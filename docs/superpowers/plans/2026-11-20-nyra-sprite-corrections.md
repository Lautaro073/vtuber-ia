# NYRA Sprite Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corregir 20 sprites faciales de NYRA sin alterar ningún píxel fuera de ojos o boca.

**Architecture:** Crear ocho maestros visuales localizados a partir de los PNG actuales y `correcciones.png`, aplicar cada maestro mediante máscaras elípticas fijas sobre copias de los originales y validar invariantes, hashes y consistencia entre variantes. La creación visual usa imagegen; el respaldo, enmascarado y QA usan PowerShell con .NET `System.Drawing`.

**Tech Stack:** PNG RGBA, imagegen integrado, PowerShell 7, .NET `System.Drawing`, SHA-256.

**Spec:** `docs/superpowers/specs/2026-11-20-nyra-sprite-corrections-design.md`

## Global Constraints

- No usar Docker.
- Mantener 1024x1536 y alfa binario 0/255.
- No modificar píxeles fuera de las máscaras de ojos o boca.
- Respaldar los 20 originales antes de reemplazarlos.
- No tocar `sad`, `thinking` ni las bocas `a`, `ei`, `l`, `mbp`, `closed` salvo que un archivo `eye-half` reciba únicamente ojos nuevos.
- Preservar todos los cambios ajenos ya presentes en el worktree.

## Review Focus

- El backup debe contener exactamente los 20 originales y conservar sus hashes previos.
- El enmascarado no puede cambiar pelo, cejas, nariz, mejillas, lunar, cuerpo ni alfa exterior.
- Los ojos `half` deben ser idénticos en las ocho variantes de boca.
- Los visemas `o`, `u` y `fv` deben ser idénticos en los tres estados de ojos.
- Los ojos `angry` y `happy` deben ser idénticos entre sus variantes de boca abierta/cerrada.

---

### Task 1: Preflight y validadores

**Files:**
- Create: `revision-sprites/validate_corrections.ps1`
- Create: `revision-sprites/nyra_sprite_manifest.json`

- [x] **Step 1:** Capturar nombres, hashes y metadatos de los 33 PNG; declarar los 20 objetivos y los 13 protegidos.
- [x] **Step 2:** Implementar validaciones de dimensiones, alfa, backup, igualdad fuera de máscara y consistencia de rasgos.
- [x] **Step 3:** Ejecutar el validador en modo postflight contra el estado actual y comprobar que falla por las correcciones todavía ausentes.

### Task 2: Ocho maestros localizados

**Files:**
- Create: `revision-sprites/_generated_masters/*.png`

- [x] **Step 1:** Generar con imagegen un maestro para ojos `half`, otro para `angry` y otro para `happy`, usando los sprites actuales como objetivos y `correcciones.png` como referencia.
- [x] **Step 2:** Generar maestros para `mouth-o`, `mouth-u`, `mouth-fv`, sorpresa cerrada y sorpresa abierta.
- [x] **Step 3:** Inspeccionar cada maestro y rechazar cualquier resultado que no conserve estilo, registro o la intención del rasgo dentro de la máscara.

### Task 3: Respaldo y aplicación determinista

**Files:**
- Create: `revision-sprites/apply_corrections.ps1`
- Create: `public/avatar/_backup/2026-11-20_before_nyra_face_corrections/*.png`
- Modify: los 20 PNG enumerados por el manifiesto.

- [x] **Step 1:** Ejecutar el script en modo dry-run y verificar que selecciona exactamente 20 archivos.
- [x] **Step 2:** Copiar originales al backup y comprobar hashes.
- [x] **Step 3:** Aplicar ojos con elipses centradas aproximadamente en `(447,203)` y `(530,203)`, radios `(40,27)`, y bocas con elipse centrada en `(489,251)`, radios `(31,20)`.
- [x] **Step 4:** Propagar cada maestro común a todas sus variantes y guardar con los nombres originales.

### Task 4: QA final y revisión visual

**Files:**
- Create: `revision-sprites/correcciones_resultado.png`

- [x] **Step 1:** Ejecutar `validate_corrections.ps1` y exigir PASS para dimensiones, alfa, máscaras, backups, archivos protegidos y consistencia entre variantes.
- [x] **Step 2:** Crear una hoja ampliada con los rasgos finales y revisar parpadeo, angry, happy, visemas y sorpresa.
- [x] **Step 3:** Ejecutar `git status --short`, confirmar que los cambios ajenos siguen intactos y enumerar los 20 PNG reemplazados.
