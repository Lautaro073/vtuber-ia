# NYRA AI Provider Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Codex CLI and OpenCode as NYRA text engines and let the panel enable or disable all four CLI providers while actually starting and stopping their processes.

**Architecture:** Move CLI process handling behind a provider registry that owns enabled state, model catalogs, client creation, fallback eligibility, persistence, and shutdown. Existing Antigravity/Claude transports remain persistent; Codex/OpenCode use isolated one-shot child processes. `server.js` asks the registry for every role so a disabled provider can never be restarted indirectly.

**Tech Stack:** Node.js ESM, `node:test`, Express, browser-native HTML/CSS/JavaScript, Windows child processes.

**Spec:** `docs/superpowers/specs/2026-11-25-nyra-ai-providers-design.md`

## Global Constraints

- Do not use Docker, Docker Compose, Docker Desktop, or local Supabase.
- Preserve all pre-existing uncommitted edits, especially the current changes in `server.js` and `public/avatar3d/vrm.js`.
- A disabled provider must have zero live child processes and must not be eligible for role creation or fallback.
- Codex/OpenCode are text-only in this change; do not advertise them in the vision selector.
- First run enables only `NYRA_ENGINE`, or the first installed provider if that value is invalid.
- Persist runtime choices in `nyra_provider_settings.json`, ignored by Git.
- Before any commit, inspect `git diff --cached` and exclude pre-existing user changes; if a clean partial stage is not possible, leave the task uncommitted and report why.

## Review Focus

- A provider is disabled while a response is in flight: the child exits, the request settles with `ProviderDisabledError`, and no warm replacement starts.
- Enable and disable requests cross in flight: the last requested state wins and stale availability checks cannot resurrect the provider.
- The active provider disappears or fails discovery: NYRA selects the first enabled, available text model without enabling anything implicitly.
- A corrupt or partially written settings file: startup falls back to the configured engine and rewrites valid state without crashing.
- Codex/OpenCode emit malformed JSON or nonzero exit codes: the request returns a bounded error, child tracking is cleared, and other enabled providers remain usable.

---

## File Structure

- Create `lib/ai/provider-registry.js`: provider state machine, persistence, client ownership, model selection, and fallback filtering.
- Create `lib/ai/cli-clients.js`: common client contract plus Antigravity, Claude, Codex, and OpenCode process adapters.
- Create `lib/ai/model-catalog.js`: model-ID parsing/normalization and provider-specific model discovery.
- Create `tests/provider-registry.test.js`: lifecycle, persistence, race, active-model, and fallback tests.
- Create `tests/cli-clients.test.js`: child-process protocol, cancellation, malformed output, and cleanup tests.
- Create `tests/model-catalog.test.js`: prefixed/legacy IDs and discovery fallback tests.
- Create `tests/fixtures/cli-fixture.mjs`: deterministic persistent and one-shot JSONL child used by client tests.
- Modify `server.js`: construct the registry, route every NYRA role through it, add provider API, and remove direct hardcoded starts.
- Modify `public/panel.html`: provider checkboxes/status rows, model grouping/filtering, and disabled UI state.
- Modify `.gitignore`: ignore `nyra_provider_settings.json`.
- Modify `.env.example`: document `CODEX_PATH`, `OPENCODE_PATH`, and the four valid `NYRA_ENGINE` values.
- Modify `package.json`: add `test` and focused test scripts using `node --test`.

### Task 1: Model IDs and catalog discovery

**Files:**
- Create: `lib/ai/model-catalog.js`
- Create: `tests/model-catalog.test.js`
- Modify: `package.json`

**Interfaces:**
- Produces: `parseModelId(value) -> { providerId, model }`, `formatModelId(providerId, model) -> string`, `normalizeModelId(value, legacyProviderId = "agy") -> string`, `discoverProviderModels(definition) -> Promise<string[]>`, and `providerSupports(providerId, capability) -> boolean`.
- Provider prefixes: `agy`, `claude-code`, `codex-cli`, `opencode`.

- [ ] **Step 1: Write failing tests for IDs and legacy compatibility**

Add tests named `parses all four provider prefixes`, `normalizes an unprefixed legacy agy model`, `keeps provider/model slashes inside an OpenCode model`, and `rejects unknown prefixes`. Assert that `opencode:anthropic/claude-sonnet-4` preserves the full value after the first colon.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pnpm exec node --test tests/model-catalog.test.js`

Expected: FAIL because `lib/ai/model-catalog.js` does not exist.

- [ ] **Step 3: Implement model-ID helpers and capability metadata**

Implement the exact exports above. Text capability is true for all four providers; vision capability is true only for `agy` and `claude-code`.

- [ ] **Step 4: Write failing discovery tests**

Cover: visible Codex entries from a supplied `models_cache.json`, hidden Codex entries excluded, `codex-cli:default` always present, OpenCode tab/newline output converted to prefixed IDs, and discovery failure returning the provider default instead of throwing.

- [ ] **Step 5: Run the discovery tests and verify RED**

Run: `pnpm exec node --test tests/model-catalog.test.js`

Expected: FAIL on missing `discoverProviderModels` behavior.

- [ ] **Step 6: Implement discovery**

Use injected `execFile` and filesystem readers so tests do not call paid models. Antigravity runs `agy models`; Claude uses the existing static catalog; Codex reads `${CODEX_HOME || USERPROFILE/.codex}/models_cache.json` and filters `visibility === "hide"`; OpenCode runs `opencode models` and falls back to `opencode:default` on the current plugin-activation failure.

- [ ] **Step 7: Add package test commands and verify GREEN**

Add `"test": "node --test tests/*.test.js"` and `"test:providers": "node --test tests/model-catalog.test.js tests/provider-registry.test.js tests/cli-clients.test.js"`.

Run: `pnpm test -- tests/model-catalog.test.js`

Expected: all catalog tests PASS.

- [ ] **Step 8: Commit the isolated files if staging is clean**

Commit message: `feat: add AI provider model catalog`

### Task 2: Provider registry lifecycle and persistence

**Files:**
- Create: `lib/ai/provider-registry.js`
- Create: `tests/provider-registry.test.js`

**Interfaces:**
- Consumes: model helpers from Task 1.
- Produces: `ProviderDisabledError`, `ProviderRegistry`, and registry methods `initialize()`, `snapshot()`, `setEnabled(providerId, enabled)`, `registerClient(providerId, client)`, `createClient(role, modelId, options)`, `selectFallbackModel(excludedModelId, capability)`, `stopProvider(providerId)`, and `stopAll()`.
- A client contract is `{ name, model, run(text, onDelta?, timeoutMs?), stream(text), stop() }`.

- [ ] **Step 1: Write failing initialization and persistence tests**

Assert: only configured `NYRA_ENGINE` starts enabled without a settings file; a corrupt file falls back safely; saved booleans survive a second registry instance; unknown keys are discarded.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pnpm exec node --test tests/provider-registry.test.js`

Expected: FAIL because `ProviderRegistry` does not exist.

- [ ] **Step 3: Implement initialization, atomic persistence, and snapshots**

Write settings through a sibling temporary file followed by rename. `snapshot()` returns each provider's `id`, `label`, `enabled`, `available`, `running`, `status`, `error`, and `models` plus `currentModel`.

- [ ] **Step 4: Write failing process-ownership tests**

Use tracked client doubles only at the external process boundary. Assert that disabling stops every registered role and warm replacement, removes models from enabled catalogs, rejects new clients with `ProviderDisabledError`, and does not allow fallback selection to return the disabled provider.

- [ ] **Step 5: Run the lifecycle tests and verify RED**

Run: `pnpm exec node --test tests/provider-registry.test.js`

Expected: FAIL on missing stop/guard behavior.

- [ ] **Step 6: Implement client ownership and fallback filtering**

Mark the provider disabled before stopping clients. Remove stopped clients from ownership sets. `createClient` must check enabled state both before and after asynchronous factory work.

- [ ] **Step 7: Write and pass race/selection tests**

Add tests for disable-during-run, crossed enable/disable requests where the last call wins, active provider shutdown selecting the first enabled provider in order, and all providers off producing `currentModel: null` with status `IA apagada`.

Run: `pnpm exec node --test tests/provider-registry.test.js`

Expected: all registry tests PASS with no unhandled rejections.

- [ ] **Step 8: Commit the isolated files if staging is clean**

Commit message: `feat: manage AI provider lifecycle`

### Task 3: Four CLI client adapters

**Files:**
- Create: `lib/ai/cli-clients.js`
- Create: `tests/cli-clients.test.js`
- Create: `tests/fixtures/cli-fixture.mjs`

**Interfaces:**
- Consumes: the client contract from Task 2.
- Produces: `AgyClient`, `ClaudeClient`, `CodexClient`, `OpenCodeClient`, and `createClientFactory(config)`.
- All clients expose the existing `run`, `stream`, and `stop` surface so `NyraAI.chatStream` does not need provider-specific branches for text.

- [ ] **Step 1: Write failing persistent-client compatibility tests**

Drive `AgyClient` and `ClaudeClient` against `cli-fixture.mjs`; assert streamed deltas, one-job-at-a-time queuing, timeout cleanup, warm-process stop, and the existing event shapes.

- [ ] **Step 2: Run and verify RED**

Run: `pnpm exec node --test tests/cli-clients.test.js --test-name-pattern="Agy|Claude|persistent"`

Expected: FAIL because the exported clients do not exist.

- [ ] **Step 3: Move the existing Antigravity/Claude implementation into the module**

Parameterize CLI paths, work directory, timeouts, turn limits, environment, and system prompt. Preserve current flags and streaming behavior exactly.

- [ ] **Step 4: Write failing Codex/OpenCode adapter tests**

Fixture modes must cover Codex `item.completed/agent_message`, OpenCode `part.type === "text"`, malformed JSON lines, nonzero exit, long prompts delivered through stdin, `stop()` during a run, and child removal after exit.

- [ ] **Step 5: Run the one-shot tests and verify RED**

Run: `pnpm exec node --test tests/cli-clients.test.js --test-name-pattern="Codex|OpenCode|one-shot"`

Expected: FAIL because one-shot adapters are missing.

- [ ] **Step 6: Implement Codex and OpenCode clients**

Codex command: `codex exec - --json --ephemeral --skip-git-repo-check -C <isolated-work-dir> -s read-only`, plus `-m <model>` unless model is `default`. OpenCode command: `opencode run --format json --standalone`, plus `--model <model>` unless default. Send the full prompt on stdin, parse only assistant text events, cap collected stderr/error text at 500 characters, and kill the exact child on `stop()`.

- [ ] **Step 7: Verify all adapter tests GREEN**

Run: `pnpm exec node --test tests/cli-clients.test.js`

Expected: all client tests PASS and the fixture leaves no child process alive.

- [ ] **Step 8: Commit the isolated files if staging is clean**

Commit message: `feat: add Codex and OpenCode clients`

### Task 4: Route NYRA and the REST API through the registry

**Files:**
- Modify: `server.js:28-45`
- Modify: `server.js:1809-2015`
- Modify: `server.js:2059-2119`
- Modify: `server.js:2267-2354`
- Modify: `server.js:2757-2786`
- Modify: `server.js:3171-3207`
- Create: `tests/server-provider-integration.test.js`

**Interfaces:**
- Consumes: registry, catalog, and clients from Tasks 1-3.
- Produces: `GET /api/status` provider snapshot and `POST /api/providers/:providerId` with `{ enabled: boolean }`.
- The POST response is `{ ok, providers, models, currentModel, visionModel }`; invalid IDs/booleans return 400, unavailable enable returns 409.

- [ ] **Step 1: Write failing server-controller tests**

Test an exported `createProviderController({ registry, ai })` without opening a port. Cover valid enable/disable, invalid body, unavailable CLI, disabling the active provider, and all-off status.

- [ ] **Step 2: Run and verify RED**

Run: `pnpm exec node --test tests/server-provider-integration.test.js`

Expected: FAIL because the controller and endpoints are absent.

- [ ] **Step 3: Construct and initialize the registry in `server.js`**

Map `NYRA_ENGINE` legacy values `agy`, `claude`, `codex`, and `opencode` to provider IDs. Add `CODEX_PATH || "codex"`, `OPENCODE_PATH || "opencode"`, and the settings-file path. Replace direct client constructors with registry-owned clients.

- [ ] **Step 4: Route every role through registry guards**

Cover `llm`, `llmAhead`, `aux`, `vision`, `companion`, `judge`, `scribe`, and `learner`. When a preferred provider is disabled, create the role on the current enabled provider. Update image handling to branch on capability rather than `instanceof ClaudeClient`.

- [ ] **Step 5: Restrict fallback and model switching**

Replace `useClaude/useAgy` with `useModel(modelId)`. `chatStream` asks `selectFallbackModel` and never references a disabled provider. Switching to a model from a disabled provider returns 409 instead of enabling it.

- [ ] **Step 6: Add provider endpoints and enriched status**

Keep legacy unprefixed Antigravity requests accepted through normalization. Ensure the text-model list contains only enabled/available providers and the vision list contains only enabled/available vision-capable providers.

- [ ] **Step 7: Remove unconditional startup processes**

Warm only roles belonging to enabled providers. If all are off, start the HTTP/WS server normally and log `IA apagada`. Replace SIGINT's direct list with `registry.stopAll()` so newly added clients cannot become orphans.

- [ ] **Step 8: Verify controller tests and syntax**

Run: `pnpm exec node --test tests/server-provider-integration.test.js`

Run: `node --check server.js`

Expected: tests PASS and syntax check exits 0.

- [ ] **Step 9: Inspect the `server.js` diff against the pre-task snapshot**

Confirm all existing filter, goal, ignore-user, refusal, and avatar changes remain intact. Do not stage unrelated lines.

- [ ] **Step 10: Commit only clean feature hunks if possible**

Commit message: `feat: route NYRA through selectable AI providers`

### Task 5: Panel controls and persisted configuration docs

**Files:**
- Modify: `public/panel.html:128-170`
- Modify: `public/panel.html:542-552`
- Modify: `public/panel.html:859-870`
- Modify: `public/panel.html:980-1005`
- Modify: `.gitignore`
- Modify: `.env.example`
- Create: `tests/panel-provider-controls.test.js`

**Interfaces:**
- Consumes: Task 4 status and provider endpoints.
- Produces: DOM IDs `providerControls`, `provider-<id>`, `provider-status-<id>` and browser functions `renderProviders(data)`, `toggleProvider(id, enabled)`, and `refreshModelSelectors(data)`.

- [ ] **Step 1: Write failing panel contract tests**

Read `public/panel.html` as text and assert the four provider rows, accessible checkbox labels, status nodes, endpoint call, rollback on non-OK response, and disabled model controls when `models.length === 0`.

- [ ] **Step 2: Run and verify RED**

Run: `pnpm exec node --test tests/panel-provider-controls.test.js`

Expected: FAIL because provider controls are not present.

- [ ] **Step 3: Add provider rows and status styling**

Place rows above the text selector. Use the existing cyber-panel palette and preserve narrow-screen layout. Checkbox labels must remain clickable and keyboard-accessible.

- [ ] **Step 4: Implement toggle behavior**

Disable the changed checkbox during the request. On success, re-render from the returned snapshot and rebuild both selectors; on failure, restore server state and show the returned reason in the row and toast.

- [ ] **Step 5: Update status/model rendering**

Group visible text options by provider labels. Populate vision from `visionModels`, not the text list. Show `IA apagada` and disable change buttons when no provider is active.

- [ ] **Step 6: Add configuration documentation**

Ignore `/nyra_provider_settings.json`. Document `NYRA_ENGINE=agy|claude|codex|opencode`, `CODEX_PATH`, and `OPENCODE_PATH` in `.env.example` without changing the user's actual `.env`.

- [ ] **Step 7: Verify panel tests GREEN**

Run: `pnpm exec node --test tests/panel-provider-controls.test.js`

Expected: all panel contract tests PASS.

- [ ] **Step 8: Commit only clean feature files/hunks if possible**

Commit message: `feat: add provider process controls to panel`

### Task 6: End-to-end verification and handoff

**Files:**
- Modify if needed: files from Tasks 1-5 only.

**Interfaces:**
- Consumes: complete feature.
- Produces: verified implementation with no orphan CLI processes.

- [ ] **Step 1: Run the complete automated suite**

Run: `pnpm test`

Expected: all tests PASS, zero failures, zero unhandled rejections.

- [ ] **Step 2: Run syntax and startup smoke checks**

Run: `node --check server.js`

Start the server on a temporary non-production port with a temporary settings path, request `/api/status`, then stop it gracefully. Do not connect TikTok, Discord, OBS, or any production service.

Expected: server starts, returns four providers, and exits without live fixture/CLI children.

- [ ] **Step 3: Verify the two installed one-shot CLIs with minimal prompts**

Use a temporary isolated directory and the configured/default model. Send `Respondé solo: ok` once through `CodexClient` and once through `OpenCodeClient`; do not enable tools or point either CLI at the project.

Expected: each returns nonempty assistant text or a bounded provider-specific availability error. An OpenCode plugin activation failure is reported without taking down the server.

- [ ] **Step 4: Verify real process lifecycle from the registry**

With production integrations disconnected, enable one provider, issue one panel test message, disable it, and inspect only child processes created by the test server PID.

Expected: after disable, zero descendant processes for that provider; after re-enable, persistent CLIs warm or one-shot CLIs report `listo bajo demanda`.

- [ ] **Step 5: Review requirements and working-tree scope**

Compare the final diff with the spec. Confirm no Docker use, no `.env` edits, no memory/personality JSON edits caused by tests, and no pre-existing user changes reverted or staged accidentally.

- [ ] **Step 6: Report exact verification evidence**

List commands, pass/fail counts, any provider unavailable on the machine, files changed, and whether commits were intentionally skipped due to the dirty `server.js`.
