# Proveedores de IA activables para NYRA

Fecha: 2026-11-25

Estado: pendiente de aprobación

## Intención

Agregar Codex CLI y OpenCode como motores elegibles de NYRA y permitir encender o apagar, desde el panel, Antigravity, Claude Code, Codex CLI y OpenCode. El objetivo principal es que un proveedor desmarcado no conserve ni vuelva a crear procesos, para reducir el consumo de memoria y CPU.

## Comportamiento acordado

- El panel mostrará un checkbox y un estado para cada proveedor: Antigravity, Claude Code, Codex CLI y OpenCode.
- Al desmarcar un proveedor se cancelarán sus trabajos en curso, se cerrarán su proceso activo, procesos de precalentamiento y procesos auxiliares, y sus modelos desaparecerán de los selectores.
- Ningún fallback, tarea de voz, visión, memoria o seguimiento podrá volver a iniciar un proveedor desactivado.
- Al marcarlo se comprobará que el CLI exista y el proveedor volverá a quedar disponible. Antigravity y Claude podrán precalentarse; Codex y OpenCode son ejecuciones por trabajo, así que quedarán en estado “listo, bajo demanda” y arrancarán al recibir el siguiente trabajo.
- Si se desactiva el proveedor del modelo activo, NYRA pasará al primer proveedor habilitado y disponible. El orden inicial será Antigravity, Claude Code, Codex CLI y OpenCode.
- Si no queda ninguno habilitado, NYRA permanecerá encendida pero sin motor de IA: no abrirá procesos, el panel mostrará “IA apagada” y las rutas que necesiten generar texto responderán con un error controlado.
- Los cambios se guardarán en un archivo local ignorado por Git para mantenerse después de reiniciar el servidor.

## Modelos y selección

- Los modelos conservarán un identificador con prefijo de proveedor para evitar ambigüedades: `agy:...`, `claude-code:...`, `codex-cli:...` y `opencode:...`.
- Antigravity seguirá descubriendo modelos mediante `agy models`.
- Claude Code conservará el catálogo que ya ofrece NYRA.
- Codex leerá el catálogo local de modelos cuando esté disponible y siempre tendrá una opción `codex-cli:default`, que respeta el modelo configurado en Codex.
- OpenCode intentará descubrir modelos mediante `opencode models` y siempre tendrá `opencode:default` como respaldo. Si la detección falla, como ocurre actualmente por un plugin de OpenCode que no termina de activarse, el resto de NYRA seguirá funcionando.
- El selector de visión solo mostrará modelos de proveedores que soporten imágenes. La primera implementación conservará la visión actual de Antigravity y Claude; Codex/OpenCode se incorporarán primero al texto para no anunciar capacidad visual sin verificar su formato de imágenes.

## Arquitectura

- Se extraerá del archivo principal un registro de proveedores responsable de disponibilidad, estado habilitado, catálogo de modelos, creación de clientes y cierre de todos los procesos pertenecientes a cada proveedor.
- Todos los clientes tendrán el mismo contrato para ejecutar, transmitir texto y detenerse. Antigravity y Claude mantendrán su transporte persistente; Codex y OpenCode usarán procesos aislados de una sola ejecución con salida JSONL/JSON.
- Codex se ejecutará de forma efímera, en la carpeta temporal aislada de NYRA y con sandbox de solo lectura. OpenCode se ejecutará en modo independiente para que no deje un servidor global en segundo plano. Ambos recibirán las mismas instrucciones de personaje y de no usar herramientas que los motores actuales.
- Chat, respuesta anticipada, voz/grupo, juez, visión, escriba y aprendizaje pedirán clientes al registro. Si su proveedor preferido está apagado, usarán el proveedor activo; jamás crearán directamente un CLI deshabilitado.
- Los fallbacks solo recorrerán proveedores habilitados. Un proveedor desmarcado no será reactivado automáticamente aunque otro falle.
- El endpoint de estado devolverá proveedores, disponibilidad, estado habilitado, estado del proceso y modelos. Un endpoint específico aplicará los cambios de checkbox y devolverá el estado resultante.

## Interfaz

- Los cuatro checks estarán en la sección “Modelo de texto (chat)”, encima del selector.
- Cada fila mostrará nombre, checkbox y estado breve: `activo`, `listo bajo demanda`, `apagado`, `no encontrado` o `error`.
- Al cambiar un check, el panel esperará la respuesta del servidor, actualizará los modelos y mostrará un aviso. Si el CLI no existe o no puede iniciar, el checkbox volverá a desmarcado y mostrará el motivo.
- Los selectores quedarán deshabilitados cuando no haya proveedores activos.

## Persistencia y compatibilidad

- En la primera ejecución sin preferencias guardadas se habilitará únicamente el proveedor indicado por `NYRA_ENGINE`; si no es válido, se elegirá el primero instalado. Así no se abren varios proveedores por defecto.
- Las variables actuales (`NYRA_ENGINE`, `AGY_MODEL`, `CLAUDE_MODEL`, `VISION_MODEL` y rutas de CLI) seguirán funcionando como valores iniciales.
- Se añadirán variables opcionales para las rutas de Codex y OpenCode, sin exigir instalaciones nuevas si ya están en `PATH`.
- No se usará Docker ni se modificará el contenido de memoria, personalidad o conocimiento de NYRA.

## Enfoques considerados

1. **Registro de proveedores con control real del ciclo de vida — elegido.** Los checks cierran procesos, bloquean reinicios indirectos y filtran modelos. Requiere ordenar la creación de clientes que hoy está repartida por `server.js`, pero cumple el ahorro de recursos.
2. **Checks solamente visuales.** Solo ocultaría modelos; los auxiliares podrían seguir abiertos. Se descartó porque no ahorra recursos de forma confiable.
3. **Arrancar siempre los cuatro proveedores y elegir uno por solicitud.** Facilita los fallbacks y reduce latencia, pero produce exactamente el consumo que se quiere evitar.

## Pruebas y aceptación

- Las pruebas automatizadas comprobarán primero que desactivar un proveedor mata todos sus clientes y que ningún rol o fallback puede recrearlo.
- También cubrirán activación, persistencia, cambio automático del modelo activo, estado sin proveedores, filtrado de modelos y fallo de un CLI ausente.
- Se verificará la salida JSON de Codex y OpenCode con procesos cortos, sin Docker y sin probar contra producción.
- Aceptación manual: observar los procesos del sistema al marcar y desmarcar cada proveedor, enviar mensajes desde el panel y reiniciar el servidor para confirmar que se conserva la selección.

## No incluido

- Balanceo simultáneo entre proveedores.
- Instalar, autenticar o comprar planes de los CLI.
- Cambiar las reglas de personalidad de NYRA.
- Habilitar visión en Codex/OpenCode sin una prueba separada de imágenes.
