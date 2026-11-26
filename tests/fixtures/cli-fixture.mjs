import { createInterface } from "node:readline";
import http from "node:http";

const protocol = process.env.FIXTURE_PROTOCOL || "agy";
const behavior = process.env.FIXTURE_BEHAVIOR || "reply";
const delay = Number(process.env.FIXTURE_DELAY_MS || 5);
let busy = false;

const send = value => process.stdout.write(JSON.stringify(value) + "\n");

if (protocol === "opencode" && process.argv.includes("serve")) {
  const portIdx = process.argv.indexOf("--port");
  const port = portIdx >= 0 ? Number(process.argv[portIdx + 1]) : 0;
  if (behavior === "serve_fail") {
    process.stderr.write("serve failed to bind");
    process.exit(1);
  }
  if (behavior === "serve_hang") {
    setInterval(() => {}, 1000);
  } else {
    const pw = process.env.OPENCODE_SERVER_PASSWORD || "";
    const deleted = new Set();
    const srv = http.createServer((req, res) => {
      const auth = req.headers.authorization || "";
      const expected = "Basic " + Buffer.from(`opencode:${pw}`).toString("base64");
      if (pw && auth !== expected) {
        res.writeHead(401, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Unauthorized" }));
        return;
      }
      if (req.method === "GET" && req.url === "/api/session") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ data: [], deleted: [...deleted] }));
        return;
      }
      if (req.method === "DELETE" && req.url?.startsWith("/api/session/")) {
        const id = decodeURIComponent(req.url.slice("/api/session/".length));
        deleted.add(id);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
        return;
      }
      res.writeHead(404);
      res.end();
    });
    srv.listen(port, "127.0.0.1");
  }
} else if (protocol === "codex" || protocol === "opencode") {
  let input = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", chunk => { input += chunk; });
  process.stdin.on("end", () => {
    if (behavior === "hang") {
      setInterval(() => {}, 1000);
      return;
    }
    if (behavior === "error") {
      process.stderr.write("x".repeat(900));
      process.exitCode = 2;
      return;
    }
    if (behavior === "limit") { // como Codex sin cuota: el motivo va en el JSON y stderr trae ruido de conectores
      send({ type: "error", message: "You've hit your usage limit." });
      process.stderr.write("ERROR rmcp::transport::worker: AuthRequired");
      process.exitCode = 1;
      return;
    }
    if (behavior === "malformed") {
      process.stdout.write("not-json\n");
      return;
    }
    if (behavior === "empty") {
      return;
    }
    if (behavior === "server_down" && process.argv.includes("--server")) {
      send({ type: "error", message: "Could not reach server at http://127.0.0.1:1: ECONNREFUSED" });
      process.exitCode = 1;
      return;
    }
    if (protocol === "opencode" && behavior === "dup_events") {
      send({ type: "text", part: { id: "p1", type: "text", text: "Hola chat" } });
      send({ type: "text", part: { id: "p1", type: "text", text: "Hola chat" } });
      return;
    }
    if (protocol === "opencode" && behavior === "cumulative") {
      send({ type: "text", part: { id: "p1", type: "text", text: "Hola" } });
      send({ type: "text", part: { id: "p1", type: "text", text: "Hola chat" } });
      send({ type: "text", part: { id: "p1", type: "text", text: "Hola chat!" } });
      return;
    }
    if (protocol === "opencode" && behavior === "incremental") {
      send({ type: "text", part: { id: "p1", type: "text", text: "Hola " } });
      send({ type: "text", part: { id: "p1", type: "text", text: "chat!" } });
      return;
    }
    const text = `reply:${input}`;
    if (protocol === "codex")
      send({ type: "item.completed", item: { type: "agent_message", text } });
    else {
      const mode = process.argv.includes("--server") ? "srv" : "one";
      send({ type: "text", sessionID: `ses_nyra_${mode}_${Date.now()}`, part: { type: "text", text } });
    }
  });
} else {
function promptFrom(line) {
  const event = JSON.parse(line);
  return protocol === "claude"
    ? event.message?.content || ""
    : event.message?.content || "";
}

function failBusy() {
  if (protocol === "claude")
    send({ type: "result", subtype: "error", is_error: true, result: "concurrent input" });
  else
    send({ event: "result", result: { status: "ERROR", error: "concurrent input" } });
}

createInterface({ input: process.stdin }).on("line", line => {
  if (behavior === "hang") return;
  if (busy) return failBusy();
  busy = true;
  const prompt = promptFrom(line);
  setTimeout(() => {
    const response = `reply:${prompt}`;
    if (protocol === "claude") {
      send({ type: "stream_event", event: { delta: { type: "text_delta", text: "reply:" } } });
      send({ type: "stream_event", event: { delta: { type: "text_delta", text: prompt } } });
      send({ type: "result", subtype: "success", is_error: false, result: response });
    } else {
      send({ step_update: { step_type: "agent_response", text_delta: "reply:" } });
      send({ step_update: { step_type: "agent_response", text_delta: prompt } });
      send({ event: "result", result: { status: "SUCCESS", response } });
    }
    busy = false;
  }, delay);
});
}
