import { createInterface } from "node:readline";

const protocol = process.env.FIXTURE_PROTOCOL || "agy";
const behavior = process.env.FIXTURE_BEHAVIOR || "reply";
const delay = Number(process.env.FIXTURE_DELAY_MS || 5);
let busy = false;

const send = value => process.stdout.write(JSON.stringify(value) + "\n");

function promptFrom(line) {
  const event = JSON.parse(line);
  return protocol === "claude"
    ? event.message?.content || ""
    : event.message?.content || "";
}

if (protocol === "codex" || protocol === "opencode") {
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
    if (behavior === "malformed") {
      process.stdout.write("not-json\n");
      return;
    }
    const text = `reply:${input}`;
    if (protocol === "codex")
      send({ type: "item.completed", item: { type: "agent_message", text } });
    else
      send({ type: "text", part: { type: "text", text } });
  });
} else {

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
