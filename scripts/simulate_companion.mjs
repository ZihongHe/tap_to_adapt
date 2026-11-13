/** @file Command-line mobile-companion simulator for local protocol and transport checks. */
const [code, dayValue = "1", durationValue = "600", baseUrl = "http://127.0.0.1:8765"] = process.argv.slice(2);

if (!code) {
  throw new Error("Usage: node simulate_companion.mjs CODE DAY [SECONDS] [BASE_URL]");
}

const day = Number(dayValue);
const durationSeconds = Number(durationValue);
if (![1, 2, 3].includes(day) || !Number.isFinite(durationSeconds) || durationSeconds <= 0) {
  throw new Error("DAY must be 1-3 and SECONDS must be positive");
}

async function request(path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail?.message || payload.detail || response.statusText);
  return payload;
}

const login = await request("/api/login", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ code, role: "companion" }),
});
const authorization = `Bearer ${login.token}`;
const bootstrap = await request("/api/bootstrap", { headers: { authorization } });
if (!bootstrap.pilot_only) {
  throw new Error("Synthetic companion is restricted to pilot or simulation accounts");
}
const session = await request("/api/sessions/start", {
  method: "POST",
  headers: { authorization, "content-type": "application/json" },
  body: JSON.stringify({ day }),
});

const socketUrl = new URL(baseUrl);
socketUrl.protocol = socketUrl.protocol === "https:" ? "wss:" : "ws:";
socketUrl.pathname = "/ws/pair";
socketUrl.search = new URLSearchParams({ token: login.token, session_id: session.id }).toString();
const socket = new WebSocket(socketUrl);

await new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error("Companion WebSocket timed out")), 10_000);
  socket.addEventListener("open", () => {
    clearTimeout(timer);
    resolve();
  }, { once: true });
  socket.addEventListener("error", () => {
    clearTimeout(timer);
    reject(new Error("Companion WebSocket failed"));
  }, { once: true });
});

function heartbeat() {
  if (socket.readyState !== WebSocket.OPEN) return;
  const clientMs = Date.now();
  socket.send(JSON.stringify({ type: "heartbeat", client_ms: clientMs }));
  socket.send(JSON.stringify({
    type: "background_state",
    action: "ready",
    ready: true,
    synthetic: true,
    client_ms: clientMs,
  }));
}

heartbeat();
const interval = setInterval(heartbeat, 5_000);
console.log(JSON.stringify({ ready: true, session_id: session.id, day }));
await new Promise((resolve) => setTimeout(resolve, durationSeconds * 1_000));
clearInterval(interval);
socket.close(1000, "simulation complete");
