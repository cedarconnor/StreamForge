function config() {
  return {
    source_type: document.getElementById("sourceType").value,
    source_name: document.getElementById("sourceName").value,
    in_res: document.getElementById("inRes").value,
    prompt: document.getElementById("prompt").value,
    preset: document.getElementById("preset").value,
    sink: "null",
    mode: "img2img",
    fps: 30,
    seconds: 0
  };
}

function showMessage(value) {
  document.getElementById("messages").textContent =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function setPreviewAspect(elementId, width, height) {
  if (!width || !height) return;
  document.getElementById(elementId).style.setProperty("--preview-ratio", `${width} / ${height}`);
}

function setCropOverlay(elementId, cropDirection) {
  const box = document.getElementById(elementId);
  box.classList.toggle("crop", cropDirection && cropDirection !== "none");
  box.classList.toggle("sides", cropDirection === "sides");
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`${url} failed: ${res.status}`);
  return res.json();
}

async function validate() {
  const data = await postJson("/api/validate", config());
  showMessage(data);
  if (data.ok && data.source) {
    document.getElementById("inputMeta").textContent = `${data.source.width}x${data.source.height}`;
    setPreviewAspect("inputBox", data.source.width, data.source.height);
  }
  if (data.ok && data.aspect?.internal) {
    const internal = data.aspect.internal;
    document.getElementById("outputMeta").textContent = `${internal.width}x${internal.height}`;
    setPreviewAspect("outputBox", internal.width, internal.height);
    setCropOverlay("outputBox", data.aspect.crop_direction);
  }
}

async function start() {
  await postJson("/api/run/start", config());
  showMessage("started");
  await refreshStatus();
}

async function stop() {
  await postJson("/api/run/stop", {});
  showMessage("stopped");
  await refreshStatus();
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  document.getElementById("state").textContent = data.running ? "running" : "idle";
  document.getElementById("emitted").textContent = data.emitted ?? 0;
  document.getElementById("repeats").textContent = data.repeats ?? 0;
  document.getElementById("filled").textContent = data.filled ?? 0;
  document.getElementById("jitter").textContent = `${(data.jitter_ms ?? 0).toFixed(2)} ms`;
  document.getElementById("infer").textContent =
    data.infer_ms_last == null ? "none" : `${data.infer_ms_last.toFixed(1)} ms`;
  const stamp = Date.now();
  document.getElementById("inputPreview").src = `/preview/input.jpg?t=${stamp}`;
  document.getElementById("outputPreview").src = `/preview/output.jpg?t=${stamp}`;
}

document.getElementById("validate").addEventListener("click", () => validate().catch(err => showMessage(err.message)));
document.getElementById("start").addEventListener("click", () => start().catch(err => showMessage(err.message)));
document.getElementById("stop").addEventListener("click", () => stop().catch(err => showMessage(err.message)));
window.setInterval(() => refreshStatus().catch(() => {}), 1000);
refreshStatus().catch(() => {});
