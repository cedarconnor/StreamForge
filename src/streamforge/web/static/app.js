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
    seconds: 0,
    backend: document.getElementById("backend").value,
    cached_blocks: +document.getElementById("sCached").value,
    sink_token: document.getElementById("sSink").checked
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

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

async function postControl(patch) {
  try { await postJson("/api/control", patch); } catch (e) { showMessage(e.message); }
}
const postControlDebounced = debounce(postControl, 150);

function denoiseFor(ref) {
  // mirrors control.py: _lerp(DENOISE_MAX=0.95, DENOISE_MIN=0.30, ref)
  return (0.95 + (0.30 - 0.95) * ref).toFixed(2);
}

function seedControls(control) {
  if (!control) return;
  if (control.backend === "sana_streaming") {
    document.getElementById("sSteps").value = control.step;
    document.getElementById("sFlow").value = control.flow_shift;
    document.getElementById("sFlowVal").textContent = (+control.flow_shift).toFixed(1);
    document.getElementById("sMotion").value = control.motion_score;
    document.getElementById("sMotionVal").textContent = control.motion_score;
    document.getElementById("sSeed").value = control.seed;
    document.getElementById("sCached").value = control.num_cached_blocks;
    document.getElementById("sCachedVal").textContent = control.num_cached_blocks;
    document.getElementById("sSink").checked = control.sink_token;
    if (control.prompt) document.getElementById("sPrompt").value = control.prompt;
    return;
  }
  const ref = document.getElementById("refStrength");
  ref.value = control.ref_strength;
  document.getElementById("refVal").textContent = (+control.ref_strength).toFixed(2);
  document.getElementById("denoiseVal").textContent = `denoise ${denoiseFor(+control.ref_strength)}`;
  document.getElementById("textMag").value = control.text_magnitude;
  document.getElementById("tmVal").textContent = (+control.text_magnitude).toFixed(2);
  document.getElementById("steps").value = control.steps;
  document.getElementById("stepsVal").textContent = control.steps;
  document.getElementById("seed").value = control.seed;
  document.getElementById("livePrompt").value = control.prompt;
  document.getElementById("liveMode").value = control.mode;
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
  const res = await fetch("/api/status");
  const data = await res.json();
  if (data.control) seedControls(data.control);
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
  const backend = (data.control && data.control.backend) || document.getElementById("backend").value;
  const isSana = backend === "sana_streaming";
  document.getElementById("live").disabled = !data.running || isSana;
  document.getElementById("sanaLive").disabled = !data.running || !isSana;
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

const refEl = document.getElementById("refStrength");
refEl.addEventListener("input", () => {
  const v = +refEl.value;
  document.getElementById("refVal").textContent = v.toFixed(2);
  document.getElementById("denoiseVal").textContent = `denoise ${denoiseFor(v)}`;
  postControlDebounced({ ref_strength: v });
});

const tmEl = document.getElementById("textMag");
tmEl.addEventListener("input", () => {
  const v = +tmEl.value;
  document.getElementById("tmVal").textContent = v.toFixed(2);
  postControlDebounced({ text_magnitude: v });
});

const stepsEl = document.getElementById("steps");
stepsEl.addEventListener("input", () => {
  const v = +stepsEl.value;
  document.getElementById("stepsVal").textContent = v;
  postControlDebounced({ steps: v });
});

document.getElementById("reroll").addEventListener("click", () => {
  const v = Math.floor(Math.random() * 1e9);
  document.getElementById("seed").value = v;
  postControl({ seed: v });
});

document.getElementById("applyPrompt").addEventListener("click", () => {
  postControl({ prompt: document.getElementById("livePrompt").value });
});

document.getElementById("liveMode").addEventListener("change", () => {
  postControl({ mode: document.getElementById("liveMode").value });
});

// --- SANA backend panel ---
function updatePanels() {
  const isSana = document.getElementById("backend").value === "sana_streaming";
  document.getElementById("live").style.display = isSana ? "none" : "";
  document.getElementById("sanaLive").style.display = isSana ? "" : "none";
}
document.getElementById("backend").addEventListener("change", updatePanels);
updatePanels();

// HOT knobs (apply live, no reset)
const sStepsEl = document.getElementById("sSteps");
sStepsEl.addEventListener("change", () => postControl({ steps: +sStepsEl.value }));
const sFlowEl = document.getElementById("sFlow");
sFlowEl.addEventListener("input", () => {
  document.getElementById("sFlowVal").textContent = (+sFlowEl.value).toFixed(1);
  postControlDebounced({ flow_shift: +sFlowEl.value });
});
const sMotionEl = document.getElementById("sMotion");
sMotionEl.addEventListener("input", () => {
  document.getElementById("sMotionVal").textContent = sMotionEl.value;
  postControlDebounced({ motion_score: +sMotionEl.value });
});
document.getElementById("sReroll").addEventListener("click", () => {
  const v = Math.floor(Math.random() * 1e9);
  document.getElementById("sSeed").value = v;
  postControl({ seed: v });
});
// WARM knobs (trigger a state reset -> brief resync flash)
const sCachedEl = document.getElementById("sCached");
sCachedEl.addEventListener("input", () => {
  document.getElementById("sCachedVal").textContent = sCachedEl.value;
  postControlDebounced({ num_cached_blocks: +sCachedEl.value });
});
document.getElementById("sSink").addEventListener("change", () =>
  postControl({ sink_token: document.getElementById("sSink").checked }));
document.getElementById("sApplyPrompt").addEventListener("click", () =>
  postControl({ prompt: document.getElementById("sPrompt").value }));
