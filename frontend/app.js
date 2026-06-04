/* ── Flux LoRA Studio — frontend logic ─────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
};

const state = { project: null, captionPoll: null, trainPoll: null,
  aspect: { w: 1024, h: 1024 } };

/* ── Toasts & overlay ── */
function toast(msg, type = "") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 4200);
}
const overlay = (show, text = "Working…") => {
  $("overlayText").textContent = text;
  $("overlay").classList.toggle("hidden", !show);
};

/* ── Step navigation ── */
function goStep(n) {
  document.querySelectorAll(".rail-step").forEach((b) =>
    b.classList.toggle("active", b.dataset.step == n));
  document.querySelectorAll(".panel").forEach((p) =>
    p.classList.toggle("active", p.id === `panel-${n}`));
  if (n == 2) loadDataset();
  if (n == 4) { loadLoras(); loadGallery(); }
}
document.querySelectorAll(".rail-step").forEach((b) =>
  b.addEventListener("click", () => goStep(b.dataset.step)));

/* ── Health ── */
async function health() {
  try {
    const h = await api("/api/health");
    const pill = $("gpuPill");
    if (h.gpu.available) {
      pill.classList.add("ok");
      $("gpuText").textContent = `${h.gpu.name} · ${h.gpu.vram_gb}GB · ${h.gpu.compute_capability}`;
    } else {
      pill.classList.add("bad");
      $("gpuText").textContent = "no CUDA GPU";
    }
    if (!h.ai_toolkit_ready) toast("ai-toolkit not found — run runpod/setup.sh", "error");
    if (!h.hf_token_configured) toast("HF_TOKEN not set — needed for FLUX.1-dev", "error");
  } catch (e) { $("gpuText").textContent = "backend offline"; }
}

/* ── Projects ── */
async function loadProjects(select) {
  const projects = await api("/api/projects");
  const sel = $("projectSelect");
  sel.innerHTML = "";
  projects.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = `${p.name}  (${p.image_count} imgs)`;
    sel.appendChild(opt);
  });
  if (projects.length) {
    state.project = select || projects[0].name;
    sel.value = state.project;
    updateProjectMeta(projects.find((p) => p.name === state.project));
  } else {
    state.project = null;
    $("projectMeta").textContent = "no projects yet — create one";
  }
}
function updateProjectMeta(p) {
  if (!p) { $("projectMeta").textContent = ""; return; }
  $("projectMeta").textContent =
    `${p.image_count} images · ${p.captioned_count} captioned · ${p.lora_count} LoRAs`;
}
$("projectSelect").addEventListener("change", (e) => {
  state.project = e.target.value;
  loadThumbs();
});
$("newProjectBtn").addEventListener("click", async () => {
  const name = prompt("New project name (letters, digits, - or _):");
  if (!name) return;
  try {
    await api("/api/projects", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await loadProjects(name);
    loadThumbs();
    toast(`Project “${name}” created`, "success");
  } catch (e) { toast(e.message, "error"); }
});

/* ── Step 1: dataset upload ── */
const dz = $("dropzone");
$("browseBtn").addEventListener("click", (e) => { e.stopPropagation(); $("fileInput").click(); });
dz.addEventListener("click", () => $("fileInput").click());
["dragover", "dragenter"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));
$("fileInput").addEventListener("change", (e) => uploadFiles(e.target.files));

async function uploadFiles(fileList) {
  if (!state.project) { toast("Create or select a project first", "error"); return; }
  const files = [...fileList].filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  overlay(true, `Uploading ${files.length} image(s)…`);
  try {
    const r = await api(`/api/projects/${state.project}/images`, { method: "POST", body: fd });
    toast(`Added ${r.processed} image(s)` + (r.skipped_small ? `, skipped ${r.skipped_small} too small` : ""), "success");
    (r.warnings || []).slice(0, 3).forEach((w) => toast(w, "error"));
    await loadThumbs();
    await loadProjects(state.project);
  } catch (e) { toast(e.message, "error"); }
  finally { overlay(false); }
}

async function loadThumbs() {
  if (!state.project) { $("thumbGrid").innerHTML = ""; $("toCaptionBtn").disabled = true; return; }
  const data = await api(`/api/projects/${state.project}/dataset`);
  const grid = $("thumbGrid");
  grid.innerHTML = "";
  data.items.forEach((it) => {
    const div = document.createElement("div");
    div.className = "thumb";
    div.innerHTML =
      `<img src="/datasets/${state.project}/${it.image}" loading="lazy" />` +
      `<button class="thumb-del" data-img="${it.image}">✕</button>`;
    grid.appendChild(div);
  });
  grid.querySelectorAll(".thumb-del").forEach((b) =>
    b.addEventListener("click", async () => {
      await api(`/api/projects/${state.project}/images/${b.dataset.img}`, { method: "DELETE" });
      loadThumbs(); loadProjects(state.project);
    }));
  $("toCaptionBtn").disabled = data.items.length === 0;
}
$("toCaptionBtn").addEventListener("click", () => goStep(2));

/* ── Step 2: captioning ── */
async function loadDataset() {
  if (!state.project) return;
  const data = await api(`/api/projects/${state.project}/dataset`);
  const list = $("captionList");
  list.innerHTML = "";
  let allCaptioned = data.items.length > 0;
  data.items.forEach((it) => {
    if (!it.caption) allCaptioned = false;
    const row = document.createElement("div");
    row.className = "caption-item";
    row.innerHTML =
      `<img src="/datasets/${state.project}/${it.image}" />` +
      `<textarea data-img="${it.image}" rows="3" placeholder="caption…">${it.caption}</textarea>`;
    list.appendChild(row);
  });
  list.querySelectorAll("textarea").forEach((ta) =>
    ta.addEventListener("blur", async () => {
      await api(`/api/projects/${state.project}/caption/${ta.dataset.img}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caption: ta.value }),
      });
    }));
  $("toTrainBtn").disabled = !allCaptioned;
}
$("captionBtn").addEventListener("click", async () => {
  if (!state.project) return;
  const trigger = $("triggerInput").value.trim();
  $("trainTrigger").value = trigger;
  try {
    await api(`/api/projects/${state.project}/caption`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trigger_word: trigger, overwrite: $("overwriteCaptions").checked }),
    });
    $("captionProgress").classList.remove("hidden");
    pollCaption();
  } catch (e) { toast(e.message, "error"); }
});
function pollCaption() {
  clearInterval(state.captionPoll);
  state.captionPoll = setInterval(async () => {
    const s = await api("/api/caption/status");
    $("captionFill").style.width = `${s.percent || 0}%`;
    $("captionText").textContent =
      s.status === "running" ? `Captioning ${s.done}/${s.total} — ${s.current}` : s.status;
    if (s.status !== "running") {
      clearInterval(state.captionPoll);
      if (s.status === "failed") toast(`Captioning failed: ${s.error}`, "error");
      else toast("Captioning complete", "success");
      loadDataset();
    }
  }, 1000);
}
$("toTrainBtn").addEventListener("click", () => goStep(3));

/* ── Step 3: training ── */
$("stepsRange").addEventListener("input", (e) => $("stepsVal").textContent = e.target.value);
$("rankRange").addEventListener("input", (e) => $("rankVal").textContent = e.target.value);
$("trainBtn").addEventListener("click", async () => {
  if (!state.project) return;
  const body = {
    trigger_word: $("trainTrigger").value.trim(),
    steps: +$("stepsRange").value,
    learning_rate: +$("lrSelect").value,
    rank: +$("rankRange").value,
  };
  try {
    await api(`/api/projects/${state.project}/train`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("trainMonitor").classList.remove("hidden");
    $("cancelTrainBtn").classList.remove("hidden");
    $("trainBtn").disabled = true;
    pollTrain();
    toast("Training started", "success");
  } catch (e) { toast(e.message, "error"); }
});
$("cancelTrainBtn").addEventListener("click", async () => {
  await api("/api/train/cancel", { method: "POST" });
  toast("Cancelling…");
});
function pollTrain() {
  clearInterval(state.trainPoll);
  state.trainPoll = setInterval(async () => {
    const s = await api("/api/train/status");
    const badge = $("trainStatusBadge");
    badge.textContent = s.status;
    badge.className = `monitor-status ${s.status}`;
    $("trainFill").style.width = `${s.percent || 0}%`;
    $("trainStepText").textContent =
      `step ${s.step || 0}/${s.total_steps || 0} · ${s.percent || 0}% · ${fmtTime(s.elapsed_sec)}`;
    if (s.log_tail) { const log = $("trainLog"); log.textContent = s.log_tail.join("\n"); log.scrollTop = log.scrollHeight; }
    if (["completed", "failed", "cancelled", "idle"].includes(s.status)) {
      clearInterval(state.trainPoll);
      $("trainBtn").disabled = false;
      $("cancelTrainBtn").classList.add("hidden");
      if (s.status === "completed") { toast("🎉 LoRA trained! Head to Generate.", "success"); loadProjects(state.project); }
      else if (s.status === "failed") toast(`Training failed: ${s.error}`, "error");
    }
  }, 2000);
}
const fmtTime = (s) => { s = s || 0; const m = Math.floor(s / 60); return m ? `${m}m ${s % 60}s` : `${s}s`; };

/* ── Step 4: generation ── */
$("scaleRange").addEventListener("input", (e) => $("scaleVal").textContent = (+e.target.value).toFixed(2));
$("genStepsRange").addEventListener("input", (e) => $("genStepsVal").textContent = e.target.value);
$("guidanceRange").addEventListener("input", (e) => $("guidanceVal").textContent = e.target.value);
document.querySelectorAll(".seg-btn").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.aspect = { w: +b.dataset.w, h: +b.dataset.h };
  }));

async function loadLoras() {
  const loras = await api("/api/loras");
  const sel = $("loraSelect");
  sel.innerHTML = `<option value="">— base model (no LoRA) —</option>`;
  loras.forEach((l) => {
    const opt = document.createElement("option");
    opt.value = l.path;
    opt.textContent = `${l.project} / ${l.name} (${l.size_mb}MB)`;
    sel.appendChild(opt);
  });
  if (loras.length) sel.selectedIndex = 1;
}
$("genBtn").addEventListener("click", async () => {
  const prompt = $("genPrompt").value.trim();
  if (!prompt) { toast("Enter a prompt", "error"); return; }
  const body = {
    prompt, lora: $("loraSelect").value || null,
    lora_scale: +$("scaleRange").value,
    width: state.aspect.w, height: state.aspect.h,
    steps: +$("genStepsRange").value,
    guidance_scale: +$("guidanceRange").value,
  };
  overlay(true, "Generating image — this can take 20–40s…");
  try {
    await api("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await loadGallery();
    toast("Image generated", "success");
  } catch (e) { toast(e.message, "error"); }
  finally { overlay(false); }
});

async function loadGallery() {
  const imgs = await api("/api/generated");
  const grid = $("galleryGrid");
  if (!imgs.length) { grid.innerHTML = `<p class="gallery-empty">No images yet — generate your first one.</p>`; return; }
  grid.innerHTML = "";
  imgs.forEach((im) => {
    const div = document.createElement("div");
    div.className = "gallery-item";
    div.innerHTML = `<img src="${im.url}" loading="lazy" />`;
    div.addEventListener("click", () => openLightbox(im.url));
    grid.appendChild(div);
  });
}

/* ── Lightbox ── */
function openLightbox(url) {
  $("lightboxImg").src = url;
  $("lightboxDownload").href = url;
  $("lightbox").classList.remove("hidden");
}
$("lightboxClose").addEventListener("click", () => $("lightbox").classList.add("hidden"));
$("lightboxBg").addEventListener("click", () => $("lightbox").classList.add("hidden"));

/* ── Boot ── */
(async function init() {
  await health();
  await loadProjects();
  await loadThumbs();
})();
