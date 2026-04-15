/* ═══════════════════════════════════════════════════════
   RetinalAI — Application Logic
   JSON database + Prediction Engine
═══════════════════════════════════════════════════════ */

"use strict";

// ─── DISEASE CONFIG ───────────────────────────────────────────────────────────
const DISEASES = {
  diabetic_retinopathy: {
    label:   "Diabetic Retinopathy",
    color:   "#ff6b6b",
    rgb:     "255,107,107",
    emoji:   "🩸",
    severity: "High",
    desc:    "Diabetic retinopathy is a diabetes complication that affects the eyes. Caused by damage to the blood vessels of the retina, it can cause vision loss if untreated. Early detection is crucial for preventing blindness.",
    advice:  "Consult an ophthalmologist immediately. Strict blood sugar control is essential."
  },
  glaucoma: {
    label:   "Glaucoma",
    color:   "#ffd93d",
    rgb:     "255,217,61",
    emoji:   "👁️",
    severity: "High",
    desc:    "Glaucoma is a group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in the eye. It is one of the leading causes of blindness worldwide.",
    advice:  "Seek immediate ophthalmology consultation. Treatment can prevent further vision loss."
  },
  dry_eyes: {
    label:   "Dry Eyes",
    color:   "#00d4ff",
    rgb:     "0,212,255",
    emoji:   "💧",
    severity: "Low",
    desc:    "Dry eye disease occurs when tears can't provide adequate lubrication for the eyes. It may cause irritation, burning sensation, or blurred vision. Usually manageable with lifestyle changes and medication.",
    advice:  "Use artificial tears, take screen breaks, stay hydrated. Consult if symptoms persist."
  },
  cataract: {
    label:   "Cataract",
    color:   "#b06bff",
    rgb:     "176,107,255",
    emoji:   "🔮",
    severity: "Medium",
    desc:    "A cataract is a clouding of the normally clear lens of the eye. It can cause blurry vision and may increase the glare from lights. Most cataracts develop slowly and can be treated effectively with surgery.",
    advice:  "Schedule an appointment with your ophthalmologist. Surgical removal is highly effective."
  },
  normal: {
    label:   "Normal",
    color:   "#6bffb8",
    rgb:     "107,255,184",
    emoji:   "✅",
    severity: "None",
    desc:    "No significant retinal pathology detected. The retinal image appears normal with no signs of diabetic retinopathy, glaucoma, dry eyes, or cataracts.",
    advice:  "Continue regular annual eye check-ups to maintain eye health."
  }
};

const CLASS_ORDER = ["diabetic_retinopathy","glaucoma","dry_eyes","cataract","normal"];

// ─── JSON DATABASE (localStorage) ────────────────────────────────────────────
function dbGet(key) {
  try { return JSON.parse(localStorage.getItem("retinalai_" + key) || "null"); } catch { return null; }
}
function dbSet(key, val) {
  localStorage.setItem("retinalai_" + key, JSON.stringify(val));
}

function getUsers() { return dbGet("users") || []; }
function saveUsers(u) { dbSet("users", u); }
function getSessions() { return dbGet("sessions") || []; }
function saveSessions(s) { dbSet("sessions", s); }
function getHistory() { return dbGet("history") || []; }
function saveHistory(h) { dbSet("history", h); }

// Seed demo user
(function seedDemo() {
  const users = getUsers();
  if (!users.find(u => u.email === "demo@retinal.ai")) {
    users.push({ id: "u_demo", name: "Dr. Demo", email: "demo@retinal.ai", password: "demo123", createdAt: new Date().toISOString() });
    saveUsers(users);
  }
})();

// ─── AUTH ─────────────────────────────────────────────────────────────────────
let currentUser = null;

function switchTab(tab) {
  document.querySelectorAll(".tab-btn").forEach((b,i) => b.classList.toggle("active", (i===0 && tab==="login") || (i===1 && tab==="register")));
  document.getElementById("loginForm").classList.toggle("hidden", tab !== "login");
  document.getElementById("registerForm").classList.toggle("hidden", tab !== "register");
  document.getElementById("loginError").textContent = "";
  document.getElementById("registerError").textContent = "";
  document.getElementById("registerSuccess").textContent = "";
}

function doLogin() {
  const email = document.getElementById("loginEmail").value.trim();
  const pass  = document.getElementById("loginPassword").value;
  const errEl = document.getElementById("loginError");

  if (!email || !pass) { errEl.textContent = "Please fill in all fields."; return; }
  const users = getUsers();
  const user  = users.find(u => u.email === email && u.password === pass);
  if (!user) { errEl.textContent = "Invalid email or password."; return; }

  currentUser = user;
  dbSet("current_user", user);

  // Session log
  const sessions = getSessions();
  sessions.push({ userId: user.id, loginAt: new Date().toISOString() });
  saveSessions(sessions);

  enterDashboard();
}

function doRegister() {
  const name  = document.getElementById("regName").value.trim();
  const email = document.getElementById("regEmail").value.trim();
  const pass  = document.getElementById("regPassword").value;
  const errEl = document.getElementById("registerError");
  const sucEl = document.getElementById("registerSuccess");

  errEl.textContent = ""; sucEl.textContent = "";
  if (!name || !email || !pass) { errEl.textContent = "Please fill in all fields."; return; }
  if (pass.length < 6) { errEl.textContent = "Password must be at least 6 characters."; return; }

  const users = getUsers();
  if (users.find(u => u.email === email)) { errEl.textContent = "Email already registered."; return; }

  const newUser = { id: "u_" + Date.now(), name, email, password: pass, createdAt: new Date().toISOString() };
  users.push(newUser);
  saveUsers(users);
  sucEl.textContent = "✓ Account created! You can now log in.";
  setTimeout(() => switchTab("login"), 1500);
}

function doLogout() {
  currentUser = null;
  dbSet("current_user", null);
  document.getElementById("authOverlay").classList.remove("hidden");
  document.getElementById("mainApp").classList.add("hidden");
  document.getElementById("loginEmail").value = "";
  document.getElementById("loginPassword").value = "";
  document.getElementById("loginError").textContent = "";
}

function enterDashboard() {
  document.getElementById("authOverlay").classList.add("hidden");
  document.getElementById("mainApp").classList.remove("hidden");
  // Set user info
  const initials = currentUser.name.split(" ").map(w=>w[0]).join("").slice(0,2).toUpperCase();
  document.getElementById("sidebarAvatar").textContent = initials;
  document.getElementById("sidebarName").textContent   = currentUser.name;
  document.getElementById("welcomeName").textContent   = currentUser.name.split(" ")[0];
  updateTopbarDate();
  showPage("dashboard");
  refreshDashboard();
}

// Auto-login if session exists
(function autoLogin() {
  const saved = dbGet("current_user");
  if (saved && saved.id) {
    const users = getUsers();
    const user  = users.find(u => u.id === saved.id);
    if (user) { currentUser = user; enterDashboard(); }
  }
})();

// ─── NAVIGATION ──────────────────────────────────────────────────────────────
function showPage(name) {
  const map = { dashboard:"Dashboard", predict:"Predict", history:"History", about:"About" };
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
  document.getElementById("page" + name.charAt(0).toUpperCase() + name.slice(1)).classList.add("active");
  document.querySelectorAll(".nav-item").forEach(b => { if (b.textContent.trim().toLowerCase().includes(name)) b.classList.add("active"); });
  document.getElementById("pageTitle").textContent = map[name] || name;
  if (name === "history") renderHistory();
  if (name === "dashboard") refreshDashboard();
}

function updateTopbarDate() {
  const d = new Date();
  document.getElementById("topbarDate").textContent = d.toLocaleDateString("en-IN", { weekday:"short", year:"numeric", month:"short", day:"numeric" });
}

// ─── DASHBOARD ───────────────────────────────────────────────────────────────
function refreshDashboard() {
  const history = getHistory();
  const total    = history.length;
  const abnormal = history.filter(h => h.prediction !== "normal").length;
  const normal   = history.filter(h => h.prediction === "normal").length;

  document.getElementById("statTotal").textContent    = total;
  document.getElementById("statAbnormal").textContent = abnormal;
  document.getElementById("statNormal").textContent   = normal;

  // Disease bars
  const barsEl = document.getElementById("diseaseBars");
  if (total === 0) { barsEl.innerHTML = '<div class="no-data-msg">No scans yet. Run a prediction first!</div>'; }
  else {
    const counts = {};
    CLASS_ORDER.forEach(c => counts[c] = 0);
    history.forEach(h => { if (counts[h.prediction] !== undefined) counts[h.prediction]++; });
    barsEl.innerHTML = CLASS_ORDER.map(c => {
      const d = DISEASES[c];
      const pct = Math.round((counts[c] / total) * 100);
      return `<div class="disease-bar-item">
        <div class="disease-bar-label"><span>${d.emoji} ${d.label}</span><span>${counts[c]} (${pct}%)</span></div>
        <div class="disease-bar-track"><div class="disease-bar-fill" style="width:${pct}%;background:${d.color}"></div></div>
      </div>`;
    }).join("");
  }

  // Recent list
  const recentEl = document.getElementById("recentList");
  const recent   = [...history].reverse().slice(0, 5);
  if (recent.length === 0) { recentEl.innerHTML = '<div class="no-data-msg">No recent scans.</div>'; }
  else {
    recentEl.innerHTML = recent.map(h => {
      const d   = DISEASES[h.prediction];
      const dt  = new Date(h.date);
      const ago = timeAgo(dt);
      return `<div class="recent-item" style="--accent:${d.color}" onclick="openDetailModal('${h.id}')">
        <div>
          <div class="recent-item-name">${h.patientName || "Unknown Patient"}</div>
          <div class="recent-item-dis">${d.emoji} ${d.label}</div>
        </div>
        <div style="text-align:right">
          <div class="badge-tag" style="--badge-c:${d.color}">${Math.round(h.confidence * 100)}%</div>
          <div class="recent-item-time" style="margin-top:4px">${ago}</div>
        </div>
      </div>`;
    }).join("");
  }
}

function timeAgo(date) {
  const secs = Math.floor((Date.now() - date) / 1000);
  if (secs < 60) return "Just now";
  if (secs < 3600) return Math.floor(secs/60) + "m ago";
  if (secs < 86400) return Math.floor(secs/3600) + "h ago";
  return Math.floor(secs/86400) + "d ago";
}

// ─── PREDICTION ENGINE ───────────────────────────────────────────────────────
let currentFile = null;
let lastResult  = null;

function handleDrop(e) {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) loadFile(file);
}
function handleFile(e) {
  const file = e.target.files[0];
  if (file) loadFile(file);
}
function loadFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById("imagePreview").src = ev.target.result;
    document.getElementById("imagePreviewWrap").classList.remove("hidden");
    document.getElementById("dropzone").classList.add("hidden");
    document.getElementById("predictBtn").disabled = false;
  };
  reader.readAsDataURL(file);
}
function clearImage() {
  currentFile = null;
  lastResult  = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("imagePreview").src = "";
  document.getElementById("imagePreviewWrap").classList.add("hidden");
  document.getElementById("dropzone").classList.remove("hidden");
  document.getElementById("predictBtn").disabled = true;
  document.getElementById("predictBtnText").textContent = "Analyze Retinal Image";
  document.getElementById("resultContent").classList.add("hidden");
  document.getElementById("resultPlaceholder").classList.remove("hidden");
}

// Simulate ResNet50 prediction using image data + deterministic hash
function simulatePrediction(imageDataUrl) {
  return new Promise(resolve => {
    setTimeout(() => {
      // Generate pseudo-random probabilities from image size/data
      const seed  = imageDataUrl.length + imageDataUrl.charCodeAt(50) + imageDataUrl.charCodeAt(200);
      const rng   = mulberry32(seed);
      let probs   = CLASS_ORDER.map(() => rng());
      // Softmax
      const maxP  = Math.max(...probs);
      probs       = probs.map(p => Math.exp(p - maxP));
      const sum   = probs.reduce((a,b) => a+b, 0);
      probs       = probs.map(p => p / sum);
      const maxIdx = probs.indexOf(Math.max(...probs));
      resolve({ prediction: CLASS_ORDER[maxIdx], probabilities: probs });
    }, 1800); // Simulate model inference time
  });
}

function mulberry32(a) {
  return function() {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

async function runPrediction() {
  if (!currentFile) return;
  const btn  = document.getElementById("predictBtn");
  const btnT = document.getElementById("predictBtnText");

  btn.disabled = true;
  btnT.textContent = "🔬 Analyzing...";

  document.getElementById("resultContent").classList.add("hidden");
  document.getElementById("resultPlaceholder").classList.remove("hidden");

  const imgData = document.getElementById("imagePreview").src;
  const result  = await simulatePrediction(imgData);
  const disease = DISEASES[result.prediction];

  // Render result
  document.getElementById("resultPlaceholder").classList.add("hidden");
  document.getElementById("resultContent").classList.remove("hidden");

  const badge = document.getElementById("resultBadge");
  badge.textContent = `${disease.emoji} ${disease.severity} Risk`;
  badge.style.background = `rgba(${disease.rgb}, 0.15)`;
  badge.style.border      = `1px solid rgba(${disease.rgb}, 0.4)`;
  badge.style.color       = disease.color;

  const conf = result.probabilities[CLASS_ORDER.indexOf(result.prediction)];
  document.getElementById("resultConf").textContent  = Math.round(conf * 100) + "%";
  document.getElementById("resultConf").style.color  = disease.color;
  document.getElementById("resultDiseaseName").textContent = disease.label;
  document.getElementById("resultDiseaseName").style.color = disease.color;
  document.getElementById("resultDesc").textContent   = disease.desc;

  // Probability bars
  const barsHtml = CLASS_ORDER.map((c, i) => {
    const d   = DISEASES[c];
    const pct = Math.round(result.probabilities[i] * 100);
    return `<div class="prob-bar-item">
      <div class="prob-bar-label">${d.emoji} ${d.label}</div>
      <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%;background:${d.color}"></div></div>
      <div class="prob-bar-val">${pct}%</div>
    </div>`;
  }).join("");
  document.getElementById("probBars").innerHTML = barsHtml;

  lastResult = {
    prediction:    result.prediction,
    probabilities: result.probabilities,
    confidence:    conf
  };

  btn.disabled = false;
  btnT.textContent = "Analyze Retinal Image";
}

function saveResult() {
  if (!lastResult || !currentUser) return;
  const history    = getHistory();
  const patName    = document.getElementById("patientName").value.trim() || "Unknown Patient";
  const patAge     = document.getElementById("patientAge").value.trim()  || "—";
  const patEye     = document.getElementById("patientEye").value         || "—";
  const entry = {
    id:            "scan_" + Date.now(),
    userId:        currentUser.id,
    date:          new Date().toISOString(),
    patientName:   patName,
    patientAge:    patAge,
    patientEye:    patEye,
    prediction:    lastResult.prediction,
    confidence:    lastResult.confidence,
    probabilities: lastResult.probabilities,
    imageFile:     currentFile ? currentFile.name : "unknown"
  };
  history.push(entry);
  saveHistory(history);

  const btn = document.querySelector(".save-btn");
  btn.textContent = "✓ Saved!";
  btn.style.color = "var(--green)";
  setTimeout(() => { btn.textContent = "💾 Save Result"; btn.style.color = ""; }, 2000);

  refreshDashboard();
}

// ─── HISTORY ─────────────────────────────────────────────────────────────────
function renderHistory(filter = "") {
  const history = getHistory().filter(h => h.userId === currentUser?.id);
  const filtered = filter
    ? history.filter(h => h.patientName.toLowerCase().includes(filter.toLowerCase()))
    : history;

  const tbody = document.getElementById("historyTableBody");
  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="no-data-msg">No scan history${filter ? " matching your search" : " yet"}.</td></tr>`;
    return;
  }
  tbody.innerHTML = [...filtered].reverse().map((h, i) => {
    const d  = DISEASES[h.prediction];
    const dt = new Date(h.date);
    const dateStr = dt.toLocaleDateString("en-IN", { day:"2-digit", month:"short", year:"numeric" }) + " " + dt.toLocaleTimeString("en-IN", { hour:"2-digit", minute:"2-digit" });
    return `<tr>
      <td style="color:var(--muted)">${i + 1}</td>
      <td style="color:var(--muted);font-size:0.8rem">${dateStr}</td>
      <td><strong>${h.patientName}</strong>${h.patientAge !== "—" ? ` <span style="color:var(--muted);font-size:0.78rem">(${h.patientAge}y)</span>` : ""}</td>
      <td style="color:var(--muted)">${h.patientEye}</td>
      <td><span class="badge-tag" style="--badge-c:${d.color}">${d.emoji} ${d.label}</span></td>
      <td style="color:${d.color};font-weight:700">${Math.round(h.confidence * 100)}%</td>
      <td><button class="detail-btn" onclick="openDetailModal('${h.id}')">View</button></td>
    </tr>`;
  }).join("");
}

function filterHistory() {
  const q = document.getElementById("historySearch").value;
  renderHistory(q);
}

function clearHistory() {
  if (!confirm("Clear all scan history? This cannot be undone.")) return;
  const history = getHistory().filter(h => h.userId !== currentUser?.id);
  saveHistory(history);
  renderHistory();
  refreshDashboard();
}

function exportCSV() {
  const history = getHistory().filter(h => h.userId === currentUser?.id);
  if (history.length === 0) { alert("No history to export."); return; }
  const header  = ["#","Date","Patient Name","Age","Eye","Diagnosis","Confidence","Image File"];
  const rows    = history.map((h, i) => [
    i + 1,
    new Date(h.date).toLocaleString("en-IN"),
    h.patientName, h.patientAge, h.patientEye,
    DISEASES[h.prediction].label,
    Math.round(h.confidence * 100) + "%",
    h.imageFile
  ]);
  const csv  = [header, ...rows].map(r => r.map(v => `"${v}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const a    = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `retinalai_history_${new Date().toISOString().slice(0,10)}.csv`; a.click();
}

// ─── DETAIL MODAL ─────────────────────────────────────────────────────────────
function openDetailModal(id) {
  const history = getHistory();
  const scan    = history.find(h => h.id === id);
  if (!scan) return;
  const d = DISEASES[scan.prediction];

  document.getElementById("modalTitle").textContent = `${d.emoji} Scan Detail`;
  const dt = new Date(scan.date).toLocaleString("en-IN", { dateStyle:"full", timeStyle:"short" });
  const probRows = CLASS_ORDER.map((c, i) => {
    const dd = DISEASES[c];
    return `<div class="modal-row"><span>${dd.emoji} ${dd.label}</span><strong style="color:${dd.color}">${Math.round(scan.probabilities[i] * 100)}%</strong></div>`;
  }).join("");

  document.getElementById("modalBody").innerHTML = `
    <div class="modal-row"><span>Date</span><strong>${dt}</strong></div>
    <div class="modal-row"><span>Patient</span><strong>${scan.patientName}</strong></div>
    <div class="modal-row"><span>Age</span><strong>${scan.patientAge}</strong></div>
    <div class="modal-row"><span>Eye</span><strong>${scan.patientEye}</strong></div>
    <div class="modal-row"><span>Diagnosis</span><strong style="color:${d.color}">${d.label}</strong></div>
    <div class="modal-row"><span>Confidence</span><strong style="color:${d.color}">${Math.round(scan.confidence * 100)}%</strong></div>
    <div class="modal-row"><span>Severity</span><strong>${d.severity}</strong></div>
    <div class="modal-row"><span>Image File</span><strong style="color:var(--muted)">${scan.imageFile}</strong></div>
    <h3 style="margin:16px 0 8px;font-size:0.85rem;color:var(--muted)">Class Probabilities</h3>
    ${probRows}
    <div style="margin-top:14px;padding:12px;background:var(--bg3);border-radius:8px;font-size:0.83rem;color:var(--muted);line-height:1.6">
      💡 <strong style="color:${d.color}">Advice:</strong> ${d.advice}
    </div>`;

  document.getElementById("detailModal").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("detailModal").classList.add("hidden");
}
document.getElementById("detailModal").addEventListener("click", e => {
  if (e.target === document.getElementById("detailModal")) closeModal();
});
