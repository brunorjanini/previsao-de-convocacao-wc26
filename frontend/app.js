/* ── Configuração ─────────────────────────────────────────────── */
const API = "http://localhost:8000";

/* ── Elementos DOM ────────────────────────────────────────────── */
const photoUpload  = document.getElementById("photo-upload");
const photoPreview = document.getElementById("photo-preview");
const posSelect    = document.getElementById("position-select");
const generateBtn  = document.getElementById("generate-btn");
const resetBtn     = document.getElementById("reset-btn");

const inputPanel   = document.getElementById("input-panel");
const resultPanel  = document.getElementById("result-panel");

const cardOverall  = document.getElementById("card-overall");
const cardPosition = document.getElementById("card-position");
const cardPhoto    = document.getElementById("card-photo");
const caIds        = { pace: "ca-pac", shooting: "ca-sho", passing: "ca-pas",
                        dribbling: "ca-dri", defending: "ca-def", physic: "ca-phy" };

const verdictBadge = document.getElementById("verdict-badge");
const verdictText  = document.getElementById("verdict-text");
const verdictProb  = document.getElementById("verdict-prob");

/* ── Estado ───────────────────────────────────────────────────── */
let photoDataUrl = null;

/* ── Carrega posições do backend ─────────────────────────────── */
async function loadPositions() {
  try {
    const res  = await fetch(`${API}/positions`);
    const data = await res.json();
    posSelect.innerHTML = data.positions
      .map(p => `<option value="${p}">${p}</option>`)
      .join("");
  } catch {
    posSelect.innerHTML = `<option value="">Backend offline</option>`;
    showError("Backend não encontrado. Inicie com: uvicorn main:app --reload (na pasta backend/)");
  }
}

/* ── Foto ────────────────────────────────────────────────────── */
photoUpload.addEventListener("change", () => {
  const file = photoUpload.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    photoDataUrl = e.target.result;
    photoPreview.src = photoDataUrl;
  };
  reader.readAsDataURL(file);
});

/* ── Sliders ─────────────────────────────────────────────────── */
document.querySelectorAll(".slider-row").forEach(row => {
  const input = row.querySelector("input[type='range']");
  const val   = row.querySelector(".attr-val");

  function syncSlider() {
    const pct = (input.value / 99) * 100;
    input.style.setProperty("--pct", pct + "%");
    val.textContent = input.value;
  }

  input.addEventListener("input", syncSlider);
  syncSlider();
});

/* ── Coleta atributos dos sliders ────────────────────────────── */
function getAttrs() {
  const attrs = {};
  document.querySelectorAll(".slider-row").forEach(row => {
    const attr  = row.dataset.attr;
    const value = parseInt(row.querySelector("input").value);
    attrs[attr] = value;
  });
  return attrs;
}

/* ── Cor do overall ──────────────────────────────────────────── */
function overallColor(ov) {
  if (ov >= 85) return "#d4af37";   // dourado
  if (ov >= 75) return "#c0c0c0";   // prata
  if (ov >= 65) return "#cd7f32";   // bronze
  return "#88c0d0";                  // azul claro
}

/* ── Gera carta ──────────────────────────────────────────────── */
function renderCard(attrs, position, overall) {
  cardOverall.textContent  = overall;
  cardOverall.style.background = `linear-gradient(180deg, ${overallColor(overall)}, #7a5c00)`;
  cardOverall.style.webkitBackgroundClip = "text";
  cardOverall.style.webkitTextFillColor  = "transparent";

  cardPosition.textContent = position;

  const photo = photoDataUrl || "placeholder.svg";
  cardPhoto.src    = photo;
  photoPreview.src = photo;

  Object.entries(caIds).forEach(([attr, id]) => {
    document.getElementById(id).textContent = attrs[attr] ?? 0;
  });
}

/* ── Render veredito ─────────────────────────────────────────── */
function renderVerdict(data) {
  const pct = Math.round(data.probability * 100);
  if (data.convocated) {
    verdictBadge.textContent = "🏆";
    verdictText.textContent  = "CONVOCADO!";
    verdictText.className    = "verdict-text convocado";
    verdictProb.textContent  = `${pct}% de chance de convocação`;
  } else {
    verdictBadge.textContent = "❌";
    verdictText.textContent  = "NÃO CONVOCADO";
    verdictText.className    = "verdict-text nao-convocado";
    verdictProb.textContent  = `Apenas ${pct}% de chance de convocação`;
  }
}

/* ── Botão Gerar ─────────────────────────────────────────────── */
generateBtn.addEventListener("click", async () => {
  const position = posSelect.value;
  if (!position) { showError("Selecione uma posição."); return; }

  const attrs = getAttrs();

  generateBtn.disabled    = true;
  generateBtn.textContent = "⏳ Calculando...";

  try {
    const res = await fetch(`${API}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ position, ...attrs }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Erro na API");
    }
    const data = await res.json();

    renderCard(attrs, position, data.overall ?? attrs.overall ?? 70);
    renderVerdict(data);

    inputPanel.hidden  = true;
    resultPanel.hidden = false;
    resultPanel.scrollIntoView({ behavior: "smooth" });

  } catch (e) {
    showError(e.message);
  } finally {
    generateBtn.disabled    = false;
    generateBtn.textContent = "⚽ Gerar Carta";
  }
});

/* ── Botão Reset ─────────────────────────────────────────────── */
resetBtn.addEventListener("click", () => {
  inputPanel.hidden  = false;
  resultPanel.hidden = true;
  window.scrollTo({ top: 0, behavior: "smooth" });
});

/* ── Toast de erro ───────────────────────────────────────────── */
function showError(msg) {
  let toast = document.getElementById("error-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "error-toast";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.display = "block";
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { toast.style.display = "none"; }, 5000);
}

/* ── Init ────────────────────────────────────────────────────── */
loadPositions();
