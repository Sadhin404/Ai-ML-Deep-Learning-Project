// ---------- App logic ----------
const $ = id => document.getElementById(id);
const dz = $('dropzone'), fi = $('fileInput');
let currentFile = null, currentResult = null, viewMode = 'original', bboxOn = true;

const CLASS_LIST = ['glioma', 'meningioma', 'notumor', 'pituitary'];
const DISPLAY = { glioma: 'Glioma', meningioma: 'Meningioma', notumor: 'No Tumor', pituitary: 'Pituitary' };
const CLASS_HEX = { glioma: '#e11d48', meningioma: '#d97706', notumor: '#059669', pituitary: '#7c3aed' };

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) { showError('Please choose an image file (PNG or JPG).'); return; }
  currentFile = file;
  hideError();
  $('previewImg').src = URL.createObjectURL(file);
  $('fileName').textContent = file.name;
  dz.classList.add('hidden');
  $('previewBox').classList.remove('hidden');
  $('analyzeBtn').disabled = false;
  document.querySelectorAll('.sample').forEach(s => s.classList.remove('active'));
  fileGate = checkMRIImage(file);
  fileGate.then(ok => {
    if (currentFile !== file) return; // a newer file replaced this one meanwhile
    if (!ok) {
      showError(NOT_SCAN_MSG);
      $('analyzeBtn').disabled = true;
    }
  }).catch(() => {});
}
function clearFile() {
  currentFile = null;
  fileGate = null;
  hideError();
  fi.value = '';
  $('previewBox').classList.add('hidden');
  dz.classList.remove('hidden');
  $('analyzeBtn').disabled = true;
  document.querySelectorAll('.sample').forEach(s => s.classList.remove('active'));
}
function showError(msg) { const e = $('errorBox'); e.textContent = msg; e.classList.remove('hidden'); }
function hideError() { $('errorBox').classList.add('hidden'); }

dz.addEventListener('click', () => fi.click());
dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); } });
fi.addEventListener('change', () => setFile(fi.files[0]));
['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', e => setFile(e.dataTransfer.files[0]));
$('removeBtn').addEventListener('click', clearFile);

document.querySelectorAll('.sample').forEach(btn => {
  btn.addEventListener('click', async () => {
    document.querySelectorAll('.sample').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    const name = btn.dataset.sample;
    const res = await fetch('samples/' + name);
    const blob = await res.blob();
    setFile(new File([blob], name, { type: 'image/jpeg' }));
    btn.classList.add('active');
  });
});

$('analyzeBtn').addEventListener('click', async () => {
  if (!currentFile) return;
  if (fileGate && !(await fileGate)) {
    showError(NOT_SCAN_MSG);
    currentResult = null;
    $('resultBox').classList.add('hidden');
    $('emptyState').classList.remove('hidden');
    return;
  }
  const btn = $('analyzeBtn');
  const started = Date.now();
  btn.disabled = true;
  btn.querySelector('.btn-label').textContent = 'Scanning…';
  btn.querySelector('.spinner').classList.remove('hidden');
  $('previewBox').classList.add('scanning');
  hideError();
  try {
    const data = await runBrowserInference(currentFile);
    currentResult = data;
    const wait = 1400 - (Date.now() - started);
    if (wait > 0) await new Promise(r => setTimeout(r, wait));
    renderResult(data);
  } catch (err) {
    showError(err.message || 'Something went wrong. Please try again.');
  } finally {
    $('previewBox').classList.remove('scanning');
    btn.disabled = false;
    btn.querySelector('.btn-label').textContent = 'Analyze image';
    btn.querySelector('.spinner').classList.add('hidden');
  }
});

function renderResult(d) {
  $('emptyState').classList.add('hidden');
  const box = $('resultBox');
  box.classList.remove('hidden');
  const pc = $('predClass');
  pc.textContent = d.predicted_display;
  pc.className = 'pred-class ' + d.predicted_class;

  // confidence ring
  const circ = 226.2, pct = d.confidence;
  const fg = $('ringFg');
  fg.style.stroke = CLASS_HEX[d.predicted_class];
  fg.style.strokeDashoffset = circ;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    fg.style.strokeDashoffset = circ * (1 - pct);
  }));
  let n = 0; const target = Math.round(pct * 100);
  const iv = setInterval(() => {
    n += Math.max(1, Math.round(target / 24));
    if (n >= target) { n = target; clearInterval(iv); }
    $('confNum').textContent = n + '%';
  }, 34);

  // probability bars
  const bars = $('bars');
  bars.innerHTML = '';
  for (const cls of CLASS_LIST) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `<span class="bar-name">${DISPLAY[cls]}</span>
      <div class="bar-track"><div class="bar-fill ${cls}"></div></div>
      <span class="bar-val">${(d.probabilities[cls] * 100).toFixed(1)}%</span>`;
    bars.appendChild(row);
    const fill = row.querySelector('.bar-fill');
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.width = (d.probabilities[cls] * 100).toFixed(1) + '%';
    }));
  }

  // viewer
  setView('original');
  const rb = $('reportBtn');
  rb.removeAttribute('href');
  rb.onclick = e => { e.preventDefault(); downloadReport(); };
  box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateBBox() {
  const box = $('bbox');
  if (!currentResult || !currentResult.bbox || !bboxOn || viewMode === 'heatmap') {
    box.classList.add('hidden');
    return;
  }
  const b = currentResult.bbox;
  box.className = 'bbox ' + currentResult.predicted_class;
  box.style.left = (b.x * 100) + '%';
  box.style.top = (b.y * 100) + '%';
  box.style.width = (b.w * 100) + '%';
  box.style.height = (b.h * 100) + '%';
  $('bboxLabel').textContent = currentResult.predicted_class === 'notumor'
    ? 'Focus region'
    : 'Detected region · ' + Math.round(currentResult.confidence * 100) + '%';
}

function setView(mode) {
  viewMode = mode;
  document.querySelectorAll('.seg-btn[data-view]').forEach(b => b.classList.toggle('active', b.dataset.view === mode));
  if (!currentResult) return;
  const imgs = currentResult.images;
  const base = $('viewBase'), top = $('viewTop');
  $('blendRow').style.display = mode === 'overlay' ? 'flex' : 'none';
  if (mode === 'original') {
    base.src = 'data:image/png;base64,' + imgs.original;
    top.classList.add('hidden');
  } else if (mode === 'heatmap') {
    base.src = 'data:image/png;base64,' + imgs.heatmap;
    top.classList.add('hidden');
  } else {
    base.src = 'data:image/png;base64,' + imgs.original;
    top.src = 'data:image/png;base64,' + imgs.overlay;
    top.classList.remove('hidden');
    top.style.opacity = $('blend').value / 100;
  }
  updateBBox();
}
$('bboxToggle').addEventListener('click', () => {
  bboxOn = !bboxOn;
  $('bboxToggle').classList.toggle('active', bboxOn);
  updateBBox();
});
document.querySelectorAll('.seg-btn[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
$('blend').addEventListener('input', e => { $('viewTop').style.opacity = e.target.value / 100; });

$('againBtn').addEventListener('click', () => {
  currentResult = null;
  $('resultBox').classList.add('hidden');
  $('emptyState').classList.remove('hidden');
  clearFile();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ---------- MRI check: refuse non-MRI images before the AI runs ----------
// The classifier only knows Glioma/Meningioma/No Tumor/Pituitary, so without
// this gate ANY picture (a photo, a landscape) would get a confident label.
// Real MRI scans are pure greyscale with rich texture; colour photos, blank
// images and smooth graphics fail these checks. Thresholds calibrated on 40
// real MRI scans: colour fraction 0.000, mean saturation 0, bright fraction
// <= 0.24, speckle >= 3.0, flat fraction <= 0.60, luminance std >= 32 — the
// limits below keep wide margins on every metric.
const NOT_SCAN_MSG = "⚠️ This doesn't look like a brain MRI scan. Analysis only works on brain MRI images — please upload a valid MRI scan.";
let fileGate = null;

async function checkMRIImage(file) {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, 256 / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale)), h = Math.max(1, Math.round(bmp.height * scale));
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const N = w * h;
  let color = 0, satSum = 0, bright = 0, lumSum = 0;
  const lum = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = d[4 * i], g = d[4 * i + 1], b = d[4 * i + 2];
    const sat = Math.max(r, g, b) - Math.min(r, g, b);
    if (sat > 32) color++;
    satSum += sat;
    const L = 0.299 * r + 0.587 * g + 0.114 * b;
    lum[i] = L; lumSum += L;
    if (L > 200) bright++;
  }
  const mean = lumSum / N;
  let varSum = 0, spkSum = 0, flat = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, diff = lum[i] - mean;
      varSum += diff * diff;
      if (x + 1 < w) { spkSum += Math.abs(lum[i + 1] - lum[i]); if (lum[i + 1] === lum[i]) flat++; }
    }
  }
  const lumStd = Math.sqrt(varSum / N);
  const speckle = spkSum / (N - h);
  const colorFrac = color / N, meanSat = satSum / N, brightFrac = bright / N;
  const flatFrac = flat / (N - h);
  if (colorFrac > 0.05 || meanSat > 14) return false; // colour photo, not a greyscale MRI
  if (brightFrac > 0.35) return false;                // mostly white (document / screenshot)
  if (speckle < 1.2 || lumStd < 12) return false;     // blank or smooth graphic — no scan texture
  if (flatFrac > 0.70) return false;                  // huge flat areas (UI screenshot) — measured: 40 real MRI scans <= 0.60
  return true;
}

// ---------- In-browser AI engine (ONNX Runtime Web) ----------
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const MODEL_FILE = 'model.onnx';
const MODEL_TAG = 'densenet201';
let ortSession = null, camWeights = null;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const CHANNELS = 1920;

async function ensureModel() {
  if (ortSession) return;
  const lbl = document.querySelector('#analyzeBtn .btn-label');
  if (lbl) lbl.textContent = 'Loading AI model…';
  camWeights = await (await fetch('cam_weights.json')).json();
  ortSession = await ort.InferenceSession.create(MODEL_FILE, { executionProviders: ['wasm'] });
  if (lbl) lbl.textContent = 'Scanning…';
}

function jetColor(x) {
  const cl = v => Math.max(0, Math.min(1, v));
  return [cl(1.5 - Math.abs(4 * x - 3)) * 255, cl(1.5 - Math.abs(4 * x - 2)) * 255, cl(1.5 - Math.abs(4 * x - 1)) * 255];
}

function focusBox(cam, W, H) {
  let thr = 0.55, mask = [];
  for (let t = 0; t < 2; t++) {
    mask = [];
    let count = 0;
    for (let i = 0; i < W * H; i++) if (cam[i] >= thr) { mask.push(i); count++; }
    if (count >= 0.01 * W * H || t === 1) break;
    thr = 0.40;
  }
  let x0 = W, x1 = 0, y0 = H, y1 = 0;
  for (const i of mask) {
    const x = i % W, y = (i / W) | 0;
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  let bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  if (bw > 0.85 * W || bh > 0.85 * H) {
    let peak = 0, pv = -1;
    for (let i = 0; i < W * H; i++) if (cam[i] > pv) { pv = cam[i]; peak = i; }
    const px = peak % W, py = (peak / W) | 0;
    bw = Math.floor(0.46 * W); bh = Math.floor(0.46 * H);
    x0 = px - (bw >> 1); y0 = py - (bh >> 1); x1 = x0 + bw; y1 = y0 + bh;
  }
  const pad = 8;
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad);
  x1 = Math.min(W - 1, x1 + pad); y1 = Math.min(H - 1, y1 + pad);
  return { x: x0 / W, y: y0 / H, w: (x1 - x0 + 1) / W, h: (y1 - y0 + 1) / H };
}

async function runBrowserInference(file) {
  await ensureModel();
  const bmp = await createImageBitmap(file);
  const S = 224;
  const cv = document.createElement('canvas'); cv.width = cv.height = S;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, S, S);
  const px = ctx.getImageData(0, 0, S, S).data;
  const N = S * S;
  const input = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    input[i] = (px[4 * i] / 255 - MEAN[0]) / STD[0];
    input[N + i] = (px[4 * i + 1] / 255 - MEAN[1]) / STD[1];
    input[2 * N + i] = (px[4 * i + 2] / 255 - MEAN[2]) / STD[2];
  }
  const tensor = new ort.Tensor('float32', input, [1, 3, S, S]);
  const out = await ortSession.run({ input: tensor });
  const logits = out.logits.data;           // [1,4]
  const acts = out.acts.data;               // [1,1920,7,7] post-ReLU feature maps
  const mx = Math.max(...logits);
  const exps = [...logits].map(v => Math.exp(v - mx));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(v => v / sum);
  const pred = probs.indexOf(Math.max(...probs));

  // Analytic Grad-CAM: for this GAP + linear head, the classifier weights of
  // the predicted class ARE the Grad-CAM channel weights (verified against
  // true gradient-based Grad-CAM: correlation 1.0000).
  const W7 = camWeights.models[MODEL_TAG][pred];
  const cam7 = new Float32Array(49);
  for (let k = 0; k < CHANNELS; k++) {
    const w = W7[k];
    if (w === 0) continue;
    const off = k * 49;
    for (let j = 0; j < 49; j++) cam7[j] += w * acts[off + j];
  }
  let mn = Infinity, mxv = -Infinity;
  for (let j = 0; j < 49; j++) { cam7[j] = Math.max(0, cam7[j]); if (cam7[j] < mn) mn = cam7[j]; if (cam7[j] > mxv) mxv = cam7[j]; }
  for (let j = 0; j < 49; j++) cam7[j] = (cam7[j] - mn) / (mxv - mn + 1e-8);

  // Upsample CAM 7x7 -> 224x224 (bilinear)
  const small = document.createElement('canvas'); small.width = small.height = 7;
  const sctx = small.getContext('2d');
  const simg = sctx.createImageData(7, 7);
  for (let j = 0; j < 49; j++) { const v = Math.round(cam7[j] * 255); simg.data[4 * j] = simg.data[4 * j + 1] = simg.data[4 * j + 2] = v; simg.data[4 * j + 3] = 255; }
  sctx.putImageData(simg, 0, 0);
  const big = document.createElement('canvas'); big.width = big.height = S;
  const bctx = big.getContext('2d', { willReadFrequently: true });
  bctx.imageSmoothingEnabled = true;
  bctx.drawImage(small, 0, 0, S, S);
  const bdata = bctx.getImageData(0, 0, S, S).data;
  const camBig = new Float32Array(N);
  for (let i = 0; i < N; i++) camBig[i] = bdata[4 * i] / 255;

  // Heatmap + overlay canvases
  const heatCv = document.createElement('canvas'); heatCv.width = heatCv.height = S;
  const hctx = heatCv.getContext('2d');
  const himg = hctx.createImageData(S, S);
  const overCv = document.createElement('canvas'); overCv.width = overCv.height = S;
  const octx = overCv.getContext('2d');
  const oimg = octx.createImageData(S, S);
  for (let i = 0; i < N; i++) {
    const [r, g, b] = jetColor(camBig[i]);
    himg.data[4 * i] = r; himg.data[4 * i + 1] = g; himg.data[4 * i + 2] = b; himg.data[4 * i + 3] = 255;
    oimg.data[4 * i] = 0.55 * px[4 * i] + 0.45 * r;
    oimg.data[4 * i + 1] = 0.55 * px[4 * i + 1] + 0.45 * g;
    oimg.data[4 * i + 2] = 0.55 * px[4 * i + 2] + 0.45 * b;
    oimg.data[4 * i + 3] = 255;
  }
  hctx.putImageData(himg, 0, 0);
  octx.putImageData(oimg, 0, 0);
  const b64 = c => c.toDataURL('image/png').split(',')[1];

  const probabilities = {};
  CLASS_LIST.forEach((c, i) => probabilities[c] = probs[i]);
  return {
    id: 'local',
    bbox: focusBox(camBig, S, S),
    predicted_class: CLASS_LIST[pred],
    predicted_display: DISPLAY[CLASS_LIST[pred]],
    confidence: probs[pred],
    probabilities,
    images: { original: b64(cv), heatmap: b64(heatCv), overlay: b64(overCv) }
  };
}

// ---------- Client-side PDF report (jsPDF) ----------
const MEANING = {
  glioma: 'A tumour pattern most consistent with a glioma — a tumour arising from the glial cells within the brain tissue itself.',
  meningioma: 'A tumour pattern most consistent with a meningioma — a tumour arising from the meninges, the membranes covering the brain. Usually slow-growing and benign.',
  notumor: 'No tumour pattern identified — the scan appearance matches the healthy, tumour-free cases.',
  pituitary: 'A tumour pattern most consistent with a pituitary tumour — a growth in the pituitary gland beneath the brain, which can affect hormones and vision.'
};
const NEXT_STEP = {
  glioma: 'Prompt referral to a neurosurgeon / neuro-oncologist is recommended. Only a biopsy can confirm the exact tumour type.',
  meningioma: 'Referral to a neurosurgeon for evaluation and follow-up imaging, as advised by the treating physician, is recommended.',
  notumor: 'Routine follow-up as advised by the treating physician. Any persistent symptom should still be examined clinically.',
  pituitary: 'Referral to an endocrinologist and a neurosurgeon is recommended; hormone blood tests and a vision check are usually advised.'
};
const CLASS_TINT = { glioma: '#fff1f2', meningioma: '#fffbeb', notumor: '#ecfdf5', pituitary: '#f5f3ff' };

function donutPNG(frac, colorHex) {
  const S = 300, cv = document.createElement('canvas'); cv.width = cv.height = S;
  const g = cv.getContext('2d');
  g.lineWidth = 30;
  g.strokeStyle = '#e5e7eb';
  g.beginPath(); g.arc(S / 2, S / 2, S / 2 - 16, 0, Math.PI * 2); g.stroke();
  g.strokeStyle = colorHex;
  g.beginPath(); g.arc(S / 2, S / 2, S / 2 - 16, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2); g.stroke();
  g.fillStyle = '#131722'; g.font = 'bold 60px Helvetica, Arial, sans-serif'; g.textAlign = 'center';
  g.fillText((frac * 100).toFixed(1) + '%', S / 2, S / 2 + 4);
  g.fillStyle = '#6b7280'; g.font = '29px Helvetica, Arial, sans-serif';
  g.fillText('confidence', S / 2, S / 2 + 44);
  return cv.toDataURL('image/png');
}

function downloadReport() {
  const d = currentResult;
  if (!d) return;
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const W = 210, L = 16, R = 16, CW = W - L - R, RX = W - R;
  const ink = '#131722', grey = '#6b7280', hair = '#e6e8ec', indigo = '#4f46e5', violet = '#7c3aed', card = '#f6f7f9';
  const accent = CLASS_HEX[d.predicted_class], tint = CLASS_TINT[d.predicted_class];
  const rid = 'BTAI-' + Math.random().toString(16).slice(2, 8).toUpperCase();
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'long', year: 'numeric' }) + ', ' +
    now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
  const pct = v => (v * 100).toFixed(1) + '%';

  // ---- Header (light) ----
  doc.setFillColor(indigo); doc.roundedRect(L, 12, 11, 11, 3, 3, 'F');
  doc.setDrawColor('#ffffff'); doc.setLineWidth(0.55);
  doc.circle(L + 5.5, 17.5, 3.1, 'S');
  doc.line(L + 5.5, 14.4, L + 5.5, 20.6);
  doc.setFont('helvetica', 'bold'); doc.setFontSize(15.5); doc.setTextColor(ink);
  doc.text('Brain Tumor MRI AI', L + 14.5, 18.6);
  doc.setFont('helvetica', 'normal'); doc.setFontSize(9.5); doc.setTextColor(grey);
  doc.text('Explainable Classification Report', L + 14.5, 23.4);
  doc.setFont('helvetica', 'bold'); doc.setFontSize(10.5); doc.setTextColor(ink);
  doc.text(rid, RX, 18.6, { align: 'right' });
  doc.setFont('helvetica', 'normal'); doc.setFontSize(9.5); doc.setTextColor(grey);
  doc.text(dateStr, RX, 23.4, { align: 'right' });
  doc.setFillColor(indigo); doc.rect(L, 27.5, CW * 0.55, 1.1, 'F');
  doc.setFillColor(violet); doc.rect(L + CW * 0.55, 27.5, CW * 0.45, 1.1, 'F');

  // ---- Meta chips ----
  const chips = [['MODEL', 'DenseNet-201'], ['TEST ACCURACY', '99.47%'], ['MACRO F1', '99.46%'], ['TEST IMAGES', '1,311']];
  const cw = (CW - 9) / 4;
  chips.forEach(([k, v], i) => {
    const x = L + i * (cw + 3);
    doc.setFillColor(card); doc.roundedRect(x, 32, cw, 10.5, 2.2, 2.2, 'F');
    doc.setFont('helvetica', 'bold'); doc.setFontSize(6.8); doc.setTextColor(grey);
    doc.text(k, x + cw / 2, 36.3, { align: 'center' });
    doc.setFontSize(9.3); doc.setTextColor(ink);
    doc.text(v, x + cw / 2, 40.8, { align: 'center' });
  });

  // ---- Prediction hero ----
  doc.setFillColor(tint); doc.roundedRect(L, 47, CW, 37, 3.5, 3.5, 'F');
  doc.setFillColor(accent); doc.roundedRect(L, 47, 4.2, 37, 2, 2, 'F');
  doc.setFont('helvetica', 'bold'); doc.setFontSize(8.5); doc.setTextColor(grey);
  doc.text('AI PREDICTION', L + 9, 54.5);
  doc.setFontSize(31); doc.setTextColor(accent);
  doc.text(d.predicted_display.toUpperCase(), L + 9, 66.5);
  doc.setFont('helvetica', 'normal'); doc.setFontSize(10); doc.setTextColor(ink);
  doc.splitTextToSize(MEANING[d.predicted_class], 124).forEach((ln, i) => doc.text(ln, L + 9, 74 + i * 4.6));
  doc.addImage(donutPNG(d.confidence, accent), 'PNG', RX - 33, 51, 29, 29);

  // ---- Class probabilities ----
  doc.setFont('helvetica', 'bold'); doc.setFontSize(11.5); doc.setTextColor(ink);
  doc.text('Class probabilities', L, 94.5);
  doc.setDrawColor(hair); doc.setLineWidth(0.25); doc.line(L, 97, RX, 97);
  let y = 104.5;
  for (const cls of CLASS_LIST) {
    const v = d.probabilities[cls], col = CLASS_HEX[cls];
    doc.setFillColor(col); doc.circle(L + 2, y - 1.1, 1.7, 'F');
    doc.setFont('helvetica', 'normal'); doc.setFontSize(10); doc.setTextColor(ink);
    doc.text(DISPLAY[cls], L + 6.5, y);
    doc.setFillColor('#e9ebef'); doc.roundedRect(L + 32, y - 3.4, 130, 4.4, 2.2, 2.2, 'F');
    if (v >= 0.005) { doc.setFillColor(col); doc.roundedRect(L + 32, y - 3.4, Math.max(130 * v, 4.4), 4.4, 2.2, 2.2, 'F'); }
    doc.setFont('helvetica', 'bold'); doc.text(pct(v), RX, y, { align: 'right' });
    doc.setFont('helvetica', 'normal');
    y += 8.2;
  }

  // ---- Visual explanation ----
  doc.setFont('helvetica', 'bold'); doc.setFontSize(11.5); doc.setTextColor(ink);
  doc.text('Visual explanation — Grad-CAM', L, 136.5);
  doc.setDrawColor(hair); doc.setLineWidth(0.25); doc.line(L, 139, RX, 139);
  const s = (CW - 8) / 3, iy = 143;
  [['original', 'Original scan'], ['heatmap', 'Grad-CAM heatmap'], ['overlay', 'Overlay']].forEach(([key, label], i) => {
    const x = L + i * (s + 4);
    doc.addImage('data:image/png;base64,' + d.images[key], 'PNG', x, iy, s, s);
    doc.setFont('helvetica', 'normal'); doc.setFontSize(8.5); doc.setTextColor(grey);
    doc.text(label, x + s / 2, iy + s + 4.8, { align: 'center' });
  });
  const bb = d.bbox;
  if (bb) {
    const bx = L + bb.x * s, by = iy + bb.y * s;
    doc.setDrawColor(accent); doc.setLineWidth(0.55);
    doc.rect(bx, by, bb.w * s, bb.h * s, 'S');
    const lbl = (d.predicted_class === 'notumor' ? 'FOCUS REGION' : 'DETECTED REGION') + ' · ' + Math.round(d.confidence * 100) + '%';
    doc.setFont('helvetica', 'bold'); doc.setFontSize(6.2);
    const lw = doc.getTextWidth(lbl) + 4;
    let ly = by - 6.2; if (ly < iy) ly = by + 1.2;
    doc.setFillColor(accent); doc.roundedRect(bx, ly, lw, 5.4, 1.4, 1.4, 'F');
    doc.setTextColor('#ffffff'); doc.text(lbl, bx + 2, ly + 3.7);
  }
  doc.setFont('helvetica', 'normal'); doc.setFontSize(9); doc.setTextColor(grey);
  doc.text('Red and yellow influenced the decision most. The heatmap is a sensitivity map, not medical proof.', L, iy + s + 9.6);

  // ---- Next step ----
  const nsY = iy + s + 15;
  doc.setFillColor(card); doc.roundedRect(L, nsY, CW, 16, 2.8, 2.8, 'F');
  doc.setFont('helvetica', 'bold'); doc.setFontSize(8); doc.setTextColor(indigo);
  doc.text('RECOMMENDED NEXT STEP', L + 5, nsY + 4.8);
  doc.setFont('helvetica', 'normal'); doc.setFontSize(9.8); doc.setTextColor(ink);
  doc.splitTextToSize(NEXT_STEP[d.predicted_class], CW - 10).forEach((ln, i) => doc.text(ln, L + 5, nsY + 10.1 + i * 4.4));

  // ---- Disclaimer ----
  const dcY = nsY + 20;
  doc.setFillColor('#fdf3f3'); doc.roundedRect(L, dcY, CW, 16.5, 2.8, 2.8, 'F');
  doc.setFont('helvetica', 'bold'); doc.setFontSize(8.5); doc.setTextColor('#9f1239');
  doc.text('IMPORTANT — NOT A MEDICAL DIAGNOSIS', L + 5, dcY + 4.8);
  doc.setFont('helvetica', 'normal'); doc.setFontSize(9); doc.setTextColor(ink);
  doc.splitTextToSize('Generated by a research AI system (final-year thesis project). It supports, but never replaces, a qualified doctor. Only clinical examination and biopsy can confirm a diagnosis.', CW - 10)
    .forEach((ln, i) => doc.text(ln, L + 5, dcY + 9.9 + i * 4.2));

  // ---- Footer ----
  doc.setDrawColor(hair); doc.setLineWidth(0.25); doc.line(L, 281.5, RX, 281.5);
  doc.setFont('helvetica', 'bold'); doc.setFontSize(8.6); doc.setTextColor(ink);
  doc.text('CSE, First Capital University of Bangladesh', L, 285.3);
  doc.setFont('helvetica', 'normal'); doc.setFontSize(8.4); doc.setTextColor(grey);
  doc.text('Dataset: Brain Tumor MRI Dataset — Nickparvar, Kaggle', L, 288.9);
  doc.save('Brain_Tumor_Report_' + rid + '.pdf');
}
