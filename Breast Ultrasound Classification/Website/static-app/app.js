// ---------- Animated background: drifting dots, links + sonar pulse rings ----------
(() => {
  const cv = document.getElementById('bg'), ctx = cv.getContext('2d');
  let W, H, pts = [], rings = [], frame = 0;
  const resize = () => {
    W = cv.width = innerWidth; H = cv.height = innerHeight;
    const n = Math.min(70, Math.floor(W * H / 24000));
    pts = Array.from({ length: n }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - .5) * .22, vy: (Math.random() - .5) * .22,
      r: Math.random() * 1.6 + .6
    }));
  };
  resize(); addEventListener('resize', resize);
  (function loop() {
    frame++;
    ctx.clearRect(0, 0, W, H);
    // sonar rings (ultrasound pulses), spawned slowly from a few centres
    if (frame % 150 === 0 && rings.length < 5) {
      rings.push({ x: W * (0.15 + Math.random() * 0.7), y: H * (0.15 + Math.random() * 0.6), r: 0, a: .30 });
    }
    rings = rings.filter(g => g.a > 0.01);
    for (const g of rings) {
      g.r += 1.15; g.a *= 0.994;
      ctx.beginPath(); ctx.arc(g.x, g.y, g.r, 0, 7);
      ctx.strokeStyle = `rgba(56,225,255,${g.a})`; ctx.lineWidth = 1.4; ctx.stroke();
      ctx.beginPath(); ctx.arc(g.x, g.y, g.r * 0.62, 0, 7);
      ctx.strokeStyle = `rgba(139,92,246,${g.a * 0.7})`; ctx.stroke();
    }
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 7);
      ctx.fillStyle = 'rgba(125,211,252,.5)'; ctx.fill();
    }
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const a = pts[i], b = pts[j], d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d < 120) {
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = `rgba(56,189,248,${(1 - d / 120) * .14})`; ctx.lineWidth = 1; ctx.stroke();
      }
    }
    requestAnimationFrame(loop);
  })();
})();

// ---------- Hero typewriter (rotating one-liners) ----------
(() => {
  const el = document.getElementById('typewriter');
  if (!el) return;
  const lines = [
    'Explainable AI for breast ultrasound',
    'EfficientNet-B0 · Grad-CAM · BUSI dataset',
    '92.31% accuracy on the held-out test set',
    'Upload a scan — download a full PDF report'
  ];
  let li = 0, ci = 0, deleting = false;
  (function tick() {
    const line = lines[li];
    el.textContent = line.slice(0, ci);
    let delay = deleting ? 26 : 52;
    if (!deleting && ci === line.length) { delay = 2100; deleting = true; }
    else if (deleting && ci === 0) { deleting = false; li = (li + 1) % lines.length; delay = 420; }
    else ci += deleting ? -1 : 1;
    setTimeout(tick, delay);
  })();
})();

// ---------- Gentle 3D tilt on the main card (desktop pointers only) ----------
(() => {
  if (!matchMedia('(pointer:fine)').matches) return;
  const card = document.getElementById('mainCard');
  card.addEventListener('mousemove', e => {
    const r = card.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5;
    card.style.transform = `perspective(1400px) rotateY(${x * 2.2}deg) rotateX(${-y * 2.2}deg)`;
  });
  card.addEventListener('mouseleave', () => card.style.transform = '');
})();

// ---------- App logic ----------
const $ = id => document.getElementById(id);
const dz = $('dropzone'), fi = $('fileInput');
let currentFile = null, currentResult = null, viewMode = 'original', bboxOn = true;

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
}
function clearFile() {
  currentFile = null;
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
    setFile(new File([blob], name, { type: 'image/png' }));
    btn.classList.add('active');
  });
});

$('analyzeBtn').addEventListener('click', async () => {
  if (!currentFile) return;
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
    const wait = 1650 - (Date.now() - started);
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
  for (const cls of ['normal', 'benign', 'malignant']) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `<span class="bar-name">${cls}</span>
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
  $('bboxLabel').textContent = currentResult.predicted_class === 'normal'
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

// ---------- In-browser AI engine (ONNX Runtime Web) ----------
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
let ortSession = null, camWeights = null;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const CLASS_LIST = ['normal', 'benign', 'malignant'];
const DISPLAY = { normal: 'Normal', benign: 'Benign', malignant: 'Malignant' };

async function ensureModel() {
  if (ortSession) return;
  const lbl = document.querySelector('#analyzeBtn .btn-label');
  if (lbl) lbl.textContent = 'Loading AI model…';
  camWeights = await (await fetch('cam_weights.json')).json();
  ortSession = await ort.InferenceSession.create('model.onnx', { executionProviders: ['wasm'] });
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
  const logits = out.logits.data, acts = out.acts.data; // acts: [1,1280,7,7]
  const mx = Math.max(...logits);
  const exps = [...logits].map(v => Math.exp(v - mx));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(v => v / sum);
  const pred = probs.indexOf(Math.max(...probs));

  // Analytic Grad-CAM: final classifier weights act as the channel weights.
  const W7 = camWeights.weights[pred];
  const cam7 = new Float32Array(49);
  for (let k = 0; k < 1280; k++) {
    const w = W7[k];
    if (w === 0) continue;
    const off = k * 49;
    for (let j = 0; j < 49; j++) cam7[j] += w * acts[off + j];
  }
  let mn = Infinity, mxv = -Infinity;
  for (let j = 0; j < 49; j++) { cam7[j] = Math.max(0, cam7[j]); if (cam7[j] < mn) mn = cam7[j]; if (cam7[j] > mxv) mxv = cam7[j]; }
  for (let j = 0; j < 49; j++) cam7[j] = (cam7[j] - mn) / (mxv - mn + 1e-8);

  // Upsample CAM 7x7 -> 224x224 (bilinear, like the server version)
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

// ---------- Client-side PDF report (jsPDF, same layout as the server report) ----------
const MEANING = {
  normal: 'No lesion-like pattern was identified in this scan. The tissue appearance is consistent with the Normal cases the model was trained on.',
  benign: 'The scan shows a lesion whose appearance is most consistent with a benign (non-cancerous) finding, such as a cyst or fibroadenoma pattern.',
  malignant: 'The scan shows a lesion whose appearance is most consistent with a malignant (cancerous) finding. This result needs prompt review by a qualified radiologist and confirmation through further clinical examination and biopsy.'
};
const NEXT_STEP = {
  normal: 'Routine follow-up as advised by the treating physician. Any new symptom should still be examined clinically.',
  benign: 'Clinical correlation and a follow-up ultrasound, as advised by the treating physician, are recommended.',
  malignant: 'Urgent referral to a specialist is recommended. Only a biopsy can confirm a diagnosis.'
};
const CLASS_HEX = { normal: '#059669', benign: '#d97706', malignant: '#e11d48' };

function downloadReport() {
  const d = currentResult;
  if (!d) return;
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const W = 210, L = 18, R = 18, CW = W - L - R;
  const ink = '#16242f', grey = '#6b8296', hair = '#dce7ee';
  const accent = CLASS_HEX[d.predicted_class];
  const rid = 'BUAI-' + Math.random().toString(16).slice(2, 8).toUpperCase();
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'long', year: 'numeric' }) + ', ' +
    now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
  const pct = v => (v * 100).toFixed(2) + '%';

  doc.setFillColor('#0b2537'); doc.rect(0, 0, W, 27, 'F');
  doc.setFillColor('#0e7490'); doc.rect(0, 27, W, 1.1, 'F');
  doc.setFillColor('#ffffff'); doc.setFont('helvetica', 'bold'); doc.setFontSize(16);
  doc.text('Breast Ultrasound Analysis Report', L, 12);
  doc.setFont('helvetica', 'normal'); doc.setFontSize(9); doc.setTextColor('#a9c9dc');
  doc.text('Explainable AI  ·  EfficientNet-B0 + Grad-CAM', L, 18.5);
  doc.setTextColor('#ffffff'); doc.text(rid, W - R, 12, { align: 'right' });
  doc.setTextColor('#a9c9dc'); doc.text(dateStr, W - R, 18.5, { align: 'right' });

  let y = 38;
  doc.setFontSize(10);
  const infoRow = (k, v) => {
    doc.setTextColor(grey); doc.text(k, L, y);
    doc.setTextColor(ink); doc.text(v, L + 32, y); y += 6;
  };
  infoRow('Uploaded file', (currentFile ? currentFile.name : 'scan.png').slice(0, 52));
  infoRow('Analysis', 'Single-image classification with Grad-CAM explanation');
  infoRow('Model accuracy', '92.31% accuracy  ·  92.46% macro F1  ·  98.07% ROC-AUC  (held-out test set)');

  y += 2;
  doc.setFillColor('#f3f8fb'); doc.roundedRect(L, y, CW, 20, 3.5, 3.5, 'F');
  doc.setFillColor(accent); doc.roundedRect(L, y, 4.2, 20, 2, 2, 'F');
  doc.setFontSize(8.5); doc.setTextColor(grey); doc.text('AI PREDICTION', L + 9, y + 6);
  doc.setFont('helvetica', 'bold'); doc.setFontSize(18); doc.setTextColor(accent);
  doc.text(d.predicted_display.toUpperCase(), L + 9, y + 15);
  doc.setFont('helvetica', 'normal'); doc.setFontSize(8.5); doc.setTextColor(grey);
  doc.text('CONFIDENCE', W - R - 7, y + 6, { align: 'right' });
  doc.setFont('helvetica', 'bold'); doc.setFontSize(15); doc.setTextColor(ink);
  doc.text(pct(d.confidence), W - R - 7, y + 15, { align: 'right' });
  doc.setFont('helvetica', 'normal');

  y += 32;
  const section = t => {
    doc.setFont('helvetica', 'bold'); doc.setFontSize(11); doc.setTextColor(ink);
    doc.text(t, L, y); y += 3;
    doc.setDrawColor(hair); doc.setLineWidth(0.25); doc.line(L, y, W - R, y); y += 4;
    doc.setFont('helvetica', 'normal');
  };
  section('Class probabilities');
  const barX = L + 28, barW = CW - 28 - 17;
  for (const cls of CLASS_LIST) {
    doc.setFontSize(10); doc.setTextColor(ink);
    doc.text(cls[0].toUpperCase() + cls.slice(1), L, y + 3.5);
    doc.setFillColor('#e3edf3'); doc.roundedRect(barX, y, barW, 4.6, 2.3, 2.3, 'F');
    doc.setFillColor(CLASS_HEX[cls]);
    doc.roundedRect(barX, y, Math.max(barW * d.probabilities[cls], 2.3), 4.6, 2.3, 2.3, 'F');
    doc.setFont('helvetica', 'bold'); doc.text(pct(d.probabilities[cls]), W - R, y + 3.5, { align: 'right' });
    doc.setFont('helvetica', 'normal');
    y += 7.6;
  }

  y += 4;
  section('Visual explanation — Grad-CAM');
  const img = 50, gap = (CW - img * 3) / 2;
  [['original', 'Original scan'], ['heatmap', 'Grad-CAM heatmap'], ['overlay', 'Overlay']].forEach(([key, label], i) => {
    const x = L + i * (img + gap);
    doc.addImage('data:image/png;base64,' + d.images[key], 'PNG', x, y, img, img);
    doc.setFontSize(8.5); doc.setTextColor(grey);
    doc.text(label, x + img / 2, y + img + 4.8, { align: 'center' });
  });
  y += img + 10;
  doc.setFontSize(9); doc.setTextColor(grey);
  doc.splitTextToSize('Red and yellow areas influenced the decision most; blue areas least. The heatmap shows where the model looked — it is a sensitivity map, not medical proof.', CW)
    .forEach(ln => { doc.text(ln, L, y); y += 4.6; });

  y += 5;
  section('Interpretation');
  doc.setFontSize(10); doc.setTextColor(ink);
  doc.splitTextToSize(MEANING[d.predicted_class], CW).forEach(ln => { doc.text(ln, L, y); y += 5.2; });
  y += 3.5;
  doc.setFont('helvetica', 'bold'); doc.text('Recommended next step', L, y); y += 5.6;
  doc.setFont('helvetica', 'normal');
  doc.splitTextToSize(NEXT_STEP[d.predicted_class], CW).forEach(ln => { doc.text(ln, L, y); y += 5.2; });

  y += 4;
  doc.setFillColor('#fdf3f3'); doc.roundedRect(L, y, CW, 15.5, 3, 3, 'F');
  doc.setFont('helvetica', 'bold'); doc.setFontSize(9); doc.setTextColor('#9f1239');
  doc.text('IMPORTANT — NOT A MEDICAL DIAGNOSIS', L + 5, y + 5.5);
  doc.setFont('helvetica', 'normal'); doc.setTextColor(ink);
  doc.splitTextToSize('This report was generated by a research AI system (final-year thesis project). It supports, but never replaces, examination by a qualified doctor. Only clinical examination and biopsy can confirm a diagnosis.', CW - 10)
    .slice(0, 2).forEach((ln, i) => doc.text(ln, L + 5, y + 10.5 + i * 4.6));

  doc.setDrawColor(hair); doc.line(L, 282, W - R, 282);
  doc.setFontSize(8.5); doc.setTextColor(grey);
  doc.text('Breast Ultrasound AI — developed by MD. Shadhin Khan (ID 0572310005101003), CSE, First Capital University of Bangladesh.', L, 286.5);
  doc.text('Dataset: BUSI — Al-Dhabyani et al., Data in Brief (2020).', L, 290.7);
  doc.save('Ultrasound_Report_' + rid + '.pdf');
}
