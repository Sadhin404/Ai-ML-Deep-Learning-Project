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
    const res = await fetch('/static/samples/' + name);
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
    const fd = new FormData();
    fd.append('file', currentFile);
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Analysis failed. Please try again.');
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
  $('reportBtn').href = '/api/report/' + d.id;
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
