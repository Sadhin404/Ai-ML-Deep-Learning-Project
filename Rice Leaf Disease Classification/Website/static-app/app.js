// Rice Leaf Disease Classification — in-browser ONNX inference + analytic Grad-CAM.
// Result shows the disease name and the Grad-CAM map only (no scores).
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const DISPLAY = {
  Bacterialblight: 'Bacterial Blight',
  Blast: 'Blast',
  Brownspot: 'Brown Spot',
  Tungro: 'Tungro'
};
const STEPS = {
  Bacterialblight: [
    'Use healthy seeds and bacterial blight-resistant rice varieties.',
    'Avoid excessive nitrogen fertilizer and maintain balanced nutrient application.',
    'Monitor affected fields regularly and consult an agricultural expert for proper management.'
  ],
  Blast: [
    'Use disease-free seeds and blast-resistant rice varieties.',
    'Apply balanced fertilizers and avoid excessive nitrogen application.',
    'Consult an agricultural expert about approved fungicides when disease symptoms appear.'
  ],
  Brownspot: [
    'Use healthy seeds and treat seeds before planting.',
    'Maintain adequate water supply and ensure balanced soil nutrition, especially potassium.',
    'Remove infected crop residues and regularly monitor disease development.'
  ],
  Tungro: [
    'Cultivate rice varieties resistant to tungro disease.',
    'Remove infected plants and regularly monitor Green Leafhopper populations.',
    'Follow Integrated Pest Management (IPM) practices to control insect vectors.'
  ]
};
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const S = 224;

const sessions = {};   // modelKey -> ort session
const camData = {};    // modelKey -> {weight, classes}
let currentImage = null; // HTMLImageElement ready to draw

const $ = id => document.getElementById(id);
const drop = $('drop'), fileInput = $('fileInput'), preview = $('preview'),
      dropHint = $('dropHint'), analyseBtn = $('analyseBtn'),
      statusEl = $('status'), resultEl = $('result');

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('dragover'); });
drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('dragover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

function setFile(file) {
  if (!file.type.startsWith('image/')) { statusEl.textContent = 'Please choose an image file.'; return; }
  const url = URL.createObjectURL(file);
  loadImage(url);
}

function loadImage(src) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    currentImage = img;
    preview.src = img.src;
    preview.hidden = false;
    dropHint.hidden = true;
    resultEl.hidden = true;
    statusEl.textContent = '';
    analyseBtn.disabled = false;
  };
  img.onerror = () => { statusEl.textContent = 'Could not load that image.'; };
  img.src = src;
}

document.querySelectorAll('.sample').forEach(btn => {
  btn.addEventListener('click', () => loadImage(btn.dataset.src));
});

function selectedModel() {
  return document.querySelector('input[name="model"]:checked').value;
}

async function getSession(key) {
  if (!sessions[key]) {
    statusEl.textContent = 'Loading model…';
    sessions[key] = await ort.InferenceSession.create(`model_${key}.onnx`,
      { executionProviders: ['wasm'] });
    const r = await fetch(`cam_weights_${key}.json`);
    camData[key] = await r.json();
  }
  return sessions[key];
}

// Notebook eval preprocessing: Resize(256) on the shorter side, CenterCrop(224),
// ImageNet normalisation. Returns {tensor, canvas224}.
function preprocess(img) {
  const scale = 256 / Math.min(img.naturalWidth, img.naturalHeight);
  const sw = Math.round(img.naturalWidth * scale);
  const sh = Math.round(img.naturalHeight * scale);
  const c256 = document.createElement('canvas');
  c256.width = sw; c256.height = sh;
  c256.getContext('2d').drawImage(img, 0, 0, sw, sh);
  const c224 = document.createElement('canvas');
  c224.width = S; c224.height = S;
  const ctx = c224.getContext('2d');
  ctx.drawImage(c256, (sw - S) / 2, (sh - S) / 2, S, S, 0, 0, S, S);
  const data = ctx.getImageData(0, 0, S, S).data;
  const input = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) {
    for (let c = 0; c < 3; c++) {
      input[c * S * S + i] = (data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return { tensor: new ort.Tensor('float32', input, [1, 3, S, S]), canvas224: c224 };
}

// JET colormap, t in [0,1] -> [r,g,b]
function jet(t) {
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// Analytic Grad-CAM: with a GAP -> Linear head, the Grad-CAM channel weights for
// the predicted class are exactly that class's classifier weights.
function renderCam(feats, weights, origCanvas) {
  const C = 1280, H = 7, Wd = 7;
  const cam = new Float32Array(H * Wd);
  let max = 0;
  for (let i = 0; i < H * Wd; i++) {
    let v = 0;
    for (let k = 0; k < C; k++) v += weights[k] * feats[k * H * Wd + i];
    cam[i] = Math.max(0, v);
    if (cam[i] > max) max = cam[i];
  }
  const small = document.createElement('canvas');
  small.width = Wd; small.height = H;
  const sctx = small.getContext('2d');
  const simg = sctx.createImageData(Wd, H);
  for (let i = 0; i < H * Wd; i++) {
    const v = Math.round((max > 0 ? cam[i] / max : 0) * 255);
    simg.data[i * 4] = v; simg.data[i * 4 + 1] = v; simg.data[i * 4 + 2] = v; simg.data[i * 4 + 3] = 255;
  }
  sctx.putImageData(simg, 0, 0);
  const big = document.createElement('canvas');
  big.width = S; big.height = S;
  const bctx = big.getContext('2d');
  bctx.imageSmoothingEnabled = true;
  bctx.drawImage(small, 0, 0, S, S);
  const heat = bctx.getImageData(0, 0, S, S).data;
  const orig = origCanvas.getContext('2d').getImageData(0, 0, S, S).data;
  const out = document.createElement('canvas');
  out.width = S; out.height = S;
  const octx = out.getContext('2d');
  const oimg = octx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    const [r, g, b] = jet(heat[i * 4] / 255);
    oimg.data[i * 4]     = Math.min(255, 0.5 * orig[i * 4]     + 0.5 * r);
    oimg.data[i * 4 + 1] = Math.min(255, 0.5 * orig[i * 4 + 1] + 0.5 * g);
    oimg.data[i * 4 + 2] = Math.min(255, 0.5 * orig[i * 4 + 2] + 0.5 * b);
    oimg.data[i * 4 + 3] = 255;
  }
  octx.putImageData(oimg, 0, 0);
  return out.toDataURL('image/png');
}

analyseBtn.addEventListener('click', async () => {
  if (!currentImage) return;
  const key = selectedModel();
  analyseBtn.disabled = true;
  try {
    const session = await getSession(key);
    statusEl.textContent = 'Analysing…';
    const { tensor, canvas224 } = preprocess(currentImage);
    const out = await session.run({ input: tensor });
    const logits = out.logits.data;
    let pred = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[pred]) pred = i;
    const className = camData[key].classes[pred];
    $('diseaseName').textContent = DISPLAY[className] || className;
    const stepsList = $('stepsList');
    stepsList.innerHTML = '';
    (STEPS[className] || []).forEach(step => {
      const li = document.createElement('li');
      li.textContent = step;
      stepsList.appendChild(li);
    });
    $('origImg').src = canvas224.toDataURL('image/png');
    $('camImg').src = renderCam(out.features.data, camData[key].weight[pred], canvas224);
    resultEl.hidden = false;
    statusEl.textContent = '';
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Something went wrong while analysing. Please try again.';
  } finally {
    analyseBtn.disabled = false;
  }
});
