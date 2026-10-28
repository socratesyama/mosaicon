importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js');
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const TPL = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
let det, rec, ep;

async function mk(list, eps) {
  let err;
  for (const s of list) { try { return await ort.InferenceSession.create(typeof s === 'string' ? s : new Uint8Array(s), { executionProviders: eps }); } catch (e) { err = e; } }
  throw err;
}
async function init(m) { // 手元から渡されたモデル > models/ の大きい方 > 小さい方
  const dets = m.det ? [m.det] : ['models/det_10g.onnx', 'models/det_500m.onnx'];
  const recs = m.rec ? [m.rec] : ['models/w600k_r50.onnx', 'models/w600k_mbf.onnx'];
  for (const eps of [['webgpu', 'wasm'], ['wasm']]) {
    try { det = await mk(dets, eps); rec = await mk(recs, eps); ep = eps[0]; return; } catch (e) { if (eps.length === 1) throw e; }
  }
}

const dc = new OffscreenCanvas(64, 64), dx = dc.getContext('2d', { willReadFrequently: true });
async function detect(bmp, size, thr) {
  const sc = Math.min(size / bmp.width, size / bmp.height);
  dc.width = dc.height = size;
  dx.fillStyle = '#000'; dx.fillRect(0, 0, size, size);
  dx.drawImage(bmp, 0, 0, Math.round(bmp.width * sc), Math.round(bmp.height * sc));
  const d = dx.getImageData(0, 0, size, size).data, n = size * size, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) f[c * n + i] = (d[4 * i + c] - 127.5) / 128;
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, size, size]) });
  const o = det.outputNames.map(k => out[k].data), faces = [];
  [8, 16, 32].forEach((st, k) => {
    const fw = size / st, S = o[k], B = o[k + 3], K = o[k + 6];
    for (let i = 0; i < fw * fw * 2; i++) {
      if (S[i] < thr) continue;
      const loc = i >> 1, cx = (loc % fw) * st, cy = ((loc / fw) | 0) * st;
      const kps = [];
      for (let j = 0; j < 5; j++) kps.push((cx + K[10 * i + 2 * j] * st) / sc, (cy + K[10 * i + 2 * j + 1] * st) / sc);
      faces.push({
        score: S[i], kps,
        box: [(cx - B[4 * i] * st) / sc, (cy - B[4 * i + 1] * st) / sc, (cx + B[4 * i + 2] * st) / sc, (cy + B[4 * i + 3] * st) / sc],
      });
    }
  });
  faces.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const a of faces) {
    if (keep.every(b => iou(a.box, b.box) < 0.4)) keep.push(a);
  }
  return keep;
}
function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = w * h;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9);
}

const rc = new OffscreenCanvas(112, 112), rx = rc.getContext('2d', { willReadFrequently: true });
async function embed(bmp, k) {
  let mx = 0, my = 0, tx = 0, ty = 0;
  for (let j = 0; j < 5; j++) { mx += k[2 * j] / 5; my += k[2 * j + 1] / 5; tx += TPL[j][0] / 5; ty += TPL[j][1] / 5; }
  let na = 0, nb = 0, den = 0;
  for (let j = 0; j < 5; j++) {
    const px = k[2 * j] - mx, py = k[2 * j + 1] - my, qx = TPL[j][0] - tx, qy = TPL[j][1] - ty;
    na += px * qx + py * qy; nb += px * qy - py * qx; den += px * px + py * py;
  }
  const a = na / den, b = nb / den;
  rx.setTransform(1, 0, 0, 1, 0, 0); rx.clearRect(0, 0, 112, 112);
  rx.setTransform(a, b, -b, a, tx - (a * mx - b * my), ty - (b * mx + a * my));
  rx.drawImage(bmp, 0, 0);
  rx.setTransform(1, 0, 0, 1, 0, 0);
  const d = rx.getImageData(0, 0, 112, 112).data, n = 112 * 112, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) f[c * n + i] = (d[4 * i + c] - 127.5) / 127.5;
  const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, 112, 112]) });
  const e = Float32Array.from(out[rec.outputNames[0]].data);
  let nr = 0; for (const v of e) nr += v * v; nr = Math.sqrt(nr) + 1e-9;
  return e.map(v => v / nr);
}

onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'init') { await init(m); postMessage({ type: 'ready', ep }); return; }
    const faces = await detect(m.bmp, m.size, m.thr);
    if (m.type === 'ref') {
      const f = faces.sort((a, b) => (b.box[2] - b.box[0]) - (a.box[2] - a.box[0]))[0];
      postMessage({ id: m.id, emb: f ? await embed(m.bmp, f.kps) : null });
    } else {
      for (const f of faces) f.emb = f.score >= 0.5 ? await embed(m.bmp, f.kps) : null;
      postMessage({ id: m.id, faces });
    }
    m.bmp.close();
  } catch (e) { postMessage({ id: m.id, error: String(e) }); }
};
