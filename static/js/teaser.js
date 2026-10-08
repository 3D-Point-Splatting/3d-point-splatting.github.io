// Animated teaser: ONE stage. The scene (the ColoRadar staircase mesh, the
// cascade radar at its held-out pose) is the main view throughout; each
// method family plays on it, then puts its methods on a floating Pareto card
// (top right), then the stage is snapshotted and flies into the dock (bottom
// left) as that family's teaser tile.
//
//   0 Scene         the whole mesh, the radar
//   1 Ray tracing   wavefront from the Tx, rays to the first hits, three
//                   single-bounce return paths; Sionna-RT + mmIR join the plot
//   2 Splatting     Gaussians on the scene; the camera tilts to bird's-eye and
//                   they collapse in elevation onto the range-azimuth map;
//                   DART + Radar Fields + RadarSplat join the plot
//   3 3DPS          the point primitives sweep out from the radar; 3DPS joins
//   4 Frontier      the baseline surface (the Pareto
//                   video's own geometry, colours and camera); 3DPS above it
//   5 Applications
//
// Scene data is the viewer's (tools/export_viewer.py); the plot's numbers,
// the surface comes from static/data/pareto.json (tools/gen_pareto_data.py,
// asserted against the deck's p5b_pareto.py). The plot's palette, emission
// strengths, colour map and camera are the deck's and gen_pareto_web.py's.
//
// Every scene layer is a pure function of one state object S (applyState),
// and the plot of P (applyPlot): the timeline only tweens S and P, so a docked
// tile can be re-rendered from its chapter's final state on a theme change.
import * as THREE from './three.module.min.js';
import { OrbitControls } from './OrbitControls.js';

const root = document.getElementById('teaser-anim');
const stage = document.getElementById('t-stage');
const sceneWrap = document.getElementById('t-scene');
const pCard = document.getElementById('t-pareto');
const pWrap = document.getElementById('pareto-wrap');
const headEl = document.getElementById('t-head');
const SCENE = 'seq_1_frame_185';
const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const D2R = Math.PI / 180;

const theme = () => (window.tdpsTheme && window.tdpsTheme()) || 'light';
const S2L = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
const toThree = (x, y, z) => new THREE.Vector3(x, z, -y);   // radar world -> three y-up (viewer.js)

// ── palettes ─────────────────────────────────────────────────────────────
// Stage: the tile background is the page's --tile-bg; the deck's Tx cyan,
// Rx amber and the return-path P1/P2/P3 colours; the teaser's pastel splats
// and pink points.
const STAGE = {
  light: { bg: 0xf5f5f7, mesh: 0xb4b8c2, tx: 0x1d9bd6, rx: 0xe8920c, ray: 0x1d9bd6, wave: 0x3aaee6,
           pts: 0xd8439a, ink: '#1f2024' },
  dark:  { bg: 0x26272c, mesh: 0x5e626d, tx: 0x40c8ff, rx: 0xffa62b, ray: 0x40c8ff, wave: 0x40c8ff,
           pts: 0xf07ab8, ink: '#e8e8ec' },
};
const PATHS = [0xab82ff, 0xef476f, 0x06d6a0];                 // palette.P1/P2/P3
const SPLAT = [0xf4a9a8, 0xf7d794, 0x9fd3e6, 0xb5a7e3, 0xa8e0c4, 0xe8a6c8, 0xf2c38f, 0x86bcd9];
// Plot: dark = the deck palette; light = gen_pareto_web.py's chrome swap.
const PLOT = {
  light: { ink: 0x1f2024, axis: 0x5b5c63, grid: 0xd4d4da, ours: 0x1d8a3a, dim: 0x777882, bg: 0xf5f5f7 },
  dark:  { ink: 0xe8e8ec, axis: 0xaaaab4, grid: 0x3c3c44, ours: 0x8ceb8c, dim: 0x9696a0, bg: 0x1a1a1e },
};
const AMBER = 0xff9e28;                                        // palette.TX_COLOR (workflow.RXC)
// Blender composited the semi-transparent emission over the tile in LINEAR
// light; WebGL blends in the sRGB framebuffer, which darkens it (the plasma
// surface came out deep purple instead of the video's pastel). So the
// composite is precomputed in linear and drawn opaque.
function over(lin, alpha, bgHex) {
  const b = new THREE.Color(bgHex);
  return new THREE.Color(alpha * lin.r + (1 - alpha) * b.r, alpha * lin.g + (1 - alpha) * b.g, alpha * lin.b + (1 - alpha) * b.b);
}
function emission(hex, strength) {             // Blender emission under 'Standard'
  const c = new THREE.Color(hex);              // r160: linear working space
  return new THREE.Color(Math.min(1, c.r * strength), Math.min(1, c.g * strength), Math.min(1, c.b * strength));
}

// ── easing / tweening, with chapter cancellation and pause ────────────────
const ease = {
  io: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  out: (t) => 1 - Math.pow(1 - t, 3),
  back: (t) => 1 + 2.4 * Math.pow(t - 1, 3) + 1.4 * Math.pow(t - 1, 2),
  lin: (t) => t,
};
let runId = 0, paused = false, started = false, lastTouch = 0;
const IDLE_MS = 12000;
class Cancel extends Error {}
function frame() { return new Promise((r) => requestAnimationFrame(r)); }
async function hold(ms, id) {
  let t = 0, last = performance.now();
  while (t < ms) {
    await frame();
    if (id !== runId) throw new Cancel();
    const now = performance.now();
    if (!paused) t += now - last;
    last = now;
  }
}
async function tween(ms, id, fn, e = ease.io) {
  let t = 0, last = performance.now();
  fn(0);
  while (t < ms) {
    await frame();
    if (id !== runId) throw new Cancel();
    const now = performance.now();
    if (!paused) t += now - last;
    last = now;
    fn(e(Math.min(1, t / ms)));
  }
  fn(1);
}
const lerp = (a, b, t) => a + (b - a) * t;
const lerpV = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

function textSprite(text, colour, px = 64, weight = 600, family = 'system-ui, sans-serif') {
  const pad = 10, c = document.createElement('canvas');
  const g = c.getContext('2d');
  g.font = `${weight} ${px}px ${family}`;
  c.width = Math.ceil(g.measureText(text).width) + pad * 2;
  c.height = Math.ceil(px * 1.25) + pad * 2;
  const h = c.getContext('2d');
  h.font = `${weight} ${px}px ${family}`;
  h.fillStyle = colour; h.textBaseline = 'middle';
  h.fillText(text, pad, c.height / 2);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthTest: false, depthWrite: false }));
  s.userData.aspect = c.width / c.height;
  s.renderOrder = 10;
  return s;
}
function sizeSprite(s, h) { s.scale.set(h * s.userData.aspect, h, 1); }
function disc() {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d'), grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, '#fff'); grd.addColorStop(0.7, '#fff'); grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.beginPath(); g.arc(32, 32, 32, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function dispose(g) {
  while (g.children.length) {
    const o = g.children.pop();
    o.traverse && o.traverse((n) => {
      n.geometry && n.geometry.dispose();
      if (n.material) { n.material.map && n.material.map.dispose(); n.material.dispose(); }
    });
  }
}
// a rod from a to b that grows from a (scale.y = progress)
function rod(a, b, radius, colour, opacity = 1) {
  const d = new THREE.Vector3().subVectors(b, a), L = d.length();
  const g = new THREE.CylinderGeometry(radius, radius, L, 10, 1, true);
  g.translate(0, L / 2, 0);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: colour, transparent: opacity < 1, opacity }));
  m.position.copy(a);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  return m;
}

// ══════════════════════════════════════════════════════════════════════════
function boot() {
  let renderer, prenderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    prenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) { root.classList.add('no-webgl'); return; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  prenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  prenderer.setClearColor(0x000000, 0);
  sceneWrap.appendChild(renderer.domElement);
  pWrap.appendChild(prenderer.domElement);

  // ── the stage scene ────────────────────────────────────────────────────
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.05, 200);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.enabled = false;
  const amb = new THREE.AmbientLight(0xffffff, 1.35);
  const sun = new THREE.DirectionalLight(0xffffff, 1.35);
  scene.add(amb, sun);
  const sprite = disc();
  const G = { radar: new THREE.Group(), rt: new THREE.Group(), gs: new THREE.Group(), fan: new THREE.Group(),
              pts: new THREE.Group(), pth: new THREE.Group() };
  Object.values(G).forEach((g) => scene.add(g));
  let mesh = null, M = null;                         // mesh, and the radar/scene geometry
  const RT = {}, GS = {}, PT = {};
  const VIEWS = {};

  // ── the plot ───────────────────────────────────────────────────────────
  const pscene = new THREE.Scene();
  const pcam = new THREE.PerspectiveCamera(30, 5 / 4, 0.1, 200);
  const pctl = new OrbitControls(pcam, prenderer.domElement);
  pctl.enableDamping = true; pctl.enablePan = true; pctl.enableZoom = true;
  pctl.zoomSpeed = 0.75; pctl.panSpeed = 0.8;
  pctl.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY,
                        RIGHT: THREE.MOUSE.PAN };
  const panKey = (e) => {
    pctl.mouseButtons.LEFT = (e.metaKey || e.ctrlKey) ? THREE.MOUSE.PAN
                                                      : THREE.MOUSE.ROTATE;
  };
  addEventListener('keydown', panKey); addEventListener('keyup', panKey);
  addEventListener('blur', () => { pctl.mouseButtons.LEFT = THREE.MOUSE.ROTATE; });
  const PG = { axes: new THREE.Group(), surf: new THREE.Group(), marks: new THREE.Group() };
  Object.values(PG).forEach((g) => pscene.add(g));
  let pdata = null, marks = new Map(), surfMat = null;
  let pUserMoved = false;
  pctl.addEventListener('start', () => { pUserMoved = true; });

  // ── state ──────────────────────────────────────────────────────────────
  const fresh = () => ({ cam: null, rt: { r: 0, paths: 0 }, gs: { grow: 0, drop: 0, fan: 0, fade: 0 },
                         pts: { frac: 0, path: 0 } });
  let S = fresh();
  const P = { show: {}, surf: 0 };

  // ════════════════════════════════════════════════════════════════════
  // scene construction
  // ════════════════════════════════════════════════════════════════════
  function buildScene(meta, buf) {
    const o = meta.offsets;
    const view = (T, k) => new T(buf, o[k][0], o[k][1] / T.BYTES_PER_ELEMENT);
    const pos = view(Float32Array, 'pos'), nrm = view(Uint8Array, 'nrm');
    const mv = view(Float32Array, 'mv'), mi = view(Uint16Array, 'mi');
    const nv = meta.mesh.vertices, mp = new Float32Array(nv * 3);
    for (let i = 0; i < nv; i++) { const v = toThree(mv[3 * i], mv[3 * i + 1], mv[3 * i + 2]); v.toArray(mp, 3 * i); }
    const mg = new THREE.BufferGeometry();
    mg.setAttribute('position', new THREE.BufferAttribute(mp, 3));
    mg.setIndex(new THREE.BufferAttribute(mi.slice(), 1));
    mg.computeVertexNormals();
    mesh = new THREE.Mesh(mg, new THREE.MeshLambertMaterial({ color: STAGE[theme()].mesh, side: THREE.DoubleSide }));
    scene.add(mesh);

    const poses = meta.poses;
    const p = poses.find((q) => q.test) || poses[Math.floor(poses.length / 2)];
    const tx = toThree(p.pos[0], p.pos[1], p.pos[2]);
    const ax = toThree(p.az[0], p.az[1], p.az[2]).normalize();
    const bo = toThree(p.bore[0], p.bore[1], p.bore[2]).normalize();
    const el = toThree(p.el[0], p.el[1], p.el[2]).normalize();
    const b0 = meta.bbox_min, b1 = meta.bbox_max;
    const centre = toThree((b0[0] + b1[0]) / 2, (b0[1] + b1[1]) / 2, (b0[2] + b1[2]) / 2);
    const span = toThree(b1[0] - b0[0], b1[1] - b0[1], b1[2] - b0[2]).length();
    // the real array is ~10 cm across, invisible at scene scale: the glyph
    // is magnified, Tx and Rx either side of the board as in the deck
    const txp = tx.clone().addScaledVector(ax, 0.16), rxp = tx.clone().addScaledVector(ax, -0.16);
    M = { tx: txp, rx: rxp, o: tx, ax, bo, el, centre, span, halfAz: meta.fov.az_deg * 0.5 * D2R,
          halfEl: meta.fov.el_deg * 0.5 * D2R, rmax: 12.5, up: new THREE.Vector3(0, 1, 0) };
    sun.position.copy(centre).add(new THREE.Vector3(span * 0.3, span * 0.8, span * 0.5));
    sun.target.position.copy(centre); scene.add(sun.target);

    // camera views: everything is relative to the radar pose and the scene box
    const at = (b, u, a) => tx.clone().addScaledVector(bo, b).addScaledVector(M.up, u).addScaledVector(ax, a);
    const flat = (v) => v.toArray();
    VIEWS.wide = { p: flat(at(-6.2, 6.4, 3.2)), t: flat(at(6.0, -0.6, 0)), u: [0, 1, 0] };
    VIEWS.rt = { p: flat(at(-5.6, 3.3, 3.4)), t: flat(at(6.0, -0.4, -0.9)), u: [0, 1, 0] };
    VIEWS.gs = { p: flat(at(-3.6, 4.4, 4.0)), t: flat(at(6.0, -0.4, -0.6)), u: [0, 1, 0] };
    const fanC = tx.clone().addScaledVector(bo, M.rmax * 0.5);
    const bevUp = bo.clone().setY(0).normalize();
    // bird's-eye: the fan centred a little right of and below centre, clear of
    // the head (top left), the plot card (top right) and the dock (bottom left)
    const bevR = new THREE.Vector3().crossVectors(bevUp, new THREE.Vector3(0, 1, 0)).normalize();
    const bevT = fanC.clone().addScaledVector(bevR, -2.6).addScaledVector(bevUp, 1.0);
    VIEWS.bev = { p: flat(bevT.clone().add(new THREE.Vector3(0, 27.0, 0)).addScaledVector(bevUp, -0.8)),
                  t: flat(bevT), u: bevUp.toArray() };
    VIEWS.pts = { p: flat(at(-4.6, 3.4, -3.6)), t: flat(at(5.8, -0.3, 0.6)), u: [0, 1, 0] };
    // ONE camera for all three docked screenshots, so the tiles differ only
    // in the primitive, not the viewpoint. Each act pans here before its
    // shot is taken, and FINAL (the skip path) uses it too.
    VIEWS.shot = VIEWS.pts;

    buildRadar();
    buildRays(meta);
    buildSplats(pos, nrm, meta.points);
    buildFan();
    buildPoints(pos, meta.points);
  }

  function buildRadar() {
    dispose(G.radar);
    const C = STAGE[theme()];
    const ball = (p, col) => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.085, 20, 14),
      new THREE.MeshBasicMaterial({ color: col })); m.position.copy(p); return m; };
    G.radar.add(ball(M.tx, C.tx), ball(M.rx, C.rx));
    const lt = textSprite('Tx', '#' + new THREE.Color(C.tx).getHexString(), 64, 700);
    const lr = textSprite('Rx', '#' + new THREE.Color(C.rx).getHexString(), 64, 700);
    sizeSprite(lt, 0.22); sizeSprite(lr, 0.22);
    lt.position.copy(M.tx).add(new THREE.Vector3(0, 0.26, 0)).addScaledVector(M.ax, 0.12);
    lr.position.copy(M.rx).add(new THREE.Vector3(0, 0.26, 0)).addScaledVector(M.ax, -0.12);
    G.radar.add(lt, lr);
  }

  // Ray tracing: a stratified grid of rays over the radar's field of view,
  // traced against the mesh once (first hit only); sorted by hit distance so
  // the hits a wavefront of radius r has reached are always a prefix.
  function buildRays() {
    dispose(G.rt);
    const C = STAGE[theme()];
    const ray = new THREE.Raycaster(), dirs = [];
    const NA = 34, NE = 11, jr = mulberry(11);
    for (let i = 0; i < NA; i++) {
      for (let j = 0; j < NE; j++) {
        const a = -M.halfAz + (2 * M.halfAz) * (i + jr()) / NA, e = -M.halfEl * 1.6 + (3.2 * M.halfEl) * (j + jr()) / NE;
        dirs.push(new THREE.Vector3().addScaledVector(M.bo, Math.cos(a) * Math.cos(e))
          .addScaledVector(M.ax, Math.sin(a) * Math.cos(e)).addScaledVector(M.el, Math.sin(e)).normalize());
      }
    }
    const rays = dirs.map((d) => {
      ray.set(M.tx, d); ray.far = M.rmax;
      const h = ray.intersectObject(mesh, false)[0];
      let face = 0;
      if (h && h.face) { const n = h.face.normal.clone(); face = Math.abs(n.dot(d)); }
      return { d, dist: h ? h.distance : M.rmax, hit: !!h, face };
    }).sort((a, b) => a.dist - b.dist);
    RT.rays = rays;
    const lp = new Float32Array(rays.length * 6);
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(lp, 3));
    RT.lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: C.ray, transparent: true, opacity: 0.5 }));
    const hits = rays.filter((r) => r.hit);
    const hp = new Float32Array(hits.length * 3);
    hits.forEach((r, i) => M.tx.clone().addScaledVector(r.d, r.dist).toArray(hp, 3 * i));
    const hg = new THREE.BufferGeometry(); hg.setAttribute('position', new THREE.BufferAttribute(hp, 3));
    RT.hits = new THREE.Points(hg, new THREE.PointsMaterial({ color: C.ray, size: 0.13, map: sprite, alphaTest: 0.5 }));
    RT.hitDist = hits.map((r) => r.dist);
    // the wavefront: the field-of-view patch of a sphere about the Tx
    const ga = [], idx = [], NA2 = 28, NE2 = 8;
    for (let i = 0; i <= NA2; i++) for (let j = 0; j <= NE2; j++) {
      const a = -M.halfAz + 2 * M.halfAz * i / NA2, e = -M.halfEl * 1.6 + 3.2 * M.halfEl * j / NE2;
      const d = new THREE.Vector3().addScaledVector(M.bo, Math.cos(a) * Math.cos(e))
        .addScaledVector(M.ax, Math.sin(a) * Math.cos(e)).addScaledVector(M.el, Math.sin(e));
      ga.push(d.x, d.y, d.z);
    }
    for (let i = 0; i < NA2; i++) for (let j = 0; j < NE2; j++) {
      const k = i * (NE2 + 1) + j; idx.push(k, k + NE2 + 1, k + 1, k + 1, k + NE2 + 1, k + NE2 + 2);
    }
    const wg = new THREE.BufferGeometry(); wg.setAttribute('position', new THREE.Float32BufferAttribute(ga, 3)); wg.setIndex(idx);
    RT.wave = new THREE.Mesh(wg, new THREE.MeshBasicMaterial({ color: C.wave, transparent: true, opacity: 0.2,
      side: THREE.DoubleSide, depthWrite: false }));
    RT.wave.position.copy(M.tx);
    // three single-bounce return paths: the most radar-facing hits, one in
    // each third of the hit distances
    const third = Math.max(1, Math.floor(hits.length / 3));
    RT.paths = [0, 1, 2].map((k) => {
      const seg = hits.slice(k * third, (k + 1) * third);
      const best = seg.reduce((m, r) => (r.face > m.face ? r : m), seg[0]);
      const p = M.tx.clone().addScaledVector(best.d, best.dist);
      const g = new THREE.Group();
      g.add(rod(M.tx, p, 0.02, PATHS[k]), rod(p, M.rx, 0.02, PATHS[k]));
      g.userData.dist = best.dist;
      return g;
    });
    G.rt.add(RT.lines, RT.hits, RT.wave, ...RT.paths);
  }

  // Splatting: anisotropic Gaussians on the fitted points (pastel, as the
  // teaser tile), oriented by the point normals; "drop" collapses them in
  // elevation onto the radar's range-azimuth plane.
  function buildSplats(pos, nrm, N) {
    dispose(G.gs);
    const step = 17, n = Math.floor(N / step);
    const geo = new THREE.SphereGeometry(1, 14, 10);
    const mat = new THREE.MeshLambertMaterial({ transparent: true, opacity: 1 });
    const im = new THREE.InstancedMesh(geo, mat, n);
    GS.items = [];
    const rng = mulberry(7);
    const col = new THREE.Color();
    for (let k = 0; k < n; k++) {
      const i = k * step;
      const p = toThree(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
      const nr = toThree(nrm[3 * i] / 127.5 - 1, nrm[3 * i + 1] / 127.5 - 1, nrm[3 * i + 2] / 127.5 - 1).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), nr);
      const sc = new THREE.Vector3(0.13 + 0.15 * rng(), 0.10 + 0.12 * rng(), 0.045);
      const v = p.clone().sub(M.tx);
      const pr = p.clone().addScaledVector(M.el, -v.dot(M.el));      // elevation collapsed
      GS.items.push({ p, pr, q, sc, delay: rng() * 0.6, range: v.length(),
                      az: Math.atan2(v.dot(M.ax), v.dot(M.bo)) });
      im.setColorAt(k, col.setHex(SPLAT[k % SPLAT.length]));
    }
    im.instanceColor.needsUpdate = true;
    GS.mesh = im; GS.mat = mat;
    G.gs.add(im);
  }

  // The range-azimuth map the splats land on, drawn as a polar fan in the
  // radar's plane: every collapsed splat adds a Gaussian at its (range,
  // azimuth); log-compressed and coloured like the paper's RA maps.
  function buildFan() {
    dispose(G.fan);
    const NAz = 96, NR = 96, img = new Float32Array(NAz * NR);
    for (const s of GS.items) {
      if (Math.abs(s.az) > M.halfAz || s.range > M.rmax) continue;
      const ia = (s.az / M.halfAz * 0.5 + 0.5) * (NAz - 1), ir = s.range / M.rmax * (NR - 1);
      for (let a = Math.max(0, Math.floor(ia - 3)); a <= Math.min(NAz - 1, Math.ceil(ia + 3)); a++)
        for (let r = Math.max(0, Math.floor(ir - 3)); r <= Math.min(NR - 1, Math.ceil(ir + 3)); r++)
          img[r * NAz + a] += Math.exp(-((a - ia) ** 2 + (r - ir) ** 2) / 3.2);
    }
    const mx = Math.max(...img), data = new Uint8Array(NAz * NR * 4);
    for (let i = 0; i < img.length; i++) {
      const v = img[i] > 0 ? Math.max(0, 1 + Math.log10(img[i] / mx) / 2.2) : 0;   // 22 dB
      const [r, g, b] = hot(v);
      data.set([r, g, b, 255], 4 * i);
    }
    const tex = new THREE.DataTexture(data, NAz, NR, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace; tex.magFilter = THREE.LinearFilter; tex.needsUpdate = true;
    const pos = [], uv = [], idx = [], SA = 48, SR = 32;
    for (let i = 0; i <= SA; i++) for (let j = 0; j <= SR; j++) {
      const a = -M.halfAz + 2 * M.halfAz * i / SA, r = M.rmax * j / SR;
      const p = M.o.clone().addScaledVector(M.bo, r * Math.cos(a)).addScaledVector(M.ax, r * Math.sin(a));
      pos.push(p.x, p.y, p.z); uv.push(i / SA, j / SR);
    }
    for (let i = 0; i < SA; i++) for (let j = 0; j < SR; j++) {
      const k = i * (SR + 1) + j; idx.push(k, k + SR + 1, k + 1, k + 1, k + SR + 1, k + SR + 2);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); g.setIndex(idx);
    GS.fanMat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0, side: THREE.DoubleSide,
                                              depthTest: false, depthWrite: false });
    const fan = new THREE.Mesh(g, GS.fanMat); fan.renderOrder = 4;
    const lab = textSprite('range-azimuth map', STAGE[theme()].ink, 64, 700);
    sizeSprite(lab, 0.62);
    const eA = -M.halfAz;
    lab.position.copy(M.o).addScaledVector(M.bo, M.rmax * 0.62 * Math.cos(eA))
      .addScaledVector(M.ax, M.rmax * 0.62 * Math.sin(eA) - 2.2);
    lab.material.opacity = 0; GS.fanLab = lab;
    G.fan.add(fan, lab);
  }

  // 3DPS: the fitted points, ordered by range so they sweep out from the radar
  function buildPoints(pos, N) {
    dispose(G.pts); dispose(G.pth);
    const C = STAGE[theme()];
    const order = [];
    for (let i = 0; i < N; i++) {
      const p = toThree(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
      order.push({ p, d: p.distanceTo(M.tx) });
    }
    order.sort((a, b) => a.d - b.d);
    const arr = new Float32Array(N * 3);
    order.forEach((o, i) => o.p.toArray(arr, 3 * i));
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    PT.points = new THREE.Points(g, new THREE.PointsMaterial({ color: C.pts, size: 0.075, map: sprite, alphaTest: 0.5 }));
    PT.n = N;
    G.pts.add(PT.points);
    // one Tx -> point -> Rx path, to a point near the boresight at mid range
    let best = order[0], bs = -1;
    for (const o of order) {
      const v = o.p.clone().sub(M.tx); const c = v.normalize().dot(M.bo);
      if (o.d > 4 && o.d < 7 && c > bs) { bs = c; best = o; }
    }
    PT.path = new THREE.Group();
    PT.path.add(rod(M.tx, best.p, 0.02, C.tx), rod(best.p, M.rx, 0.02, C.rx));
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    dot.position.copy(best.p); PT.path.add(dot);
    G.pth.add(PT.path);
  }

  // ════════════════════════════════════════════════════════════════════
  // the stage as a function of S
  // ════════════════════════════════════════════════════════════════════
  const _v = new THREE.Vector3();
  function applyState(st) {
    if (!M) return;
    if (st.cam) {
      camera.position.fromArray(st.cam.p); camera.up.fromArray(st.cam.u).normalize();
      camera.lookAt(_v.fromArray(st.cam.t));
      controls.target.fromArray(st.cam.t);
    }
    // ray tracing
    const r = st.rt.r;
    G.rt.visible = r > 0.001 || st.rt.paths > 0;
    if (G.rt.visible) {
      const a = RT.lines.geometry.attributes.position.array;
      RT.rays.forEach((ry, i) => {
        M.tx.toArray(a, 6 * i);
        _v.copy(M.tx).addScaledVector(ry.d, Math.min(r, ry.dist)).toArray(a, 6 * i + 3);
      });
      RT.lines.geometry.attributes.position.needsUpdate = true;
      let k = 0; while (k < RT.hitDist.length && RT.hitDist[k] <= r) k++;
      RT.hits.geometry.setDrawRange(0, k);
      RT.wave.scale.setScalar(Math.max(0.01, r));
      RT.wave.visible = r > 0.05 && r < M.rmax * 0.999;
      RT.wave.material.opacity = 0.24 * (1 - r / M.rmax) + 0.03;
      RT.lines.material.opacity = 0.6 - 0.15 * st.rt.paths;
      RT.paths.forEach((g, i) => {
        const t = Math.max(0, Math.min(1, st.rt.paths * 3 - i));
        g.visible = t > 0;
        g.children[0].scale.y = Math.min(1, t * 2); g.children[1].scale.y = Math.max(0.001, t * 2 - 1);
        g.children[1].visible = t > 0.5;
      });
    }
    // splats
    const gs = st.gs;
    G.gs.visible = gs.grow > 0 && gs.fade < 1;
    if (G.gs.visible) {
      const m4 = new THREE.Matrix4(), sc = new THREE.Vector3(), q = new THREE.Quaternion(), flatQ = new THREE.Quaternion()
        .setFromUnitVectors(new THREE.Vector3(0, 0, 1), M.el);
      GS.items.forEach((it, k) => {
        const g = ease.back(Math.max(0, Math.min(1, (gs.grow - it.delay) / 0.4)));
        const d = ease.io(Math.max(0, Math.min(1, gs.drop * 1.25 - it.delay * 0.4)));
        _v.copy(it.p).lerp(it.pr, d);
        q.copy(it.q).slerp(flatQ, d);
        sc.copy(it.sc).multiplyScalar(Math.max(0.0001, g));
        sc.z = Math.max(0.0001, g * lerp(it.sc.z, 0.012, d));
        m4.compose(_v, q, sc);
        GS.mesh.setMatrixAt(k, m4);
      });
      GS.mesh.instanceMatrix.needsUpdate = true;
      GS.mat.opacity = 1 - gs.fade;
      GS.mat.transparent = gs.fade > 0;
    }
    G.fan.visible = gs.fan > 0;
    if (G.fan.visible) { GS.fanMat.opacity = 0.94 * gs.fan; GS.fanLab.material.opacity = gs.fan; }
    // points
    G.pts.visible = st.pts.frac > 0;
    if (G.pts.visible) PT.points.geometry.setDrawRange(0, Math.floor(PT.n * st.pts.frac));
    G.pth.visible = st.pts.path > 0;
    if (G.pth.visible) {
      const t = st.pts.path;
      PT.path.children[0].scale.y = Math.min(1, t * 2);
      PT.path.children[1].visible = t > 0.5;
      PT.path.children[1].scale.y = Math.max(0.001, t * 2 - 1);
    }
  }

  // the final state of each chapter, for a tile re-render after a theme flip
  const FINAL = {
    mc: () => ({ cam: VIEWS.shot, rt: { r: M.rmax * 0.78, paths: 1 }, gs: { grow: 0, drop: 0, fan: 0, fade: 0 }, pts: { frac: 0, path: 0 } }),
    inr: () => ({ cam: VIEWS.shot, rt: { r: 0, paths: 0 }, gs: { grow: 1.6, drop: 0, fan: 0, fade: 0 }, pts: { frac: 0, path: 0 } }),
    pts: () => ({ cam: VIEWS.shot, rt: { r: 0, paths: 0 }, gs: { grow: 0, drop: 0, fan: 0, fade: 0 }, pts: { frac: 1, path: 1 } }),
  };
  function cropRect() {                              // stage px: 4:3 at full height, centred on the band
    const W = stage.clientWidth, H = stage.clientHeight, cw = Math.min(W, H * 4 / 3);
    return { x: Math.max(0, Math.min(W - cw, midX * W - cw / 2)), y: 0, w: cw, h: H };
  }
  const cropCanvas = document.createElement('canvas');
  function grab() {
    const src = renderer.domElement, k = src.width / stage.clientWidth, r = cropRect();
    cropCanvas.width = Math.round(r.w * k); cropCanvas.height = Math.round(r.h * k);
    cropCanvas.getContext('2d').drawImage(src, r.x * k, r.y * k, r.w * k, r.h * k, 0, 0, cropCanvas.width, cropCanvas.height);
    return cropCanvas.toDataURL('image/jpeg', 0.9);
  }
  function snapshot(st) {
    applyState(st); renderer.render(scene, camera);
    const url = grab();
    applyState(S);
    return url;
  }

  // ════════════════════════════════════════════════════════════════════
  // the plot (deck geometry: x = log runtime 8.4, y = memory 6.0 deep,
  // z = accuracy 5.2 up; three.js: (x, acc, -mem))
  // ════════════════════════════════════════════════════════════════════
  // PDIST 19.2 framed the axes but clipped their captions against the
  // card; the captions reach x = -1.6 and acc = -1.5, ~12% beyond the box
  const XL = 8.4, YL = 6.0, ZL = 5.2, PDIST = 21.8;
  const pv = (x, mem, acc) => new THREE.Vector3(x, acc, -mem);
  function px(t, d) { return (Math.log10(t) - Math.log10(d.t[0])) / (Math.log10(d.t[1]) - Math.log10(d.t[0])) * XL; }
  const pza = (a, d) => (a - d.a[0]) / (d.a[1] - d.a[0]) * ZL;
  const pym = (g, d) => (g - d.m[0]) / (d.m[1] - d.m[0]) * YL;
  function plotCam(yaw, pitch, dist) {         // the deck's cam_key, mapped to three.js
    const T = pv((-1.6 + XL + 0.35) / 2, (0 + YL + 0.35) / 2,
                 (-1.5 + ZL + 0.85) / 2 + 0.35);
    const ya = yaw * D2R, pa = pitch * D2R;
    const off = pv(-Math.sin(ya) * Math.cos(pa) * dist, -Math.cos(ya) * Math.cos(pa) * dist, Math.sin(pa) * dist);
    return { p: T.clone().add(off), t: T };
  }
  // The surface is CONTEXT, so it is desaturated: a two-stop slate ramp,
  // dark at low accuracy and light at high, carrying the height reading
  // without competing with the red / amber / green method markers.
  const SURF_RAMP = {
    light: [[0x34, 0x44, 0x5c], [0xb8, 0xc9, 0xdc]],
    dark:  [[0x26, 0x33, 0x47], [0x8f, 0xa6, 0xc0]],
  };
  function surfColour(t) {
    const [a, b] = SURF_RAMP[theme()];
    const u = Math.max(0, Math.min(1, t));
    const c = new THREE.Color();
    c.setRGB(...[0, 1, 2].map((k) => {
      const v = (a[k] + (b[k] - a[k]) * u) / 255;
      return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    }));
    return c;
  }

  function buildPlot() {
    [PG.axes, PG.surf, PG.marks].forEach(dispose);
    marks.clear();
    const C = PLOT[theme()], d = pdata.domain;
    const hex = (h) => '#' + new THREE.Color(h).getHexString();
    // axes (thin rods, the deck's tubes), grid on the two back faces
    const ax = new THREE.MeshBasicMaterial({ color: C.axis });
    const axis = (a, b) => { const m = rod(a, b, 0.022, C.axis); PG.axes.add(m); };
    axis(pv(0, 0, 0), pv(XL + 0.35, 0, 0)); axis(pv(0, 0, 0), pv(0, 0, ZL + 0.35)); axis(pv(0, 0, 0), pv(0, YL + 0.35, 0));
    ax.dispose();
    const gl = [];
    for (const t of d.t_ticks) { const x = px(t, d); gl.push(...pv(x, 0, 0).toArray(), ...pv(x, 0, ZL).toArray()); }
    for (const a of d.a_ticks) { const z = pza(a, d); gl.push(...pv(0, 0, z).toArray(), ...pv(XL, 0, z).toArray());
                                 gl.push(...pv(0, 0, z).toArray(), ...pv(0, YL, z).toArray()); }
    for (const m of d.m_ticks) { const y = pym(m, d); gl.push(...pv(0, y, 0).toArray(), ...pv(0, y, ZL).toArray()); }
    // the acc = 0 floor (runtime x memory) -- the third face, previously bare
    for (const t of d.t_ticks) { const x = px(t, d); gl.push(...pv(x, 0, 0).toArray(), ...pv(x, YL, 0).toArray()); }
    for (const m of d.m_ticks) { const y = pym(m, d); gl.push(...pv(0, y, 0).toArray(), ...pv(XL, y, 0).toArray()); }
    PG.axes.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(gl, 3)),
      new THREE.LineBasicMaterial({ color: C.grid })));
    const put = (txt, p, h, col = C.axis, w = 500) => { const s = textSprite(txt, hex(col), 64, w); sizeSprite(s, h); s.position.copy(p); PG.axes.add(s); };
    for (const t of d.t_ticks) put(String(t), pv(px(t, d), -0.62, -0.46), 0.56);
    for (const a of d.a_ticks) put(a.toFixed(1), pv(-0.78, 0, pza(a, d)), 0.56);
    for (const m of d.m_ticks.slice(1)) put(String(m), pv(-0.62, pym(m, d), -0.26), 0.50);
    put('wall-clock  (min, log)', pv(XL * 0.60, -1.45, -1.30), 0.62, C.ink, 500);
    put('peak GPU  (GB)', pv(-1.55, YL * 0.62, -0.95), 0.62, C.ink, 500);
    put('held-out |RA| Pearson', pv(-1.2, 0, ZL + 0.85), 0.62, C.ink, 500);
    // surface: the 28 x 28 fit on a low-chroma cool ramp (see SURF_RAMP)
    const s = pdata.surface, nu = s.t.length, nv = s.m.length, pos = [], col = [], idx = [];
    const lo = s.acc[0][0], hi = s.acc[nv - 1][nu - 1];
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const a = s.acc[j][i];
      pos.push(...pv(px(s.t[i], d), pym(s.m[j], d), pza(a, d)).toArray());
      const o = over(surfColour((a - lo) / (hi - lo)), 0.62, C.bg);
      col.push(o.r, o.g, o.b);
    }
    for (let j = 0; j < nv - 1; j++) for (let i = 0; i < nu - 1; i++) {
      const k = j * nu + i; idx.push(k, k + 1, k + nu, k + 1, k + nu + 1, k + nu);
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    sg.setAttribute('color', new THREE.Float32BufferAttribute(col, 3)); sg.setIndex(idx);
    surfMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0,
                                            side: THREE.DoubleSide, depthWrite: true,
                                            polygonOffset: true, polygonOffsetFactor: 3, polygonOffsetUnits: 3 });
    PG.surf.add(new THREE.Mesh(sg, surfMat));
    // methods: emission spheres (baselines DIM_TEXT 1.7, mmIR amber 1.7,
    // 3DPS NORMAL_COLOR 2.4 and x1.15); hollow = no measured memory, alpha 0.32
    const R = 0.20;
    for (const m of pdata.methods) {
      const ours = m.name === pdata.ours;
      const hollow = m.gb === null || m.name === 'mmIR';
      const base = ours ? C.ours : (m.name === 'mmIR' ? AMBER : C.dim);
      const g = new THREE.Group();
      // no measured memory -> sit it where the FIT puts it, not on the
      // m = 0 wall where it floated clear of the surface (see pareto.json)
      const gb = m.gb === null ? pdata.dart_on_surface_gb : m.gb;
      g.position.copy(pv(px(m.min, d), pym(gb, d), pza(m.acc, d)));
      const e = emission(base, ours ? 2.4 : 1.7);
      const mat = new THREE.MeshBasicMaterial({ color: hollow ? over(e, 0.32, C.bg) : e });
      g.add(new THREE.Mesh(new THREE.SphereGeometry(R * (ours ? 1.15 : 1), 24, 16), mat));
      const lab = textSprite(m.name, hex(ours ? emission(C.ours, 2.0).getHex() : C.ink), 64, ours ? 700 : 500);
      sizeSprite(lab, ours ? 0.90 : 0.71);          // +22% for legibility
      const LO = { 'DART': [0.85, 0.45], 'Radar Fields': [0.0, 0.66], 'RadarSplat': [1.05, 0.48],
                   'Sionna-RT': [-1.05, 0.48], 'mmIR': [0.0, -0.66], '3DPS': [0.0, 0.78] }[m.name] || [0, 0.66];
      lab.position.set(LO[0], LO[1], 0);
      g.add(lab);
      // the "joins the plot" ring
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.26, 0.32, 40),
        new THREE.MeshBasicMaterial({ color: ours ? C.ours : (m.name === 'mmIR' ? AMBER : C.axis), transparent: true,
                                      opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
      g.add(ring);
      g.userData = { ring, lab };
      PG.marks.add(g); marks.set(m.name, g);
    }
    applyPlot();
  }
  let wig = 0;
  function applyPlot() {
    if (!pdata) return;
    marks.forEach((g, nm) => {
      const v = P.show[nm] || 0;
      g.visible = v > 0;
      const s = ease.back(Math.min(1, v * 1.4));
      g.children[0].scale.setScalar(Math.max(0.001, s));
      g.userData.lab.material.opacity = Math.max(0, Math.min(1, v * 2 - 0.6));
      const rv = Math.max(0, Math.min(1, v));
      g.userData.ring.material.opacity = rv < 1 ? 0.8 * (1 - rv) : 0;
      g.userData.ring.scale.setScalar(1 + 2.2 * rv);
      g.userData.ring.quaternion.copy(pcam.quaternion);
    });
    if (surfMat) { surfMat.opacity = P.surf; surfMat.transparent = P.surf < 1; PG.surf.visible = P.surf > 0; }
  }

  // ════════════════════════════════════════════════════════════════════
  // head, dock, apps
  // ════════════════════════════════════════════════════════════════════
  async function head(title, colour, id) {
    headEl.classList.add('out');
    await hold(260, id);
    headEl.textContent = title || '';
    headEl.style.setProperty('--h', colour || 'var(--fg)');
    if (title) headEl.classList.remove('out');
  }
  const slot = (fam) => root.querySelector(`.t-slot[data-fam="${fam}"]`);
  function dockInstant(fam) {
    const s = slot(fam);
    s.querySelector('img').src = snapshot(FINAL[fam]());
    s.classList.add('on');
  }
  async function dockFly(fam, id) {
    const s = slot(fam), shot = s.querySelector('.t-shot');
    renderer.render(scene, camera);
    const url = grab(), c0 = cropRect();
    const st = stage.getBoundingClientRect(), tr = shot.getBoundingClientRect();
    const fly = document.createElement('div');
    fly.className = 't-fly';
    fly.style.setProperty('--c', getComputedStyle(s).getPropertyValue('--c'));
    Object.assign(fly.style, { left: c0.x + 'px', top: c0.y + 'px', width: c0.w + 'px', height: c0.h + 'px' });
    const im = new Image(); im.src = url; fly.appendChild(im);
    stage.appendChild(fly);
    await frame(); await frame();
    Object.assign(fly.style, { left: (tr.left - st.left) + 'px', top: (tr.top - st.top) + 'px',
      width: tr.width + 'px', height: tr.height + 'px', borderRadius: '8px', borderWidth: '2px' });
    try { await hold(1050, id); } finally {
      s.querySelector('img').src = url;
      s.classList.add('on');
      fly.remove();
    }
  }
  function apps(on) {
    root.classList.toggle('apps', on);
    root.querySelectorAll('.t-app').forEach((el, i) => {
      clearTimeout(el._t);
      if (on) el._t = setTimeout(() => el.classList.add('on'), 250 + i * 420);
      else el.classList.remove('on');
    });
  }

  // ════════════════════════════════════════════════════════════════════
  // chapters: play (animated) and finish (instant final state)
  // ════════════════════════════════════════════════════════════════════
  const RED = 'var(--t-red)', GREEN = 'var(--t-green)';
  const camTo = async (v, ms, id) => {
    const a = S.cam || VIEWS.wide;
    await tween(ms, id, (t) => { S.cam = { p: lerpV(a.p, v.p, t), t: lerpV(a.t, v.t, t), u: lerpV(a.u, v.u, t) }; });
  };
  const addMarks = async (names, id) => {
    for (const nm of names) {
      await tween(780, id, (t) => { P.show[nm] = t; }, ease.lin);
      await hold(160, id);
    }
  };
  const CH = [
    { // 0 Scene
      async play(id) {
        await head('', null, id);
        const a = { ...VIEWS.shot, p: lerpV(VIEWS.shot.p, VIEWS.wide.p, 0.45) }, b = VIEWS.shot;
        S.cam = a;
        await tween(2600, id, (t) => { S.cam = { p: lerpV(a.p, b.p, t), t: a.t, u: a.u }; });
      },
      finish() { S.cam = VIEWS.shot; },
    },
    { // 1 Ray tracing
      async play(id) {
        await head('Prior Methods: Mesh + MC Ray-Tracing', RED, id);
        await camTo(VIEWS.shot, 900, id);      // a no-op unless the user skipped in
        await tween(2900, id, (t) => { S.rt.r = t * M.rmax * 0.78; }, ease.lin);
        await tween(1500, id, (t) => { S.rt.paths = t; }, ease.lin);
        pCard.classList.add('on');
        await hold(350, id);
        await addMarks(['Sionna-RT', 'mmIR'], id);
        await hold(450, id);
        await dockFly('mc', id);
        await tween(500, id, (t) => { S.rt.r = M.rmax * 0.78 * (1 - t); S.rt.paths = 1 - t; });
      },
      finish() { P.show['Sionna-RT'] = P.show.mmIR = 1; pCard.classList.add('on'); dockInstant('mc'); S.rt = { r: 0, paths: 0 }; S.cam = VIEWS.shot; },
    },
    { // 2 Splatting
      async play(id) {
        await head('Prior Methods: NeRF/3DGS Primitives', RED, id);
        await camTo(VIEWS.shot, 900, id);      // a no-op unless the user skipped in
        await tween(1900, id, (t) => { S.gs.grow = t * 1.6; }, ease.lin);
        await hold(400, id);
        await camTo(VIEWS.bev, 2200, id);
        await tween(2200, id, (t) => { S.gs.drop = t; S.gs.fan = Math.max(0, (t - 0.35) / 0.65); }, ease.lin);
        await tween(900, id, (t) => { S.gs.fade = 0.55 * t; });
        await addMarks(['DART', 'Radar Fields', 'RadarSplat'], id);
        await hold(400, id);
        await tween(800, id, (t) => { S.gs.fan = 1 - t; S.gs.drop = 1 - t;
                                      S.gs.fade = 0.55 * (1 - t); }, ease.lin);
        await camTo(VIEWS.shot, 1200, id);
        await hold(250, id);
        await dockFly('inr', id);
        await tween(500, id, (t) => { S.gs.fade = t; });
        S.gs = { grow: 0, drop: 0, fan: 0, fade: 0 };
      },
      finish() { ['DART', 'Radar Fields', 'RadarSplat'].forEach((n) => { P.show[n] = 1; }); dockInstant('inr');
                 S.gs = { grow: 0, drop: 0, fan: 0, fade: 0 }; S.cam = VIEWS.shot; },
    },
    { // 3 3DPS
      async play(id) {
        await head('3DPS: Point Primitives', GREEN, id);
        await camTo(VIEWS.shot, 900, id);      // a no-op: act 2 panned back here
        await tween(2600, id, (t) => { S.pts.frac = t; }, ease.out);
        await tween(1100, id, (t) => { S.pts.path = t; }, ease.lin);
        await addMarks(['3DPS'], id);
        await hold(450, id);
        await dockFly('pts', id);
      },
      finish() { P.show['3DPS'] = 1; dockInstant('pts'); S.pts = { frac: 1, path: 1 }; S.cam = VIEWS.shot; },
    },
    { // 4 Frontier
      async play(id) {
        await head('', null, id);
        pCard.classList.add('glow');
        await tween(1400, id, (t) => { P.surf = t; });
        await camTo(VIEWS.wide, 2200, id);
        await hold(2600, id);
        pCard.classList.remove('glow');
      },
      finish() { P.surf = 1; S.cam = VIEWS.wide; },
    },
    { // 5 Applications
      async play(id) {
        await head('', null, id);
        apps(true);
        await hold(4200, id);
      },
      finish() { apps(true); },
    },
  ];

  function resetAll() {
    S = fresh(); S.cam = VIEWS.wide;
    for (const k of Object.keys(P.show)) P.show[k] = 0;
    P.surf = 0;
    pCard.classList.remove('on', 'glow');
    root.querySelectorAll('.t-slot').forEach((s) => s.classList.remove('on'));
    root.querySelectorAll('.t-fly').forEach((f) => f.remove());
    apps(false);
  }
  async function playFrom(ch) {
    started = true;
    const id = ++runId;
    paused = false;
    controls.enabled = false;
    resetAll();
    for (let k = 0; k < ch; k++) CH[k].finish();
    try {
      for (let k = ch; k < CH.length; k++) await CH[k].play(id);
      controls.enabled = true;
      // hold the last frame; replay once nobody has touched the plot or the
      // scene for IDLE_MS
      lastTouch = Math.max(lastTouch, performance.now());
      while (performance.now() - lastTouch < IDLE_MS) await hold(500, id);
      playFrom(0);
    } catch (e) { if (!(e instanceof Cancel)) throw e; }
  }
  function showFinal() {
    started = true;
    ++runId; resetAll();
    CH.forEach((c) => c.finish());
    controls.enabled = true;
    headEl.textContent = ''; headEl.classList.add('out');
  }

  // ── theme, resize, controls ──────────────────────────────────────────
  function applyTheme() {
    if (!M) return;
    const C = STAGE[theme()];
    renderer.setClearColor(C.bg, 1);
    mesh.material.color.setHex(C.mesh);
    buildRadar();
    RT.lines.material.color.setHex(C.ray); RT.hits.material.color.setHex(C.ray); RT.wave.material.color.setHex(C.wave);
    PT.points.material.color.setHex(C.pts);
    PT.path.children[0].material.color.setHex(C.tx); PT.path.children[1].material.color.setHex(C.rx);
    const lab = GS.fanLab; G.fan.remove(lab); lab.material.map.dispose(); lab.material.dispose();
    const nl = textSprite('range-azimuth map', C.ink, 64, 700); sizeSprite(nl, 0.62);
    nl.position.copy(lab.position); nl.material.opacity = lab.material.opacity; GS.fanLab = nl; G.fan.add(nl);
    if (pdata) buildPlot();
    root.querySelectorAll('.t-slot.on').forEach((s) => { s.querySelector('img').src = snapshot(FINAL[s.dataset.fam]()); });
  }
  window.addEventListener('tdps-theme', applyTheme);
  const dockEl = document.getElementById('t-dock');
  let midX = 0.5;                                    // centre of the free band, as a stage fraction
  function resize() {
    const w = sceneWrap.clientWidth, h = sceneWrap.clientHeight;
    if (w && h) {
      renderer.setSize(w, h, false); camera.aspect = w / h;
      const st = stage.getBoundingClientRect(), cd = pCard.getBoundingClientRect();
      const band = Math.max(1, cd.left - st.left);
      const dockH = dockEl.getBoundingClientRect().height;
      midX = (band / 2) / st.width;
      // centre the scene in the band left of the card, and lift it clear of
      // the row of shots along the bottom
      camera.setViewOffset(w, h, -(midX - 0.5) * w, dockH / 2, w, h);
      camera.updateProjectionMatrix();
      headEl.style.left = (midX * 100) + '%';
      // keep the title inside that band whatever the string length
      headEl.style.maxWidth = Math.max(90, band - 22) + 'px';
    }
    const pw = pWrap.clientWidth, ph = pWrap.clientHeight;
    if (pw && ph) {
      prenderer.setSize(pw, ph, false); pcam.aspect = pw / ph;
      // the deck's 52 mm lens is 38.2 deg horizontal; keep the plot's
      // horizontal extent, so the card's taller aspect only adds headroom
      const hf = 2 * Math.atan(18 / 52);
      pcam.fov = 2 * Math.atan(Math.tan(hf / 2) / pcam.aspect) / D2R;
      pcam.updateProjectionMatrix();
    }
  }
  window.addEventListener('resize', resize);
  const touch = () => { lastTouch = performance.now(); };
  [pWrap, sceneWrap].forEach((el) => {
    el.addEventListener('pointerdown', touch);
    el.addEventListener('pointermove', (e) => { if (e.buttons) touch(); });
    el.addEventListener('wheel', touch, { passive: true });
  });

  // ── load and go ──────────────────────────────────────────────────────
  Promise.all([
    fetch(`./static/viewer/${SCENE}.json`).then((r) => r.json()),
    fetch(`./static/viewer/${SCENE}.bin`).then((r) => r.arrayBuffer()),
    fetch('./static/data/pareto.json').then((r) => r.json()),
  ]).then(([meta, buf, pd]) => {
    pdata = pd;
    buildScene(meta, buf);
    renderer.setClearColor(STAGE[theme()].bg, 1);
    buildPlot();
    resize();
    S.cam = VIEWS.wide;
    const pc = plotCam(26, 17, PDIST);
    pcam.position.copy(pc.p); pctl.target.copy(pc.t); pcam.lookAt(pc.t);
    // the reader may already have picked a chapter: only auto-start an idle stage
    const go = () => { if (!started) (REDUCED ? showFinal() : playFrom(0)); };
    window.__teaser = { playFrom, showFinal, state: () => ({ S, P, runId }), setPaused: (v) => { paused = v; } };
    if (!('IntersectionObserver' in window)) { go(); return; }
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); go(); } },
                                        { rootMargin: '-15% 0px' });
    io.observe(stage);
  }).catch((err) => {
    console.error('[teaser] could not start:', err);
    root.classList.add('no-webgl');
  });

  (function loop() {
    requestAnimationFrame(loop);
    if (!M) return;
    applyState(S);
    if (controls.enabled) controls.update();
    renderer.render(scene, camera);
    if (pdata) {
      // the deck's slow wiggle about its 3-D vantage, until the reader drags
      wig += 0.0045;
      if (!pUserMoved) {
        const pc = plotCam(26 + 4.4 * Math.cos(wig), 17 + 3.2 * Math.sin(wig), PDIST);
        pcam.position.copy(pc.p); pctl.target.copy(pc.t);
      }
      pctl.update();
      applyPlot();
      prenderer.render(pscene, pcam);
    }
  })();
}

function hot(v) {                                   // black -> red -> yellow -> white
  const r = Math.min(1, v * 2.2), g = Math.max(0, Math.min(1, v * 2.2 - 0.9)), b = Math.max(0, Math.min(1, v * 3 - 2.1));
  return [Math.round(255 * r), Math.round(255 * g), Math.round(255 * b)];
}
function mulberry(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
                 t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// LAST: boot() reads module-level consts, so it runs after the module body
if (root && stage) boot();
