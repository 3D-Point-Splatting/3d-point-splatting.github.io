// Interactive fitted 3DPS scene. Loads static/viewer/<scene>.{json,bin}
// (tools/export_viewer.py) once the section scrolls into view. Everything
// shown is the shipped checkpoint and the shipped alignment configs: the
// 20,000 fitted oriented points, the decimated LiDAR mesh, and the nine
// cascaded-radar poses of the trajectory. Each scene is exported in its
// OWN frame -- origin at that scene's test-frame TX1, yaw removed about
// the array plane normal. Radar world (x right, y boresight, z up) maps
// to three.js y-up as (x, z, -y).
import * as THREE from './three.module.min.js';
import { OrbitControls } from './OrbitControls.js';

const wrap = document.getElementById('viewer-wrap');
const loading = document.getElementById('viewer-loading');
const chips = document.getElementById('viewer-chips');
const raBox = document.getElementById('ra-overlay');
const raBar = document.getElementById('ra-slider');
const raTicks = document.getElementById('ra-ticks');
const raRange = document.getElementById('ra-range');
const raLabel = document.getElementById('ra-label');
const raGt = document.getElementById('ra-gt');
const raRd = document.getElementById('ra-rd');
if (wrap) init();

function init() {
  let started = false;
  const go = () => { if (!started) { started = true; start(); } };
  if (!('IntersectionObserver' in window)) { go(); return; }
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); go(); }
  }, { rootMargin: '200px' });
  io.observe(wrap);
  // Some environments never deliver an IntersectionObserver callback (a
  // headless render with no compositor frames, for one), which would leave
  // the viewer stuck on "loading". Fall back to a plain geometry check.
  const near = () => {
    const r = wrap.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight || 0;
    return r.top < vh + 200 && r.bottom > -200;
  };
  const check = () => {
    if (started || !near()) return;
    io.disconnect();
    window.removeEventListener('scroll', check, true);
    window.removeEventListener('resize', check);
    go();
  };
  window.addEventListener('scroll', check, true);
  window.addEventListener('resize', check);
  setTimeout(check, 0);
  setTimeout(check, 1500);
}

const BG = { light: 0xf5f5f7, dark: 0x26272c };
const GREY = { light: [120, 122, 132], dark: [168, 170, 180] };
const MESH_C = { light: 0x9aa0ad, dark: 0x6d7280 };
const TRAIN_C = 0x2f7fd0, TEST_C = 0xe0584c, SEL_C = 0x1d8a3a;
const D2R = Math.PI / 180;

// three.js r160 colour management is on by default: Color.setHex() converts
// sRGB -> linear working space, but values written straight into a colour
// BufferAttribute are taken to be LINEAR ALREADY and re-encoded on output.
// These palettes are sRGB, so the cloud was rendering far lighter than
// authored -- measured against the light tile, the grey cloud came out at
// 1.82:1 instead of the 3.92:1 its values specify, and normals at 1.44:1.
// A 256-entry LUT puts them in the working space the attribute expects.
const S2L = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const v = i / 255;
  S2L[i] = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}
const s2l = (x) => S2L[x < 0 ? 0 : x > 255 ? 255 : x | 0];
// Residual per-theme corrections, once the colour space is right:
//   normals are rgb=(n+1)/2, inherently pastel -- 37% of them still sat
//   under 2:1 on the light tile, so they are darkened (hue preserved).
//   inferno's low end is near-black, which is what the DARK tile cannot
//   show -- 98% under 3:1 -- so it is lifted there.
const NRM_GAIN_LIGHT = 0.75;
const MAT_LIFT_DARK = 0.15;

function toThree(x, y, z) { return new THREE.Vector3(x, z, -y); }

function discTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d'); const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(0.72, 'rgba(255,255,255,1)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.beginPath(); g.arc(32, 32, 32, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

function start() {
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ antialias: true }); }
  catch (e) {
    loading.textContent = 'This browser cannot create a WebGL context, so the 3-D viewer is unavailable.';
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  wrap.insertBefore(renderer.domElement, wrap.firstChild);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.05, 500);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 1.5));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(1, 2, 1.5);
  scene.add(sun);
  const sprite = discTexture();

  let data = null, pts = null, meshObj = null;
  const poseGroup = new THREE.Group(), fovGroup = new THREE.Group(),
        trajGroup = new THREE.Group();
  scene.add(poseGroup, fovGroup, trajGroup);
  let colourMode = 'geometry';
  const show = { mesh: false, poses: true, fov: false, ra: false };
  const FOV_BASE = 0.10;            // every wedge, when none is selected
  let selectedIdx = -1;
  const poseMarks = [];

  function theme() { return (window.tdpsTheme && window.tdpsTheme()) || 'light'; }
  function applyTheme() {
    scene.background = new THREE.Color(BG[theme()]);
    if (data) { colourPoints(); if (meshObj) meshObj.material.color.setHex(MESH_C[theme()]); }
  }
  window.addEventListener('tdps-theme', applyTheme);
  applyTheme();

  function resize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  function colourPoints() {
    const N = data.meta.points, col = pts.geometry.attributes.color.array;
    const light = theme() === 'light';
    const grey = GREY[theme()];
    const k = light ? NRM_GAIN_LIGHT : 1;          // darken pastel normals
    const t = light ? 0 : MAT_LIFT_DARK;           // lift inferno's low end
    for (let i = 0; i < N; i++) {
      let r, g, b;
      if (colourMode === 'normals') {
        r = data.nrm[3 * i] * k; g = data.nrm[3 * i + 1] * k; b = data.nrm[3 * i + 2] * k;
      } else if (colourMode === 'materials') {
        r = data.mat[3 * i]; g = data.mat[3 * i + 1]; b = data.mat[3 * i + 2];
        r += (255 - r) * t; g += (255 - g) * t; b += (255 - b) * t;
      } else { r = grey[0]; g = grey[1]; b = grey[2]; }
      col[3 * i] = s2l(r); col[3 * i + 1] = s2l(g); col[3 * i + 2] = s2l(b);
    }
    pts.geometry.attributes.color.needsUpdate = true;
  }

  // ── radar poses, their fields of view, and the trajectory ──────────────
  // Each pose carries its full orthonormal board frame (azimuth, boresight,
  // elevation) from the config, ported from the 3DPS submission's
  // extract_board_frame_from_config. Orienting by the whole frame -- not
  // just the boresight -- is what makes the FOV's wide-azimuth /
  // narrow-elevation cross-section land the right way round; aligning the
  // axis alone leaves the roll about the boresight arbitrary.
  // Basis X = azimuth, Y = boresight, Z = elevation is right-handed and
  // matches build_fov_wireframe's _sph_to_xyz.
  function poseBasis(p) {
    const ax = toThree(p.az[0], p.az[1], p.az[2]).normalize();
    const by = toThree(p.bore[0], p.bore[1], p.bore[2]).normalize();
    const ez = toThree(p.el[0], p.el[1], p.el[2]).normalize();
    return new THREE.Matrix4().makeBasis(ax, by, ez);
  }

  function buildSensor() {
    [poseGroup, fovGroup, trajGroup].forEach((g) => {
      while (g.children.length) {
        const c = g.children.pop();
        c.traverse && c.traverse((o) => {
          o.geometry && o.geometry.dispose(); o.material && o.material.dispose();
        });
      }
    });
    const m = data.meta, span = sceneSpan();
    // the real board is ~10 cm across, invisible against a ~16 m scene, so
    // it is drawn at a uniform magnification; the element layout stays true
    const reach = Math.max(2.0, span * 0.30);
    const halfAz = m.fov.az_deg * 0.5 * D2R, halfEl = m.fov.el_deg * 0.5 * D2R;

    poseMarks.length = 0;
    m.poses.forEach((p, i) => {
      const at = toThree(p.pos[0], p.pos[1], p.pos[2]);
      const basis = poseBasis(p);
      const col = p.test ? TEST_C : TRAIN_C;

      // the sensor: one sphere, blue for a training pose and red for the
      // held-out one, with a translucent boresight arrow so the viewpoint's
      // ORIENTATION is readable without the arrow hiding the scene behind it
      const g = new THREE.Group();
      g.position.copy(at);
      g.quaternion.setFromRotationMatrix(basis);
      const r = span * 0.005;          // a third of v2: the markers dominated
      const sm = new THREE.MeshLambertMaterial({ color: col, transparent: true });
      sm.userData.base = 1.0;
      g.add(new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), sm));
      const bl = span * 0.0283;        // scaled with the sphere, same shape
      const am = new THREE.MeshBasicMaterial({ color: col, transparent: true,
                                               opacity: 0.45, depthWrite: false });
      am.userData.base = 0.45;
      const shaft = new THREE.Mesh(
        new THREE.CylinderGeometry(r * 0.26, r * 0.26, bl * 0.76, 10), am);
      shaft.position.set(0, bl * 0.38, 0);
      g.add(shaft);
      const tip = new THREE.Mesh(
        new THREE.ConeGeometry(r * 0.62, bl * 0.26, 12), am);
      tip.position.set(0, bl * 0.89, 0);
      g.add(tip);
      g.userData = { idx: i, pose: p, mats: [sm, am], bore: [shaft, tip] };
      poseGroup.add(g);
      poseMarks.push(g);

      // the FOV: apex on the board, wide in azimuth (local X), narrow in
      // elevation (local Z) -- the squash now follows the real axes
      const rad = reach * Math.tan(halfAz);
      const cg = new THREE.ConeGeometry(rad, reach, 32, 1, true);
      cg.translate(0, -reach / 2, 0);      // apex to the origin, along -Y
      cg.rotateX(Math.PI);                 // flip to +Y, the boresight
      cg.scale(1, 1, Math.tan(halfEl) / Math.tan(halfAz));
      const cm = new THREE.MeshBasicMaterial({
        color: col, transparent: true, opacity: FOV_BASE,
        side: THREE.DoubleSide, depthWrite: false });
      const fg = new THREE.Group();
      fg.add(new THREE.Mesh(cg, cm));
      fg.position.copy(at);
      fg.quaternion.setFromRotationMatrix(basis);
      fg.userData = { fill: cm };
      fovGroup.add(fg);
    });
    buildTicks();

    if (m.poses.length > 1) {
      const pts3 = m.poses.map((p) => toThree(p.pos[0], p.pos[1], p.pos[2]));
      const curve = new THREE.CatmullRomCurve3(pts3);
      const tube = new THREE.Mesh(
        new THREE.TubeGeometry(curve, 120, Math.max(0.03, span * 0.003), 8, false),
        new THREE.MeshBasicMaterial({ color: TRAIN_C, transparent: true,
          opacity: 0.38, depthWrite: false }));
      trajGroup.add(tube);
    }
    applyVisibility();
  }

  function sceneSpan() {
    const b0 = data.meta.bbox_min, b1 = data.meta.bbox_max;
    return Math.max(b1[0] - b0[0], b1[1] - b0[1], b1[2] - b0[2]);
  }

  function applyVisibility() {
    if (meshObj) meshObj.visible = show.mesh;
    poseGroup.visible = show.poses;
    // the cone already states orientation, so the arrow would be redundant
    poseMarks.forEach((g) => (g.userData.bore || []).forEach(
      (o) => { o.visible = !show.fov; }));
    fovGroup.visible = show.fov && show.poses;
    trajGroup.visible = show.poses;   // the path IS part of the pose set
    const on = show.ra && show.poses && !!data;
    if (raBox) raBox.hidden = !on;
    if (raBar) raBar.hidden = !on;
    if (on && selectedIdx < 0) selectIdx(defaultIdx());
    else if (!on) selectIdx(-1);
    else applyFocus();        // e.g. the FOV toggle, with a pose already picked
  }

  // ── the RA maps, overlaid on the canvas ───────────────────────────────
  function defaultIdx() {
    const k = data.meta.poses.findIndex((p) => p.test);
    return k < 0 ? 0 : k;
  }

  function buildTicks() {
    if (!raTicks) return;
    raTicks.innerHTML = '';
    raRange.max = String(data.meta.poses.length - 1);
    const n = data.meta.poses.length;
    data.meta.poses.forEach((p, i) => {
      const d = document.createElement('span');
      d.style.background = p.test ? '#e0584c' : '#2f7fd0';   // 4 blue, 1 red, 4 blue
      d.style.left = `${(i / (n - 1)) * 100}%`;              // dead under the thumb
      raTicks.appendChild(d);
    });
  }

  // the picked pose doubles AND everything else drops in opacity -- size
  // alone was not enough to pull one pose out of nine. The FOV cones get a
  // far harder split than the markers: nine overlapping wedges turn into a
  // haze, so the unselected ones go to a fifth of their resting value while
  // the selected one is pushed up to twice it.
  const DIM = 0.32;                 // markers, unselected
  const SEL = 2.0;                  // markers, selected (clamped at 1)
  const FOV_SEL = 0.22;             // the one wedge still drawn
  function applyFocus() {
    poseMarks.forEach((g, i) => {
      const sel = i === selectedIdx;
      const off = selectedIdx >= 0 && !sel;
      g.scale.setScalar(sel ? 2 : 1);
      (g.userData.mats || []).forEach((m) => {
        const b = m.userData.base;
        m.opacity = off ? b * DIM : Math.min(1, sel ? b * SEL : b);
        m.depthWrite = m.opacity >= 1;
      });
      const fg = fovGroup.children[i];
      if (fg) {
        // the other eight wedges are not dimmed, they are REMOVED: eight
        // double-sided surfaces at 0.02 still sum to a haze the selected
        // one has to out-shout, and no opacity ratio fixes that
        fg.visible = !off;
        fg.userData.fill.opacity = sel ? FOV_SEL : FOV_BASE;
      }
    });
  }

  function selectIdx(k) {
    selectedIdx = k;
    applyFocus();
    if (k < 0 || !data) return;
    const p = data.meta.poses[k];
    if (raRange) raRange.value = String(k);
    if (raLabel) {
      const corr = (p.corr === null || p.corr === undefined) ? ''
        : `: corr ${Number(p.corr).toFixed(3)}`;
      raLabel.textContent =
        `frame #${k + 1} [${p.test ? 'held out' : 'train'}]${corr}`;
    }
    if (raGt && p.gt) raGt.src = `./static/viewer/${p.gt}`;
    if (raRd && p.rd) raRd.src = `./static/viewer/${p.rd}`;
  }

  // during an orbit/zoom the maps fade out so the scene reads through them,
  // then fade back in once the interaction stops
  // 'start'/'end' fire on both drag and wheel; 'change' is avoided because
  // damping keeps emitting it after the user has let go
  let settle = null;
  function dimOn() { clearTimeout(settle); [raBox, raBar].forEach((e) => e && e.classList.add('dim')); }
  function dimOff() {
    clearTimeout(settle);
    settle = setTimeout(() => {
      [raBox, raBar].forEach((e) => e && e.classList.remove('dim'));
    }, 380);
  }
  controls.addEventListener('start', dimOn);
  controls.addEventListener('end', dimOff);

  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  function pick(ev) {
    const t = ev.changedTouches ? ev.changedTouches[0] : ev;
    const r = renderer.domElement.getBoundingClientRect();
    ndc.x = ((t.clientX - r.left) / r.width) * 2 - 1;
    ndc.y = -((t.clientY - r.top) / r.height) * 2 + 1;
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObjects(poseGroup.children, true);
    if (hit.length) {
      let o = hit[0].object;
      while (o && !o.userData.pose) o = o.parent;      // climb to the marker
      if (o && o.parent === poseGroup) {
        if (!show.ra) { show.ra = true; raChip && raChip.classList.add('on'); }
        selectIdx(o.userData.idx);
        applyVisibility();
      }
    }
  }
  // only treat it as a click if the pointer barely moved (orbiting drags)
  let downAt = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!downAt) return;
    if (Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5 && show.poses) pick(e);
    downAt = null;
  });

  async function load(name) {
    loading.style.display = 'flex';
    selectedIdx = -1;
    try {
      const meta = await (await fetch(`./static/viewer/${name}.json`)).json();
      const buf = await (await fetch(`./static/viewer/${name}.bin`)).arrayBuffer();
      const o = meta.offsets;
      const view = (T, k) => new T(buf, o[k][0], o[k][1] / T.BYTES_PER_ELEMENT);
      data = { meta, pos: view(Float32Array, 'pos'), nrm: view(Uint8Array, 'nrm'),
               mat: view(Uint8Array, 'mat'), mv: view(Float32Array, 'mv'),
               mi: view(Uint16Array, 'mi') };

      if (pts) { scene.remove(pts); pts.geometry.dispose(); }
      const N = meta.points, g = new THREE.BufferGeometry(), p = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        p[3 * i] = data.pos[3 * i];
        p[3 * i + 1] = data.pos[3 * i + 2];
        p[3 * i + 2] = -data.pos[3 * i + 1];
      }
      g.setAttribute('position', new THREE.BufferAttribute(p, 3));
      g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
      pts = new THREE.Points(g, new THREE.PointsMaterial({
        size: 0.055, vertexColors: true, map: sprite,
        alphaTest: 0.5, transparent: false, sizeAttenuation: true }));
      scene.add(pts);

      // the LiDAR mesh, for context behind the fitted points
      if (meshObj) { scene.remove(meshObj); meshObj.geometry.dispose(); }
      const nv = meta.mesh.vertices, mg = new THREE.BufferGeometry();
      const mp = new Float32Array(nv * 3);
      for (let i = 0; i < nv; i++) {
        mp[3 * i] = data.mv[3 * i];
        mp[3 * i + 1] = data.mv[3 * i + 2];
        mp[3 * i + 2] = -data.mv[3 * i + 1];
      }
      mg.setAttribute('position', new THREE.BufferAttribute(mp, 3));
      mg.setIndex(new THREE.BufferAttribute(data.mi.slice(), 1));
      mg.computeVertexNormals();
      meshObj = new THREE.Mesh(mg, new THREE.MeshLambertMaterial({
        color: MESH_C[theme()], transparent: true, opacity: 0.30,
        side: THREE.DoubleSide, depthWrite: false }));
      scene.add(meshObj);

      buildSensor();

      const b0 = meta.bbox_min, b1 = meta.bbox_max;
      const c = toThree((b0[0] + b1[0]) / 2, (b0[1] + b1[1]) / 2, (b0[2] + b1[2]) / 2);
      const span = sceneSpan();
      controls.target.copy(c);
      camera.position.set(c.x + span * 0.35, c.y + span * 0.30, c.z + span * 0.75);
      controls.update();
      colourPoints();
      loading.style.display = 'none';
    } catch (err) {
      loading.textContent = 'Could not load this scene.';
    }
  }

  // ── chips ──────────────────────────────────────────────────────────────
  const sceneChips = chips.querySelectorAll('.chip[data-scene]');
  const colChips = chips.querySelectorAll('.chip[data-color]');
  const togChips = chips.querySelectorAll('.chip[data-toggle]');
  const raChip = chips.querySelector('.chip[data-toggle="ra"]');
  if (raRange) raRange.addEventListener('input', () => {
    selectIdx(Number(raRange.value));
  });
  sceneChips.forEach((c) => c.addEventListener('click', () => {
    sceneChips.forEach((d) => d.classList.toggle('on', d === c)); load(c.dataset.scene);
  }));
  colChips.forEach((c) => c.addEventListener('click', () => {
    colChips.forEach((d) => d.classList.toggle('on', d === c));
    colourMode = c.dataset.color; if (data) colourPoints();
  }));
  togChips.forEach((c) => {
    c.classList.toggle('on', !!show[c.dataset.toggle]);
    c.addEventListener('click', () => {
      show[c.dataset.toggle] = !show[c.dataset.toggle];
      c.classList.toggle('on', show[c.dataset.toggle]);
      if (c.dataset.toggle === 'poses' && !show.poses) show.ra = false;
      applyVisibility();
    });
  });
  sceneChips[0].classList.add('on'); colChips[0].classList.add('on');
  load(sceneChips[0].dataset.scene);

  (function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  })();
}
