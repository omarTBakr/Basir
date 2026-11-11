// 3D map tab: the live map of a phone over /ws/map (snapshot, then new voxels), a saved MLflow run, or a local .ply.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const $ = (id) => document.getElementById(id);
const box = $("map"), hud = $("hud");
const params = new URLSearchParams(location.search);
const store = (k, v) => { try { v === undefined ? (v = localStorage.getItem(k)) : localStorage.setItem(k, v); } catch (e) {} return v; };

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
box.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c0e10);
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 200);
camera.position.set(3, 3, 3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.GridHelper(10, 50, 0x2e3237, 0x1d2024));
const world = new THREE.Group();  // map data in its own frame; rotated so its up axis is three.js +Y
scene.add(world);

// Points: preallocated buffers, grown by doubling; `count` drawn.
let capacity = 0, count = 0, positions, colours, rgb;
const geometry = new THREE.BufferGeometry();
const points = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.01, vertexColors: true }));
world.add(points);
const trailGeometry = new THREE.BufferGeometry();
const trail = new THREE.Line(trailGeometry, new THREE.LineBasicMaterial({ color: 0xff4d4d }));
world.add(trail);
const cameraMarker = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.12, 4), new THREE.MeshBasicMaterial({ color: 0xffd23f, wireframe: true }));
cameraMarker.visible = false;
world.add(cameraMarker);
let trailPoints = [];

function reserve(n) {
  if (n <= capacity) return;
  const size = Math.max(n, capacity * 2, 1 << 18);
  const p = new Float32Array(size * 3), c = new Float32Array(size * 3), r = new Uint8Array(size * 3);
  if (positions) { p.set(positions.subarray(0, count * 3)); c.set(colours.subarray(0, count * 3)); r.set(rgb.subarray(0, count * 3)); }
  positions = p; colours = c; rgb = r; capacity = size;
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
}

function clearPoints() {
  count = 0; trailPoints = []; geometry.setDrawRange(0, 0);
  trailGeometry.setFromPoints([]); cameraMarker.visible = false; updateStats();
}

function colourOf(i) {
  if ($("cmode").value === "rgb") return [rgb[i * 3] / 255, rgb[i * 3 + 1] / 255, rgb[i * 3 + 2] / 255];
  const up = $("up").value === "z" ? 2 : 1;
  const t = THREE.MathUtils.clamp((positions[i * 3 + up] + 0.2) / 2.4, 0, 1);
  const c = new THREE.Color().setHSL(0.7 - 0.7 * t, 0.9, 0.55);
  return [c.r, c.g, c.b];
}

function append(xyz, cols) {
  const n = xyz.length / 3;
  reserve(count + n);
  positions.set(xyz, count * 3);
  rgb.set(cols, count * 3);
  for (let i = count; i < count + n; i++) colours.set(colourOf(i), i * 3);
  const first = count === 0;
  count += n;
  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
  geometry.setDrawRange(0, count);
  geometry.computeBoundingSphere();
  if (first) frame();
  updateStats();
}

function recolour() {
  for (let i = 0; i < count; i++) colours.set(colourOf(i), i * 3);
  if (geometry.attributes.color) geometry.attributes.color.needsUpdate = true;
}

function unpack(buffer) {  // "P" | u32 n | float32 xyz * n | uint8 rgb * n
  const view = new DataView(buffer);
  const n = view.getUint32(1, true);
  const xyz = new Float32Array(buffer.slice(5, 5 + n * 12));
  return [xyz, new Uint8Array(buffer, 5 + n * 12, n * 3)];
}

function frame() {
  const sphere = geometry.boundingSphere;
  if (!sphere) return;
  const centre = sphere.center.clone().applyMatrix4(world.matrixWorld);
  controls.target.copy(centre);
  camera.position.copy(centre).add(new THREE.Vector3(1, 0.9, 1).multiplyScalar(Math.max(sphere.radius, 0.5) * 1.6));
}

function setTrail(list) {
  trailPoints = list.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  trailGeometry.setFromPoints(trailPoints);
  const last = trailPoints[trailPoints.length - 1];
  cameraMarker.visible = !!last;
  if (last) cameraMarker.position.copy(last);
}

function updateStats(s = {}) {
  $("s-points").textContent = count.toLocaleString();
  if (s.keyframes !== undefined) $("s-keyframes").textContent = s.keyframes;
  if (s.fuse_ms !== undefined) $("s-fuse").textContent = `${s.fuse_ms} ms`;
}

function applyUp() {
  world.rotation.set($("up").value === "z" ? -Math.PI / 2 : 0, 0, 0);
  world.updateMatrixWorld();
  store("basir-map-up", $("up").value);
}

// ---- sources -------------------------------------------------------------
let source = "live", socket = null, device = params.get("device") || "";

function closeSocket() { if (socket) { const s = socket; socket = null; s.close(); } }

function openLive() {
  closeSocket(); clearPoints();
  if (!device) { hud.textContent = "Waiting for a phone with depth and pose…"; return; }
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/map/${encodeURIComponent(device)}`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => {
    if (typeof e.data !== "string") return append(...unpack(e.data));
    const m = JSON.parse(e.data);
    if (m.type === "meta") { setTrail(m.trail); updateStats(m); }
    if (m.type === "stats") {
      trailPoints.push(new THREE.Vector3(...m.camera)); setTrail(trailPoints.map((v) => [v.x, v.y, v.z]));
      if ($("follow").checked) controls.target.lerp(cameraMarker.getWorldPosition(new THREE.Vector3()), 0.25);
      updateStats(m);
    }
    if (m.type === "reset") clearPoints();
    if (m.type === "resync") openLive();
    hud.textContent = `● live · ${device} · ${count.toLocaleString()} points`;
  };
  ws.onopen = () => { hud.textContent = `● live · ${device} · waiting for keyframes`; };
  ws.onclose = () => { if (socket === ws) setTimeout(() => source === "live" && socket === ws && openLive(), 2000); };
  socket = ws;
}

async function openRun(runId) {
  closeSocket(); clearPoints();
  if (!runId) return;
  hud.textContent = "Loading saved run…";
  const response = await fetch(`/api/map/runs/${runId}/cloud`);
  if (!response.ok) { hud.textContent = `Could not load the run (${response.status})`; return; }
  append(...unpack(await response.arrayBuffer()));
  hud.textContent = `saved run · ${$("run").selectedOptions[0].textContent} · ${count.toLocaleString()} points`;
}

function openFile(file) {
  closeSocket(); clearPoints();
  const reader = new FileReader();
  reader.onload = () => {
    const g = new PLYLoader().parse(reader.result);
    const xyz = g.attributes.position.array;
    const n = xyz.length / 3, cols = new Uint8Array(n * 3);
    if (g.attributes.color) for (let i = 0; i < n * 3; i++) cols[i] = g.attributes.color.array[i] * 255;
    else cols.fill(200);
    append(new Float32Array(xyz), cols);
    hud.textContent = `file · ${file.name} · ${count.toLocaleString()} points`;
  };
  reader.readAsArrayBuffer(file);
}

async function refreshDevices() {
  try {
    const live = await (await fetch("/api/live")).json();
    const names = [...new Set([...live.devices.map((d) => d.device), ...(await (await fetch("/api/map/sessions")).json()).map((s) => s.device)])];
    if (!device && names.length) { device = names[0]; if (source === "live") openLive(); }
    const options = names.includes(device) || !device ? names : [device, ...names];
    $("device").innerHTML = options.length ? options.map((n) => `<option ${n === device ? "selected" : ""}>${n}</option>`).join("") : '<option value="">–</option>';
  } catch (e) { /* server restarting */ }
}

async function refreshRuns() {
  const runs = await (await fetch("/api/map/runs")).json();
  $("run").innerHTML = '<option value="">–</option>' + runs.map((r) =>
    `<option value="${r.run_id}">${r.name}${r.points ? ` · ${Math.round(r.points).toLocaleString()} pts` : ""}</option>`).join("");
}

function setSource(next) {
  source = next;
  document.querySelectorAll("#source button").forEach((b) => b.classList.toggle("on", b.dataset.source === next));
  $("reset").disabled = $("save").disabled = next !== "live";
  if (next === "live") openLive();
  if (next === "run") { closeSocket(); clearPoints(); hud.textContent = "Pick a saved run on the right."; refreshRuns(); }
  if (next === "file") $("file").click();
}

document.querySelectorAll("#source button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.source)));
$("file").addEventListener("change", (e) => e.target.files[0] && openFile(e.target.files[0]));
$("device").addEventListener("change", () => { device = $("device").value; setSource("live"); });
$("run").addEventListener("change", () => { source = "run"; document.querySelectorAll("#source button").forEach((b) => b.classList.toggle("on", b.dataset.source === "run")); openRun($("run").value); });
$("reset").addEventListener("click", () => device && fetch(`/api/map/${encodeURIComponent(device)}/reset`, { method: "POST" }));
$("save").addEventListener("click", async () => {
  if (!device) return;
  $("save").disabled = true; $("saved").textContent = "Saving…";
  const r = await fetch(`/api/map/${encodeURIComponent(device)}/save`, { method: "POST" });
  const body = await r.json();
  $("saved").textContent = r.ok ? `Saved ${body.voxels.toLocaleString()} points to ${body.ply}${body.run_id ? ` (MLflow run ${body.run_id.slice(0, 8)})` : ""}` : body.detail;
  $("save").disabled = false;
});
$("psize").addEventListener("input", () => { points.material.size = 0.005 * +$("psize").value; });
$("cmode").addEventListener("change", recolour);
$("up").addEventListener("change", () => { applyUp(); recolour(); });
$("trail").addEventListener("change", () => { trail.visible = cameraMarker.visible = $("trail").checked; });

function resize() { const w = box.clientWidth, h = box.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(box);
$("up").value = store("basir-map-up") || "z";
applyUp();
points.material.size = 0.005 * +$("psize").value;
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
refreshDevices();
setInterval(refreshDevices, 4000);
setSource("live");
