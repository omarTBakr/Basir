// Basir phone page: pair with the laptop, then stream colour (+ depth and pose with WebXR AR) over the wire protocol v2.
(() => {
  const $ = (id) => document.getElementById(id);
  const store = {
    get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} },
    del: (k) => { try { localStorage.removeItem(k); } catch (e) {} },
  };
  const AR_FPS = 5, CAMERA_FPS = 10, COLOUR_WIDTH = 640, DEPTH_WIDTH = 160, MAX_BUFFERED = 1 << 20;

  const model = (navigator.userAgentData && navigator.userAgentData.platform) || (navigator.userAgent.match(/;\s*([^;)]+)\s+Build/) || [])[1] || "phone";
  let device = store.get("basir-device");
  if (!device) { device = `phone-${Math.random().toString(36).slice(2, 6)}`; store.set("basir-device", device); }
  $("who").textContent = `${model} · ${device}`;

  function show(id) {
    for (const s of ["connecting", "pairing", "ready", "streaming", "error"]) $(`s-${s}`).hidden = s !== id;
  }
  function fail(text) { $("error-text").textContent = text; show("error"); }

  // ---- pairing ------------------------------------------------------------
  async function pairFlow() {
    if (store.get("basir-token")) return ready();
    show("connecting");
    let ask;
    try {
      const r = await fetch("/pair", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ device, label: model }) });
      if (!r.ok) throw new Error((await r.json()).detail || r.status);
      ask = await r.json();
    } catch (e) { return fail(`Could not ask the laptop to pair (${e.message}). Is Basir running there?`); }
    $("code").textContent = ask.code;
    show("pairing");
    while (true) {
      await new Promise((ok) => setTimeout(ok, 1000));
      let status;
      try { status = await (await fetch(`/pair/${ask.request_id}`)).json(); } catch (e) { continue; }
      if (status.status === "approved") { store.set("basir-token", status.token); return ready(); }
      if (status.status === "denied") return fail("The laptop denied this phone.");
      if (status.detail) return pairFlow();  // expired: ask again
    }
  }

  async function ready() {
    show("ready");
    let ar = false;
    try { ar = !!(navigator.xr && await navigator.xr.isSessionSupported("immersive-ar")); } catch (e) {}
    $("start-ar").hidden = !ar;
    $("ready-text").textContent = ar
      ? "This phone is paired and supports AR: stream colour, depth and its position, so the laptop can build the 3D map."
      : "This phone is paired. Its browser has no AR (WebXR), so it can send colour only.";
    $("ar-note").textContent = ar ? "AR needs Google Play Services for AR (ARCore). Depth needs a phone with the ARCore Depth API." : "For depth and the 3D map use Chrome on an ARCore phone.";
  }

  // ---- wire protocol v2 -------------------------------------------------------
  let ws = null, seq = 0, sent = 0, sentTimes = [], busy = false;
  function openSocket() {
    return new Promise((resolve, reject) => {
      const url = `wss://${location.host}/ws/stream/${encodeURIComponent(device)}?token=${encodeURIComponent(store.get("basir-token"))}`;
      const socket = new WebSocket(url);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => resolve(socket);
      socket.onerror = () => reject(new Error("connection failed"));
      socket.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.latency_ms !== undefined) $("latency").textContent = Math.round(m.latency_ms);
      };
      socket.onclose = (e) => {
        if (ws !== socket) return;
        stop();
        if (e.code === 1008) { store.del("basir-token"); fail("The laptop no longer knows this phone. Tap Try again to pair it again."); }
        else fail(`The connection closed (${e.code}). Tap Try again.`);
      };
    });
  }

  function message(header, jpeg, depth) {
    if (depth) header.jpeg_bytes = jpeg.byteLength;
    const head = new TextEncoder().encode(JSON.stringify(header));
    const out = new Uint8Array(4 + head.length + jpeg.byteLength + (depth ? depth.byteLength : 0));
    new DataView(out.buffer).setUint32(0, head.length, false);
    out.set(head, 4);
    out.set(new Uint8Array(jpeg), 4 + head.length);
    if (depth) out.set(new Uint8Array(depth.buffer), 4 + head.length + jpeg.byteLength);
    return out.buffer;
  }

  function send(header, jpeg, depth) {
    if (!ws || ws.readyState !== 1 || ws.bufferedAmount > MAX_BUFFERED) return;
    ws.send(message({ seq: seq++, timestamp: performance.now() / 1000, ...header }, jpeg, depth));
    sent++;
    const now = performance.now();
    sentTimes = sentTimes.filter((t) => now - t < 2000).concat(now);
    $("sent").textContent = sent;
    $("fps").textContent = (sentTimes.length / 2).toFixed(1);
    $("overlay-text").textContent = `Streaming · ${(sentTimes.length / 2).toFixed(1)} frames/s · latency ${$("latency").textContent} ms`;
  }

  const jpegOf = (canvas) => new Promise((ok) => canvas.toBlob((b) => b.arrayBuffer().then(ok), "image/jpeg", 0.7));

  // ---- camera only (any browser) ----------------------------------------------
  let stream = null, timer = null;
  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      ws = await openSocket();
    } catch (e) { return fail(`Could not start the camera (${e.message}).`); }
    const video = $("video");
    video.srcObject = stream; video.hidden = false; await video.play();
    const canvas = document.createElement("canvas");
    $("mode-text").textContent = "Camera: colour only (no depth or position).";
    $("depth").textContent = "none";
    show("streaming");
    timer = setInterval(async () => {
      if (busy || !video.videoWidth) return;
      busy = true;
      canvas.width = COLOUR_WIDTH; canvas.height = Math.round(COLOUR_WIDTH * video.videoHeight / video.videoWidth);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      send({}, await jpegOf(canvas), null);
      busy = false;
    }, 1000 / CAMERA_FPS);
  }

  // ---- WebXR AR: colour (camera-access), depth (depth-sensing), pose -------------
  let session = null;
  const VERT = "attribute vec2 p; varying vec2 uv; void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }";
  const FRAG = "precision mediump float; varying vec2 uv; uniform sampler2D t; void main() { gl_FragColor = texture2D(t, vec2(uv.x, 1.0 - uv.y)); }";

  function program(gl) {
    const make = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const prog = gl.createProgram();
    gl.attachShader(prog, make(gl.VERTEX_SHADER, VERT)); gl.attachShader(prog, make(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    return { prog, buffer, loc: gl.getAttribLocation(prog, "p") };
  }

  function readColour(gl, glp, texture, width, height, target) {
    if (!target.fb || target.w !== width || target.h !== height) {
      target.tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, target.tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      target.fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target.tex, 0);
      Object.assign(target, { w: width, h: height, pixels: new Uint8ClampedArray(width * height * 4) });
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb);
    gl.viewport(0, 0, width, height);
    gl.useProgram(glp.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.bindBuffer(gl.ARRAY_BUFFER, glp.buffer);
    gl.enableVertexAttribArray(glp.loc);
    gl.vertexAttribPointer(glp.loc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, target.pixels);
    return new ImageData(target.pixels, width, height);
  }

  function intrinsics(p, width, height) {  // WebXR projection (column-major) -> pinhole, OpenCV image axes
    return { fx: p[0] * width / 2, fy: p[5] * height / 2, cx: (1 - p[8]) * width / 2, cy: (1 + p[9]) * height / 2, width, height };
  }

  function poseCv(m) {  // camera->world, WebXR camera (x right, y up, -z forward) -> OpenCV camera (y down, z forward), row-major
    const s = [1, -1, -1, 1], out = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out.push(m[c * 4 + r] * s[c]);
    return out;
  }

  function depthGrid(info, width, height) {  // metres -> uint16 mm, sampled on a view-aligned grid
    const out = new Uint16Array(width * height);
    for (let j = 0; j < height; j++) for (let i = 0; i < width; i++) {
      const d = info.getDepthInMeters((i + 0.5) / width, (j + 0.5) / height);
      out[j * width + i] = d > 0 && d < 65 ? Math.round(d * 1000) : 0;
    }
    return out;
  }

  async function startAR() {
    try {
      session = await navigator.xr.requestSession("immersive-ar", {
        requiredFeatures: ["local"],
        optionalFeatures: ["depth-sensing", "camera-access", "dom-overlay"],
        depthSensing: { usagePreference: ["cpu-optimized"], dataFormatPreference: ["luminance-alpha", "float32"] },
        domOverlay: { root: $("overlay") },
      });
      ws = await openSocket();
    } catch (e) { if (session) session.end(); return fail(`Could not start AR (${e.message}).`); }
    $("overlay").hidden = false;
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl", { xrCompatible: true });
    await gl.makeXRCompatible();
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
    const space = await session.requestReferenceSpace("local");
    const binding = typeof XRWebGLBinding === "function" ? new XRWebGLBinding(session, gl) : null;
    const glp = program(gl), target = {}, out = document.createElement("canvas");
    const hasDepth = (session.enabledFeatures || []).includes("depth-sensing");
    $("mode-text").textContent = `AR: colour${hasDepth ? ", depth" : ""} and position.`;
    $("depth").textContent = hasDepth ? "on" : "off";
    show("streaming");
    let last = 0;
    session.addEventListener("end", () => { session = null; $("overlay").hidden = true; if (ws) stop(); });
    session.requestAnimationFrame(function onFrame(time, frame) {
      if (!session) return;
      session.requestAnimationFrame(onFrame);
      const pose = frame.getViewerPose(space);
      if (!pose || busy || time - last < 1000 / AR_FPS) return;
      const view = pose.views[0];
      if (!view.camera || !binding) return;
      last = time; busy = true;
      const w = COLOUR_WIDTH, h = Math.round(COLOUR_WIDTH * view.camera.height / view.camera.width);
      const image = readColour(gl, glp, binding.getCameraImage(view.camera), w, h, target);
      gl.bindFramebuffer(gl.FRAMEBUFFER, session.renderState.baseLayer.framebuffer);
      let depth = null, depthMeta = null;
      if (hasDepth) {
        try {
          const info = frame.getDepthInformation(view);
          if (info) {
            const dh = Math.round(DEPTH_WIDTH * h / w);
            depth = depthGrid(info, DEPTH_WIDTH, dh);
            depthMeta = { width: DEPTH_WIDTH, height: dh, scale: 0.001 };
          }
        } catch (e) { /* depth not ready yet */ }
      }
      const header = { pose: poseCv(view.transform.matrix), intrinsics: intrinsics(view.projectionMatrix, w, h), depth: depthMeta };
      out.width = w; out.height = h;
      out.getContext("2d").putImageData(image, 0, 0);
      jpegOf(out).then((jpeg) => { send(header, jpeg, depth); busy = false; });
    });
  }

  function stop() {
    const socket = ws; ws = null;
    if (socket) socket.close();
    if (timer) clearInterval(timer);
    timer = null;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    if (session) session.end();
    $("video").hidden = true;
    busy = false;
    ready();
  }

  $("start-ar").addEventListener("click", startAR);
  $("start-camera").addEventListener("click", startCamera);
  $("stop").addEventListener("click", stop);
  $("overlay-stop").addEventListener("click", stop);
  $("retry").addEventListener("click", pairFlow);
  if (!window.isSecureContext) fail("Open this page with https:// (the address the laptop shows), not http://.");
  else pairFlow();
})();
