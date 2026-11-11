// Live tab: one /ws/view socket for the selected phone; colour (C + JPEG) and depth (D + PNG) side by side.
(() => {
  const $ = (id) => document.getElementById(id);
  const select = $("device"), colour = $("colour"), depth = $("depth");
  let socket = null, paused = false, device = new URLSearchParams(location.search).get("device") || "";
  const urls = { C: null, D: null };

  function show(tag, blob) {
    const img = tag === "C" ? colour : depth;
    if (urls[tag]) URL.revokeObjectURL(urls[tag]);
    urls[tag] = URL.createObjectURL(blob);
    img.src = urls[tag];
    if (tag === "C") $("waiting").hidden = true;
    if (tag === "D") $("no-depth").hidden = true;
  }

  function stats(s) {
    document.querySelectorAll("#stats [data-k]").forEach((el) => {
      const v = s[el.dataset.k];
      el.textContent = v === undefined ? "–" : v === true ? "tracking" : v === false ? "none" : v;
    });
    $("fps-tag").textContent = s.connected ? `${s.shown_fps ?? 0} fps` : "offline";
    $("colour-tag").textContent = s.colour ? `Colour · ${s.colour}` : "Colour";
    $("depth-tag").textContent = s.depth ? `Depth · ${s.depth}` : "Depth";
    $("no-depth").hidden = !s.connected || !!s.depth;
    if (!s.connected) { $("waiting").hidden = false; $("waiting").textContent = `${device} is not connected.`; }
  }

  function connect() {
    if (socket) socket.close();
    socket = null;
    if (!device || paused) return;
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/view/${encodeURIComponent(device)}`);
    ws.binaryType = "blob";
    ws.onopen = () => ws.send(JSON.stringify({ depth_max_m: +$("range").value }));
    ws.onmessage = async (e) => {
      if (typeof e.data === "string") return stats(JSON.parse(e.data));
      const tag = String.fromCharCode(new Uint8Array(await e.data.slice(0, 1).arrayBuffer())[0]);
      show(tag, e.data.slice(1, e.data.size, tag === "C" ? "image/jpeg" : "image/png"));
    };
    ws.onclose = () => { if (socket === ws) setTimeout(connect, 1500); };
    socket = ws;
  }

  async function refreshDevices() {
    try {
      const live = await (await fetch("/api/live")).json();
      const names = live.devices.map((d) => d.device);
      if (!device && names.length) { device = names[0]; connect(); }
      const options = names.includes(device) || !device ? names : [device, ...names];
      select.innerHTML = options.length
        ? options.map((n) => `<option ${n === device ? "selected" : ""}>${n}</option>`).join("")
        : '<option value="">No phone connected</option>';
    } catch (e) { /* server restarting */ }
  }

  select.addEventListener("change", () => {
    device = select.value;
    history.replaceState(null, "", device ? `?device=${encodeURIComponent(device)}` : location.pathname);
    $("waiting").hidden = false; $("waiting").textContent = "Connecting…";
    connect();
  });
  document.querySelectorAll("#mode button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#mode button").forEach((x) => x.classList.toggle("on", x === b));
    $("colour-view").hidden = b.dataset.mode === "depth";
    $("depth-view").hidden = b.dataset.mode === "colour";
    $("streams").classList.toggle("single", b.dataset.mode !== "both");
  }));
  $("pause").addEventListener("click", () => {
    paused = !paused;
    $("pause").textContent = paused ? "Resume" : "Pause";
    connect();
  });
  $("range").addEventListener("input", () => {
    const m = (+$("range").value).toFixed(1);
    $("range-label").textContent = `${m} m`;
    $("range-tag").textContent = `0 – ${m} m`;
    if (socket && socket.readyState === 1) socket.send(JSON.stringify({ depth_max_m: +m }));
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) { socket && socket.close(); socket = null; } else connect(); });

  refreshDevices();
  setInterval(refreshDevices, 3000);
  connect();
})();
