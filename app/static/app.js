const dom = {
  statusPill: document.getElementById("status-pill"),
  sourceLabel: document.getElementById("source-label"),
  todayTotal: document.getElementById("today-total"),
  sessionTotal: document.getElementById("session-total"),
  activeTracks: document.getElementById("active-tracks"),
  fps: document.getElementById("fps"),
  leftValue: document.getElementById("left-value"),
  rightValue: document.getElementById("right-value"),
  leftBar: document.getElementById("left-bar"),
  rightBar: document.getElementById("right-bar"),
  dominance: document.getElementById("dominance"),
  throughput: document.getElementById("throughput"),
  leftRatio: document.getElementById("left-ratio"),
  rightRatio: document.getElementById("right-ratio"),
  distanceTotal: document.getElementById("distance-total"),
  distanceCaption: document.getElementById("distance-caption"),
  distanceProgressBar: document.getElementById("distance-progress-bar"),
  distanceList: document.getElementById("distance-list"),
  canvas: document.getElementById("trend-canvas"),
  rangePills: document.getElementById("range-pills"),
  restartBtn: document.getElementById("restart-btn"),
  resetTodayBtn: document.getElementById("reset-today-btn"),
};

const ctx = dom.canvas.getContext("2d");
let currentRangeSec = 60;

const DISTANCE_REFERENCES = [
  { label: "Kosice -> Presov (approx)", meters: 31000 },
  { label: "Kosice -> Poprad (approx)", meters: 79000 },
  { label: "Kosice -> Bratislava (approx)", meters: 312000 },
  { label: "Eiffel Tower height", meters: 330 },
  { label: "Burj Khalifa height", meters: 828 },
];

function setStatus(online, text) {
  dom.statusPill.textContent = text;
  dom.statusPill.classList.toggle("offline", !online);
}

function formatNumber(value) {
  return new Intl.NumberFormat("uk-UA").format(value || 0);
}

function formatDistanceMeters(meters) {
  if (meters >= 1000) return `${(meters / 1000).toFixed(2)} km`;
  return `${meters.toFixed(0)} m`;
}

function renderDistanceComparisons(distanceMeters) {
  const lines = DISTANCE_REFERENCES.map((ref) => {
    const ratio = ref.meters > 0 ? distanceMeters / ref.meters : 0;
    return {
      label: ref.label,
      text: `${(ratio * 100).toFixed(1)}%`,
      ratio,
    };
  });

  lines.sort((a, b) => Math.abs(a.ratio - 1.0) - Math.abs(b.ratio - 1.0));
  const nearest = lines[0];

  dom.distanceCaption.textContent = `Progress vs ${nearest.label}: ${nearest.text}`;
  const progress = Math.max(0, Math.min(100, nearest.ratio * 100));
  dom.distanceProgressBar.style.width = `${progress}%`;

  dom.distanceList.innerHTML = "";
  lines.forEach((item) => {
    const li = document.createElement("li");
    const title = document.createElement("strong");
    const value = document.createElement("span");
    title.textContent = item.label;
    value.textContent = item.text;
    li.appendChild(title);
    li.appendChild(value);
    dom.distanceList.appendChild(li);
  });
}

function renderVerticalInsight(verticalMeters) {
  const floors = verticalMeters / 3.0;
  const floorLine = document.createElement("li");
  const title = document.createElement("strong");
  const value = document.createElement("span");
  title.textContent = "Equivalent building floors (3m)";
  value.textContent = `${floors.toFixed(1)} floors`;
  floorLine.appendChild(title);
  floorLine.appendChild(value);
  dom.distanceList.appendChild(floorLine);
}

function pickHistory(history, rangeSec) {
  if (!Array.isArray(history) || history.length === 0) return [];
  const now = Date.now();
  return history.filter((item) => {
    if (!item.timestamp) return false;
    const ts = new Date(item.timestamp).getTime();
    return now - ts <= rangeSec * 1000;
  });
}

function drawSeries(points, key, color, minValue, maxValue, pad, w, h) {
  if (points.length < 2) return;
  const t0 = new Date(points[0].timestamp).getTime();
  const t1 = new Date(points[points.length - 1].timestamp).getTime();
  const duration = Math.max(1, t1 - t0);

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;

  points.forEach((p, i) => {
    const t = new Date(p.timestamp).getTime();
    const x = pad + ((t - t0) / duration) * (w - pad * 2);
    const value = Number(p[key] || 0);
    const y = h - pad - ((value - minValue) / Math.max(1, maxValue - minValue)) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

function drawTrend(history) {
  const width = dom.canvas.width;
  const height = dom.canvas.height;
  const pad = 18;

  ctx.clearRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(122, 146, 177, 0.22)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i += 1) {
    const y = pad + ((height - pad * 2) * i) / 3;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  if (!history.length) return;

  const values = history.flatMap((p) => [
    Number(p.left || 0),
    Number(p.right || 0),
    Number(p.session_total || 0),
  ]);
  const minValue = Math.min(...values, 0);
  const maxValue = Math.max(...values, 1);

  drawSeries(history, "left", "#32d882", minValue, maxValue, pad, width, height);
  drawSeries(history, "right", "#ffb547", minValue, maxValue, pad, width, height);
  drawSeries(history, "session_total", "#7ea8ff", minValue, maxValue, pad, width, height);
}

function updateMetrics(data) {
  const left = Number(data.left || 0);
  const right = Number(data.right || 0);
  const total = left + right;

  dom.sourceLabel.textContent = data.source || "Unknown source";
  dom.todayTotal.textContent = formatNumber(data.today_total);
  dom.sessionTotal.textContent = formatNumber(data.session_total);
  dom.activeTracks.textContent = formatNumber(data.active_tracks);
  dom.fps.textContent = Number(data.fps || 0).toFixed(1);

  dom.leftValue.textContent = formatNumber(left);
  dom.rightValue.textContent = formatNumber(right);

  const leftPct = total > 0 ? (left / total) * 100 : 0;
  const rightPct = total > 0 ? (right / total) * 100 : 0;
  dom.leftBar.style.width = `${leftPct}%`;
  dom.rightBar.style.width = `${rightPct}%`;

  let dominance = "Balanced";
  if (leftPct - rightPct > 12) dominance = "Left dominates";
  if (rightPct - leftPct > 12) dominance = "Right dominates";
  dom.dominance.textContent = dominance;

  dom.leftRatio.textContent = `${leftPct.toFixed(1)}%`;
  dom.rightRatio.textContent = `${rightPct.toFixed(1)}%`;

  const distanceMeters = Number(data.session_distance_m || 0);
  const verticalMeters = Number(data.session_distance_vertical_m || 0);
  const stairsTotal = Number(data.session_steps || 0);
  dom.distanceTotal.textContent = `Session: ${formatDistanceMeters(distanceMeters)}`;
  renderDistanceComparisons(distanceMeters);
  dom.distanceCaption.textContent += ` (session ${formatNumber(stairsTotal)} steps)`;
  renderVerticalInsight(verticalMeters);

  const history = pickHistory(data.history || [], currentRangeSec);
  if (history.length >= 2) {
    const first = history[0];
    const last = history[history.length - 1];
    const dtMinutes = Math.max(
      1 / 60,
      (new Date(last.timestamp).getTime() - new Date(first.timestamp).getTime()) / 60000,
    );
    const delta = Math.max(0, Number(last.session_total || 0) - Number(first.session_total || 0));
    dom.throughput.textContent = (delta / dtMinutes).toFixed(1);
  } else {
    dom.throughput.textContent = "0.0";
  }

  drawTrend(history);
}

async function fetchMetrics() {
  try {
    const res = await fetch("/api/metrics", { cache: "no-store" });
    const data = await res.json();

    if (data.running) {
      setStatus(true, "Live");
    } else if (data.error) {
      setStatus(false, `Error: ${data.error}`);
    } else {
      setStatus(false, "Stopped");
    }

    updateMetrics(data);
  } catch (err) {
    setStatus(false, "API unreachable");
  }
}

function bindEvents() {
  dom.rangePills.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      currentRangeSec = Number(button.dataset.range);
      dom.rangePills.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      fetchMetrics();
    });
  });

  dom.restartBtn.addEventListener("click", async () => {
    dom.restartBtn.disabled = true;
    dom.restartBtn.textContent = "Restarting...";
    try {
      await fetch("/api/restart");
      setStatus(true, "Restart requested");
    } finally {
      setTimeout(() => {
        dom.restartBtn.disabled = false;
        dom.restartBtn.textContent = "Restart Stream";
      }, 800);
    }
  });

  dom.resetTodayBtn.addEventListener("click", async () => {
    dom.resetTodayBtn.disabled = true;
    dom.resetTodayBtn.textContent = "Resetting...";
    try {
      await fetch("/api/reset-today");
      fetchMetrics();
    } finally {
      setTimeout(() => {
        dom.resetTodayBtn.disabled = false;
        dom.resetTodayBtn.textContent = "Reset Today";
      }, 500);
    }
  });
}

bindEvents();
fetchMetrics();
setInterval(fetchMetrics, 1000);
