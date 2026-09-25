async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

let ridesByDate = {};
let chart = null;

const syncStatus = document.getElementById("sync-status");
const statsEl = document.getElementById("stats");

document.getElementById("sync-btn").addEventListener("click", async () => {
  syncStatus.textContent = "Syncing...";
  try {
    const result = await api("/api/intervals/sync", { method: "POST" });
    syncStatus.textContent = `Synced ${result.oldest} to ${result.newest}: ${result.weigh_ins} weigh-ins, ${result.ride_days} ride days.`;
    await refresh();
  } catch (e) {
    syncStatus.textContent = "Sync failed: " + e.message;
  }
});
function stat(label, value, sub) {
  return `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function lbWk(perDay) {
  const v = perDay * 7;
  return `${v > 0 ? "+" : ""}${v.toFixed(2)} lb/wk`;
}

function zNote(value, se) {
  return Math.abs(value / se) > 1.96 ? "distinguishable from zero" : "not yet distinguishable from zero";
}

function renderStats(latest) {
  if (!latest) {
    statsEl.innerHTML = '<div class="stat-grid"><div class="stat empty">No weigh-ins yet — sync from intervals.icu to get started.</div></div>';
    return;
  }

  const trendStat = latest.trend === undefined
    ? stat("Trend", "&mdash;", "needs a second weigh-in")
    : stat("Trend", lbWk(latest.trend),
        `&plusmn;${(latest.se_trend * 7).toFixed(2)}/wk &middot; at ~${latest.ride_kcal_forecast.toFixed(0)} ride kcal/day (weekly EWMA) &middot; ${zNote(latest.trend, latest.se_trend)}`);

  statsEl.innerHTML = `
    <div class="stat-grid">
      ${stat("Filtered weight", `${latest.x.toFixed(1)} lb`, `&plusmn;${latest.se_x.toFixed(2)}`)}
      ${trendStat}
      ${stat("Baseline (&beta;)", lbWk(latest.beta), `&plusmn;${(latest.se_beta * 7).toFixed(2)}/wk &middot; with no riding &middot; ${zNote(latest.beta, latest.se_beta)}`)}
      ${stat("Ride kcal kept off (&kappa;)", `${(latest.kappa * 100).toFixed(0)}%`, `&plusmn;${(latest.se_kappa * 100).toFixed(0)}% &middot; share of ride kcal not eaten back`)}
      ${stat("Water-weight (e)", `${latest.e.toFixed(2)} lb`, "AR(1) transient")}
      ${stat("Fat mass", `${latest.fat.toFixed(1)} lb`, `&plusmn;${latest.se_fat.toFixed(1)} &middot; from Garmin Index bioimpedance`)}
    </div>
    <div class="caveat">Trend = &beta; &minus; &kappa; &times; forecast ride kcal / 3500, where the forecast is an EWMA of your recent 7-day blocks of riding (most recent week weighted 1, then 0.7, 0.49, &hellip;). &kappa; only becomes identifiable once ride volume varies over time. Fat mass is currently an independent estimate, not yet coupled into the weight/trend dynamics.</div>
  `;
}

function renderChart(entries, trajectory) {
  const labels = trajectory.map(r => r.date);
  const raw = entries.map(e => e.weight);
  const filtered = trajectory.map(r => r.x);

  const styles = getComputedStyle(document.documentElement);
  const rawColor = styles.getPropertyValue("--raw-point").trim();
  const accentColor = styles.getPropertyValue("--accent").trim();
  const textColor = styles.getPropertyValue("--text").trim();
  const borderColor = styles.getPropertyValue("--border").trim();

  if (chart) chart.destroy();
  chart = new Chart(document.getElementById("chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Raw weight", data: raw, borderColor: rawColor, backgroundColor: rawColor, pointRadius: 3, showLine: false },
        { label: "Filtered trend", data: filtered, borderColor: accentColor, backgroundColor: accentColor, pointRadius: 0, borderWidth: 2, tension: 0.15 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      color: textColor,
      scales: {
        x: { ticks: { color: textColor, maxRotation: 0 }, grid: { color: borderColor } },
        y: { title: { display: true, text: "lb", color: textColor }, ticks: { color: textColor }, grid: { color: borderColor } },
      },
      plugins: { legend: { labels: { color: textColor } } },
    },
  });
}

function prevDayRideKcal(dateStr) {
  const d = new Date(dateStr + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() - 1);
  return ridesByDate[d.toISOString().slice(0, 10)];
}

function renderTable(entries) {
  const tbody = document.querySelector("#entries-table tbody");
  tbody.innerHTML = "";
  for (const e of [...entries].reverse()) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${e.date}</td><td>${e.weight}</td><td>${e.body_fat_pct ?? ""}</td><td>${prevDayRideKcal(e.date) ?? ""}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function refresh() {
  const entries = await api("/api/entries");
  ridesByDate = await api("/api/rides");

  const filterResult = await api("/api/filter");
  renderStats(filterResult.latest);
  renderChart(entries, filterResult.trajectory);
  renderTable(entries);
}

refresh();
