async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

let entriesByDate = {};
let chart = null;

const dateInput = document.getElementById("date");
const weightInput = document.getElementById("weight");
const calInInput = document.getElementById("cal_in");
const calOutInput = document.getElementById("cal_out");
const bodyFatInput = document.getElementById("body_fat_pct");
const syncStatus = document.getElementById("sync-status");
const statsEl = document.getElementById("stats");

function loadFormForDate(dateStr) {
  const entry = entriesByDate[dateStr];
  weightInput.value = entry ? entry.weight : "";
  calInInput.value = entry && entry.cal_in !== null ? entry.cal_in : "";
  calOutInput.value = entry && entry.cal_out !== null ? entry.cal_out : "";
  bodyFatInput.value = entry && entry.body_fat_pct !== null ? entry.body_fat_pct : "";
}

dateInput.valueAsDate = new Date();
dateInput.addEventListener("change", (ev) => loadFormForDate(ev.target.value));

document.getElementById("sync-btn").addEventListener("click", async () => {
  syncStatus.textContent = "Syncing...";
  try {
    const result = await api("/api/whoop/sync", { method: "POST" });
    if (result.updated.length === 0) {
      syncStatus.textContent = "Nothing to backfill — no entries missing cal_out for a completed day.";
    } else {
      const list = result.updated.map(u => `${u.date} (${u.cal_out.toFixed(0)} kcal)`).join(", ");
      syncStatus.textContent = `Backfilled: ${list}`;
    }
    await refresh();
  } catch (e) {
    syncStatus.textContent = "Sync failed: " + e.message;
  }
});

document.getElementById("entry-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = {
    date: dateInput.value,
    weight: parseFloat(weightInput.value),
    cal_in: calInInput.value ? parseFloat(calInInput.value) : null,
    cal_out: calOutInput.value ? parseFloat(calOutInput.value) : null,
    body_fat_pct: bodyFatInput.value ? parseFloat(bodyFatInput.value) : null,
  };
  await api("/api/entries", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  syncStatus.textContent = "";
  await refresh();
});

async function deleteEntry(date) {
  await api(`/api/entries/${date}`, { method: "DELETE" });
  await refresh();
}

function stat(label, value, sub) {
  return `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function renderStats(latest) {
  if (!latest) {
    statsEl.innerHTML = '<div class="stat-grid"><div class="stat empty">No entries yet — log today\'s weight to get started.</div></div>';
    return;
  }
  const betaWk = (latest.beta * 7).toFixed(2);
  const betaSeWk = (latest.se_beta * 7).toFixed(2);
  const distinguishable = Math.abs(latest.beta_z) > 1.96;

  statsEl.innerHTML = `
    <div class="stat-grid">
      ${stat("Filtered weight", `${latest.x.toFixed(1)} lb`, `&plusmn;${latest.se_x.toFixed(2)}`)}
      ${stat("Trend (&beta;)", `${betaWk} lb/wk`, `&plusmn;${betaSeWk}/wk &middot; ${distinguishable ? "distinguishable from zero" : "not yet distinguishable from zero"}`)}
      ${stat("Bias (b)", `${latest.b.toFixed(0)} kcal/day`, `&plusmn;${latest.se_b.toFixed(0)}`)}
      ${stat("Water-weight (e)", `${latest.e.toFixed(2)} lb`, "AR(1) transient")}
      ${stat("Fat mass", `${latest.fat.toFixed(1)} lb`, `&plusmn;${latest.se_fat.toFixed(1)} &middot; from Garmin Index bioimpedance`)}
    </div>
    <div class="caveat">&beta; may reflect residual/unexplained trend rather than the whole trend, since tracked calorie balance already explains most calorie-driven change — see plan Section 6. Fat mass is currently an independent estimate, not yet coupled into the weight/trend dynamics.</div>
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

function renderTable(entries) {
  const tbody = document.querySelector("#entries-table tbody");
  tbody.innerHTML = "";
  for (const e of [...entries].reverse()) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${e.date}</td><td>${e.weight}</td><td>${e.cal_in ?? ""}</td><td>${e.cal_out ?? ""}</td><td>${e.body_fat_pct ?? ""}</td>
      <td><button class="icon" title="Delete" data-date="${e.date}">&times;</button></td>
    `;
    tr.querySelector("button").addEventListener("click", () => deleteEntry(e.date));
    tbody.appendChild(tr);
  }
}

async function refresh() {
  const entries = await api("/api/entries");
  entriesByDate = Object.fromEntries(entries.map(e => [e.date, e]));
  loadFormForDate(dateInput.value);

  const filterResult = await api("/api/filter");
  renderStats(filterResult.latest);
  renderChart(entries, filterResult.trajectory);
  renderTable(entries);
}

refresh();
