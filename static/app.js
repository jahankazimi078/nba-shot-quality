const seasons = ["2024-25", "2023-24", "2022-23"];
const state = {
  season: "2024-25",
  view: "overview",
  playerA: "",
  playerB: "",
  archetype: "All",
};
const data = {};
const dataFiles = [
  "player_profiles",
  "season_summary",
  "archetype_summary",
  "rapm_pooled",
  "coaching_did_summary",
  "coaching_did_results",
  "model_evidence",
  "data_manifest",
];
const syncRender = new URLSearchParams(window.location.search).get("render") === "sync";
const stringFields = new Set([
  "season",
  "player_name",
  "archetype",
  "team_abbr",
  "coach_out",
  "change_date",
  "metric",
  "asset",
  "category",
  "description",
  "download_label",
  "file",
  "name",
  "section",
  "top_player",
  "bottom_player",
]);
const zoneFields = [
  ["restricted_area_share", "At the rim"],
  ["paint_non_ra_share", "Paint"],
  ["mid_range_share", "Midrange"],
  ["corner_3_share", "Corner 3"],
  ["above_break_3_share", "Above-break 3"],
];
const metricLabels = {
  xpts_100poss: "Shot quality allowed (xPts / 100 poss)",
  pts_100poss: "Points allowed / 100 poss",
  xpts_100shots: "xPts allowed / 100 shots",
};
const archetypeNotes = {
  "Perimeter Spacers": "Most of their shots are threes, usually above the break. Think floor-spacing wings.",
  "Corner Spacers": "Split between corner threes and finishes near the rim, with almost no midrange. Classic role-player diet.",
  "Rim Pressure": "Two out of three shots come at the basket. Mostly bigs and hard-driving guards.",
  "Midrange Creators": "The most shots from the in-between areas. Often the players creating their own looks.",
  "Balanced Shot Diet": "No single zone dominates. What this group looks like shifts a bit from season to season.",
};
// Hand-written captions for the report figures, keyed by file name.
const figureNotes = {
  "calibration_2024-25.png": [
    "Is the model honest?",
    "Shots grouped by predicted make chance vs. how often they actually went in. Points sitting on the diagonal means a 40% shot really goes in about 40% of the time.",
  ],
  "poe_stability_2023-24_vs_2024-25.png": [
    "Does shot-making carry over?",
    "Each dot is a player's POE/100 in 2023-24 against 2024-25. r = 0.58 means it's largely a real skill, not a hot streak.",
  ],
  "poe_vs_rts_2024-25.png": [
    "How does it compare to TS%?",
    "POE/100 against relative true shooting. They agree (r = 0.66), but the gaps are the interesting part: efficient players who get easy looks vs. those who earn it.",
  ],
  "rapm_off_vs_poe_2024-25.png": [
    "Cross-check between models",
    "Offensive RAPM comes from a completely separate lineup model, yet it lines up with POE/100 (r ≈ 0.7). Two methods, same signal.",
  ],
  "rapm_face_validity_2024-25.png": [
    "Defense vs. tracking data",
    "Defensive RAPM against the NBA's own defended-FG tracking. The correlation is moderate (r ≈ 0.4), and the top of the list is the rim protectors you'd expect.",
  ],
  "rapm_stability_2023-24_vs_2024-25.png": [
    "The honest weak spot",
    "Offense repeats year to year; defense mostly doesn't (r ≈ 0.12). Shot defense is noisy, so the defensive numbers are pooled over three seasons.",
  ],
  "coaching_event_study.png": [
    "Every firing, lined up",
    "Each team's defense before and after the coach was fired, relative to the rest of the league over the same dates.",
  ],
};

const app = document.querySelector("#app");
const seasonSelect = document.querySelector("#seasonSelect");
const tabButtons = Array.from(document.querySelectorAll(".tabs button"));
const tip = document.querySelector("#tip");

function applyUrlState() {
  const params = new URLSearchParams(window.location.search);
  state.season = params.get("season") || state.season;
  state.view = params.get("view") || state.view;
  state.playerA = params.get("playerA") || state.playerA;
  state.playerB = params.get("playerB") || state.playerB;
  state.archetype = params.get("archetype") || state.archetype;
}

function writeUrlState() {
  const params = new URLSearchParams({
    season: state.season,
    view: state.view,
  });
  if (state.playerA) params.set("playerA", state.playerA);
  if (state.playerB) params.set("playerB", state.playerB);
  if (state.archetype && state.archetype !== "All") params.set("archetype", state.archetype);
  history.replaceState(null, "", `${location.pathname}?${params.toString()}`);
}

function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && inQuotes && next === '"') {
      field += '"';
      i += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(field);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  row.push(field);
  if (row.some((value) => value !== "")) rows.push(row);
  const headers = rows.shift() || [];
  return rows.map((values) => {
    const out = {};
    headers.forEach((header, index) => {
      const value = values[index] ?? "";
      const numeric = value !== "" && !stringFields.has(header) && !Number.isNaN(Number(value));
      out[header] = numeric ? Number(value) : value;
    });
    return out;
  });
}

async function loadCSV(name) {
  const response = await fetch(`data/${name}.csv`);
  if (!response.ok) throw new Error(`Could not load data/${name}.csv`);
  return parseCSV(await response.text());
}

function loadCSVSync(name) {
  const request = new XMLHttpRequest();
  request.open("GET", `data/${name}.csv`, false);
  request.send(null);
  if (request.status < 200 || request.status >= 300) throw new Error(`Could not load data/${name}.csv`);
  return parseCSV(request.responseText);
}

async function loadAll() {
  const loaded = await Promise.all(dataFiles.map((name) => loadCSV(name).then((rows) => [name, rows])));
  loaded.forEach(([name, rows]) => {
    data[name] = rows;
  });
}

function loadAllSync() {
  dataFiles.forEach((name) => {
    data[name] = loadCSVSync(name);
  });
}

// RAPM rows only carry last names; borrow full names from the player profiles.
function attachFullNames() {
  const names = new Map(data.player_profiles.map((row) => [row.player_id, row.player_name]));
  data.rapm_pooled.forEach((row) => {
    row.player_name = names.get(row.player_id) || row.player_name;
  });
}

function fmt(value, digits = 1, signed = false) {
  if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return "–";
  const number = Number(value);
  const text = Math.abs(number).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  if (number < 0 && Number(text.replace(/,/g, "")) !== 0) return `−${text}`;
  return signed && number > 0 ? `+${text}` : text;
}

function pct(value, digits = 1, signed = false) {
  if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return "–";
  return `${fmt(Number(value) * 100, digits, signed)}%`;
}

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function playerRows(season = state.season) {
  return data.player_profiles.filter((row) => row.season === season);
}

function shotRows(season = state.season) {
  return data[`shot_map_sample_${season}`] || [];
}

async function ensureShotRows(season = state.season) {
  const key = `shot_map_sample_${season}`;
  if (data[key]) return data[key];
  data[key] = syncRender ? loadCSVSync(key) : await loadCSV(key);
  return data[key];
}

function byPoe(rows, desc = true) {
  return [...rows].sort((a, b) => (desc ? b.poe_per_100 - a.poe_per_100 : a.poe_per_100 - b.poe_per_100));
}

function setView(view) {
  state.view = view;
  writeUrlState();
  render();
}

function initControls() {
  seasonSelect.innerHTML = seasons.map((season) => `<option value="${season}">${season}</option>`).join("");
  seasonSelect.value = state.season;
  seasonSelect.addEventListener("change", () => {
    state.season = seasonSelect.value;
    writeUrlState();
    render();
  });
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  // One shared tooltip for every chart mark that carries a data-tip.
  document.addEventListener("pointermove", (event) => {
    const target = event.target.closest?.("[data-tip]");
    if (!target) {
      tip.classList.remove("on");
      return;
    }
    tip.innerHTML = target.getAttribute("data-tip");
    tip.style.left = `${event.clientX}px`;
    tip.style.top = `${event.clientY}px`;
    tip.classList.add("on");
  });
}

function updateTabs() {
  tabButtons.forEach((button) => {
    button.setAttribute("aria-selected", button.dataset.view === state.view ? "true" : "false");
  });
  seasonSelect.value = state.season;
  // Season only matters for the player views.
  document.querySelector(".season").style.visibility = ["overview", "compare", "archetypes"].includes(state.view)
    ? "visible"
    : "hidden";
}

function renderHeroStats() {
  const shots = data.season_summary.reduce((sum, row) => sum + row.shots, 0);
  const events = new Set(data.coaching_did_results.map((row) => `${row.season}-${row.team_abbr}`)).size;
  document.querySelector("#heroStats").innerHTML = [
    stat(fmt(shots, 0), "shots modeled"),
    stat(fmt(data.player_profiles.length, 0), "player-seasons"),
    stat("r = 0.58", "year-to-year repeatability of <b>POE</b>"),
    stat(String(events), "mid-season coach firings studied"),
  ].join("");
}

function stat(value, label) {
  return `<div class="stat"><span class="v">${value}</span><span class="k">${label}</span></div>`;
}

function head(idx, title) {
  return `<div class="section-head"><span class="idx">${idx}</span><h2>${title}</h2></div>`;
}

function term(name, text) {
  return `<div><dt>${name}</dt><dd>${text}</dd></div>`;
}

function table(rows, columns, { rank = false } = {}) {
  if (!rows.length) return `<div class="empty">Nothing to show for this selection.</div>`;
  const cls = (col) => [col.align === "l" ? "l" : "", col.cls || ""].join(" ").trim();
  const headCells = (rank ? `<th class="l"></th>` : "") + columns.map((col) => `<th class="${cls(col)}">${col.label}</th>`).join("");
  const body = rows
    .map(
      (row, i) =>
        `<tr>${rank ? `<td class="l rank">${i + 1}</td>` : ""}${columns
          .map((col) => `<td class="${cls(col)}">${col.format ? col.format(row[col.key], row) : esc(row[col.key])}</td>`)
          .join("")}</tr>`,
    )
    .join("");
  return `<div class="tbl-wrap"><table><thead><tr>${headCells}</tr></thead><tbody>${body}</tbody></table></div>`;
}

/* ── chart helpers ─────────────────────────────────────── */

function svgEl(name, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function add(parent, name, attrs = {}, text) {
  const el = parent.appendChild(svgEl(name, attrs));
  if (text !== undefined) el.textContent = text;
  return el;
}

function scale(value, domainMin, domainMax, rangeMin, rangeMax) {
  if (domainMax === domainMin) return (rangeMin + rangeMax) / 2;
  return rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) ticks.push(Number(t.toFixed(10)));
  return ticks;
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function poeColor(value, band = 2) {
  if (value > band) return css("--above");
  if (value < -band) return css("--below");
  return css("--neutral");
}

// Dot + interval chart. rows: [{label, value, lo, hi, sub?}]
function intervalChart(container, rows, { title, unit = "", signedColor = true, labelWidth = 170, band = 0 } = {}) {
  if (!rows.length) {
    container.innerHTML = `<div class="empty">Nothing to show for this selection.</div>`;
    return;
  }
  const rowH = 26;
  const width = 760;
  const margin = { top: 8, right: 56, bottom: 30, left: labelWidth };
  const height = margin.top + rows.length * rowH + margin.bottom;
  const lo = Math.min(0, ...rows.map((r) => r.lo ?? r.value));
  const hi = Math.max(0, ...rows.map((r) => r.hi ?? r.value));
  const pad = (hi - lo) * 0.04;
  const ticks = niceTicks(lo - pad, hi + pad, 6);
  const dMin = Math.min(ticks[0], lo - pad);
  const dMax = Math.max(ticks[ticks.length - 1], hi + pad);
  const x = (v) => scale(v, dMin, dMax, margin.left, width - margin.right);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title });
  const plotBottom = height - margin.bottom;
  ticks.forEach((t) => {
    add(svg, "line", { class: t === 0 ? "axl" : "gridline", x1: x(t), x2: x(t), y1: margin.top, y2: plotBottom });
    add(svg, "text", { class: "axt", x: x(t), y: plotBottom + 16, "text-anchor": "middle" }, fmt(t, Math.abs(ticks[1] - ticks[0]) < 1 ? 1 : 0, true));
  });
  rows.forEach((row, i) => {
    const cy = margin.top + i * rowH + rowH / 2;
    const color = signedColor ? poeColor(row.value, band) : css("--body");
    const g = add(svg, "g", { class: "hit", "data-tip": row.tip || `<b>${esc(row.label)}</b> ${fmt(row.value, 1, true)}${unit}` });
    add(g, "rect", { x: 0, y: cy - rowH / 2, width, height: rowH, fill: "transparent" });
    add(g, "text", { class: "lbl", x: margin.left - 12, y: cy + 4, "text-anchor": "end" }, row.label);
    if (row.lo !== undefined && row.hi !== undefined) {
      add(g, "line", { x1: x(row.lo), x2: x(row.hi), y1: cy, y2: cy, stroke: color, "stroke-width": 2, "stroke-linecap": "round", opacity: 0.45 });
    }
    add(g, "circle", { cx: x(row.value), cy, r: 4.5, fill: color, stroke: css("--surface"), "stroke-width": 2 });
    add(g, "text", { class: "val", x: x(Math.max(row.hi ?? row.value, row.value)) + 10, y: cy + 3.5 }, `${fmt(row.value, 1, true)}${unit}`);
  });
  container.replaceChildren(svg);
}

function scatterChart(container, rows, { title = "Scatter", labelTop = 6 } = {}) {
  if (!rows.length) {
    container.innerHTML = `<div class="empty">No players match this filter.</div>`;
    return;
  }
  const width = 760;
  const height = 420;
  const margin = { top: 16, right: 20, bottom: 44, left: 48 };
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title });
  const xs = rows.map((row) => row.avg_distance_ft);
  const xMin = Math.max(0, Math.floor(Math.min(...xs) - 1));
  const xMax = Math.ceil(Math.max(...xs) + 1);
  const yMax = Math.min(1, Math.max(...rows.map((row) => row.three_pa_rate)) + 0.05);
  const x = (v) => scale(v, xMin, xMax, margin.left, width - margin.right);
  const y = (v) => scale(v, 0, yMax, height - margin.bottom, margin.top);

  niceTicks(0, yMax, 5).forEach((t) => {
    add(svg, "line", { class: t === 0 ? "axl" : "gridline", x1: margin.left, x2: width - margin.right, y1: y(t), y2: y(t) });
    add(svg, "text", { class: "axt", x: margin.left - 8, y: y(t) + 3.5, "text-anchor": "end" }, pct(t, 0));
  });
  niceTicks(xMin, xMax, 6).forEach((t) => {
    add(svg, "text", { class: "axt", x: x(t), y: height - margin.bottom + 16, "text-anchor": "middle" }, `${t} ft`);
  });
  add(svg, "text", { class: "axt", x: width - margin.right, y: height - 6, "text-anchor": "end" }, "avg. shot distance →");
  add(svg, "text", { class: "axt", x: margin.left, y: margin.top - 4 }, "↑ share of shots that are threes");

  // Draw low-POE first so standouts sit on top.
  [...rows].sort((a, b) => Math.abs(a.poe_per_100) - Math.abs(b.poe_per_100)).forEach((row) => {
    const radius = scale(Math.min(row.attempts, 1600), 200, 1600, 3.5, 10);
    add(svg, "circle", {
      class: "hit",
      cx: x(row.avg_distance_ft),
      cy: y(row.three_pa_rate),
      r: radius,
      fill: poeColor(row.poe_per_100),
      "fill-opacity": 0.75,
      stroke: css("--surface"),
      "stroke-width": 1.5,
      "data-tip": `<b>${esc(row.player_name)}</b><br>${fmt(row.poe_per_100, 1, true)} POE/100 · ${fmt(row.attempts, 0)} FGA`,
    });
  });
  byPoe(rows).slice(0, labelTop).forEach((row) => {
    add(svg, "text", { class: "lbl halo", x: x(row.avg_distance_ft) + 9, y: y(row.three_pa_rate) - 7 }, row.player_name);
  });
  container.replaceChildren(svg);
}

function drawCourt(svg, fx, fy) {
  const c = { class: "court" };
  add(svg, "rect", { ...c, x: fx(-25), y: fy(41.75), width: fx(25) - fx(-25), height: fy(-5.25) - fy(41.75) });
  add(svg, "rect", { ...c, x: fx(-8), y: fy(13.75), width: fx(8) - fx(-8), height: fy(-5.25) - fy(13.75) });
  add(svg, "circle", { ...c, cx: fx(0), cy: fy(13.75), r: fx(6) - fx(0) });
  add(svg, "circle", { ...c, cx: fx(0), cy: fy(0), r: fx(0.75) - fx(0) });
  add(svg, "line", { ...c, x1: fx(-3), x2: fx(3), y1: fy(-1.25), y2: fy(-1.25) });
  add(svg, "path", { ...c, d: `M ${fx(-4)} ${fy(0)} A ${fx(4) - fx(0)} ${fx(4) - fx(0)} 0 0 1 ${fx(4)} ${fy(0)}` });
  const r3 = fx(23.75) - fx(0);
  add(svg, "line", { ...c, x1: fx(-22), x2: fx(-22), y1: fy(-5.25), y2: fy(8.95) });
  add(svg, "line", { ...c, x1: fx(22), x2: fx(22), y1: fy(-5.25), y2: fy(8.95) });
  add(svg, "path", { ...c, d: `M ${fx(-22)} ${fy(8.95)} A ${r3} ${r3} 0 0 1 ${fx(22)} ${fy(8.95)}` });
  add(svg, "path", { ...c, d: `M ${fx(-6)} ${fy(41.75)} A ${fx(6) - fx(0)} ${fx(6) - fx(0)} 0 0 1 ${fx(6)} ${fy(41.75)}` });
}

function shotMap(container, rows, title) {
  if (!rows.length) {
    container.innerHTML = `<div class="empty">No sampled shots for this player.</div>`;
    return;
  }
  const width = 500;
  const height = 470;
  const fx = (v) => scale(v, -25, 25, 0, width);
  const fy = (v) => scale(v, -5.25, 41.75, height, 0);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title });
  drawCourt(svg, fx, fy);
  [...rows]
    .filter((row) => row.loc_y_ft <= 41.75)
    .sort((a, b) => a.shot_made - b.shot_made)
    .forEach((row) => {
      const made = row.shot_made === 1;
      add(svg, "circle", {
        cx: fx(row.loc_x_ft),
        cy: fy(row.loc_y_ft),
        r: 3.2,
        fill: made ? css("--above") : "none",
        stroke: made ? css("--surface") : css("--below"),
        "stroke-width": made ? 0.8 : 1.1,
        "fill-opacity": 0.8,
        "stroke-opacity": made ? 1 : 0.7,
        "data-tip": `${made ? "Make" : "Miss"} · ${fmt(row.shot_distance_ft, 0)} ft<br>worth ${fmt(row.xpoints, 2)} xPts`,
      });
    });
  container.replaceChildren(svg);
}

// Horizontal zone-share bars with a tick for the league average.
function shotMix(container, row, leagueAvg) {
  if (!row) {
    container.innerHTML = `<div class="empty">No shot mix for this player.</div>`;
    return;
  }
  const width = 500;
  const rowH = 28;
  const margin = { left: 104, right: 48, top: 4 };
  const height = margin.top + zoneFields.length * rowH + 6;
  const max = Math.max(0.5, ...zoneFields.map(([key]) => Math.max(row[key] || 0, leagueAvg[key] || 0)));
  const x = (v) => scale(v, 0, max, margin.left, width - margin.right);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${row.player_name} shot mix` });
  zoneFields.forEach(([key, label], i) => {
    const cy = margin.top + i * rowH + rowH / 2;
    const value = row[key] || 0;
    const g = add(svg, "g", { class: "hit", "data-tip": `<b>${label}</b> ${pct(value, 0)} of shots<br>league avg ${pct(leagueAvg[key], 0)}` });
    add(g, "rect", { x: 0, y: cy - rowH / 2, width, height: rowH, fill: "transparent" });
    add(g, "text", { class: "lbl", x: margin.left - 12, y: cy + 4, "text-anchor": "end" }, label);
    add(g, "rect", { x: margin.left, y: cy - 6, width: x(max) - margin.left, height: 12, rx: 3, fill: css("--surface-2") });
    add(g, "rect", { x: margin.left, y: cy - 6, width: Math.max(2, x(value) - margin.left), height: 12, rx: 3, fill: css("--muted") });
    add(g, "line", { x1: x(leagueAvg[key]), x2: x(leagueAvg[key]), y1: cy - 10, y2: cy + 10, stroke: css("--text"), "stroke-width": 1.5 });
    add(g, "text", { class: "val", x: x(max) + 8, y: cy + 3.5 }, pct(value, 0));
  });
  container.replaceChildren(svg);
}

/* ── views ─────────────────────────────────────────────── */

function overview() {
  const rows = playerRows();
  const qualified = rows.filter((row) => row.attempts >= 400);
  const leaders = byPoe(qualified).slice(0, 15);
  const trailers = byPoe(qualified, false).slice(0, 5).reverse();
  const top = leaders[0];
  app.innerHTML = `
    <section>
      ${head("01", "The idea")}
      <div class="prose">
        <p>Field-goal percentage treats every shot the same, and they aren't. A wide-open layup goes in about two
        thirds of the time; a fadeaway from 18 feet, far less. So a player's efficiency mixes two things: how good
        their shots are, and how well they make them.</p>
        <p>To pull those apart, I trained a LightGBM model on every regular-season shot from 2022–23 through 2024–25
        to estimate how often an average NBA player makes a shot from that spot, in that situation. Multiply by the
        shot's point value and you get its <strong>expected points</strong>. Anything a player scores above that is
        shot-making, and that's what the rest of this site is about.</p>
      </div>
      <dl class="terms">
        ${term("xPoints", "What the average player would score on the same shot: make probability × 2 or 3.")}
        ${term("POE", "Points over expected. Actual points minus xPoints. Positive means they beat the shot.")}
        ${term("POE / 100", "POE per 100 attempts, so a 400-shot bench guard and a 1,500-shot star sit on the same scale.")}
        ${term("rTS%", "True shooting % relative to league average. Useful context, but it includes free throws, which POE doesn't.")}
      </dl>
    </section>
    <section>
      ${head("02", `Best shot-makers, ${state.season}`)}
      <p class="note">Players with at least 400 shots, ranked by POE/100. The faint line is a 95% bootstrap interval,
      so you can see who's clearly above average and who's still within the noise. ${esc(top.player_name)} led the
      league at ${fmt(top.poe_per_100, 1, true)} points per 100 shots.</p>
      <div class="panel">
        <div id="leaderChart" class="chart"></div>
        <div class="legend"><span><i class="above"></i>Clearly above expected</span><span><i class="neutral"></i>Within ±2</span><span><i class="below"></i>Below expected</span></div>
      </div>
      <h3>…and the bottom five</h3>
      <div class="panel"><div id="trailerChart" class="chart"></div></div>
      <h3>The numbers behind it</h3>
      ${leaderboardTable(leaders)}
    </section>
    <section>
      ${head("03", "Where they shoot from")}
      <p class="note">Every qualified player by average shot distance and three-point rate. Dot size is shot volume,
      color is POE/100. Good shot-makers show up everywhere on this map, which is the point: style and skill are
      separate questions.</p>
      <div class="panel">
        <div id="overviewScatter" class="chart"></div>
        <div class="legend"><span><i class="above"></i>POE/100 above +2</span><span><i class="neutral"></i>Within ±2</span><span><i class="below"></i>Below −2</span></div>
      </div>
    </section>
  `;
  const toRow = (row) => ({
    label: row.player_name,
    value: row.poe_per_100,
    lo: row.poe_ci_low,
    hi: row.poe_ci_high,
    tip: `<b>${esc(row.player_name)}</b> ${fmt(row.poe_per_100, 1, true)} POE/100<br>95% CI ${fmt(row.poe_ci_low, 1, true)} to ${fmt(row.poe_ci_high, 1, true)}`,
  });
  intervalChart(document.querySelector("#leaderChart"), leaders.map(toRow), { title: "Top POE per 100", band: 2 });
  intervalChart(document.querySelector("#trailerChart"), trailers.map(toRow), { title: "Bottom POE per 100", band: 2 });
  scatterChart(document.querySelector("#overviewScatter"), qualified, { title: "Shot distance vs. three-point rate" });
}

function leaderboardTable(rows) {
  return table(
    rows,
    [
      { key: "player_name", label: "Player", align: "l", cls: "name" },
      { key: "archetype", label: "Shot diet", align: "l" },
      { key: "attempts", label: "FGA", format: (v) => fmt(v, 0) },
      { key: "poe_per_100", label: "POE/100", format: (v) => fmt(v, 1, true) },
      { key: "pps", label: "Pts/shot", format: (v) => fmt(v, 2) },
      { key: "xpps", label: "xPts/shot", format: (v) => fmt(v, 2) },
      { key: "rel_ts_pct", label: "rTS%", format: (v) => pct(v, 1, true) },
    ],
    { rank: true },
  );
}

function defaultPair(rows) {
  const has = (name) => rows.some((row) => row.player_name === name);
  const preferred = [["Nikola Jokić", "Stephen Curry"], ["Shai Gilgeous-Alexander", "Stephen Curry"]];
  const pair = preferred.find(([a, b]) => has(a) && has(b));
  if (pair) return pair;
  const top = byPoe(rows);
  return [top[0]?.player_name || "", top[1]?.player_name || top[0]?.player_name || ""];
}

function compare() {
  const rows = playerRows().filter((row) => row.attempts >= 200).sort((a, b) => a.player_name.localeCompare(b.player_name));
  if (!rows.length) {
    app.innerHTML = `<section>${head("01", "Compare two players")}<div class="empty">No players for ${state.season}.</div></section>`;
    return;
  }
  const [defA, defB] = defaultPair(rows);
  if (!rows.some((row) => row.player_name === state.playerA)) state.playerA = defA;
  if (!rows.some((row) => row.player_name === state.playerB)) state.playerB = defB;
  const options = rows.map((row) => `<option value="${esc(row.player_name)}">${esc(row.player_name)}</option>`).join("");
  const a = rows.find((row) => row.player_name === state.playerA);
  const b = rows.find((row) => row.player_name === state.playerB);
  const leagueAvg = Object.fromEntries(
    zoneFields.map(([key]) => [key, rows.reduce((s, r) => s + (r[key] || 0) * r.attempts, 0) / rows.reduce((s, r) => s + r.attempts, 0)]),
  );
  app.innerHTML = `
    <section>
      ${head("01", "Compare two players")}
      <p class="note">Two players can land on the same efficiency in very different ways. Pick any two with 200+ shots
      this season: the numbers show how much they beat their shots by, and the bars and court show which shots
      they're taking.</p>
      <div class="controls">
        <label>Player A <select id="playerA">${options}</select></label>
        <label>Player B <select id="playerB">${options}</select></label>
      </div>
      <div class="cols">
        ${playerPanel(a, 0)}
        ${playerPanel(b, 1)}
      </div>
      <p class="note" id="compareStatus" style="margin-top:1rem">Loading shot charts…</p>
    </section>
  `;
  document.querySelector("#playerA").value = state.playerA;
  document.querySelector("#playerB").value = state.playerB;
  ["A", "B"].forEach((key) => {
    document.querySelector(`#player${key}`).addEventListener("change", (event) => {
      state[`player${key}`] = event.target.value;
      writeUrlState();
      render();
    });
  });
  [a, b].forEach((player, index) => shotMix(document.querySelector(`#mix${index}`), player, leagueAvg));
  ensureShotRows(state.season).then(() => {
    const status = document.querySelector("#compareStatus");
    if (!status) return;
    status.innerHTML = `Shot charts show a fixed sample of each player's attempts so they stay readable. Filled
      <span style="color:var(--above)">blue</span> dots are makes, <span style="color:var(--below)">red</span> rings are misses.
      The white tick on each bar is the league average.`;
    [a, b].forEach((player, index) => {
      const playerShots = shotRows().filter((row) => row.player_id === player.player_id);
      shotMap(document.querySelector(`#map${index}`), playerShots, `${player.player_name} shot chart`);
    });
  });
}

function playerPanel(row, index) {
  return `
    <div>
      <h3 style="margin-top:0">${esc(row.player_name)}</h3>
      <p class="faint" style="font-size:0.8125rem;margin:-0.6rem 0 1rem">${esc(row.archetype)} · ${fmt(row.attempts, 0)} shots</p>
      <div class="stats two">
        ${stat(fmt(row.poe_per_100, 1, true), "POE / 100 shots")}
        ${stat(`${fmt(row.pps, 2)} <span class="faint" style="font-size:0.9rem">vs ${fmt(row.xpps, 2)}</span>`, "pts per shot vs. expected")}
        ${stat(pct(row.ts_pct, 1), `true shooting (${pct(row.rel_ts_pct, 1, true)} vs lg)`)}
        ${stat(pct(row.three_pa_rate, 0), "of shots are threes")}
      </div>
      <div class="panel">
        <div class="fig-t">Shot diet</div>
        <div class="fig-s">Share of attempts by zone</div>
        <div id="mix${index}" class="chart"></div>
      </div>
      <div class="panel" style="margin-top:1rem">
        <div class="fig-t">Shot chart</div>
        <div class="fig-s">Sampled attempts, ${state.season}</div>
        <div id="map${index}" class="chart"></div>
      </div>
    </div>
  `;
}

function archetypes() {
  const rows = playerRows().filter((row) => row.attempts >= 200);
  const names = ["All", ...new Set(rows.map((row) => row.archetype).sort())];
  if (!names.includes(state.archetype)) state.archetype = "All";
  const filtered = state.archetype === "All" ? rows : rows.filter((row) => row.archetype === state.archetype);
  const summary = data.archetype_summary.filter((row) => row.season === state.season);
  app.innerHTML = `
    <section>
      ${head("01", "Shot diets")}
      <div class="prose">
        <p>I clustered players (k-means) using only <em>where</em> they shoot from: zone shares, average distance,
        three-point rate. Performance isn't an input, so every group has good and bad shooters in it. That makes
        the groups useful for fair comparisons: who's the best spacer, who's the most efficient rim attacker.</p>
      </div>
      <ul class="dash">
        ${summary.map((row) => `<li><strong>${esc(row.archetype)}</strong> <span class="faint">(${row.players})</span>: ${archetypeNotes[row.archetype] || ""}</li>`).join("")}
      </ul>
    </section>
    <section>
      ${head("02", "The map")}
      <div class="controls">
        <label>Show <select id="archetypeSelect">${names.map((name) => `<option value="${esc(name)}">${name === "All" ? "Every group" : esc(name)}</option>`).join("")}</select></label>
        <span class="chip">${filtered.length} players · 200+ shots</span>
      </div>
      <div class="panel">
        <div id="archetypeScatter" class="chart"></div>
        <div class="legend"><span><i class="above"></i>POE/100 above +2</span><span><i class="neutral"></i>Within ±2</span><span><i class="below"></i>Below −2</span></div>
      </div>
    </section>
    <section>
      ${head("03", state.archetype === "All" ? "Best shot-makers overall" : `Best ${esc(state.archetype)}`)}
      ${leaderboardTable(byPoe(filtered).slice(0, 12))}
      <h3>How the groups compare</h3>
      ${table(summary, [
        { key: "archetype", label: "Group", align: "l", cls: "name" },
        { key: "players", label: "Players", format: (v) => fmt(v, 0) },
        { key: "avg_poe_per_100", label: "Avg POE/100", format: (v) => fmt(v, 1, true) },
        { key: "avg_distance_ft", label: "Avg dist", format: (v) => `${fmt(v, 1)} ft` },
        { key: "avg_three_pa_rate", label: "3PA rate", format: (v) => pct(v, 0) },
        { key: "avg_rim_rate", label: "Rim rate", format: (v) => pct(v, 0) },
      ])}
    </section>
  `;
  document.querySelector("#archetypeSelect").value = state.archetype;
  document.querySelector("#archetypeSelect").addEventListener("change", (event) => {
    state.archetype = event.target.value;
    writeUrlState();
    render();
  });
  scatterChart(document.querySelector("#archetypeScatter"), filtered, { title: "Shot diet map", labelTop: 5 });
}

function figure(file, wide = false) {
  const row = data.model_evidence.find((r) => r.name === file);
  if (!row) return "";
  const [title, caption] = figureNotes[file] || [file, ""];
  return `<figure${wide ? ' class="wide"' : ""}><div class="figpanel"><img src="${row.asset}" alt="${esc(title)}" loading="lazy" /></div>
    <figcaption><strong>${title}.</strong> ${caption}</figcaption></figure>`;
}

function evidence() {
  const rapm = data.rapm_pooled.filter((row) => row.def_shots >= 5000);
  const byNet = [...rapm].sort((a, b) => b.net_rapm - a.net_rapm);
  const byDef = [...rapm].sort((a, b) => b.def_rapm - a.def_rapm);
  app.innerHTML = `
    <section>
      ${head("01", "Does it hold up?")}
      <div class="prose">
        <p>A leaderboard is easy to make. The harder part is showing it measures something real. These are the
        checks I ran, including the one that didn't come out great.</p>
      </div>
      <div class="stats" style="margin-bottom:2.5rem">
        ${stat("0.225", "Brier score on held-out games <b>(0.248 = guessing)</b>")}
        ${stat("±1.5pp", "worst calibration miss in any shot zone")}
        ${stat("0.58", "POE/100 correlation, one season to the next")}
        ${stat("0.66", "correlation with relative true shooting")}
      </div>
      <p class="note">Training and scoring are grouped by game, and every shot is scored by a model that never saw that
      game. Without that, a player's hot night would leak into their own baseline and shrink their POE.</p>
      <div class="figgrid" style="margin-top:1.5rem">
        ${figure("calibration_2024-25.png", true)}
        ${figure("poe_stability_2023-24_vs_2024-25.png")}
        ${figure("poe_vs_rts_2024-25.png")}
      </div>
    </section>
    <section>
      ${head("02", "Beyond the shooter: RAPM")}
      <div class="prose">
        <p>Shot quality also depends on who else is on the floor. I rebuilt every lineup from play-by-play
        substitutions and fit a ridge regression (RAPM) that splits each shot's result across the ten players
        on the court. Offense measures how much a player lifts their team's shot-making; defense, how much they
        drag the opponent's down.</p>
        <p>Offense is solid. Defense is noisier than I'd like: it only sees field-goal attempts and has to share credit
        among five defenders, so the ratings below are pooled across all three seasons. The top of the defensive list
        still passes the eye test (rim protectors like Jaren Jackson Jr., Wembanyama, and Holmgren).</p>
      </div>
      <div class="figgrid" style="margin:1.5rem 0 1rem">
        ${figure("rapm_off_vs_poe_2024-25.png")}
        ${figure("rapm_face_validity_2024-25.png")}
        ${figure("rapm_stability_2023-24_vs_2024-25.png", true)}
      </div>
      <div class="cols" style="margin-top:2rem">
        <div>
          <h3>Top overall impact</h3>
          ${rapmTable(byNet.slice(0, 12), "net_rapm")}
        </div>
        <div>
          <h3>Top shot defenders</h3>
          ${rapmTable(byDef.slice(0, 12), "def_rapm")}
        </div>
      </div>
      <p class="note" style="margin-top:1rem">Pooled 2022–25, players who defended 5,000+ shots. Units are points per 100 shots;
      positive is good on both ends. This covers shot quality only, not turnovers, rebounding, or free throws.</p>
    </section>
  `;
}

function rapmTable(rows, key) {
  const cols = [
    { key: "player_name", label: "Player", align: "l", cls: "name" },
    { key: "net_rapm", label: "Net", format: (v) => fmt(v, 1, true) },
    { key: "off_rapm", label: "Off", format: (v) => fmt(v, 1, true) },
    { key: "def_rapm", label: "Def", format: (v) => fmt(v, 1, true) },
  ];
  const focus = cols.find((col) => col.key === key);
  focus.format = (v) => `<span style="color:var(--text)">${fmt(v, 1, true)}</span>`;
  return table(rows, cols, { rank: true });
}

function coaching() {
  const summary = data.coaching_did_summary;
  const events = data.coaching_did_results.filter((row) => row.window === 20);
  const eventRows = Object.values(
    events.reduce((acc, row) => {
      const id = `${row.season}-${row.team_abbr}`;
      acc[id] ||= { season: row.season, team_abbr: row.team_abbr, coach_out: row.coach_out, change_date: row.change_date };
      acc[id][row.metric] = row.did;
      return acc;
    }, {}),
  ).sort((a, b) => a.change_date.localeCompare(b.change_date));
  const n = summary[0]?.n_events ?? eventRows.length;
  app.innerHTML = `
    <section>
      ${head("01", "Does firing the coach fix the defense?")}
      <div class="prose">
        <p>Teams that fire their coach mid-season often look better afterward. But teams usually fire a coach after a
        bad stretch, and bad stretches tend to end on their own. So I compared each team's defense before and after
        the firing against every team that <em>didn't</em> change coaches over the same dates (a
        difference-in-differences design).</p>
        <p>I also used the model to split "points allowed" into two parts: the quality of shots the defense gave up,
        and everything else (makes, free throws, turnovers).</p>
      </div>
      <ul class="dash">
        <li><strong>Points allowed dropped</strong> by about 2.5 per 100 possessions relative to other teams. That's a real
        improvement if it holds, but the interval still just crosses zero.</li>
        <li><strong>Shot quality allowed barely moved.</strong> The new coaches weren't forcing noticeably worse shots.
        Whatever changed, it wasn't mainly shot selection.</li>
        <li><strong>It isn't just a bounce-back.</strong> These defenses weren't in freefall before the firing, so
        simple regression to the mean doesn't explain it.</li>
      </ul>
      <p class="note">With only ${n} firings in three seasons, every interval here is wide. I'd read this as a
      lean, not a conclusion.</p>
    </section>
    <section>
      ${head("02", "The estimates")}
      <p class="note">Change relative to control teams, with 95% intervals (bootstrap, clustered by firing). Left of zero
      means the defense got better. Windows are the number of games before and after.</p>
      <div class="panel">
        <div id="coachingChart" class="chart"></div>
        <div class="legend"><span><i class="above"></i>Defense improved</span><span><i class="below"></i>Defense got worse</span></div>
      </div>
      <h3>Each firing, 20-game window</h3>
      ${table(eventRows, [
        { key: "season", label: "Season", align: "l" },
        { key: "team_abbr", label: "Team", align: "l", cls: "name" },
        { key: "coach_out", label: "Coach fired", align: "l" },
        { key: "change_date", label: "Date", align: "l" },
        { key: "pts_100poss", label: "Pts allowed", format: (v) => fmt(v, 1, true) },
        { key: "xpts_100poss", label: "Shot quality", format: (v) => fmt(v, 1, true) },
      ])}
      <p class="note" style="margin-top:0.75rem">Per 100 possessions, vs. control teams. Negative = better defense.</p>
      <div style="margin-top:2rem">${figure("coaching_event_study.png")}</div>
    </section>
  `;
  const rows = ["pts_100poss", "xpts_100poss"].flatMap((metric) =>
    summary
      .filter((row) => row.metric === metric)
      .sort((a, b) => a.window - b.window)
      .map((row) => ({
        label: `${metric === "pts_100poss" ? "Points allowed" : "Shot quality allowed"} · ${row.window} games`,
        value: row.pooled_did,
        lo: row.event_ci_low,
        hi: row.event_ci_high,
        tip: `<b>${metricLabels[metric]}</b>, ${row.window}-game window<br>${fmt(row.pooled_did, 2, true)} (95% CI ${fmt(row.event_ci_low, 1, true)} to ${fmt(row.event_ci_high, 1, true)})`,
      })),
  );
  const chart = document.querySelector("#coachingChart");
  intervalChart(chart, rows, { title: "Coaching change estimates", labelWidth: 210, band: 0.5 });
  // In this chart negative is good, so swap the encoding.
  chart.querySelectorAll("circle, line[stroke-linecap]").forEach((el) => {
    const attr = el.tagName === "circle" ? "fill" : "stroke";
    const v = el.getAttribute(attr);
    if (v === css("--above")) el.setAttribute(attr, css("--below"));
    else if (v === css("--below")) el.setAttribute(attr, css("--above"));
  });
}

const fileNotes = {
  shots: "Every shot in the season with its location, result, xPoints, and POE.",
  shot_map_sample: "A fixed per-player sample of shots, used for the shot charts.",
  leaderboard: "Player-season POE with intervals, TS%, and rTS%.",
  player_profiles: "Everything on the player views: POE, shot-zone shares, and shot diet group.",
  season_summary: "Season totals and the top and bottom shot-makers.",
  archetype_summary: "Averages for each shot-diet group.",
  rapm_pooled: "Offensive, defensive, and net RAPM with intervals, pooled 2022–25.",
  coaching_did_results: "Per-firing difference-in-differences estimates.",
  coaching_did_summary: "Pooled coaching estimates with intervals.",
  model_evidence: "Index of the validation figures.",
};

function dataView() {
  const files = [...data.data_manifest]
    .map((row) => {
      const name = row.file.split("/").pop();
      const stem = name.replace(/\.csv$/, "").replace(/_\d{4}-\d{2}$/, "");
      return { ...row, name, description: fileNotes[stem] || row.description };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
  app.innerHTML = `
    <section>
      ${head("01", "Data")}
      <div class="prose">
        <p>Everything on this site is plain CSV, so you can check my numbers or use them for your own work. The
        full per-season shot files are big (~19 MB each); the pipeline that produces them is on
        <a href="https://github.com/jahankazimi078/nba-shot-quality" target="_blank" rel="noopener">GitHub</a>.</p>
      </div>
      ${table(files, [
        { key: "name", label: "File", align: "l", format: (v, row) => `<a href="${row.file}">${esc(v)}</a>` },
        { key: "description", label: "What's in it", align: "l", format: (v) => `<span style="white-space:normal">${esc(v)}</span>` },
        { key: "bytes", label: "Size", format: (v) => (v >= 1e6 ? `${fmt(v / 1e6, 1)} MB` : `${fmt(v / 1e3, 0)} KB`) },
      ])}
    </section>
  `;
}

function render() {
  updateTabs();
  tip.classList.remove("on");
  const renderers = { overview, compare, archetypes, evidence, coaching, data: dataView };
  (renderers[state.view] || overview)();
  app.focus({ preventScroll: true });
}

function showError(error) {
  app.innerHTML = `<div class="error"><strong>The data didn't load.</strong><br />${esc(error.message)}<br />If you're running
    this locally, start it with <code>make app</code> from the project root.</div>`;
}

function start() {
  attachFullNames();
  renderHeroStats();
  render();
  writeUrlState();
}

applyUrlState();
initControls();
if (syncRender) {
  try {
    loadAllSync();
    start();
  } catch (error) {
    showError(error);
  }
} else {
  loadAll().then(start).catch(showError);
}
