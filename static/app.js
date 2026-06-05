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
  ["restricted_area_share", "Rim"],
  ["paint_non_ra_share", "Paint"],
  ["mid_range_share", "Mid"],
  ["corner_3_share", "Corner 3"],
  ["above_break_3_share", "Above break 3"],
];
const colors = {
  primary: "#0f766e",
  comparison: "#6f5aa8",
  warm: "#d45b4c",
  gold: "#b98b2d",
  good: "#138a5b",
  bad: "#b6423a",
  neutral: "#aeb8bf",
  line: "#d6dde2",
  ink: "#17212b",
};

const app = document.querySelector("#app");
const seasonSelect = document.querySelector("#seasonSelect");
const tabButtons = Array.from(document.querySelectorAll(".tabs button"));

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

function fmt(value, digits = 1, signed = false) {
  if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  const number = Number(value);
  const text = number.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return signed && number > 0 ? `+${text}` : text;
}

function pct(value, digits = 1, signed = false) {
  if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${fmt(Number(value) * 100, digits, signed)}%`;
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
  if (syncRender) {
    data[key] = loadCSVSync(key);
    return data[key];
  }
  data[key] = await loadCSV(key);
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
}

function updateTabs() {
  tabButtons.forEach((button) => {
    button.setAttribute("aria-selected", button.dataset.view === state.view ? "true" : "false");
  });
  seasonSelect.value = state.season;
}

function metric(label, value, detail = "") {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong>${detail ? `<em>${detail}</em>` : ""}</div>`;
}

function explainCard(title, text) {
  return `<article class="explain-card"><h3>${title}</h3><p>${text}</p></article>`;
}

function metricGuide() {
  return `
    <section class="panel">
      <div class="section-head">
        <div>
          <h2>How to read the metrics</h2>
          <p>These definitions are written for basketball readers first. Higher is usually better unless the card says otherwise.</p>
        </div>
      </div>
      <div class="explain-grid">
        ${explainCard("xPoints", "The number of points an average NBA shooter would be expected to score from the same shot. A corner three, a layup, and a contested-looking long two do not start from the same baseline.")}
        ${explainCard("POE", "Points over expected. This is actual field-goal points minus xPoints. Positive POE means the player scored more than expected from the shots he took.")}
        ${explainCard("POE / 100", "POE scaled to 100 shot attempts. This makes high-volume stars and lower-volume specialists easier to compare on the same scale.")}
        ${explainCard("PPS and xPPS", "PPS is actual points per shot. xPPS is expected points per shot. If PPS is above xPPS, the player beat the model's expectation.")}
        ${explainCard("TS% and rTS%", "True Shooting % includes free throws and threes. rTS% is how far a player was above or below league-average TS%. It is useful context, but it is not the same shot universe as POE.")}
        ${explainCard("Shot archetype", "A style group based only on where and how far a player shoots. It is not a ranking. Use it to compare players with similar shot diets.")}
        ${explainCard("RAPM", "Regularized adjusted plus-minus for shot quality. It estimates how team or opponent shot quality changed when a player was on the floor, while sharing credit with teammates.")}
        ${explainCard("Coaching DiD", "Difference-in-differences compares a team's before/after change to the rest of the league over the same calendar window. Negative defensive values mean the team allowed less after the change.")}
      </div>
    </section>
  `;
}

function answersBlock() {
  return `
    <section class="panel">
      <div class="section-head">
        <div>
          <h2>What this app answers</h2>
          <p>Start here if you have 90 seconds. Each tab is built around one reviewer or analyst question.</p>
        </div>
      </div>
      <div class="explain-grid answers-grid">
        ${explainCard("Shooter skill", "Who scored more than expected after accounting for shot value and location? Use POE/100 as the first leaderboard read.")}
        ${explainCard("Shot diet", "How does a player get those shots? Compare rim, paint, midrange, corner three, and above-break three shares.")}
        ${explainCard("Player comparison", "Which two players create similar value in different ways? The Compare tab pairs outcome cards with shot maps and shot mix.")}
        ${explainCard("Model evidence", "Do the metrics behave like signal? Evidence shows calibration, POE stability, POE vs rTS%, and RAPM diagnostics.")}
        ${explainCard("Coaching study", "What changed after in-season firings? The DiD view separates directional results from claims the sample cannot support.")}
      </div>
    </section>
  `;
}

function table(rows, columns, limit = rows.length) {
  const shown = rows.slice(0, limit);
  if (!shown.length) {
    return `<div class="empty-state">No rows are available for this selection.</div>`;
  }
  const head = columns.map((col) => `<th>${col.label}</th>`).join("");
  const body = shown
    .map(
      (row) =>
        `<tr>${columns
          .map((col) => `<td>${col.format ? col.format(row[col.key], row) : row[col.key]}</td>`)
          .join("")}</tr>`,
    )
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function svgEl(name, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function scale(value, domainMin, domainMax, rangeMin, rangeMax) {
  if (domainMax === domainMin) return (rangeMin + rangeMax) / 2;
  return rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function poeColor(value) {
  if (value > 2) return colors.good;
  if (value < -2) return colors.bad;
  return colors.neutral;
}

function barChart(container, rows, options) {
  if (!rows.length) {
    container.innerHTML = `<div class="empty-state">No chart rows are available for this selection.</div>`;
    return;
  }
  const width = 760;
  const height = Math.max(260, rows.length * 30 + 48);
  const margin = { top: 22, right: 54, bottom: 26, left: 156 };
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": options.title });
  const values = rows.map((row) => row[options.value]);
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const zero = scale(0, min, max, margin.left, width - margin.right);
  rows.forEach((row, index) => {
    const y = margin.top + index * 30;
    const x = scale(Math.min(0, row[options.value]), min, max, margin.left, width - margin.right);
    const x2 = scale(Math.max(0, row[options.value]), min, max, margin.left, width - margin.right);
    svg.appendChild(svgEl("text", { x: margin.left - 8, y: y + 18, "text-anchor": "end", "font-size": 12, fill: colors.ink }))
      .textContent = row.player_name;
    svg.appendChild(
      svgEl("rect", {
        x,
        y,
        width: Math.max(1, x2 - x),
        height: 20,
        rx: 4,
        fill: row[options.value] >= 0 ? colors.primary : colors.warm,
      }),
    );
    svg.appendChild(svgEl("text", { x: x2 + 6, y: y + 15, "font-size": 12, fill: colors.ink })).textContent = fmt(
      row[options.value],
      1,
      true,
    );
  });
  svg.appendChild(svgEl("line", { x1: zero, x2: zero, y1: 12, y2: height - 22, stroke: colors.ink, "stroke-width": 1 }));
  container.replaceChildren(svg);
}

function scatterChart(container, rows, options = {}) {
  if (!rows.length) {
    container.innerHTML = `<div class="empty-state">No players match this filter.</div>`;
    return;
  }
  const width = 760;
  const height = 460;
  const margin = { top: 24, right: 28, bottom: 54, left: 58 };
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": options.title || "Scatter" });
  const xs = rows.map((row) => row.avg_distance_ft);
  const ys = rows.map((row) => row.three_pa_rate);
  const xMin = Math.min(...xs) - 1;
  const xMax = Math.max(...xs) + 1;
  const yMin = 0;
  const yMax = Math.min(1, Math.max(...ys) + 0.08);

  for (let tick = 0; tick <= 5; tick += 1) {
    const yValue = yMin + ((yMax - yMin) / 5) * tick;
    const y = scale(yValue, yMin, yMax, height - margin.bottom, margin.top);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: colors.line }));
    svg.appendChild(svgEl("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", "font-size": 11, fill: "#60707c" }))
      .textContent = pct(yValue, 0);
  }

  rows.forEach((row) => {
    const x = scale(row.avg_distance_ft, xMin, xMax, margin.left, width - margin.right);
    const y = scale(row.three_pa_rate, yMin, yMax, height - margin.bottom, margin.top);
    const radius = scale(Math.min(row.attempts, 1600), 200, 1600, 4, 14);
    svg.appendChild(
      svgEl("circle", {
        cx: x,
        cy: y,
        r: radius,
        fill: poeColor(row.poe_per_100),
        opacity: 0.72,
        stroke: colors.ink,
        "stroke-width": 0.6,
      }),
    ).appendChild(svgEl("title")).textContent = `${row.player_name}: ${fmt(row.poe_per_100, 1, true)} POE/100`;
  });

  byPoe(rows).slice(0, 6).forEach((row) => {
    const x = scale(row.avg_distance_ft, xMin, xMax, margin.left, width - margin.right);
    const y = scale(row.three_pa_rate, yMin, yMax, height - margin.bottom, margin.top);
    svg.appendChild(svgEl("text", { x: x + 8, y: y - 8, "font-size": 11, fill: colors.ink })).textContent =
      row.player_name;
  });
  svg.appendChild(svgEl("text", { x: width / 2, y: height - 12, "text-anchor": "middle", "font-size": 12, fill: "#60707c" }))
    .textContent = "Average shot distance";
  svg.appendChild(
    svgEl("text", {
      x: 16,
      y: height / 2,
      transform: `rotate(-90 16 ${height / 2})`,
      "text-anchor": "middle",
      "font-size": 12,
      fill: "#60707c",
    }),
  ).textContent = "3PA rate";
  container.replaceChildren(svg);
}

function drawCourt(svg, width, height) {
  const cx = (x) => scale(x, -27, 27, 0, width);
  const cy = (y) => scale(y, -6, 49, height, 0);
  const line = (x1, y1, x2, y2) => svg.appendChild(svgEl("line", { x1: cx(x1), y1: cy(y1), x2: cx(x2), y2: cy(y2), stroke: colors.ink, "stroke-width": 1.8 }));
  svg.appendChild(svgEl("rect", { x: cx(-25), y: cy(47), width: cx(25) - cx(-25), height: cy(-4) - cy(47), fill: "none", stroke: colors.ink, "stroke-width": 1.8 }));
  svg.appendChild(svgEl("circle", { cx: cx(0), cy: cy(0), r: 7, fill: "none", stroke: colors.ink, "stroke-width": 1.8 }));
  svg.appendChild(svgEl("rect", { x: cx(-8), y: cy(15), width: cx(8) - cx(-8), height: cy(-4) - cy(15), fill: "none", stroke: colors.ink, "stroke-width": 1.8 }));
  line(-22, -4, -22, 10);
  line(22, -4, 22, 10);
  svg.appendChild(svgEl("path", { d: `M ${cx(-22)} ${cy(10)} A ${cx(23.75) - cx(0)} ${cy(0) - cy(23.75)} 0 0 1 ${cx(22)} ${cy(10)}`, fill: "none", stroke: colors.ink, "stroke-width": 1.8 }));
}

function shotMap(container, rows, title) {
  if (!rows.length) {
    container.innerHTML = `<div class="empty-state">No sampled shots are available for this player.</div>`;
    return;
  }
  const width = 520;
  const height = 500;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title });
  drawCourt(svg, width, height);
  rows.forEach((row) => {
    const cx = scale(row.loc_x_ft, -27, 27, 0, width);
    const cy = scale(row.loc_y_ft, -6, 49, height, 0);
    svg.appendChild(
      svgEl("circle", {
        cx,
        cy,
        r: 3.1,
        fill: row.poe > 0 ? colors.good : colors.bad,
        opacity: 0.58,
        stroke: colors.ink,
        "stroke-width": 0.25,
      }),
    );
  });
  container.replaceChildren(svg);
}

function shotMix(container, row) {
  if (!row) {
    container.innerHTML = `<div class="empty-state">No shot mix is available for this player.</div>`;
    return;
  }
  const width = 520;
  const height = 230;
  const margin = { top: 20, right: 10, bottom: 46, left: 38 };
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${row.player_name} shot mix` });
  const max = Math.max(0.45, ...zoneFields.map(([key]) => row[key] || 0));
  zoneFields.forEach(([key, label], index) => {
    const x = margin.left + index * ((width - margin.left - margin.right) / zoneFields.length) + 8;
    const barW = (width - margin.left - margin.right) / zoneFields.length - 16;
    const barH = scale(row[key] || 0, 0, max, 0, height - margin.top - margin.bottom);
    svg.appendChild(
      svgEl("rect", {
        x,
        y: height - margin.bottom - barH,
        width: barW,
        height: barH,
        rx: 5,
        fill: [colors.primary, "#4d99a2", colors.gold, colors.warm, colors.comparison][index],
      }),
    );
    svg.appendChild(svgEl("text", { x: x + barW / 2, y: height - 20, "text-anchor": "middle", "font-size": 11, fill: "#60707c" }))
      .textContent = label;
    svg.appendChild(svgEl("text", { x: x + barW / 2, y: height - margin.bottom - barH - 6, "text-anchor": "middle", "font-size": 11, fill: colors.ink }))
      .textContent = pct(row[key] || 0, 0);
  });
  container.replaceChildren(svg);
}

function overview() {
  const rows = playerRows();
  const summary = data.season_summary.find((row) => row.season === state.season);
  const qualified = rows.filter((row) => row.attempts >= 400);
  const top = byPoe(qualified)[0];
  const bottom = byPoe(qualified, false)[0];
  app.innerHTML = `
    ${answersBlock()}
    <section class="metric-strip">
      ${metric("Shots", fmt(summary.shots, 0), "field-goal attempts")}
      ${metric("Qualified players", fmt(summary.qualified_players_400_fga, 0), "400+ FGA")}
      ${metric("Best POE / 100", fmt(top.poe_per_100, 1, true), top.player_name)}
      ${metric("Lowest POE / 100", fmt(bottom.poe_per_100, 1, true), bottom.player_name)}
    </section>
    ${metricGuide()}
    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Shot-quality adjusted scoring leaders</h2>
          <p>POE rewards made-shot value above expected points. Read this as: who scored more than an average NBA shooter would have from the same shot locations and shot values.</p>
        </div>
        <div class="legend"><span>Positive POE</span><span class="bad">Negative POE</span><span class="neutral">Near expected</span></div>
      </div>
      <div id="topBars" class="chart"></div>
    </section>
    <section class="grid-two">
      <div class="panel">
        <h2>Top 15 player seasons</h2>
        ${leaderboardTable(byPoe(qualified).slice(0, 15))}
      </div>
      <div class="panel">
        <h2>Archetype map</h2>
        <p class="note">Left-to-right shows average shot distance. Bottom-to-top shows how often the player shoots threes. Color shows whether scoring beat expectation.</p>
        <div id="overviewScatter" class="chart"></div>
      </div>
    </section>
  `;
  barChart(document.querySelector("#topBars"), byPoe(qualified).slice(0, 14), {
    title: "Top POE per 100 players",
    value: "poe_per_100",
  });
  scatterChart(document.querySelector("#overviewScatter"), qualified, { title: "Archetypes by distance and 3PA rate" });
}

function leaderboardTable(rows) {
  return table(rows, [
    { key: "player_name", label: "Player" },
    { key: "archetype", label: "Archetype" },
    { key: "attempts", label: "FGA", format: (v) => fmt(v, 0) },
    { key: "poe_per_100", label: "POE/100", format: (v) => fmt(v, 1, true) },
    { key: "pps", label: "PPS", format: (v) => fmt(v, 3) },
    { key: "xpps", label: "xPPS", format: (v) => fmt(v, 3) },
    { key: "rel_ts_pct", label: "rTS%", format: (v) => pct(v, 1, true) },
  ]);
}

function compare() {
  const rows = playerRows().filter((row) => row.attempts >= 200).sort((a, b) => a.player_name.localeCompare(b.player_name));
  if (!rows.length) {
    app.innerHTML = `<section class="panel"><h2>Compare two players</h2><div class="empty-state">No qualified players are available for ${state.season}.</div></section>`;
    return;
  }
  if (!state.playerA || !rows.some((row) => row.player_name === state.playerA)) state.playerA = rows[0]?.player_name || "";
  if (!state.playerB || !rows.some((row) => row.player_name === state.playerB)) state.playerB = rows.find((row) => row.player_name !== state.playerA)?.player_name || state.playerA;
  const options = rows.map((row) => `<option value="${row.player_name}">${row.player_name}</option>`).join("");
  const a = rows.find((row) => row.player_name === state.playerA);
  const b = rows.find((row) => row.player_name === state.playerB);
  app.innerHTML = `
    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Compare two players</h2>
          <p>Use this view to separate shot-making from shot selection. The cards show outcomes; the bars and maps show the kind of shots behind those outcomes.</p>
        </div>
      </div>
      <div class="control-row">
        <label>Player A<select id="playerA">${options}</select></label>
        <label>Player B<select id="playerB">${options}</select></label>
      </div>
      <div id="compareStatus" class="note">Preparing the representative shot-map sample for ${state.season}.</div>
      <div class="grid-two">
        ${playerPanel(a, "A")}
        ${playerPanel(b, "B")}
      </div>
    </section>
  `;
  document.querySelector("#playerA").value = state.playerA;
  document.querySelector("#playerB").value = state.playerB;
  document.querySelector("#playerA").addEventListener("change", (event) => {
    state.playerA = event.target.value;
    writeUrlState();
    render();
  });
  document.querySelector("#playerB").addEventListener("change", (event) => {
    state.playerB = event.target.value;
    writeUrlState();
    render();
  });
  [a, b].forEach((player, index) => {
    shotMix(document.querySelector(`#mix${index}`), player);
  });
  ensureShotRows(state.season).then(() => {
    const status = document.querySelector("#compareStatus");
    if (status) status.textContent = "Shot maps use a deterministic sample for readability. Green shots beat expectation; red shots fell short.";
    [a, b].forEach((player, index) => {
      const playerShots = shotRows().filter((row) => row.player_id === player.player_id);
      shotMap(document.querySelector(`#map${index}`), playerShots, `${player.player_name} shot map`);
    });
  });
}

function playerPanel(row, label) {
  const delta = row.pps - row.xpps;
  return `
    <div class="panel-subtle">
      <div class="section-head">
        <div>
          <h2>${row.player_name}</h2>
          <p>${row.archetype}</p>
        </div>
        <strong>${label}</strong>
      </div>
      <div class="metric-strip">
        ${metric("POE / 100", fmt(row.poe_per_100, 1, true))}
        ${metric("PPS - xPPS", fmt(delta, 3, true))}
        ${metric("3PA rate", pct(row.three_pa_rate, 1))}
        ${metric("Rim rate", pct(row.rim_rate, 1))}
      </div>
      <p class="note">POE/100 is the cleanest shot-making summary. PPS minus xPPS is the same idea per individual shot. Shot mix explains whether the player gets value at the rim, from threes, or from tougher middle areas.</p>
      <h3>Shot mix</h3>
      <div id="mix${label === "A" ? 0 : 1}" class="chart"></div>
      <h3>Shot map sample</h3>
      <div id="map${label === "A" ? 0 : 1}" class="chart shot-map"></div>
    </div>
  `;
}

function archetypes() {
  const rows = playerRows().filter((row) => row.attempts >= 200);
  const names = ["All", ...new Set(rows.map((row) => row.archetype).sort())];
  if (!names.includes(state.archetype)) state.archetype = "All";
  const filtered = state.archetype === "All" ? rows : rows.filter((row) => row.archetype === state.archetype);
  app.innerHTML = `
    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Shot diet archetypes</h2>
          <p>Archetypes group players by shot style, not by quality. A rim-pressure player and a perimeter spacer can both be excellent, but they create value in different ways.</p>
        </div>
        <label>Archetype<select id="archetypeSelect">${names.map((name) => `<option value="${name}">${name}</option>`).join("")}</select></label>
      </div>
      <div class="explain-grid compact">
        ${explainCard("Perimeter Spacers", "Players whose shot diet leans heavily toward threes, especially above-the-break attempts.")}
        ${explainCard("Rim Pressure", "Players who create a large share of attempts at the basket, where expected points are usually high.")}
        ${explainCard("Midrange Creators", "Players who take more self-created or in-between shots. These can be valuable when the player consistently beats expectation.")}
        ${explainCard("Balanced Shot Diet", "Players without one dominant zone. Their value is easier to judge by pairing POE with the full shot mix.")}
      </div>
      <div id="archetypeScatter" class="chart"></div>
    </section>
    <section class="grid-two">
      <div class="panel">
        <h2>Archetype summary</h2>
        ${archetypeSummaryTable(data.archetype_summary.filter((row) => row.season === state.season))}
      </div>
      <div class="panel">
        <h2>Best within filter</h2>
        ${leaderboardTable(byPoe(filtered).slice(0, 12))}
      </div>
    </section>
  `;
  document.querySelector("#archetypeSelect").value = state.archetype;
  document.querySelector("#archetypeSelect").addEventListener("change", (event) => {
    state.archetype = event.target.value;
    writeUrlState();
    render();
  });
  scatterChart(document.querySelector("#archetypeScatter"), filtered, { title: "Archetype scatter" });
}

function archetypeSummaryTable(rows) {
  return table(rows, [
    { key: "archetype", label: "Archetype" },
    { key: "players", label: "Players", format: (v) => fmt(v, 0) },
    { key: "avg_poe_per_100", label: "POE/100", format: (v) => fmt(v, 1, true) },
    { key: "avg_distance_ft", label: "Dist", format: (v) => fmt(v, 1) },
    { key: "avg_three_pa_rate", label: "3PA", format: (v) => pct(v, 1) },
    { key: "avg_rim_rate", label: "Rim", format: (v) => pct(v, 1) },
  ]);
}

function evidence() {
  const reportGroups = data.model_evidence.reduce((acc, row) => {
    acc[row.section] ||= [];
    acc[row.section].push(row);
    return acc;
  }, {});
  const rapm = [...data.rapm_pooled].sort((a, b) => b.net_rapm - a.net_rapm);
  app.innerHTML = `
    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Model evidence</h2>
          <p>These figures answer whether the numbers are trustworthy. Calibration checks whether expected points line up with reality; stability checks whether the metric persists year to year; RAPM checks whether on-floor impact agrees with independent signals.</p>
        </div>
      </div>
      <div class="explain-grid compact">
        ${explainCard("Calibration", "If the model says a group of shots should score about 1.05 points each, the actual average should be close to 1.05. Good calibration makes POE fairer.")}
        ${explainCard("Stability", "A noisy metric disappears from one year to the next. A useful skill metric should show some year-to-year persistence.")}
        ${explainCard("RAPM validation", "RAPM is harder because five players share the floor. The validation plots show whether the estimated impact passes basic reasonableness checks.")}
        ${explainCard("RAPM conclusion", "Use RAPM as directional shot-quality impact. Do not read it as all-in player value, because it excludes turnovers, free throws, rebounding, and role context.")}
      </div>
      ${Object.entries(reportGroups)
        .map(
          ([section, rows]) => `
            <h3>${section}</h3>
            <div class="image-grid">
              ${rows
                .slice(0, 4)
                .map((row) => `<figure><img class="evidence-img" src="${row.asset}" alt="${row.name}" /><figcaption class="caption">${row.name}</figcaption></figure>`)
                .join("")}
            </div>
          `,
        )
        .join("")}
    </section>
    <section class="grid-two">
      <div class="panel">
        <h2>Top net RAPM</h2>
        <p class="note">Positive net RAPM means the player's teams tended to create better shot-quality outcomes while he was on the floor, after regularization.</p>
        ${rapmTable(rapm.slice(0, 15))}
      </div>
      <div class="panel">
        <h2>Lowest net RAPM</h2>
        <p class="note">Treat this as shot-quality impact, not total player value. It does not include turnovers, rebounding, free throws, or playoff context.</p>
        ${rapmTable(rapm.slice(-15).reverse())}
      </div>
    </section>
  `;
}

function rapmTable(rows) {
  return table(rows, [
    { key: "player_name", label: "Player" },
    { key: "net_rapm", label: "Net", format: (v) => fmt(v, 2, true) },
    { key: "off_rapm", label: "Off", format: (v) => fmt(v, 2, true) },
    { key: "def_rapm", label: "Def", format: (v) => fmt(v, 2, true) },
    { key: "def_shots", label: "Def shots", format: (v) => fmt(v, 0) },
  ]);
}

function coaching() {
  const summary = data.coaching_did_summary;
  const detail = data.coaching_did_results.filter((row) => row.metric === "xpts_100poss");
  app.innerHTML = `
    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Coaching-change difference-in-differences</h2>
          <p>This asks whether a team allowed fewer points or easier shots after a coaching change, compared with the rest of the league over the same dates. Negative estimates mean the defense improved relative to league drift.</p>
        </div>
      </div>
      <div class="explain-grid compact">
        ${explainCard("Treated team", "The team that changed coaches during the season.")}
        ${explainCard("Control teams", "Teams in the same season that did not make an in-season coaching change.")}
        ${explainCard("Window", "The number of games before and after the coaching change. W=10 compares about ten games before with about ten games after.")}
        ${explainCard("Confidence interval", "A range of plausible values. If it crosses zero, the evidence is not strong enough to call the effect clearly positive or negative.")}
        ${explainCard("What to conclude", "Actual defensive rating moved in the expected direction after firings, but the intervals are wide. The honest read is directional, sample-limited evidence.")}
        ${explainCard("What not to conclude", "Do not treat the pooled estimate as proof that firing a coach caused better defense. Seven events cannot support that strong a claim.")}
      </div>
      <div id="coachingBars" class="chart"></div>
    </section>
    <section class="grid-two">
      <div class="panel">
        <h2>Pooled estimates</h2>
        ${coachingSummaryTable(summary)}
      </div>
      <div class="panel">
        <h2>Headline event details</h2>
        ${coachingDetailTable(detail)}
      </div>
    </section>
  `;
  barChart(
    document.querySelector("#coachingBars"),
    summary.map((row) => ({ player_name: `${row.metric} W${row.window}`, poe_per_100: row.pooled_did })),
    { title: "Pooled coaching DiD", value: "poe_per_100" },
  );
}

function coachingSummaryTable(rows) {
  return table(rows, [
    { key: "metric", label: "Metric" },
    { key: "window", label: "W", format: (v) => fmt(v, 0) },
    { key: "n_events", label: "N", format: (v) => fmt(v, 0) },
    { key: "pooled_did", label: "DiD", format: (v) => fmt(v, 2, true) },
    { key: "event_ci_low", label: "CI low", format: (v) => fmt(v, 2, true) },
    { key: "event_ci_high", label: "CI high", format: (v) => fmt(v, 2, true) },
  ]);
}

function coachingDetailTable(rows) {
  return table(rows, [
    { key: "season", label: "Season" },
    { key: "team_abbr", label: "Team" },
    { key: "coach_out", label: "Coach out" },
    { key: "window", label: "W", format: (v) => fmt(v, 0) },
    { key: "did", label: "DiD", format: (v) => fmt(v, 2, true) },
  ]);
}

function exportUseCase(file) {
  if (file.includes("coaching")) return "Coaching study";
  if (file.includes("rapm") || file.includes("model_evidence")) return "Model evidence";
  if (file.includes("shots_") || file.includes("shot_map_sample_")) return "Shot-level analysis";
  return "Player analysis";
}

function dataView() {
  const groupOrder = ["Player analysis", "Shot-level analysis", "Model evidence", "Coaching study"];
  const files = [...data.data_manifest]
    .map((row) => ({
      ...row,
      category: exportUseCase(row.file),
      download_label: row.file.split("/").pop(),
    }))
    .sort((a, b) => a.file.localeCompare(b.file));
  const grouped = groupOrder.map((category) => [category, files.filter((row) => row.category === category)]);
  app.innerHTML = `
    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Data exports</h2>
          <p>Every CSV used by the app is listed below. Full shot exports are large audit files; the dashboard renders sampled shot-map files so browser interactions stay readable.</p>
        </div>
      </div>
      <div class="explain-grid compact">
        ${explainCard("Player analysis", "Use player profiles, leaderboards, season summaries, and archetype summaries for player-level review.")}
        ${explainCard("Shot-level analysis", "Use full shots files for audits and sampled shot-map files for browser-friendly visualization.")}
        ${explainCard("Model evidence", "Use RAPM and report-index files to audit validation plots and impact estimates.")}
        ${explainCard("Coaching study", "Use DiD results and pooled summaries for the in-season coaching-change analysis.")}
      </div>
    </section>
    <section class="data-grid">
      ${grouped
        .map(
          ([category, rows]) => `
            <div class="panel">
              <h2>${category}</h2>
              ${table(rows, [
                { key: "download_label", label: "File" },
                { key: "bytes", label: "Size", format: (v) => `${fmt(v / 1_000_000, 2)} MB` },
                { key: "description", label: "Contents" },
                { key: "file", label: "Download", format: (v, row) => `<a href="${v}">${row.download_label}</a>` },
              ])}
            </div>
          `,
        )
        .join("")}
    </section>
  `;
}

function render() {
  updateTabs();
  const renderers = { overview, compare, archetypes, evidence, coaching, data: dataView };
  (renderers[state.view] || overview)();
  app.focus({ preventScroll: true });
}

applyUrlState();
initControls();
if (syncRender) {
  try {
    loadAllSync();
    render();
  } catch (error) {
    app.innerHTML = `<section class="error"><strong>Dashboard data failed to load.</strong><br />${error.message}<br />Run <code>make app</code> from the project root and open the local URL.</section>`;
  }
} else {
  loadAll()
    .then(() => {
      render();
      writeUrlState();
    })
    .catch((error) => {
      app.innerHTML = `<section class="error"><strong>Dashboard data failed to load.</strong><br />${error.message}<br />Run <code>make app</code> from the project root and open the local URL.</section>`;
    });
}
