/* Timestral × M5 static demo. All forecasts are precomputed; this script only loads and displays them. */
"use strict";

const S = {
  lang: "en", meta: null, series: null, keyIndex: new Map(), scaleSq: null, learn: null,
  chunks: new Map(), current: null, tab: "explore",
};
const REF_COLOR = { snaive: "--c-snaive", tsb: "--c-tsb", ets: "--c-ets", winner: "--c-winner" };
const LIEU = { 2: "state", 3: "store", 6: "state", 7: "state", 8: "store", 9: "store", 11: "state", 12: "store" };
const PRODUIT = { 4: "cat", 5: "dept", 6: "cat", 7: "dept", 8: "cat", 9: "dept" };
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ text & numbers
function t(key, vars) {
  let s = (I18N[S.lang] && I18N[S.lang][key]) ?? I18N.en[key] ?? key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(v);
  return s;
}
const locale = () => (S.lang === "fr" ? "fr-FR" : "en-US");
const fnum = (x, d = 3) => (x == null || !isFinite(x) ? "–" : x.toLocaleString(locale(), { minimumFractionDigits: d, maximumFractionDigits: d }));
const fint = (x) => (x == null || !isFinite(x) ? "–" : Math.round(x).toLocaleString(locale()));
function fpct(x, signed = false) {
  if (x == null || !isFinite(x)) return "–";
  const v = Math.round(x * 100);
  const s = (signed && v > 0 ? "+" : "") + v.toLocaleString(locale()).replace("-", "−");
  return S.lang === "fr" ? `${s} %` : `${s}%`;
}
const narrow = () => window.matchMedia("(max-width: 600px)").matches;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function cfgLabel(c) {
  return `${c.model.replace("Timestral-", "Timestral ")} · ${t("cov_" + c.covariates)}`;
}
const refLabel = (k) => t("ref_" + k);
function methodLabel(m) {
  const key = "m_" + m;
  if (I18N.en[key]) return t(key);
  const c = S.meta.configs.find((x) => x.name === m);
  return c ? cfgLabel(c) : m;
}

// ------------------------------------------------------------------ data
async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
async function chunk(prefix, c, k) {
  const key = `${prefix}/${c}`;
  if (!S.chunks.has(key)) {
    S.chunks.set(key, fetch(`data/${prefix}_${c}.bin`).then(async (r) => {
      if (!r.ok) throw new Error(`${prefix}_${c}: ${r.status}`);
      const buf = await r.arrayBuffer();
      const n = buf.byteLength / (4 + 2 * k);
      return { n, scale: new Float32Array(buf, 0, n), codes: new Uint16Array(buf, 4 * n, n * k) };
    }));
  }
  return S.chunks.get(key);
}
async function row(prefix, i, k) {
  const C = S.meta.chunk, c = Math.floor(i / C), j = i - c * C;
  const ch = await chunk(prefix, c, k);
  const out = new Float64Array(k), sc = ch.scale[j] / 65535;
  for (let m = 0; m < k; m++) out[m] = ch.codes[j * k + m] * sc;
  return out;
}
async function seriesData(i) {
  const H = S.meta.hist_days, h = S.meta.horizon, refs = S.meta.refs;
  const v = await row("base/base", i, H + h + refs.length * h);
  const out = { hist: v.subarray(0, H), actual: v.subarray(H, H + h), ref: {} };
  refs.forEach((r, k) => { out.ref[r] = v.subarray(H + h + k * h, H + h + (k + 1) * h); });
  return out;
}
async function forecastData(cfg, i) {
  const h = S.meta.horizon, nq = S.meta.display_q.length;
  const v = await row(`${cfg}/q`, i, nq * h + h);
  const q = [];
  for (let a = 0; a < nq; a++) q.push(v.subarray(a * h, (a + 1) * h));
  return { q, mean: v.subarray(nq * h, nq * h + h) };
}
function quantileAt(fc, u) {
  const L = S.meta.display_q, h = S.meta.horizon, out = new Float64Array(h);
  let k = 0;
  while (k < L.length - 2 && u > L[k + 1]) k++;
  const w = Math.min(1, Math.max(0, (u - L[k]) / (L[k + 1] - L[k])));
  for (let d = 0; d < h; d++) out[d] = fc.q[k][d] + w * (fc.q[k + 1][d] - fc.q[k][d]);
  return out;
}

// ------------------------------------------------------------------ metrics
const sum = (a) => a.reduce((x, y) => x + y, 0);
function metrics(i, f, y) {
  let se = 0, ae = 0;
  for (let d = 0; d < y.length; d++) { se += (f[d] - y[d]) ** 2; ae += Math.abs(f[d] - y[d]); }
  const tot = sum(y);
  return {
    rmsse: Math.sqrt(se / y.length / Math.max(S.scaleSq[i], 1e-12)),
    wape: tot > 0 ? ae / tot : NaN, bias: tot > 0 ? (sum(f) - tot) / tot : NaN, total: sum(f),
  };
}

// ------------------------------------------------------------------ dates
function dayDate(offset) {
  const d = new Date(S.meta.start_date + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + offset);
  return d.toISOString().slice(0, 10);
}

// ------------------------------------------------------------------ series selection
function fillSelect(sel, values, labels, value) {
  sel.innerHTML = values.map((v, k) => `<option value="${esc(v)}">${esc(labels ? labels[k] : v)}</option>`).join("");
  if (value != null && values.includes(value)) sel.value = value;
}
function updateSelectors(keep = {}) {
  const lv = +$("sel-level").value, m = S.meta;
  const loc = LIEU[lv], prod = PRODUIT[lv], isItem = lv >= 10;
  $("f-loc").hidden = !loc; $("f-prod").hidden = !prod;
  $("f-dept").hidden = !isItem; $("f-item").hidden = !isItem;
  if (loc) {
    $("lbl-loc").textContent = t(loc === "state" ? "state" : "store");
    fillSelect($("sel-loc"), loc === "state" ? m.states : m.stores, null, keep.loc ?? $("sel-loc").value);
  }
  if (prod) {
    $("lbl-prod").textContent = t(prod === "cat" ? "category" : "department");
    fillSelect($("sel-prod"), prod === "cat" ? m.cats : m.depts, null, keep.prod ?? $("sel-prod").value);
  }
  if (isItem) {
    const dept = keep.dept ?? ($("sel-dept").value || "FOODS_3");
    fillSelect($("sel-dept"), m.depts, null, dept);
    fillSelect($("sel-item"), m.items_by_dept[$("sel-dept").value], null, keep.item ?? $("sel-item").value);
  }
}
function currentIndex() {
  const lv = +$("sel-level").value, loc = $("sel-loc").value, prod = $("sel-prod").value, item = $("sel-item").value;
  let a1 = "X", a2 = "X";
  if (lv === 1) a1 = "Total";
  else if (lv === 2 || lv === 3) a1 = loc;
  else if (lv === 4 || lv === 5) a1 = prod;
  else if (lv <= 9) { a1 = loc; a2 = prod; }
  else if (lv === 10) a1 = item;
  else if (lv === 11) { a1 = loc; a2 = item; }
  else { a1 = item; a2 = loc; }
  return S.keyIndex.get(`${lv}|${a1}|${a2}`);
}
function selectIndex(i) {
  const lv = S.series.level[i], a1 = S.series.a1[i], a2 = S.series.a2[i];
  $("sel-level").value = String(lv);
  const keep = {};
  if (lv === 2 || lv === 3) keep.loc = a1;
  else if (lv === 4 || lv === 5) keep.prod = a1;
  else if (lv >= 6 && lv <= 9) { keep.loc = a1; keep.prod = a2; }
  else if (lv === 10) keep.item = a1;
  else if (lv === 11) { keep.loc = a1; keep.item = a2; }
  else if (lv === 12) { keep.item = a1; keep.loc = a2; }
  if (keep.item) keep.dept = keep.item.split("_").slice(0, 2).join("_");
  updateSelectors(keep);
}
function seriesName(i) {
  const lv = S.series.level[i], parts = [S.series.a1[i], S.series.a2[i]].filter((p) => p !== "X" && p !== "Total");
  return parts.length ? `${t("lv_names")[lv - 1]} · ${parts.join(" · ")}` : t("total_series");
}

// ------------------------------------------------------------------ explore rendering
function kpi(label, value, sub = "", tone = "") {
  return `<div class="kpi ${tone}"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`;
}
function plotLayout(extra = {}) {
  const ink = css("--ink"), muted = css("--muted"), grid = css("--grid");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: css("--font-body"), color: ink, size: 12 },
    margin: { l: 52, r: 14, t: 40, b: 40 }, hovermode: "x unified",
    legend: { orientation: "h", y: -0.16, x: 0, font: { size: 11.5, color: muted } },
    xaxis: { gridcolor: grid, linecolor: grid, tickfont: { color: muted } },
    yaxis: { gridcolor: grid, zerolinecolor: grid, tickfont: { color: muted }, rangemode: "tozero" },
  }, extra);
}
const plotConfig = { displayModeBar: false, responsive: true };

async function renderExplore() {
  const i = currentIndex();
  if (i == null || !S.meta.configs.length) return;
  const cfg = $("sel-config").value, c = S.meta.configs.find((x) => x.name === cfg);
  const [sd, fc] = await Promise.all([seriesData(i), forecastData(cfg, i)]);
  if (i !== currentIndex() || cfg !== $("sel-config").value) return; // a newer selection is pending
  S.current = { i, sd, fc, c };
  const y = sd.actual, med = quantileAt(fc, 0.5);
  const rows = { median: metrics(i, med, y), mean: metrics(i, fc.mean, y) };
  for (const r of S.meta.refs) rows[r] = metrics(i, sd.ref[r], y);
  const bestT = Math.min(rows.median.rmsse, rows.mean.rmsse);
  const stat = S.meta.refs.filter((r) => r !== "winner");
  const bestStat = stat.reduce((a, b) => (rows[b].rmsse < rows[a].rmsse ? b : a), stat[0]);
  const lo80 = quantileAt(fc, 0.1), hi80 = quantileAt(fc, 0.9);
  let in80 = 0;
  for (let d = 0; d < y.length; d++) if (y[d] >= lo80[d] - 1e-9 && y[d] <= hi80[d] + 1e-9) in80++;

  const cards = [kpi(t("k_rmsse"), fnum(bestT), t("k_lower"))];
  if (bestStat) {
    const g = 1 - bestT / rows[bestStat].rmsse;
    cards.push(kpi(t("k_vs", { name: refLabel(bestStat) }), fpct(g, true), t("k_best_stat", { v: fnum(rows[bestStat].rmsse) }), g > 0 ? "good" : "bad"));
  }
  if (rows.winner) {
    const g = 1 - bestT / rows.winner.rmsse;
    cards.push(kpi(t("k_vs_winner"), fpct(g, true), `RMSSE ${fnum(rows.winner.rmsse)}`, g > 0 ? "good" : "bad"));
  }
  cards.push(kpi(t("k_band"), t("k_band_v", { n: in80 }), t("k_band_sub"), in80 >= 18 && in80 <= 26 ? "good" : "bad"));
  cards.push(kpi(t("k_bias"), fpct(rows.median.bias, true), t("k_bias_sub")));
  cards.push(kpi(t("k_speed"), `${fint(c.ms_per_series)} ms`, t("k_speed_sub", { model: c.model.replace("Timestral-", "") })));
  $("kpis").innerHTML = cards.join("");

  const zs = sd.hist.filter((v) => v < 0.5).length / sd.hist.length;
  const meanWins = rows.mean.rmsse < rows.median.rmsse;
  $("reading").innerHTML = t("reading", {
    zs: fpct(zs), verdict: t(bestStat && bestT < rows[bestStat].rmsse ? "beats" : "trails"), n: in80,
    which: t(meanWins ? "w_mean" : "w_median"), extra: zs > 0.4 && meanWins ? t("reading_extra") : ".",
  });

  const names = { median: t("median"), mean: t("mean") };
  for (const r of S.meta.refs) names[r] = refLabel(r);
  let html = `<thead><tr><th>${t("col_method")}</th><th class="num">RMSSE</th><th class="num">WAPE</th><th class="num">${t("col_bias")}</th><th class="num">${t("col_total")}</th></tr></thead><tbody>`;
  for (const [k, v] of Object.entries(rows)) {
    html += `<tr class="${k === "median" || k === "mean" ? "hl" : ""}"><td>${esc(names[k])}</td><td class="num">${fnum(v.rmsse)}</td><td class="num">${fpct(v.wape)}</td><td class="num">${fpct(v.bias, true)}</td><td class="num">${fint(v.total)}</td></tr>`;
  }
  html += `<tr><td>${t("actual")}</td><td class="num">–</td><td class="num">–</td><td class="num">–</td><td class="num">${fint(sum(y))}</td></tr></tbody>`;
  $("err-table").innerHTML = html;
  drawForecast();
  drawService();
}

function drawForecast() {
  if (!S.current || typeof Plotly === "undefined") return;
  const { i, sd, fc } = S.current, H = S.meta.hist_days, h = S.meta.horizon;
  const days = +$("sel-days").value;
  const hx = [], fx = [];
  for (let d = H - days; d < H; d++) hx.push(dayDate(d));
  for (let d = H; d < H + h; d++) fx.push(dayDate(d));
  const acc = css("--accent"), rgb = hexToRgb(acc);
  const band = (lo, hi, name, a) => ({
    x: fx.concat([...fx].reverse()), y: Array.from(hi).concat(Array.from(lo).reverse()), fill: "toself",
    fillcolor: `rgba(${rgb},${a})`, line: { width: 0 }, name, hoverinfo: "skip", type: "scatter",
  });
  const traces = [
    { x: hx, y: Array.from(sd.hist.subarray(H - days)), name: t("history"), mode: "lines", line: { color: css("--c-history"), width: 1.4 } },
    band(quantileAt(fc, 0.01), quantileAt(fc, 0.99), t("band98"), 0.13),
    band(quantileAt(fc, 0.1), quantileAt(fc, 0.9), t("band80"), 0.28),
    { x: fx, y: Array.from(quantileAt(fc, 0.5)), name: t("median"), mode: "lines", line: { color: acc, width: 2.6 } },
  ];
  for (const box of document.querySelectorAll("#compare input:checked")) {
    const r = box.value;
    traces.push({ x: fx, y: Array.from(sd.ref[r]), name: refLabel(r), mode: "lines", line: { color: css(REF_COLOR[r]), width: 1.6, dash: "dot" } });
  }
  traces.push({ x: fx, y: Array.from(sd.actual), name: t("actual_test"), mode: "lines+markers", line: { color: css("--c-actual"), width: 2 }, marker: { size: 4 } });
  const muted = css("--muted");
  const shapes = [{ type: "line", x0: fx[0], x1: fx[0], yref: "paper", y0: 0, y1: 1, line: { color: muted, width: 1, dash: "dash" } }];
  const annotations = [{ x: fx[0], y: 1.02, yref: "paper", text: t("forecast_start"), showarrow: false, xanchor: "left", font: { size: 11, color: muted } }];
  for (const [off, name] of narrow() ? [] : S.meta.events) {
    if (off >= H) annotations.push({ x: dayDate(off), y: 0.97, yref: "paper", text: esc(name), showarrow: false, textangle: -90, yanchor: "top", xshift: -6, font: { size: 9.5, color: muted } });
  }
  Plotly.react("chart", traces, plotLayout({
    title: { text: esc(seriesName(i)), font: { size: 15, family: css("--font-body") }, x: 0, xanchor: "left" },
    shapes, annotations, yaxis: { ...plotLayout().yaxis, title: { text: t("units_day"), font: { size: 12, color: muted } } },
    xaxis: { ...plotLayout().xaxis, tickformat: S.lang === "fr" ? "%d/%m" : "%b %d" },
  }), plotConfig);
}

function drawService() {
  if (!S.current || typeof Plotly === "undefined") return;
  const { sd, fc } = S.current, H = S.meta.hist_days, h = S.meta.horizon;
  const target = +$("svc").value, u = target / 100, y = sd.actual;
  $("svc-out").textContent = target;
  const lq = quantileAt(fc, u), med = quantileAt(fc, 0.5);
  let cov = 0, covM = 0, short = 0, surplus = 0;
  for (let d = 0; d < h; d++) {
    if (y[d] <= lq[d] + 1e-9) cov++;
    if (y[d] <= med[d] + 1e-9) covM++;
    short += Math.max(y[d] - lq[d], 0); surplus += Math.max(lq[d] - y[d], 0);
  }
  const tot = sum(y), fill = tot > 0 ? 1 - short / tot : NaN;
  $("svc-kpis").innerHTML = [
    kpi(t("s_covered"), fpct(cov / h), t("s_covered_sub", { t: target, n: cov }), Math.abs(cov / h - u) <= 0.1 ? "good" : "bad"),
    kpi(t("s_fill"), fpct(fill), t("s_fill_sub", { v: fint(short) })),
    kpi(t("s_surplus"), fint(surplus), t("s_surplus_sub")),
    kpi(t("s_median"), fpct(covM / h), t("s_median_sub")),
  ].join("");
  const fx = [];
  for (let d = H; d < H + h; d++) fx.push(dayDate(d));
  const bad = css("--bad"), hist = css("--c-history"), acc = css("--accent");
  Plotly.react("svc-chart", [
    { x: fx, y: Array.from(y), type: "bar", name: t("actual_demand"), marker: { color: Array.from(y).map((v, d) => (v > lq[d] + 1e-9 ? bad : hist)) } },
    { x: fx, y: Array.from(lq), name: t("stock_at", { t: target }), mode: "lines", line: { color: acc, width: 2.4, shape: "hv" } },
    { x: fx, y: Array.from(med), name: t("median_short"), mode: "lines", line: { color: acc, width: 1.2, dash: "dot", shape: "hv" } },
  ], plotLayout({ margin: { l: 52, r: 14, t: 10, b: 40 }, bargap: 0.25,
    yaxis: { ...plotLayout().yaxis, title: { text: t("units"), font: { size: 12, color: css("--muted") } } },
    xaxis: { ...plotLayout().xaxis, tickformat: S.lang === "fr" ? "%d/%m" : "%b %d" } }), plotConfig);
}

function hexToRgb(hex) {
  const m = hex.replace("#", "").match(/.{2}/g);
  return m ? m.map((x) => parseInt(x, 16)).join(",") : "232,89,12";
}

// ------------------------------------------------------------------ benchmark
function bestVariant(c) {
  const m = c.wrmsse["median|bottom-up"], e = c.wrmsse["mean|bottom-up"];
  return e < m ? { v: "mean", s: e } : { v: "median", s: m };
}
function renderBenchmark() {
  const m = S.meta, cfgs = m.configs;
  const base = Object.fromEntries(m.baselines.filter((b) => b.metric === "wrmsse").map((b) => [b.method, b.score]));
  const winner = Object.entries(base).find(([k]) => k.startsWith("M5 #1 "));
  if (cfgs.length) {
    const best = cfgs.reduce((a, b) => (bestVariant(b).s < bestVariant(a).s ? b : a));
    const bs = bestVariant(best).s;
    const rel = (ref) => (ref ? 1 - bs / ref : NaN);
    $("bench-kpis").innerHTML = [
      kpi(t("b_best"), fnum(bs, 4), `${esc(cfgLabel(best))} · ${t("var_" + bestVariant(best).v)}`, "lead"),
      kpi(t("b_vs_ets"), fpct(rel(base.ets), true), `WRMSSE ${fnum(base.ets, 4)}`, rel(base.ets) > 0 ? "good" : "bad"),
      kpi(t("b_vs_snaive"), fpct(rel(base.snaive), true), `WRMSSE ${fnum(base.snaive, 4)}`, rel(base.snaive) > 0 ? "good" : "bad"),
      kpi(t("b_vs_winner"), fpct(rel(winner && winner[1]), true), `WRMSSE ${fnum(winner && winner[1], 4)}`, rel(winner && winner[1]) > 0 ? "good" : "bad"),
    ].join("") + uncertaintyKpi();
  } else {
    $("bench-kpis").innerHTML = "";
  }

  // leaderboard
  const rows = [];
  for (const [k, s] of Object.entries(base)) {
    const comp = k.startsWith("M5 #");
    rows.push({ name: comp ? k.replace("M5 #", "M5 #") : methodLabel(k), type: comp ? "comp" : "stat", s, point: "" });
  }
  for (const c of cfgs) { const b = bestVariant(c); rows.push({ name: cfgLabel(c), type: "tsfm", s: b.s, point: t("var_" + b.v) }); }
  rows.sort((a, b) => a.s - b.s);
  const typeTag = { stat: `<span class="tag">${t("type_stat")}</span>`, comp: `<span class="tag">${t("type_comp")}</span>`, tsfm: `<span class="tag tsfm">${t("type_tsfm")}</span>` };
  $("lb-table").innerHTML = `<thead><tr><th class="num">${t("lb_rank")}</th><th>${t("lb_method")}</th><th>${t("lb_type")}</th><th>${t("lb_point")}</th><th class="num">WRMSSE</th></tr></thead><tbody>` +
    rows.map((r, k) => `<tr class="${r.type === "tsfm" ? "hl" : ""}"><td class="num">${k + 1}</td><td>${esc(r.name)}</td><td>${typeTag[r.type]}</td><td>${esc(r.point)}</td><td class="num">${fnum(r.s, 4)}</td></tr>`).join("") + "</tbody>";

  // per-level chart
  if (typeof Plotly !== "undefined") {
    const lvl = (method, variant, mode) => {
      const out = new Array(12).fill(null);
      for (const r of m.by_level) {
        if (r.metric !== "wrmsse" || r.method !== method || r.mode !== mode || (variant != null && r.variant !== variant)) continue;
        const n = r.level === "Total" ? 0 : +r.level.replace("Level", "");
        if (n >= 1) out[n - 1] = r.score;
      }
      return out;
    };
    const small = narrow();
    const xs = small ? t("lv_names").map((_, k) => String(k + 1)) : t("lv_names").map((n, k) => `${k + 1}. ${n}`);
    const series = [];
    if (cfgs.length) {
      const best = cfgs.reduce((a, b) => (bestVariant(b).s < bestVariant(a).s ? b : a));
      series.push({ name: cfgLabel(best), y: lvl(best.name, bestVariant(best).v, "bottom-up"), color: css("--accent"), width: 3 });
    }
    series.push({ name: methodLabel("ets"), y: lvl("ets", "", "bottom-up"), color: css("--c-ets") });
    series.push({ name: methodLabel("snaive"), y: lvl("snaive", "", "bottom-up"), color: css("--c-snaive") });
    series.push({ name: methodLabel("tsb"), y: lvl("tsb", "", "bottom-up"), color: css("--c-tsb") });
    if (winner) series.push({ name: refLabel("winner"), y: lvl(winner[0], "", "bottom-up"), color: css("--c-winner") });
    Plotly.react("lv-chart", series.map((s) => ({ x: xs, y: s.y, name: s.name, mode: "lines+markers", line: { color: s.color, width: s.width || 1.8 }, marker: { size: 6 } })),
      plotLayout({
        margin: small ? { l: 44, r: 10, t: 10, b: 40 } : { l: 52, r: 14, t: 10, b: 110 }, hovermode: "x unified",
        xaxis: { ...plotLayout().xaxis, tickangle: small ? 0 : -35, type: "category",
          title: small ? { text: t("level"), font: { size: 12, color: css("--muted") } } : undefined },
        yaxis: { ...plotLayout().yaxis, title: { text: "RMSSE", font: { size: 12, color: css("--muted") } } },
        legend: small ? { orientation: "h", yanchor: "top", y: -0.22, x: 0, font: { size: 11, color: css("--muted") } }
                      : { orientation: "v", x: 1.01, y: 1, font: { size: 11.5, color: css("--muted") } },
      }), plotConfig);
  }

  renderUncertainty();

  // ablation
  const emp = m.baselines.find((b) => b.metric === "wspl");
  if (!cfgs.length) { $("ab-table").innerHTML = `<tbody><tr><td>${t("ab_empty")}</td></tr></tbody>`; }
  else {
    const noneOf = (model) => cfgs.find((c) => c.covariates === "none" && c.model === model);
    $("ab-table").innerHTML = `<thead><tr><th>${t("ab_cfg")}</th><th>${t("ab_cov")}</th><th class="num">${t("ab_bu_med")}</th><th class="num">${t("ab_bu_mean")}</th><th class="num">${t("ab_direct")}</th><th class="num">${t("ab_delta")}</th><th class="num">${t("ab_wspl")}</th><th class="num">${t("ab_speed")}</th></tr></thead><tbody>` +
      cfgs.map((c) => {
        const ref = noneOf(c.model), b = bestVariant(c), d = ref && c !== ref ? 1 - b.s / bestVariant(ref).s : null;
        const delta = d == null ? "–" : `<span class="${d > 0 ? "delta-good" : "delta-bad"}">${fpct(d, true)}</span>`;
        const dir = Math.min(c.wrmsse["median|direct"], c.wrmsse["mean|direct"]);
        return `<tr><td>${esc(c.model.replace("Timestral-", "Timestral "))}</td><td title="${esc(t("cov_desc_" + c.covariates))}">${esc(t("cov_" + c.covariates))}</td><td class="num">${fnum(c.wrmsse["median|bottom-up"], 4)}</td><td class="num">${fnum(c.wrmsse["mean|bottom-up"], 4)}</td><td class="num">${fnum(dir, 4)}</td><td class="num">${delta}</td><td class="num">${fnum(c.wspl, 4)}</td><td class="num">${fint(c.ms_per_series)}</td></tr>`;
      }).join("") + "</tbody>";
  }
  $("ab-note").innerHTML = t("ab_note", { emp: emp ? fnum(emp.score, 4) : "–" }) + "<br>" + t("ab_note_median") + "<br>" +
    ["none", "calendar", "full", "rich"].map((k) => `<strong>${esc(t("cov_" + k))}</strong>${S.lang === "fr" ? "\u00a0:" : ":"} ${esc(t("cov_desc_" + k))}`).join("<br>");
}


function bestWspl() {
  const cfgs = S.meta.configs;
  return cfgs.length ? cfgs.reduce((a, b) => (b.wspl < a.wspl ? b : a)) : null;
}
function uncertaintyKpi() {
  const best = bestWspl(), pub = (S.meta.uncertainty_published || []).find((p) => p.key === "unc_winner");
  if (!best || !pub) return "";
  const g = 1 - best.wspl / pub.wspl;
  return kpi(t("b_unc"), fpct(g, true), t("b_unc_sub", { cfg: esc(cfgLabel(best)), v: fnum(best.wspl, 3), w: fnum(pub.wspl, 3) }), g >= -0.005 ? "good" : "bad");
}
function renderUncertainty() {
  const m = S.meta, rows = [];
  for (const c of m.configs) rows.push({ name: cfgLabel(c), type: "tsfm", s: c.wspl, levels: null, cfg: c.name });
  for (const p of m.uncertainty_published || []) rows.push({ name: t(p.key), type: p.key === "unc_winner" ? "comp" : "stat", s: p.wspl, levels: p.levels });
  const emp = m.baselines.find((b) => b.metric === "wspl");
  if (emp) rows.push({ name: t("unc_emp"), type: "stat", s: emp.score });
  rows.sort((a, b) => a.s - b.s);
  const tag = { stat: `<span class="tag">${t("type_stat")}</span>`, comp: `<span class="tag">${t("type_comp")}</span>`, tsfm: `<span class="tag tsfm">${t("type_tsfm")}</span>` };
  $("unc-table").innerHTML = `<thead><tr><th class="num">${t("lb_rank")}</th><th>${t("lb_method")}</th><th>${t("lb_type")}</th><th class="num">WSPL</th></tr></thead><tbody>` +
    rows.map((r, k) => `<tr class="${r.type === "tsfm" ? "hl" : ""}"><td class="num">${k + 1}</td><td>${esc(r.name)}</td><td>${tag[r.type]}</td><td class="num">${fnum(r.s, 3)}</td></tr>`).join("") +
    `</tbody><caption class="muted" style="caption-side:bottom;text-align:left;padding:8px 12px"><a href="${esc(m.uncertainty_source || "#")}">${t("unc_source")}</a></caption>`;
  if (typeof Plotly === "undefined") return;
  const small = narrow();
  const xs = small ? t("lv_names").map((_, k) => String(k + 1)) : t("lv_names").map((n, k) => `${k + 1}. ${n}`);
  const lvl = (method) => {
    const out = new Array(12).fill(null);
    for (const r of m.by_level) if (r.metric === "wspl" && r.method === method && r.level !== "Total") out[+r.level.replace("Level", "") - 1] = r.score;
    return out;
  };
  const best = bestWspl(), series = [];
  if (best) series.push({ name: cfgLabel(best), y: lvl(best.name), color: css("--accent"), width: 3 });
  for (const p of m.uncertainty_published || []) series.push({ name: t(p.key), y: p.levels, color: css(p.key === "unc_winner" ? "--c-winner" : "--c-snaive") });
  if (emp) series.push({ name: t("unc_emp"), y: lvl(emp.method), color: css("--c-history") });
  Plotly.react("unc-chart", series.map((s) => ({ x: xs, y: s.y, name: s.name, mode: "lines+markers", line: { color: s.color, width: s.width || 1.8 }, marker: { size: 6 } })),
    plotLayout({
      margin: small ? { l: 44, r: 10, t: 10, b: 40 } : { l: 52, r: 14, t: 10, b: 110 }, hovermode: "x unified",
      xaxis: { ...plotLayout().xaxis, tickangle: small ? 0 : -35, type: "category", title: small ? { text: t("level"), font: { size: 12, color: css("--muted") } } : undefined },
      yaxis: { ...plotLayout().yaxis, title: { text: "SPL", font: { size: 12, color: css("--muted") } } },
      legend: small ? { orientation: "h", yanchor: "top", y: -0.22, x: 0, font: { size: 11, color: css("--muted") } }
                    : { orientation: "v", x: 1.01, y: 1, font: { size: 11.5, color: css("--muted") } },
    }), plotConfig);
}

// ------------------------------------------------------------------ static texts, language, tabs
function applyTexts() {
  document.documentElement.lang = S.lang;
  for (const el of document.querySelectorAll("[data-t]")) el.textContent = t(el.dataset.t);
  for (const el of document.querySelectorAll("[data-th]")) el.innerHTML = t(el.dataset.th);
  for (const b of document.querySelectorAll(".lang button")) b.setAttribute("aria-pressed", String(b.dataset.lang === S.lang));
  if (!S.meta) return;
  $("foot-text").textContent = t("footer", { date: new Date(S.meta.generated + "T00:00:00Z").toLocaleDateString(locale(), { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" }) });
  const lvSel = $("sel-level"), lv = lvSel.value || "12";
  fillSelect(lvSel, Array.from({ length: 12 }, (_, k) => String(k + 1)), t("lv_names").map((n, k) => `${k + 1}. ${n}`), lv);
  const cfgSel = $("sel-config"), cv = cfgSel.value || S.meta.default_config;
  fillSelect(cfgSel, S.meta.configs.map((c) => c.name), S.meta.configs.map(cfgLabel), cv);
  updateCfgHint();
  $("examples").innerHTML = S.meta.examples.map((e) => `<button type="button" data-i="${e.i}">${esc(t(e.key))}</button>`).join("");
  const checked = new Set([...document.querySelectorAll("#compare input:checked")].map((x) => x.value));
  const defaults = checked.size || document.querySelector("#compare input") ? checked : new Set(["snaive", "ets", "winner"]);
  $("compare").innerHTML = S.meta.refs.map((r) => `<label><input type="checkbox" value="${r}" ${defaults.has(r) ? "checked" : ""}><span class="swatch" style="border-color:var(${REF_COLOR[r]})"></span>${esc(refLabel(r))}</label>`).join("");
  updateSelectors();
  $("learn-body").innerHTML = S.learn ? S.learn[S.lang] : "";
}
function updateCfgHint() {
  const c = S.meta.configs.find((x) => x.name === $("sel-config").value);
  $("cfg-hint").textContent = c ? t("cov_desc_" + c.covariates) : "";
}
function setLang(lang) {
  S.lang = lang;
  try { localStorage.setItem("lang", lang); } catch (e) { /* storage unavailable */ }
  applyTexts();
  if (S.meta) { renderExplore(); renderBenchmark(); }
}
function setTab(tab, push = true) {
  S.tab = tab;
  for (const id of ["explore", "benchmark", "learn"]) {
    $(id).hidden = id !== tab;
    $("tab-" + id).setAttribute("aria-selected", String(id === tab));
  }
  if (push) history.replaceState(null, "", "#" + tab);
  if (typeof Plotly !== "undefined") requestAnimationFrame(() => {
    for (const id of ["chart", "svc-chart", "lv-chart", "unc-chart"]) { const el = $(id); if (el && el.data && !el.closest("[hidden]")) Plotly.Plots.resize(el); }
  });
}

// ------------------------------------------------------------------ boot
async function boot() {
  let lang = (navigator.language || "en").toLowerCase().startsWith("fr") ? "fr" : "en";
  try { lang = localStorage.getItem("lang") || lang; } catch (e) { /* storage unavailable */ }
  S.lang = lang;
  applyTexts();
  const hash = location.hash.replace("#", "");
  setTab(["explore", "benchmark", "learn"].includes(hash) ? hash : "explore", false);
  try {
    const [meta, series, learn, sq] = await Promise.all([
      getJSON("data/meta.json"), getJSON("data/series.json"), getJSON("data/learn.json"),
      fetch("data/scale_sq.bin").then((r) => { if (!r.ok) throw new Error("scale_sq"); return r.arrayBuffer(); }),
    ]);
    S.meta = meta; S.series = series; S.learn = learn; S.scaleSq = new Float32Array(sq);
    series.level.forEach((lv, i) => S.keyIndex.set(`${lv}|${series.a1[i]}|${series.a2[i]}`, i));
  } catch (err) {
    $("status").textContent = t("load_error");
    $("status").classList.add("error");
    console.error(err);
    return;
  }
  $("status").hidden = true;
  if (narrow()) { $("sel-days").value = "56"; $("days-out").textContent = "56"; }
  applyTexts();
  selectIndex(S.meta.examples[2] ? S.meta.examples[2].i : 0);
  renderBenchmark();
  await renderExplore();

  $("sel-level").addEventListener("change", () => { updateSelectors(); renderExplore(); });
  for (const id of ["sel-loc", "sel-prod", "sel-item"]) $(id).addEventListener("change", renderExplore);
  $("sel-dept").addEventListener("change", () => {
    fillSelect($("sel-item"), S.meta.items_by_dept[$("sel-dept").value]);
    renderExplore();
  });
  $("sel-config").addEventListener("change", () => { updateCfgHint(); renderExplore(); });
  $("compare").addEventListener("change", drawForecast);
  $("sel-days").addEventListener("input", () => { $("days-out").textContent = $("sel-days").value; drawForecast(); });
  $("svc").addEventListener("input", drawService);
  $("examples").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-i]");
    if (b) { selectIndex(+b.dataset.i); renderExplore(); }
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawForecast(); drawService(); renderBenchmark(); });
}

document.addEventListener("DOMContentLoaded", () => {
  for (const b of document.querySelectorAll(".lang button")) b.addEventListener("click", () => setLang(b.dataset.lang));
  for (const id of ["explore", "benchmark", "learn"]) $("tab-" + id).addEventListener("click", () => setTab(id));
  boot();
});
