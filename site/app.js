/* Timestral static demo on two benchmarks (M5 Walmart, daily; VN1, weekly).
   All forecasts are precomputed; this script only loads and displays them. */
"use strict";

const S = {
  lang: "en", bench: "m5", benches: ["m5"], tab: "explore",
  meta: null, series: null, keyIndex: new Map(), scaleSq: null, vn1: null, // data of the current benchmark
  loaded: {}, learn: null, glossary: null, chunks: new Map(), current: null, checked: {},
};
const TABS = ["explore", "benchmark", "learn", "glossary"];
const REF_COLOR = {
  snaive: "--c-snaive", tsb: "--c-tsb", ets: "--c-ets", arima: "--c-arima", winner: "--c-winner",
  naive: "--c-naive", ma12: "--c-ma12", vn1_1st: "--c-winner",
};
const DEFAULT_REFS = { m5: ["snaive", "ets", "winner"], vn1: ["naive", "vn1_1st"] };
const GLOSS = { // method → glossary entry
  snaive: "snaive", naive: "naive", ma28: "ma", ma12: "ma", ses: "ses", sba: "croston", tsb: "tsb", adida: "adida",
  imapa: "adida", ets: "ets", arima: "arima",
};
const LIEU = { 2: "state", 3: "store", 6: "state", 7: "state", 8: "store", 9: "store", 11: "state", 12: "store" };
const PRODUIT = { 4: "cat", 5: "dept", 6: "cat", 7: "dept", 8: "cat", 9: "dept" };
const $ = (id) => document.getElementById(id);
const isVN1 = () => S.bench === "vn1";

// ------------------------------------------------------------------ text & numbers
function t(key, vars) {
  const L = I18N[S.lang] || {}, E = I18N.en, bk = `${key}__${S.bench}`; // benchmark-specific text wins
  let s = L[bk] ?? E[bk] ?? L[key] ?? E[key] ?? key;
  if (vars && typeof s === "string") for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(v);
  return s;
}
const locale = () => (S.lang === "fr" ? "fr-FR" : "en-US");
const fnum = (x, d = 3) => (x == null || !isFinite(x) ? "–" : x.toLocaleString(locale(), { minimumFractionDigits: d, maximumFractionDigits: d }));
const fint = (x) => (x == null || !isFinite(x) ? "–" : Math.round(x).toLocaleString(locale()));
function fpct(x, signed = false, digits = 0) {
  if (x == null || !isFinite(x)) return "–";
  const f = 10 ** digits, v = Math.round(x * 100 * f) / f;
  const s = (signed && v > 0 ? "+" : "") + v.toLocaleString(locale(), { minimumFractionDigits: digits, maximumFractionDigits: digits }).replace("-", "−");
  return S.lang === "fr" ? `${s} %` : `${s}%`;
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
  if (I18N.en[key] || I18N.en[`${key}__${S.bench}`]) return t(key);
  const c = S.meta.configs.find((x) => x.name === m);
  return c ? cfgLabel(c) : m;
}
// method name linked to its glossary entry when there is one
function glossLink(text, anchor) {
  return anchor ? `<a href="#glossary" class="gl" data-g="${anchor}">${esc(text)}</a>` : esc(text);
}
function methodAnchor(m) {
  if (GLOSS[m]) return GLOSS[m];
  if (m === "winner" || m.startsWith("M5 #")) return "lightgbm";
  if (m.startsWith("vn1_")) return null; // VN1 top 5: varied methods, no single glossary entry
  const c = S.meta.configs.find((x) => x.name === m);
  return c ? cfgAnchor(c) : null;
}
const help = (anchor) => ` <a href="#glossary" class="gl help" data-g="${anchor}" aria-label="${esc(t("what_is"))}">?</a>`;

// ------------------------------------------------------------------ data
async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
async function chunk(prefix, c, k) {
  const key = `${S.bench}/${prefix}/${c}`;
  if (!S.chunks.has(key)) {
    S.chunks.set(key, fetch(`data/${S.bench}/${prefix}_${c}.bin`).then(async (r) => {
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
  const h = S.meta.horizon, nq = S.meta.display_q.length, c = S.meta.configs.find((x) => x.name === cfg);
  const extra = c && c.calib ? h : 0; // calibrated point forecast stored after the mean
  const v = await row(`${cfg}/q`, i, nq * h + h + extra);
  const q = [];
  for (let a = 0; a < nq; a++) q.push(v.subarray(a * h, (a + 1) * h));
  return { q, mean: v.subarray(nq * h, nq * h + h), calib: extra ? v.subarray(nq * h + h, nq * h + 2 * h) : null };
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
  const tot = sum(y), bias = sum(f) - tot;
  return {
    rmsse: S.scaleSq ? Math.sqrt(se / y.length / Math.max(S.scaleSq[i], 1e-12)) : NaN,
    wape: tot > 0 ? ae / tot : NaN, bias: tot > 0 ? bias / tot : NaN, total: sum(f),
    score: tot > 0 ? (ae + Math.abs(bias)) / tot : NaN, // official VN1 score on this series
  };
}
// the error used to rank methods on one series: RMSSE for M5, VN1 score for VN1
const err = (m) => (isVN1() ? m.score : m.rmsse);
// VN1 point forecast of a configuration: the quantile chosen on phase 1 (validation) when available, so that
// nothing is selected on the test weeks; otherwise the better of median and mean.
const bestVN1 = (c) => (c.calib ? { v: "calib", ...c.calib.test }
  : c.mean.score < c.median.score ? { v: "mean", ...c.mean } : { v: "median", ...c.median });
// the configuration put forward: best on phase 1 when calibrated, otherwise best on the test
const valScore = (c) => (c.calib ? c.calib.val.score : bestVN1(c).score);
// one reference version of Timestral across the site (the other sizes stay in the size / covariates table)
const refCfgs = () => S.meta.configs.filter((c) => !S.meta.reference_model || c.model === S.meta.reference_model);
// other foundation models (t0-gamma…), shown next to the Timestral reference version
const isTimestral = (c) => c.model.startsWith("Timestral");
const otherCfgs = () => S.meta.configs.filter((c) => !isTimestral(c));
const cfgAnchor = (c) => (c.model.startsWith("t0") ? "t0" : "timestral");
function refConfig() {
  const r = refCfgs(), d = r.find((c) => c.name === S.meta.default_config);
  if (d) return d;
  return isVN1() ? r.reduce((a, b) => (valScore(b) < valScore(a) ? b : a), r[0]) : r.reduce((a, b) => (bestVariant(b).s < bestVariant(a).s ? b : a), r[0]);
}
function varLabel(c, v) {
  return v === "calib" ? t("var_calib", { u: Math.round(c.calib.u * 100) }) : t("var_" + v);
}

// ------------------------------------------------------------------ dates
function dayDate(offset) {
  const d = new Date(S.meta.start_date + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + offset * (S.meta.freq === "W" ? 7 : 1));
  return d.toISOString().slice(0, 10);
}

// ------------------------------------------------------------------ series selection (M5)
function fillSelect(sel, values, labels, value) {
  sel.innerHTML = values.map((v, k) => `<option value="${esc(v)}">${esc(labels ? labels[k] : v)}</option>`).join("");
  if (value != null && values.map(String).includes(String(value))) sel.value = String(value);
}
function updateSelectors(keep = {}) {
  if (isVN1()) return updateVN1Selectors(keep);
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
  if (isVN1()) { const v = $("sel-product").value; return v === "" ? undefined : +v; }
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
  if (isVN1()) {
    const s = S.series;
    return updateVN1Selectors({ client: s.client[i], warehouse: s.warehouse[i], product: i });
  }
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
  if (isVN1()) {
    const s = S.series;
    return t("vn1_name", { c: s.client[i], w: s.warehouse[i], p: s.product[i] });
  }
  const lv = S.series.level[i], parts = [S.series.a1[i], S.series.a2[i]].filter((p) => p !== "X" && p !== "Total");
  return parts.length ? `${t("lv_names")[lv - 1]} · ${parts.join(" · ")}` : t("total_series");
}

// ------------------------------------------------------------------ series selection (VN1: client → warehouse → product)
function indexVN1(series) {
  // client → warehouse → [series index], every list sorted by sales over the last 52 weeks
  const tree = new Map(), vol = series.vol;
  series.client.forEach((c, i) => {
    if (!tree.has(c)) tree.set(c, { vol: 0, wh: new Map() });
    const C = tree.get(c), w = series.warehouse[i];
    C.vol += vol[i];
    if (!C.wh.has(w)) C.wh.set(w, { vol: 0, items: [] });
    const W = C.wh.get(w);
    W.vol += vol[i]; W.items.push(i);
  });
  for (const C of tree.values()) for (const W of C.wh.values()) W.items.sort((a, b) => vol[b] - vol[a]);
  return tree;
}
function updateVN1Selectors(keep = {}) {
  const tree = S.vn1, s = S.series;
  const clients = [...tree.keys()].sort((a, b) => tree.get(b).vol - tree.get(a).vol);
  const unit = t("per_year_units");
  fillSelect($("sel-client"), clients, clients.map((c) => t("client_opt", { c, n: tree.get(c).wh.size })), keep.client ?? $("sel-client").value);
  const C = tree.get(+$("sel-client").value);
  const whs = [...C.wh.keys()].sort((a, b) => C.wh.get(b).vol - C.wh.get(a).vol);
  fillSelect($("sel-wh"), whs, whs.map((w) => t("wh_opt", { w, n: C.wh.get(w).items.length })), keep.warehouse ?? $("sel-wh").value);
  const items = C.wh.get(+$("sel-wh").value).items;
  fillSelect($("sel-product"), items, items.map((i) => `${t("product")} ${s.product[i]} · ${fint(s.vol[i])} ${unit}`), keep.product ?? $("sel-product").value);
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
const plotConfigBase = { displayModeBar: false, responsive: true };
let plotLocaleReady = false;
function plotConfig() {
  // French day and month names for the date axes (Plotly ships English only)
  if (!plotLocaleReady && typeof Plotly !== "undefined") {
    Plotly.register({ moduleType: "locale", name: "fr", dictionary: {}, format: {
      days: ["dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi"],
      shortDays: ["dim.", "lun.", "mar.", "mer.", "jeu.", "ven.", "sam."],
      months: ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
      shortMonths: ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."],
      date: "%d/%m/%Y", decimal: ",", thousands: " ", grouping: [3], currency: ["", " €"] } });
    plotLocaleReady = true;
  }
  return { ...plotConfigBase, locale: S.lang === "fr" ? "fr" : "en" };
}
// date axis: weekday + date on daily charts (store teams think in weekdays), week start on weekly charts
function dateAxis() {
  const fr = S.lang === "fr";
  if (S.meta.freq === "W") return { ...plotLayout().xaxis, tickformat: fr ? "%d/%m/%y" : "%b %d '%y", hoverformat: fr ? "semaine du %d/%m/%Y" : "week of %b %d, %Y" };
  return { ...plotLayout().xaxis, tickformat: fr ? "%a %d/%m" : "%a %b %d", hoverformat: fr ? "%A %d/%m/%Y" : "%A, %b %d %Y" };
}

async function renderExplore() {
  const i = currentIndex();
  if (i == null || !S.meta.configs.length) return;
  const bench = S.bench, cfg = $("sel-config").value, c = S.meta.configs.find((x) => x.name === cfg);
  if (!c) return;
  const [sd, fc] = await Promise.all([seriesData(i), forecastData(cfg, i)]);
  if (bench !== S.bench || i !== currentIndex() || cfg !== $("sel-config").value) return; // a newer selection is pending
  S.current = { i, sd, fc, c };
  const y = sd.actual, h = y.length, med = quantileAt(fc, 0.5);
  const rows = { median: metrics(i, med, y), mean: metrics(i, fc.mean, y) };
  if (fc.calib) rows.calib = metrics(i, fc.calib, y);
  for (const r of S.meta.refs) rows[r] = metrics(i, sd.ref[r], y);
  const lo80 = quantileAt(fc, 0.1), hi80 = quantileAt(fc, 0.9);
  let in80 = 0;
  for (let d = 0; d < h; d++) if (y[d] >= lo80[d] - 1e-9 && y[d] <= hi80[d] + 1e-9) in80++;
  const e80 = Math.round(0.8 * h);
  const bandCard = kpi(t("k_band"), t("k_band_v", { n: in80, h }), t("k_band_sub", { e: e80 }), Math.abs(in80 - e80) <= Math.max(2, h / 7) ? "good" : "bad");
  const speedCard = kpi(t("k_speed"), `${fint(c.ms_per_series)} ms`, t("k_speed_sub", { model: c.model.replace("Timestral-", "") }));
  const zs = sd.hist.subarray(Math.max(0, sd.hist.length - (isVN1() ? 52 : 364))).filter((v) => v < 0.5).length / Math.min(sd.hist.length, isVN1() ? 52 : 364);
  const cards = [];

  if (isVN1()) {
    const v = bestVN1(c).v, T = rows[v];
    cards.push(kpi(t("k_score"), fnum(T.score, 3), t("k_score_sub", { mae: fpct(T.wape), bias: fpct(T.bias, true), v: varLabel(c, v) })));
    for (const [ref, key] of [["naive", "k_vs_naive"], ["vn1_1st", "k_vs_winner"]]) {
      if (!rows[ref]) continue;
      const g = 1 - T.score / rows[ref].score;
      cards.push(kpi(t(key), fpct(g, true), t("k_score_of", { v: fnum(rows[ref].score, 3) }), isFinite(g) ? (g > 0 ? "good" : "bad") : ""));
    }
    cards.push(bandCard, speedCard);
    $("kpis").innerHTML = cards.join("");
    const nv = rows.naive, verdict = nv && isFinite(T.score) && isFinite(nv.score) ? t(T.score < nv.score ? "beats" : "trails") : null;
    $("reading").innerHTML = isFinite(T.score) ? t("reading", {
      zs: fpct(zs), which: t("w_" + v), verdict: verdict ?? "–", a: fnum(T.score, 3), b: nv ? fnum(nv.score, 3) : "–",
      n: in80, h, bias: fpct(T.bias, true),
    }) : t("reading_nosales");
  } else {
    const bestT = Math.min(rows.median.rmsse, rows.mean.rmsse);
    const stat = S.meta.refs.filter((r) => r !== "winner");
    const bestStat = stat.reduce((a, b) => (rows[b].rmsse < rows[a].rmsse ? b : a), stat[0]);
    cards.push(kpi(t("k_rmsse") + help("rmsse"), fnum(bestT), t("k_lower")));
    if (bestStat) {
      const g = 1 - bestT / rows[bestStat].rmsse;
      cards.push(kpi(t("k_vs", { name: refLabel(bestStat) }), fpct(g, true), t("k_best_stat", { v: fnum(rows[bestStat].rmsse) }), g > 0 ? "good" : "bad"));
    }
    if (rows.winner) {
      const g = 1 - bestT / rows.winner.rmsse;
      cards.push(kpi(t("k_vs_winner"), fpct(g, true), `RMSSE ${fnum(rows.winner.rmsse)}`, g > 0 ? "good" : "bad"));
    }
    cards.push(bandCard);
    cards.push(kpi(t("k_bias") + help("bias"), fpct(rows.median.bias, true), t("k_bias_sub")));
    cards.push(speedCard);
    $("kpis").innerHTML = cards.join("");
    const meanWins = rows.mean.rmsse < rows.median.rmsse;
    $("reading").innerHTML = t("reading", {
      zs: fpct(zs), verdict: t(bestStat && bestT < rows[bestStat].rmsse ? "beats" : "trails"), n: in80, h,
      which: t(meanWins ? "w_mean" : "w_median"), extra: zs > 0.4 && meanWins ? t("reading_extra") : ".",
    });
  }

  // error table
  const names = { median: t("median"), mean: t("mean") };
  if (c.calib) names.calib = t("calib_name", { u: Math.round(c.calib.u * 100) });
  for (const r of S.meta.refs) names[r] = refLabel(r);
  const order = Object.keys(rows);
  let html;
  if (isVN1()) {
    html = `<thead><tr><th>${t("col_method")}</th><th class="num">${t("col_score")}</th><th class="num">MAE%</th><th class="num">${t("col_bias")}</th><th class="num">${t("col_total")}</th></tr></thead><tbody>`;
    for (const k of order) {
      const v = rows[k];
      html += `<tr class="${k === "median" || k === "mean" ? "hl" : ""}"><td>${glossLink(names[k], k === "median" || k === "mean" ? "median-mean" : methodAnchor(k))}</td><td class="num">${fnum(v.score)}</td><td class="num">${fpct(v.wape)}</td><td class="num">${fpct(v.bias, true)}</td><td class="num">${fint(v.total)}</td></tr>`;
    }
  } else {
    html = `<thead><tr><th>${t("col_method")}</th><th class="num">RMSSE</th><th class="num">WAPE</th><th class="num">${t("col_bias")}</th><th class="num">${t("col_total")}</th></tr></thead><tbody>`;
    for (const k of order) {
      const v = rows[k];
      html += `<tr class="${k === "median" || k === "mean" ? "hl" : ""}"><td>${glossLink(names[k], k === "median" || k === "mean" ? "median-mean" : methodAnchor(k))}</td><td class="num">${fnum(v.rmsse)}</td><td class="num">${fpct(v.wape)}</td><td class="num">${fpct(v.bias, true)}</td><td class="num">${fint(v.total)}</td></tr>`;
    }
  }
  html += `<tr><td>${t("actual")}</td><td class="num">–</td><td class="num">–</td><td class="num">–</td><td class="num">${fint(sum(y))}</td></tr></tbody>`;
  $("err-table").innerHTML = html;
  drawForecast();
  drawService();
}

function drawForecast() {
  if (!S.current || typeof Plotly === "undefined") return;
  const { i, sd, fc } = S.current, H = S.meta.hist_days, h = S.meta.horizon;
  const days = Math.min(+$("sel-days").value, H);
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
  if (isVN1()) {
    traces.push({ x: fx, y: Array.from(fc.mean), name: t("mean"), mode: "lines", line: { color: acc, width: 1.2, dash: "dot" } });
    const c = S.current.c;
    if (fc.calib) traces.push({ x: fx, y: Array.from(fc.calib), name: t("calib_name", { u: Math.round(c.calib.u * 100) }), mode: "lines", line: { color: css("--accent-ink"), width: 2, dash: "dash" } });
  }
  for (const box of document.querySelectorAll("#compare input:checked")) {
    const r = box.value;
    if (!sd.ref[r]) continue;
    traces.push({ x: fx, y: Array.from(sd.ref[r]), name: refLabel(r), mode: "lines", line: { color: css(REF_COLOR[r] || "--c-history"), width: 1.6, dash: "dot" } });
  }
  traces.push({ x: fx, y: Array.from(sd.actual), name: t("actual_test"), mode: "lines+markers", line: { color: css("--c-actual"), width: 2 }, marker: { size: 4 } });
  const muted = css("--muted");
  const shapes = [{ type: "line", x0: fx[0], x1: fx[0], yref: "paper", y0: 0, y1: 1, line: { color: muted, width: 1, dash: "dash" } }];
  const annotations = [{ x: fx[0], y: 1.02, yref: "paper", text: t("forecast_start"), showarrow: false, xanchor: "left", font: { size: 11, color: muted } }];
  for (const [off, name] of narrow() ? [] : S.meta.events || []) {
    if (off >= H) annotations.push({ x: dayDate(off), y: 0.97, yref: "paper", text: esc(name), showarrow: false, textangle: -90, yanchor: "top", xshift: -6, font: { size: 9.5, color: muted } });
  }
  Plotly.react("chart", traces, plotLayout({
    title: { text: esc(seriesName(i)), font: { size: 15, family: css("--font-body") }, x: 0, xanchor: "left" },
    shapes, annotations, yaxis: { ...plotLayout().yaxis, title: { text: t("units_day"), font: { size: 12, color: muted } } },
    xaxis: dateAxis(),
  }), plotConfig());
}

function drawService() {
  if (!S.current || typeof Plotly === "undefined") return;
  const { fc } = S.current, H = S.meta.hist_days, h = S.meta.horizon, y = S.current.sd.actual;
  const target = +$("svc").value, u = target / 100;
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
    kpi(t("s_covered") + help("service"), fpct(cov / h), t("s_covered_sub", { t: target, n: cov, h }), Math.abs(cov / h - u) <= 0.1 ? "good" : "bad"),
    kpi(t("s_fill"), fpct(fill), t("s_fill_sub", { v: fint(short) })),
    kpi(t("s_sold"), fint(tot), t("s_sold_sub", { h, stock: fint(sum(lq)) })),
    kpi(t("s_surplus"), fint(surplus), t("s_surplus_sub", { p: tot > 0 ? fpct(surplus / tot) : "–" })),
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
    xaxis: dateAxis() }), plotConfig());
}

function hexToRgb(hex) {
  const m = hex.replace("#", "").match(/.{2}/g);
  return m ? m.map((x) => parseInt(x, 16)).join(",") : "232,89,12";
}

// ------------------------------------------------------------------ benchmark (M5)
function bestVariant(c) {
  const m = c.wrmsse["median|bottom-up"], e = c.wrmsse["mean|bottom-up"];
  return e < m ? { v: "mean", s: e } : { v: "median", s: m };
}
const typeTag = (type) => ({
  stat: `<span class="tag">${t("type_stat")}</span>`, official: `<span class="tag">${t("type_official")}</span>`,
  comp: `<span class="tag">${t("type_comp")}</span>`, tsfm: `<span class="tag tsfm">${t("type_tsfm")}</span>`,
}[type]);
function renderBenchmark() {
  if (isVN1()) return renderBenchmarkVN1();
  const m = S.meta, cfgs = m.configs;
  const base = Object.fromEntries(m.baselines.filter((b) => b.metric === "wrmsse").map((b) => [b.method, b.score]));
  const winner = Object.entries(base).find(([k]) => k.startsWith("M5 #1 "));
  if (cfgs.length) {
    const best = refConfig();
    const bs = bestVariant(best).s;
    const rel = (ref) => (ref ? 1 - bs / ref : NaN);
    const vsCard = (key, ref) => kpi(t(key), fpct(rel(ref), true), `WRMSSE ${fnum(ref, 4)}`, rel(ref) > 0 ? "good" : "bad");
    $("bench-kpis").innerHTML = [
      kpi(t("b_ref") + help("wrmsse"), fnum(bs, 4), `${esc(cfgLabel(best))} · ${t("var_" + bestVariant(best).v)}`, "lead"),
      vsCard("b_vs_ets", base.ets),
      base.arima ? vsCard("b_vs_arima", base.arima) : vsCard("b_vs_snaive", base.snaive),
      vsCard("b_vs_winner", winner && winner[1]),
      ...otherCfgs().map((o) => kpi(t("b_vs_model", { m: esc(cfgLabel(o)) }), fpct(1 - bs / bestVariant(o).s, true), `WRMSSE ${fnum(bestVariant(o).s, 4)}`, bs < bestVariant(o).s ? "good" : "bad")),
    ].join("") + uncertaintyKpi();
  } else {
    $("bench-kpis").innerHTML = "";
  }

  // leaderboard
  const rows = [];
  for (const [k, s] of Object.entries(base)) {
    const comp = k.startsWith("M5 #");
    rows.push({ name: comp ? k : methodLabel(k), anchor: methodAnchor(k), type: comp ? "comp" : "stat", s, point: "" });
  }
  for (const c of [refConfig(), ...otherCfgs()]) { const b = bestVariant(c); rows.push({ name: cfgLabel(c), anchor: cfgAnchor(c), type: "tsfm", s: b.s, point: t("var_" + b.v) }); }
  rows.sort((a, b) => a.s - b.s);
  $("lb-table").innerHTML = `<thead><tr><th class="num">${t("lb_rank")}</th><th>${t("lb_method")}</th><th>${t("lb_type")}</th><th>${t("lb_point")}</th><th class="num">WRMSSE</th></tr></thead><tbody>` +
    rows.map((r, k) => `<tr class="${r.type === "tsfm" ? "hl" : ""}"><td class="num">${k + 1}</td><td>${glossLink(r.name, r.anchor)}</td><td>${typeTag(r.type)}</td><td>${esc(r.point)}</td><td class="num">${fnum(r.s, 4)}</td></tr>`).join("") + "</tbody>";

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
      const best = refConfig();
      series.push({ name: cfgLabel(best), y: lvl(best.name, bestVariant(best).v, "bottom-up"), color: css("--accent"), width: 3 });
      otherCfgs().forEach((o, k) => series.push({ name: cfgLabel(o), y: lvl(o.name, bestVariant(o).v, "bottom-up"), color: css("--c-t0"), width: 2.2, dash: k ? "dot" : "solid" }));
    }
    for (const k of ["ets", "arima", "snaive", "tsb"]) if (k in base) series.push({ name: methodLabel(k), y: lvl(k, "", "bottom-up"), color: css(REF_COLOR[k]) });
    if (winner) series.push({ name: refLabel("winner"), y: lvl(winner[0], "", "bottom-up"), color: css("--c-winner") });
    Plotly.react("lv-chart", series.map((s) => ({ x: xs, y: s.y, name: s.name, mode: "lines+markers", line: { color: s.color, width: s.width || 1.8, dash: s.dash || "solid" }, marker: { size: 6 } })),
      plotLayout({
        margin: small ? { l: 44, r: 10, t: 10, b: 40 } : { l: 52, r: 14, t: 10, b: 110 }, hovermode: "x unified",
        xaxis: { ...plotLayout().xaxis, tickangle: small ? 0 : -35, type: "category",
          title: small ? { text: t("level"), font: { size: 12, color: css("--muted") } } : undefined },
        yaxis: { ...plotLayout().yaxis, title: { text: "RMSSE", font: { size: 12, color: css("--muted") } } },
        legend: small ? { orientation: "h", yanchor: "top", y: -0.22, x: 0, font: { size: 11, color: css("--muted") } }
                      : { orientation: "v", x: 1.01, y: 1, font: { size: 11.5, color: css("--muted") } },
      }), plotConfig());
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
  $("ab-note").innerHTML = t("ab_size_note") + "<br>" + t("ab_note", { emp: emp ? fnum(emp.score, 4) : "–" }) + "<br>" + t("ab_note_median") + "<br>" +
    ["none", "calendar", "full", "rich"].map((k) => `<strong>${esc(t("cov_" + k))}</strong>${S.lang === "fr" ? " :" : ":"} ${esc(t("cov_desc_" + k))}`).join("<br>");
}

function bestWspl() {
  return S.meta.configs.length ? refConfig() : null;
}
function uncertaintyKpi() {
  const best = bestWspl(), pub = (S.meta.uncertainty_published || []).find((p) => p.key === "unc_winner");
  if (!best || !pub) return "";
  const g = 1 - best.wspl / pub.wspl;
  return kpi(t("b_unc") + help("wspl"), fpct(g, true), t("b_unc_sub", { cfg: esc(cfgLabel(best)), v: fnum(best.wspl, 3), w: fnum(pub.wspl, 3) }), g >= -0.005 ? "good" : "bad");
}
function renderUncertainty() {
  const m = S.meta, rows = [];
  for (const c of S.meta.configs.length ? [refConfig(), ...otherCfgs()] : []) rows.push({ name: cfgLabel(c), anchor: cfgAnchor(c), type: "tsfm", s: c.wspl, levels: null, cfg: c.name });
  for (const p of m.uncertainty_published || []) rows.push({ name: t(p.key), anchor: p.key === "unc_winner" ? "lightgbm" : "arima", type: p.key === "unc_winner" ? "comp" : "stat", s: p.wspl, levels: p.levels });
  const emp = m.baselines.find((b) => b.metric === "wspl");
  if (emp) rows.push({ name: t("unc_emp"), anchor: "empirical", type: "stat", s: emp.score });
  rows.sort((a, b) => a.s - b.s);
  $("unc-table").innerHTML = `<thead><tr><th class="num">${t("lb_rank")}</th><th>${t("lb_method")}</th><th>${t("lb_type")}</th><th class="num">WSPL</th></tr></thead><tbody>` +
    rows.map((r, k) => `<tr class="${r.type === "tsfm" ? "hl" : ""}"><td class="num">${k + 1}</td><td>${glossLink(r.name, r.anchor)}</td><td>${typeTag(r.type)}</td><td class="num">${fnum(r.s, 3)}</td></tr>`).join("") +
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
  otherCfgs().forEach((o, k) => series.push({ name: cfgLabel(o), y: lvl(o.name), color: css("--c-t0"), width: 2.2, dash: k ? "dot" : "solid" }));
  for (const p of m.uncertainty_published || []) series.push({ name: t(p.key), y: p.levels, color: css(p.key === "unc_winner" ? "--c-winner" : "--c-arima") });
  if (emp) series.push({ name: t("unc_emp"), y: lvl(emp.method), color: css("--c-history") });
  Plotly.react("unc-chart", series.map((s) => ({ x: xs, y: s.y, name: s.name, mode: "lines+markers", line: { color: s.color, width: s.width || 1.8, dash: s.dash || "solid" }, marker: { size: 6 } })),
    plotLayout({
      margin: small ? { l: 44, r: 10, t: 10, b: 40 } : { l: 52, r: 14, t: 10, b: 110 }, hovermode: "x unified",
      xaxis: { ...plotLayout().xaxis, tickangle: small ? 0 : -35, type: "category", title: small ? { text: t("level"), font: { size: 12, color: css("--muted") } } : undefined },
      yaxis: { ...plotLayout().yaxis, title: { text: "SPL", font: { size: 12, color: css("--muted") } } },
      legend: small ? { orientation: "h", yanchor: "top", y: -0.22, x: 0, font: { size: 11, color: css("--muted") } }
                    : { orientation: "v", x: 1.01, y: 1, font: { size: 11.5, color: css("--muted") } },
    }), plotConfig());
}

// ------------------------------------------------------------------ benchmark (VN1)
function renderBenchmarkVN1() {
  const m = S.meta, cfgs = m.configs, base = Object.fromEntries(m.baselines.map((b) => [b.method, b]));
  const comp = m.baselines.filter((b) => b.family === "competition").sort((a, b) => a.score - b.score);
  const stats = m.baselines.filter((b) => b.family === "benchmark");
  const bestStat = stats.reduce((a, b) => (b.score < a.score ? b : a), stats[0]);
  if (cfgs.length) {
    const best = refConfig(), B = bestVN1(best);
    const rel = (r) => (r ? 1 - B.score / r.score : NaN);
    const vs = (key, r, sub) => kpi(t(key), fpct(rel(r), true), sub, rel(r) > 0 ? "good" : "bad");
    const rank = comp.filter((c) => c.score < B.score).length + 1;
    $("bench-kpis").innerHTML = [
      kpi(t("b_ref") + help("calibration"), fnum(B.score, 4), `${esc(cfgLabel(best))} · ${varLabel(best, B.v)} · MAE ${fpct(B.mae, false, 1)} · ${t("col_bias")} ${fpct(B.bias, true, 1)}`, "lead"),
      vs("k_vs_naive", base.naive, t("k_score_of", { v: fnum(base.naive && base.naive.score, 4) })),
      bestStat ? vs("b_vs_beststat", bestStat, `${esc(methodLabel(bestStat.method))} · ${fnum(bestStat.score, 4)}`) : "",
      vs("b_vs_winner", base.vn1_1st, t("k_score_of", { v: fnum(base.vn1_1st && base.vn1_1st.score, 4) })),
      kpi(t("b_rank"), rank <= comp.length ? `#${rank}` : t("b_rank_out", { n: comp.length }), t("b_rank_sub")),
      ...otherCfgs().map((o) => { const O = bestVN1(o); return vs("b_vs_model", O, `${varLabel(o, O.v)} · ${fnum(O.score, 4)}`).replace("{m}", esc(cfgLabel(o))); }),
    ].join("");
  } else {
    $("bench-kpis").innerHTML = "";
  }

  const rows = m.baselines.map((b) => ({
    name: methodLabel(b.method), anchor: methodAnchor(b.method), point: "",
    type: b.family === "competition" ? "comp" : b.family === "official" ? "official" : "stat", ...b,
  }));
  for (const c of [refConfig(), ...otherCfgs()]) { const b = bestVN1(c); rows.push({ name: cfgLabel(c), anchor: cfgAnchor(c), type: "tsfm", point: varLabel(c, b.v), ...b }); }
  rows.sort((a, b) => a.score - b.score);
  $("lb-table").innerHTML = `<thead><tr><th class="num">${t("lb_rank")}</th><th>${t("lb_method")}</th><th>${t("lb_type")}</th><th>${t("lb_point")}</th><th class="num">${t("col_score")}</th><th class="num">MAE%</th><th class="num">${t("col_bias")}</th></tr></thead><tbody>` +
    rows.map((r, k) => `<tr class="${r.type === "tsfm" ? "hl" : ""}"><td class="num">${k + 1}</td><td>${glossLink(r.name, r.anchor)}</td><td>${typeTag(r.type)}</td><td>${esc(r.point)}</td><td class="num">${fnum(r.score, 4)}</td><td class="num">${fpct(r.mae, false, 1)}</td><td class="num">${fpct(r.bias, true, 1)}</td></tr>`).join("") + "</tbody>";

  if (!cfgs.length) { $("ab-table").innerHTML = `<tbody><tr><td>${t("ab_empty")}</td></tr></tbody>`; }
  else {
    const noneOf = (model) => cfgs.find((c) => c.covariates === "none" && c.model === model);
    const hasCal = cfgs.some((c) => c.calib);
    $("ab-table").innerHTML = `<thead><tr><th>${t("ab_cfg")}</th><th>${t("ab_cov")}</th><th class="num">${t("ab_med")}</th><th class="num">${t("ab_mean")}</th>${hasCal ? `<th class="num">${t("ab_cal")}</th>` : ""}<th class="num">${t("ab_delta")}</th><th class="num">${t("col_bias")}</th><th class="num">${t("ab_cov80")}</th><th class="num">${t("ab_speed")}</th></tr></thead><tbody>` +
      cfgs.map((c) => {
        const ref = noneOf(c.model), b = bestVN1(c), d = ref && c !== ref ? 1 - b.score / bestVN1(ref).score : null;
        const delta = d == null ? "–" : `<span class="${d > 0 ? "delta-good" : "delta-bad"}">${fpct(d, true, 1)}</span>`;
        return `<tr><td>${esc(c.model.replace("Timestral-", "Timestral "))}</td><td title="${esc(t("cov_desc_" + c.covariates))}">${esc(t("cov_" + c.covariates))}</td><td class="num">${fnum(c.median.score, 4)}</td><td class="num">${fnum(c.mean.score, 4)}</td>${hasCal ? `<td class="num">${c.calib ? `${fnum(c.calib.test.score, 4)} <span class="muted">(P${Math.round(c.calib.u * 100)})</span>` : "–"}</td>` : ""}<td class="num">${delta}</td><td class="num">${fpct(b.bias, true, 1)}</td><td class="num">${fpct(c.coverage80)}</td><td class="num">${fint(c.ms_per_series)}</td></tr>`;
      }).join("") + "</tbody>";
  }
  renderCalibration();
  renderDiagnostic();
  $("ab-note").innerHTML = t("ab_size_note") + "<br>" + t("ab_note") + "<br>" + S.meta.configs.map((c) => c.covariates).filter((k, i, a) => a.indexOf(k) === i).map((k) => `<strong>${esc(t("cov_" + k))}</strong>${S.lang === "fr" ? " :" : ":"} ${esc(t("cov_desc_" + k))}`).join("<br>");
}

function windowLabel(w) {
  if (w.label === "val") return t("cal_win_p1");
  const d0 = new Date(w.label.split("|")[0] + "T00:00:00Z"); // first forecast week: the window is one quarter
  return t("quarter", { q: Math.floor(d0.getUTCMonth() / 3) + 1, y: d0.getUTCFullYear() });
}
function renderCalibration() {
  const cfgs = S.meta.configs.filter((c) => c.calib);
  $("cal-section").hidden = !cfgs.length;
  if (!cfgs.length) return;
  const sel = cfgs.includes(refConfig()) ? refConfig() : cfgs[0], C = sel.calib, P = (u) => `P${Math.round(u * 100)}`;
  $("cal-table").innerHTML = `<thead><tr><th>${t("ab_cfg")}</th><th class="num">${t("cal_test_med")}</th><th class="num">${t("cal_u")}</th><th class="num">${t("cal_test_cal")}</th><th class="num">${t("cal_gain")}</th><th class="num">${t("col_bias")}</th><th class="num">${t("cal_p1")}</th></tr></thead><tbody>` +
    cfgs.map((c) => {
      const k = c.calib, g = 1 - k.test.score / c.median.score;
      const p1 = k.p1 ? `${P(k.p1.u)} → ${fnum(k.p1.test.score, 4)}` : "–";
      return `<tr class="${c === sel ? "hl" : ""}"><td>${esc(cfgLabel(c))}</td><td class="num">${fnum(c.median.score, 4)}</td><td class="num">${P(k.u)}</td><td class="num">${fnum(k.test.score, 4)}</td><td class="num"><span class="${g > 0 ? "delta-good" : "delta-bad"}">${fpct(g, true, 1)}</span></td><td class="num">${fpct(c.median.bias, true, 1)} → ${fpct(k.test.bias, true, 1)}</td><td class="num muted">${p1}</td></tr>`;
    }).join("") + "</tbody>";
  $("cal-note").innerHTML = t("cal_note", {
    cfg: esc(cfgLabel(sel)), u: Math.round(C.u * 100), s: fnum(C.test.score, 4),
    wins: (C.windows || []).map((w) => `${esc(windowLabel(w))} ${P(w.u)}`).join(", "),
    ub: Math.round(C.test_best_u * 100), sb: fnum(C.test_best, 4),
  });
  if (typeof Plotly === "undefined") return;
  const x = C.curve.u.map((u) => Math.round(u * 100)), acc = css("--accent"), muted = css("--muted");
  const winColors = ["--c-snaive", "--c-ets", "--c-tsb", "--c-winner"];
  const traces = (C.windows || []).map((w, k) => ({ x, y: w.curve, name: windowLabel(w), mode: "lines", line: { color: css(winColors[k % 4]), width: 1.2, dash: "dot" } }));
  traces.push({ x, y: C.curve.val, name: t("cal_curve_val"), mode: "lines", line: { color: css("--c-history"), width: 2.6 } });
  traces.push({ x, y: C.curve.test, name: t("cal_curve_test"), mode: "lines", line: { color: acc, width: 2.6 } });
  Plotly.react("cal-chart", traces, plotLayout({
    margin: { l: 56, r: 14, t: 70, b: 50 }, hovermode: "x unified",
    shapes: [50, Math.round(C.u * 100)].map((u, k) => ({ type: "line", x0: u, x1: u, yref: "paper", y0: 0, y1: 1, line: { color: k ? acc : muted, width: 1, dash: "dash" } })),
    annotations: [{ x: 50, y: 0.98, yref: "paper", yanchor: "top", text: t("median_short"), showarrow: false, xanchor: "right", xshift: -4, font: { size: 11, color: muted } },
      { x: Math.round(C.u * 100), y: 0.98, yref: "paper", yanchor: "top", text: t("cal_chosen", { u: Math.round(C.u * 100) }), showarrow: false, xanchor: "left", xshift: 4, font: { size: 11, color: acc } }],
    xaxis: { ...plotLayout().xaxis, title: { text: t("cal_x"), font: { size: 12, color: muted } } },
    yaxis: { ...plotLayout().yaxis, rangemode: "normal", title: { text: t("col_score"), font: { size: 12, color: muted } } },
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, x: 0, font: { size: 11.5, color: muted } },
  }), plotConfig());
}

function renderDiagnostic() {
  const m = S.meta, D = m.diag, cfgs = m.configs.filter((c) => c.diag);
  $("diag-section").hidden = !D || !cfgs.length;
  if (!D || !cfgs.length) return;
  const sel = cfgs.includes(refConfig()) ? refConfig() : cfgs[0], B = bestVN1(sel);
  const selName = `${cfgLabel(sel)} · ${varLabel(sel, B.v)}`;
  const segRow = (name, seg, k) => seg ? seg[k] : null;
  const pts = [
    { name: t("diag_ts_median", { cfg: cfgLabel(sel) }), seg: sel.diag.segments.median },
    sel.diag.segments.calib ? { name: selName, seg: sel.diag.segments.calib, hl: true } : null,
    D.ref_segments.vn1_1st ? { name: refLabel("vn1_1st"), seg: D.ref_segments.vn1_1st } : null,
    D.ref_segments.naive ? { name: refLabel("naive"), seg: D.ref_segments.naive } : null,
  ].filter(Boolean);
  const head = D.segment_keys.map((k, j) => `<th class="num">${t(k)}<br><span class="muted sub">${t(k + "_sub")} · ${fpct(pts[0].seg[j].share)} ${t("diag_of_sales")}</span></th>`).join("");
  $("diag-table").innerHTML = `<thead><tr><th>${t("diag_bias_by")}</th>${head}<th class="num">${t("diag_total")}</th></tr></thead><tbody>` +
    pts.map((p) => {
      const tot = p.seg.reduce((a, x) => a + x.bias, 0);
      return `<tr class="${p.hl ? "hl" : ""}"><td>${esc(p.name)}</td>${p.seg.map((x) => `<td class="num">${fpct(x.bias, true, 1)}</td>`).join("")}<td class="num"><strong>${fpct(tot, true, 1)}</strong></td></tr>`;
    }).join("") + "</tbody>";
  if (typeof Plotly === "undefined") return;
  const H = m.hist_days, h = m.horizon, nh = D.hist_totals.length;
  const hx = [], fx = [];
  for (let d = H - nh; d < H; d++) hx.push(dayDate(d));
  for (let d = H; d < H + h; d++) fx.push(dayDate(d));
  const acc = css("--accent"), muted = css("--muted");
  const traces = [
    { x: hx, y: D.hist_totals, name: t("history"), mode: "lines", line: { color: css("--c-history"), width: 1.6 } },
    { x: fx, y: D.actual_totals, name: t("actual_test"), mode: "lines+markers", line: { color: css("--c-actual"), width: 2 }, marker: { size: 4 } },
    { x: fx, y: sel.diag.totals.median, name: t("diag_ts_median", { cfg: cfgLabel(sel) }), mode: "lines", line: { color: acc, width: 1.4, dash: "dot" } },
  ];
  if (sel.diag.totals.calib) traces.push({ x: fx, y: sel.diag.totals.calib, name: selName, mode: "lines", line: { color: acc, width: 2.8 } });
  if (D.ref_totals.vn1_1st) traces.push({ x: fx, y: D.ref_totals.vn1_1st, name: refLabel("vn1_1st"), mode: "lines", line: { color: css("--c-winner"), width: 1.6, dash: "dash" } });
  if (D.ref_totals.snaive) traces.push({ x: fx, y: D.ref_totals.snaive, name: refLabel("snaive"), mode: "lines", line: { color: css("--c-snaive"), width: 1.2, dash: "dot" }, visible: "legendonly" });
  Plotly.react("diag-chart", traces, plotLayout({
    margin: { l: 60, r: 14, t: 70, b: 40 },
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, x: 0, font: { size: 11.5, color: muted } },
    shapes: [{ type: "line", x0: fx[0], x1: fx[0], yref: "paper", y0: 0, y1: 1, line: { color: muted, width: 1, dash: "dash" } }],
    yaxis: { ...plotLayout().yaxis, title: { text: t("diag_y"), font: { size: 12, color: muted } } },
    xaxis: dateAxis(),
  }), plotConfig());
}

// ------------------------------------------------------------------ static texts, language, tabs, benchmark switch
function applyTexts() {
  document.documentElement.lang = S.lang;
  document.body.dataset.bench = S.bench;
  for (const el of document.querySelectorAll("[data-t]")) el.textContent = t(el.dataset.t);
  for (const el of document.querySelectorAll("[data-th]")) el.innerHTML = t(el.dataset.th);
  for (const el of document.querySelectorAll("[data-help]")) {
    el.querySelector(".gl.help")?.remove();
    const g = el.dataset.help, k = "gl_" + g;
    el.insertAdjacentHTML("beforeend", help(I18N.en[k] ? t(k) : g));
  }
  document.title = t("title");
  for (const b of document.querySelectorAll(".lang button")) b.setAttribute("aria-pressed", String(b.dataset.lang === S.lang));
  for (const b of document.querySelectorAll(".bench-switch button")) {
    b.setAttribute("aria-pressed", String(b.dataset.bench === S.bench));
    b.hidden = !S.benches.includes(b.dataset.bench);
  }
  document.querySelector(".bench-switch").hidden = S.benches.length < 2;
  $("glossary-body").innerHTML = S.glossary ? S.glossary[S.lang] : "";
  if (!S.meta) return;
  $("foot-text").textContent = t("footer", { date: new Date(S.meta.generated + "T00:00:00Z").toLocaleDateString(locale(), { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" }) });
  if (!isVN1()) {
    const lvSel = $("sel-level"), lv = lvSel.value || "12";
    fillSelect(lvSel, Array.from({ length: 12 }, (_, k) => String(k + 1)), t("lv_names").map((n, k) => `${k + 1}. ${n}`), lv);
  }
  const cfgSel = $("sel-config"), cv = S.meta.configs.some((c) => c.name === cfgSel.value) ? cfgSel.value : S.meta.default_config;
  const rc = [...refCfgs(), ...otherCfgs()];
  fillSelect(cfgSel, rc.map((c) => c.name), rc.map(cfgLabel), rc.some((c) => c.name === cv) ? cv : refConfig().name);
  updateCfgHint();
  $("examples").innerHTML = S.meta.examples.map((e) => `<button type="button" data-i="${e.i}">${esc(t(e.key))}</button>`).join("");
  const checked = S.checked[S.bench] || new Set(DEFAULT_REFS[S.bench]);
  $("compare").innerHTML = S.meta.refs.map((r) => `<label><input type="checkbox" value="${r}" ${checked.has(r) ? "checked" : ""}><span class="swatch" style="border-color:var(${REF_COLOR[r] || "--c-history"})"></span>${esc(refLabel(r))}</label>`).join("");
  updateSelectors();
  $("learn-body").innerHTML = S.learn ? (S.learn[S.lang][S.bench] ?? S.learn[S.lang]) : "";
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
function writeHash() {
  history.replaceState(null, "", "#" + (S.bench === "m5" || S.tab === "glossary" ? S.tab : `${S.bench}/${S.tab}`));
}
function setTab(tab, push = true) {
  S.tab = tab;
  for (const id of TABS) {
    $(id).hidden = id !== tab;
    $("tab-" + id).setAttribute("aria-selected", String(id === tab));
  }
  if (push) writeHash();
  if (typeof Plotly !== "undefined") requestAnimationFrame(() => {
    for (const id of ["chart", "svc-chart", "lv-chart", "unc-chart", "cal-chart", "diag-chart"]) { const el = $(id); if (el && el.data && !el.closest("[hidden]")) Plotly.Plots.resize(el); }
  });
}
function showGlossary(anchor) {
  setTab("glossary");
  const el = anchor && $("g-" + anchor);
  if (!el) { window.scrollTo({ top: 0 }); return; }
  el.scrollIntoView({ block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 900);
}

async function loadBench(bench) {
  if (!S.loaded[bench]) {
    const files = [getJSON(`data/${bench}/meta.json`), getJSON(`data/${bench}/series.json`)];
    if (bench === "m5") files.push(fetch("data/m5/scale_sq.bin").then((r) => { if (!r.ok) throw new Error("scale_sq"); return r.arrayBuffer(); }));
    const [meta, series, sq] = await Promise.all(files);
    const L = { meta, series, scaleSq: sq ? new Float32Array(sq) : null, keyIndex: new Map(), vn1: null };
    if (bench === "m5") series.level.forEach((lv, i) => L.keyIndex.set(`${lv}|${series.a1[i]}|${series.a2[i]}`, i));
    else L.vn1 = indexVN1(series);
    S.loaded[bench] = L;
  }
  Object.assign(S, S.loaded[bench], { bench, current: null });
}
async function setBench(bench, first = false) {
  if (!first && bench === S.bench) return;
  if (!first) S.checked[S.bench] = new Set([...document.querySelectorAll("#compare input:checked")].map((x) => x.value));
  $("status").hidden = false; $("status").classList.remove("error"); $("status").textContent = t("loading");
  try { await loadBench(bench); } catch (err) {
    $("status").textContent = t("load_error"); $("status").classList.add("error"); console.error(err); return;
  }
  $("status").hidden = true;
  for (const id of ["kpis", "reading", "err-table", "svc-kpis"]) $(id).innerHTML = "";
  for (const id of ["chart", "svc-chart"]) if (typeof Plotly !== "undefined" && $(id).data) Plotly.purge($(id));
  // history slider: days on M5, weeks on VN1
  const sl = $("sel-days"), H = S.meta.hist_days;
  if (isVN1()) { sl.min = 13; sl.max = H; sl.step = 1; sl.value = narrow() ? 52 : 104; }
  else { sl.min = 28; sl.max = 364; sl.step = 7; sl.value = narrow() ? 56 : 112; }
  $("days-out").textContent = sl.value;
  applyTexts();
  const ex = S.meta.examples, start = isVN1() ? ex[0] : ex[2] || ex[0];
  selectIndex(start ? start.i : 0);
  renderBenchmark();
  writeHash();
  await renderExplore();
}

// ------------------------------------------------------------------ boot
async function boot() {
  let lang = (navigator.language || "en").toLowerCase().startsWith("fr") ? "fr" : "en";
  try { lang = localStorage.getItem("lang") || lang; } catch (e) { /* storage unavailable */ }
  S.lang = lang;
  const [h1, h2] = location.hash.replace("#", "").split("/");
  let bench = "m5", tab = h1;
  if (h2 !== undefined) { bench = h1; tab = h2; }
  applyTexts();
  setTab(TABS.includes(tab) ? tab : "explore", false);
  try {
    const [index, learn, glossary] = await Promise.all([
      getJSON("data/index.json").catch(() => ({ benches: ["m5"] })), getJSON("data/learn.json"), getJSON("data/glossary.json").catch(() => null),
    ]);
    S.benches = index.benches; S.learn = learn; S.glossary = glossary;
  } catch (err) {
    $("status").textContent = t("load_error"); $("status").classList.add("error"); console.error(err); return;
  }
  await setBench(S.benches.includes(bench) ? bench : "m5", true);
  if (S.tab === "glossary") writeHash();

  $("sel-level").addEventListener("change", () => { updateSelectors(); renderExplore(); });
  for (const id of ["sel-loc", "sel-prod", "sel-item", "sel-product"]) $(id).addEventListener("change", renderExplore);
  $("sel-dept").addEventListener("change", () => { fillSelect($("sel-item"), S.meta.items_by_dept[$("sel-dept").value]); renderExplore(); });
  $("sel-client").addEventListener("change", () => { updateVN1Selectors({ warehouse: null, product: null }); renderExplore(); });
  $("sel-wh").addEventListener("change", () => { updateVN1Selectors({ product: null }); renderExplore(); });
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
  for (const b of document.querySelectorAll(".bench-switch button")) b.addEventListener("click", () => setBench(b.dataset.bench));
  for (const id of TABS) $("tab-" + id).addEventListener("click", () => setTab(id));
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a.gl");
    if (a) { e.preventDefault(); showGlossary(a.dataset.g); }
  });
  boot();
});
