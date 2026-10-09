/* Market Monitor front end.
   Reads the JSON files in data/ (written by scripts/update_data.py) and renders every section.
   No build step: plain JavaScript + Chart.js (vendored in assets/vendor). */
(function () {
  "use strict";

  // ------------------------------------------------------------------ utilities
  const $ = (s, r = document) => r.querySelector(s);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const MINUS = "−";
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "style") n.style.cssText = v;
      else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) {
      if (c == null || c === false) continue;
      n.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return n;
  }

  const ok = (v) => v != null && Number.isFinite(v);
  function fmtNum(v, d = 2) {
    if (!ok(v)) return "–";
    return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }).replace("-", MINUS);
  }
  function signed(v, d = 2, suffix = "") {
    if (!ok(v)) return "–";
    const s = Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
    if (Number(Math.abs(v).toFixed(d)) === 0) return (0).toFixed(d) + suffix;
    return (v > 0 ? "+" : MINUS) + s + suffix;
  }
  const fmtPct = (f, d = 1) => (ok(f) ? signed(f * 100, d, "%") : "–");
  const fmtRate = (v, d = 2) => (ok(v) ? fmtNum(v, d) + "%" : "–");
  function fmtBp(pp) {
    if (!ok(pp)) return "–";
    const bp = Math.round(pp * 100);
    return (bp > 0 ? "+" : bp < 0 ? MINUS : "") + Math.abs(bp) + " bp";
  }
  function fmtCap(v) {
    if (!ok(v)) return "–";
    if (v >= 1e12) return "$" + (v / 1e12).toFixed(2) + "T";
    if (v >= 1e9) return "$" + (v / 1e9).toFixed(0) + "B";
    return "$" + (v / 1e6).toFixed(0) + "M";
  }
  const dir = (v) => (!ok(v) || v === 0 ? "flat" : v > 0 ? "up" : "down");
  function change(v, text) {
    // arrow + signed text; the color is never the only cue. A value that displays as zero is flat.
    if (!/[1-9]/.test(String(text).split("(")[0])) v = 0;
    const a =!ok(v) || v === 0 ? "" : v > 0 ? "▲ " : "▼ ";
    return el("span", { class: dir(v) }, el("span", { "aria-hidden": "true" }, a), text);
  }

  // dates: everything is UTC midnight timestamps
  function ts(s) {
    const p = String(s).split("-").map(Number);
    return Date.UTC(p[0], (p[1] || 1) - 1, p[2] || 1);
  }
  function fmtDate(t, g = "day") {
    const d = new Date(t);
    const y = d.getUTCFullYear(), m = d.getUTCMonth();
    if (g === "month") return `${MONTHS[m]} ${y}`;
    if (g === "quarter") return `Q${Math.floor(m / 3) + 1} ${y}`;
    return `${MONTHS[m]} ${d.getUTCDate()}, ${y}`;
  }
  const fmtPeriod = (s, freq) => fmtDate(ts(s), freq === "quarterly" ? "quarter" : "month");

  // ------------------------------------------------------------------ data loading
  async function getJSON(name) {
    try {
      const bust = Math.floor(Date.now() / 600000);
      const r = await fetch(`data/${name}.json?v=${bust}`, { cache: "no-cache" });
      if (!r.ok) return null;
      return await r.json();
    } catch (e) {
      return null;
    }
  }

  async function getText(path) {
    try {
      const r = await fetch(`${path}?v=${Math.floor(Date.now() / 600000)}`, { cache: "no-cache" });
      return r.ok ? await r.text() : null;
    } catch (e) {
      return null;
    }
  }

  // ------------------------------------------------------------------ Chart.js setup
  const redrawers = [];
  const PALETTE = () => ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"].map(css);

  function chartDefaults() {
    Chart.defaults.font.family = css("--font-sans") || "system-ui, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = css("--axis-ink");
    Chart.defaults.borderColor = css("--grid");
  }

  // Tooltip mode: the nearest point on the x axis for EVERY visible series (series may have different dates)
  Chart.Interaction.modes.eachNearestX = function (chart, e) {
    const pos = Chart.helpers.getRelativePosition(e, chart);
    const items = [];
    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      if (!chart.isDatasetVisible(di) || !meta.data.length) return;
      let lo = 0, hi = meta.data.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (meta.data[mid].x < pos.x) lo = mid; else hi = mid;
      }
      const idx = Math.abs(meta.data[lo].x - pos.x) <= Math.abs(meta.data[hi].x - pos.x) ? lo : hi;
      if (meta.data[idx].skip) return;
      items.push({ element: meta.data[idx], datasetIndex: di, index: idx });
    });
    return items;
  };

  Chart.register({
    id: "mmCrosshair",
    afterDatasetsDraw(chart, args, opts) {
      if (opts && opts.enabled === false) return;
      const act = chart.tooltip && chart.tooltip.getActiveElements();
      if (!act || !act.length) return;
      const x = act[0].element.x, { top, bottom } = chart.chartArea, c = chart.ctx;
      c.save(); c.strokeStyle = css("--axis"); c.lineWidth = 1;
      c.beginPath(); c.moveTo(Math.round(x) + 0.5, top); c.lineTo(Math.round(x) + 0.5, bottom); c.stroke(); c.restore();
    },
  });

  Chart.register({
    id: "mmRecessions",
    beforeDatasetsDraw(chart, args, opts) {
      const periods = (opts && opts.periods) || [];
      if (!periods.length) return;
      const xs = chart.scales.x, { top, bottom, left, right } = chart.chartArea, c = chart.ctx;
      c.save(); c.fillStyle = css((opts && opts.color) || "--recession");
      for (const [a, b] of periods) {
        if (b < xs.min || a > xs.max) continue;
        const x0 = Math.max(left, xs.getPixelForValue(a)), x1 = Math.min(right, xs.getPixelForValue(b));
        c.fillRect(x0, top, Math.max(1, x1 - x0), bottom - top);
      }
      c.restore();
    },
  });

  Chart.register({
    id: "mmRefLines",
    afterDatasetsDraw(chart, args, opts) {
      const lines = (opts && opts.lines) || [];
      if (!lines.length) return;
      const ys = chart.scales.y, { left, right, top, bottom } = chart.chartArea, c = chart.ctx;
      c.save();
      for (const ln of lines) {
        const y = ys.getPixelForValue(ln.y);
        if (y < top || y > bottom) continue;
        c.strokeStyle = css("--ink-2"); c.globalAlpha = 0.7; c.lineWidth = 1; c.setLineDash([4, 3]);
        c.beginPath(); c.moveTo(left, Math.round(y) + 0.5); c.lineTo(right, Math.round(y) + 0.5); c.stroke();
        c.setLineDash([]); c.globalAlpha = 1;
        if (ln.label) { c.font = `12px ${css("--font-sans")}`; c.fillStyle = css("--ink-2"); c.textBaseline = "bottom"; c.fillText(ln.label, left + 6, y - 3); }
      }
      c.restore();
    },
  });

  Chart.register({
    id: "mmBarLabels",
    afterDatasetsDraw(chart, args, opts) {
      if (!opts || !opts.format) return;
      const c = chart.ctx, meta = chart.getDatasetMeta(0);
      c.save(); c.font = `12px ${css("--font-sans")}`; c.fillStyle = css("--ink-2"); c.textBaseline = "middle";
      meta.data.forEach((bar, i) => c.fillText(opts.format(chart.data.datasets[0].data[i]), bar.x + 6, bar.y));
      c.restore();
    },
  });

  // calendar-aligned ticks for a linear time axis
  function timeTicks(min, max) {
    const years = (max - min) / (864e5 * 365.25);
    const step = years > 90 ? 240 : years > 45 ? 120 : years > 18 ? 60 : years > 9 ? 24 : years > 3.5 ? 12
      : years > 1.6 ? 6 : years > 0.7 ? 3 : years > 0.2 ? 1 : 0;
    const out = [];
    if (step === 0) { // under ~2 months: weekly ticks
      const d = new Date(min); let t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
      const wd = new Date(t).getUTCDay(); t += ((8 - wd) % 7) * 864e5; // next Monday
      for (; t <= max; t += 7 * 864e5) out.push({ value: t });
      return { ticks: out, step };
    }
    const d = new Date(min);
    let k = Math.ceil((d.getUTCFullYear() * 12 + d.getUTCMonth()) / step) * step;
    for (;;) {
      const t = Date.UTC(Math.floor(k / 12), k % 12, 1);
      if (t > max) break;
      if (t >= min) out.push({ value: t });
      k += step;
    }
    return { ticks: out, step };
  }
  function tickLabel(t, step) {
    const d = new Date(t);
    if (step === 0) return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
    if (step >= 12) return String(d.getUTCFullYear());
    if (d.getUTCMonth() === 0) return `${MONTHS[0]} ${d.getUTCFullYear()}`;
    return MONTHS[d.getUTCMonth()];
  }

  function tooltipStyle() {
    return {
      backgroundColor: css("--surface"), titleColor: css("--ink-2"), bodyColor: css("--ink"),
      borderColor: css("--border-strong"), borderWidth: 1, padding: 10, cornerRadius: 8, caretSize: 0,
      titleFont: { size: 12, weight: "normal" }, bodyFont: { size: 13, weight: "600" },
      usePointStyle: true, boxWidth: 14, boxHeight: 14, bodySpacing: 4,
    };
  }

  /* Line chart over time. series: [{label, color, points:[{x,y}], width, hidden}] */
  function timeLineChart(canvas, o) {
    const surface = css("--surface");
    return new Chart(canvas, {
      type: "line",
      data: {
        datasets: o.series.map((s) => ({
          label: s.label, data: s.points, borderColor: s.color, backgroundColor: s.color,
          borderWidth: s.width || 2, pointRadius: 0, pointHoverRadius: 4, pointHoverBorderWidth: 2,
          pointHoverBorderColor: surface, pointHitRadius: 8, tension: 0, spanGaps: true,
          borderJoinStyle: "round", borderCapStyle: "round", hidden: !!s.hidden,
        })),
      },
      options: {
        parsing: false, normalized: true, animation: false, responsive: true, maintainAspectRatio: false,
        interaction: { mode: "eachNearestX", intersect: false },
        layout: { padding: { top: 8, right: 4 } },
        scales: {
          x: {
            type: "linear", grid: { display: false }, border: { color: css("--axis") },
            min: o.min, max: o.max,
            afterBuildTicks: (scale) => { const r = timeTicks(scale.min, scale.max); scale.ticks = r.ticks; scale._mmStep = r.step; },
            ticks: { maxRotation: 0, autoSkip: true, autoSkipPadding: 12, callback(v) { return tickLabel(v, this._mmStep); } },
          },
          y: {
            type: o.logY ? "logarithmic" : "linear", position: "left",
            grid: { color: (ctx) => (o.zeroLine && ctx.tick && ctx.tick.value === 0 ? css("--axis") : css("--grid")), lineWidth: (ctx) => (o.zeroLine && ctx.tick && ctx.tick.value === 0 ? 1.5 : 1) },
            border: { display: false }, suggestedMin: o.suggestedMin, suggestedMax: o.suggestedMax,
            ticks: { maxTicksLimit: 7, callback: o.yTick, autoSkip: !o.logY },
            afterBuildTicks: o.logY ? (scale) => { scale.ticks = scale.ticks.filter((t) => { const m = t.value / Math.pow(10, Math.floor(Math.log10(t.value) + 1e-9)); return [1, 2, 5].some((k) => Math.abs(m - k) < 1e-6); }); } : undefined,
          },
        },
        plugins: {
          legend: { display: false },
          decimation: { enabled: true, algorithm: "min-max" },
          tooltip: Object.assign(tooltipStyle(), {
            mode: "eachNearestX", intersect: false,
            callbacks: {
              title: (items) => (items.length ? fmtDate(items[0].raw.x, o.grain || "day") : ""),
              label: (it) => ` ${o.yFmt(it.raw.y)}   ${it.dataset.label}`,
              labelPointStyle: () => ({ pointStyle: "line", rotation: 0 }),
              labelColor: (it) => ({ borderColor: it.dataset.borderColor, backgroundColor: it.dataset.borderColor, borderWidth: 3 }),
            },
          }),
          mmRecessions: { periods: o.recessions || [], color: o.shadeColor },
          mmRefLines: { lines: o.refLines || [] },
        },
      },
    });
  }

  /* A chart that can rebuild itself (theme change, range change) and keeps legend + data table in sync. */
  function managed({ canvasId, legendId, build, csvName, legendExtra = [], grain = "day", yFmtCsv }) {
    const canvas = document.getElementById(canvasId);
    const hidden = new Set();
    const api = { chart: null };
    api.rebuild = () => {
      if (api.chart) api.chart.destroy();
      api.chart = build();
      api.chart.data.datasets.forEach((ds, i) => { if (hidden.has(ds.label)) api.chart.setDatasetVisibility(i, false); });
      api.chart.update("none");
      if (legendId) drawLegend();
      if (api.renderTable && api.tableOpen()) api.renderTable();
    };
    function drawLegend() {
      const ul = document.getElementById(legendId);
      ul.replaceChildren();
      const ch = api.chart, many = ch.data.datasets.length > 1;
      if (many) {
        ch.data.datasets.forEach((ds, i) => {
          const b = el("button", { type: "button", "aria-pressed": String(ch.isDatasetVisible(i)) },
            el("span", { class: "key", style: `color:${ds.borderColor}`, "aria-hidden": "true" }), ds.label);
          b.addEventListener("click", () => {
            const vis = !ch.isDatasetVisible(i);
            ch.setDatasetVisibility(i, vis); b.setAttribute("aria-pressed", String(vis));
            if (vis) hidden.delete(ds.label); else hidden.add(ds.label);
            ch.update("none");
          });
          ul.append(el("li", null, b));
        });
      }
      for (const x of legendExtra) {
        const lab = typeof x === "string" ? x : x.label, cls = typeof x === "string" ? "key shade" : `key shade ${x.cls}`;
        ul.append(el("li", null, el("span", { class: "static" }, el("span", { class: cls, "aria-hidden": "true" }), lab)));
      }
    }
    // data table + CSV under the chart
    const det = document.querySelector(`details.datatable[data-chart="${canvasId}"]`);
    if (det) {
      api.tableOpen = () => det.open;
      api.renderTable = () => {
        const ch = api.chart, dss = ch.data.datasets;
        let rows;
        if (ch.config.type === "line" && ch.data.labels && ch.data.labels.length) {
          rows = ch.data.labels.map((lab, i) => [lab, ...dss.map((d) => d.data[i])]);
        } else {
          const xs = new Set(); dss.forEach((d) => d.data.forEach((p) => xs.add(p.x)));
          const maps = dss.map((d) => new Map(d.data.map((p) => [p.x, p.y])));
          rows = [...xs].sort((a, b) => a - b).map((x) => [x, ...maps.map((m) => m.get(x))]);
        }
        const head = ["Date", ...dss.map((d) => d.label)];
        const g = typeof grain === "function" ? grain() : grain;
        const label = (x) => (typeof x === "number" ? fmtDate(x, g) : x);
        const iso = (x) => (typeof x === "number" ? new Date(x).toISOString().slice(0, g === "day" ? 10 : 7) : x);
        const shown = rows.slice().reverse().slice(0, 60);
        const table = el("table", null,
          el("thead", null, el("tr", null, head.map((h, i) => el("th", { class: i ? "n" : null, scope: "col" }, i ? h : (ch.data.labels && ch.data.labels.length ? "Maturity" : h))))),
          el("tbody", null, shown.map((r) => el("tr", null, r.map((v, i) => el("td", { class: i ? "n" : null }, i ? (ok(v) ? (yFmtCsv ? yFmtCsv(v) : fmtNum(v, 2)) : "–") : label(v)))))));
        const btn = el("button", { class: "btn", type: "button" }, "Download CSV");
        btn.addEventListener("click", () => downloadCSV(`${csvName}.csv`, head, rows.map((r) => [iso(r[0]), ...r.slice(1)])));
        det.querySelector(".dt-body").replaceChildren(
          el("div", { class: "dt-tools" }, el("span", null, rows.length > shown.length ? `Latest ${shown.length} of ${rows.length} observations in the selected range. The CSV has all of them.` : `${rows.length} observations`), btn),
          el("div", { class: "dt-scroll" }, table));
      };
      det.addEventListener("toggle", () => { if (det.open) api.renderTable(); });
    }
    redrawers.push(api.rebuild);
    api.rebuild();
    return api;
  }

  function downloadCSV(filename, head, rows) {
    const q = (v) => { if (v == null || (typeof v === "number" && !Number.isFinite(v))) return ""; const s = String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const blob = new Blob([[head, ...rows].map((r) => r.map(q).join(",")).join("\n")], { type: "text/csv" });
    const a = el("a", { href: URL.createObjectURL(blob), download: filename });
    document.body.append(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
  }

  // range buttons
  function bindRange(name, onChange) {
    const g = document.querySelector(`[data-range="${name}"]`);
    if (!g) return null;
    const btns = [...g.querySelectorAll("button")];
    btns.forEach((b) => {
      if (!b.hasAttribute("aria-pressed")) b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => { btns.forEach((x) => x.setAttribute("aria-pressed", String(x === b))); onChange(b.dataset.r); });
    });
    const cur = btns.find((b) => b.getAttribute("aria-pressed") === "true");
    return cur ? cur.dataset.r : btns[0].dataset.r;
  }
  function rangeStart(code, last, first) {
    const d = new Date(last), y = d.getUTCFullYear(), m = d.getUTCMonth(), day = d.getUTCDate();
    if (code === "MAX") return first;
    if (code === "YTD") return Date.UTC(y - 1, 11, 31);
    const n = parseInt(code, 10);
    if (code.endsWith("M")) return Date.UTC(y, m - n, day);
    return Date.UTC(y - n, m, day);
  }
  // keep points from `start` on, plus the one just before it so lines start at the edge
  function sliceFrom(points, start) {
    let i = points.findIndex((p) => p.x >= start);
    if (i === -1) return [];
    return points.slice(Math.max(0, i - 1));
  }
  const toPoints = (pairs) => pairs.filter((p) => ok(p[1])).map((p) => ({ x: ts(p[0]), y: p[1] }));

  // sparkline (inline SVG; the numbers beside it carry the values)
  function sparkline(vals, w = 110, h = 28) {
    const v = (vals || []).filter(ok);
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.setAttribute("aria-hidden", "true");
    if (v.length < 2) return svg;
    const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
    const X = (i) => 2 + (i / (v.length - 1)) * (w - 8), Y = (y) => h - 4 - ((y - lo) / span) * (h - 8);
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", v.map((y, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(y).toFixed(1)}`).join(""));
    path.setAttribute("fill", "none"); path.setAttribute("stroke", css("--ink-2")); path.setAttribute("stroke-width", "1.25");
    path.setAttribute("stroke-linejoin", "round"); path.setAttribute("opacity", "0.8");
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", X(v.length - 1)); dot.setAttribute("cy", Y(v[v.length - 1])); dot.setAttribute("r", "3");
    dot.setAttribute("fill", v[v.length - 1] >= v[0] ? css("--gain-pole") : css("--loss-pole"));
    dot.setAttribute("stroke", css("--surface")); dot.setAttribute("stroke-width", "1.5");
    svg.append(path, dot);
    return svg;
  }

  function emptyState(container, msg) {
    if (!container) return;
    container.replaceChildren(el("div", { class: "empty" }, msg || "This data isn't available yet. It appears after the next automatic update."));
  }
  const statRow = (label, value, note) => el("div", null, el("dt", null, label), el("dd", null, value), note ? el("dd", { class: "note" }, note) : null);
  const keyFig = (k, v, n) => el("div", null, el("div", { class: "k" }, k), el("div", { class: "v num" }, v), n ? el("div", { class: "n" }, n) : null);

  // 52-week range: low ── | ── high, with a marker at today's price
  function rangeCell(low, high, last, dec, prefix = "", barOnly = false) {
    if (!ok(high) || !ok(low) || high <= low) return "–";
    const p = Math.max(0, Math.min(1, (last - low) / (high - low)));
    const short = (v) => prefix + fmtNum(v, Math.abs(v) >= 1000 ? 0 : dec);
    const bar = el("span", { class: "rng", role: "img", "aria-label": `${Math.round(p * 100)}% of the way from the 52-week low (${short(low)}) to the high (${short(high)})` }, el("i", { style: `left:${(p * 100).toFixed(1)}%` }));
    return el("span", { class: "rngcell", title: `52-week low ${short(low)} · high ${short(high)}` }, barOnly ? null : short(low), bar, barOnly ? null : short(high));
  }

  // ------------------------------------------------------------------ quote tables (indices, cross-asset)
  const SYMBOL_LABEL = { "^GSPC": "SPX", "^IXIC": "COMP", "^DJI": "DJIA", "^RUT": "RUT", "^VIX": "VIX", "CL=F": "WTI", "GC=F": "GC", "DX-Y.NYB": "DXY", "EURUSD=X": "EUR/USD", "BTC-USD": "BTC" };
  function quoteTable(box, rows, decOf, unitOf) {
    const th = (t, cls) => el("th", { class: cls || null, scope: "col" }, t);
    const head = el("tr", null, th("Name"), th("Last", "n"), th("Change", "n hide-sm"), th("% Chg", "n"),
      th("1M", "n hide-sm"), th("YTD", "n"), th("1Y", "n"), th("6-month trend", "hide-sm"), th("52-week range", "hide-sm"));
    const body = el("tbody", null, rows.map((q) => {
      const d = decOf(q), unit = unitOf ? unitOf(q) : null, abs = q.last - q.prev;
      return el("tr", null,
        el("td", null, el("span", { class: "co" }, q.name), el("span", { class: "tk" }, SYMBOL_LABEL[q.symbol] || q.symbol), unit ? el("span", { class: "sub" }, unit) : null),
        el("td", { class: "n strong" }, fmtNum(q.last, d)),
        el("td", { class: "n hide-sm" }, change(abs, signed(abs, d))),
        el("td", { class: "n" }, change(q.chg1d, fmtPct(q.chg1d, 2))),
        el("td", { class: "n hide-sm" }, change(q.chg1m, fmtPct(q.chg1m, 1))),
        el("td", { class: "n" }, change(q.ytd, fmtPct(q.ytd, 1))),
        el("td", { class: "n" }, change(q.chg1y, fmtPct(q.chg1y, 1))),
        el("td", { class: "spark hide-sm" }, sparkline(q.spark)),
        el("td", { class: "hide-sm" }, rangeCell(q.low52, q.high52, q.last, d)));
    }));
    box.replaceChildren(el("table", null, el("thead", null, head), body));
  }

  // key numbers strip in the masthead
  function renderStrip(m, mc) {
    const ul = $("#key-strip"), items = [];
    const find = (arr, sym) => (arr || []).find((q) => q.symbol === sym);
    const px = (label, q, d) => { if (q) items.push([label, fmtNum(q.last, d), q.chg1d, fmtPct(q.chg1d, 2)]); };
    if (m) {
      px("S&P 500", find(m.indices, "^GSPC"), 2); px("Nasdaq", find(m.indices, "^IXIC"), 2);
      px("Dow", find(m.indices, "^DJI"), 2); px("Russell 2000", find(m.indices, "^RUT"), 2); px("VIX", find(m.indices, "^VIX"), 2);
    }
    if (mc && mc.rates && mc.rates.DGS10) { const r = mc.rates.DGS10; items.push(["10Y Treasury", fmtRate(r.last), r.chg1d, fmtBp(r.chg1d)]); }
    if (m) {
      px("WTI crude", find(m.cross, "CL=F"), 2); px("Gold", find(m.cross, "GC=F"), 0);
      px("EUR/USD", find(m.cross, "EURUSD=X"), 4); px("Bitcoin", find(m.cross, "BTC-USD"), 0);
    }
    if (!items.length) { ul.closest(".strip-wrap").hidden = true; return; }
    ul.replaceChildren(...items.map(([k, v, ch, txt]) => {
      const c = change(ch, txt); c.className = "c " + c.className;
      return el("li", null, el("span", { class: "k" }, k), el("span", { class: "v" }, v), c);
    }));
  }

  // ------------------------------------------------------------------ sections
  function renderOverview(m) {
    const box = $("#index-table");
    if (!m || !m.indices || !m.indices.length) { emptyState(box); return; }
    const drawTable = () => quoteTable(box, m.indices, () => 2);
    drawTable(); redrawers.push(drawTable);

    const spx = m.spx;
    if (!spx || !spx.history) return;
    const pts = toPoints(spx.history);
    // 200-day moving average over the full history
    const ma = []; let sum = 0;
    for (let i = 0; i < pts.length; i++) {
      sum += pts[i].y; if (i >= 200) sum -= pts[i - 200].y;
      if (i >= 199) ma.push({ x: pts[i].x, y: sum / 200 });
    }
    const last = pts[pts.length - 1];
    const lastMA = ma.length ? ma[ma.length - 1].y : null;
    const vix = m.indices.find((q) => q.symbol === "^VIX");
    const gspc = m.indices.find((q) => q.symbol === "^GSPC");
    $("#spx-sub").textContent = `Daily closing level · latest close ${fmtDate(last.x)}`;
    const vs = lastMA ? last.y / lastMA - 1 : null;
    $("#spx-stats").replaceChildren(
      statRow("Below record high", fmtPct(spx.drawdown, 1), `Record close ${fmtNum(spx.ath, 0)} on ${fmtDate(ts(spx.athDate))}`),
      statRow("Vs. 200-day average", vs != null ? change(vs, fmtPct(vs, 1)) : "–", lastMA ? `200-day average ${fmtNum(lastMA, 0)}` : null),
      statRow("Realized volatility, 1 month", ok(spx.rv21) ? fmtNum(spx.rv21 * 100, 1) + "%" : "–", "Annualized, from daily moves"),
      statRow("Realized volatility, 3 months", ok(spx.rv63) ? fmtNum(spx.rv63 * 100, 1) + "%" : "–", null),
      statRow("Implied volatility (VIX)", vix ? fmtNum(vix.last, 1) + "%" : "–", vix ? `≈ ±${fmtNum(vix.last / Math.sqrt(12), 1)}% typical one-month move` : null),
      gspc ? statRow("52-week range", `${fmtNum(gspc.low52, 0)} – ${fmtNum(gspc.high52, 0)}`, null) : null);

    let range = bindRange("spx", (r) => { range = r; chart.rebuild(); });
    const maBox = $("#spx-ma");
    const chart = managed({
      canvasId: "spx-chart", legendId: "spx-legend", csvName: "sp500", grain: "day",
      build: () => {
        const start = rangeStart(range, last.x, pts[0].x);
        const series = [{ label: "S&P 500", color: css("--s1"), points: sliceFrom(pts, start) }];
        if (maBox.checked) series.push({ label: "200-day average", color: css("--s2"), points: sliceFrom(ma, start), width: 1.5 });
        return timeLineChart($("#spx-chart"), {
          series, min: start, max: last.x, grain: "day",
          yFmt: (v) => fmtNum(v, 2), yTick: (v) => fmtNum(v, 0),
        });
      },
    });
    maBox.addEventListener("change", () => chart.rebuild());
  }

  // diverging heat color: mix the neutral midpoint toward the gain or loss pole
  function hexRgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map((c) => c + c).join(""); const n = parseInt(h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function lum([r, g, b]) { const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); }
  function heat(v, cap) {
    const mid = hexRgb(css("--mid")), pole = hexRgb(v >= 0 ? css("--gain-pole") : css("--loss-pole"));
    const t = Math.pow(Math.min(1, Math.abs(v) / cap), 0.85) * 0.9;
    const rgb = mid.map((c, i) => Math.round(c + (pole[i] - c) * t));
    const L = lum(rgb), white = 1.05 / (L + 0.05), ink = (L + 0.05) / 0.0537;
    return { bg: `rgb(${rgb.join(",")})`, fg: white > ink ? "#ffffff" : "#0d1726" };
  }

  function renderSectors(m) {
    const box = $("#sector-table");
    if (!m || !m.sectors || !m.sectors.length) { emptyState(box); return; }
    const cols = [["chg1d", "1 day", 0.02], ["chg1w", "1 week", 0.04], ["chg1m", "1 month", 0.08], ["chg3m", "3 months", 0.12], ["ytd", "Year to date", 0.25], ["chg1y", "1 year", 0.3]];
    let sortKey = "ytd", desc = true;
    function draw() {
      const rows = m.sectors.slice().sort((a, b) => (desc ? 1 : -1) * ((ok(b[sortKey]) ? b[sortKey] : -9) - (ok(a[sortKey]) ? a[sortKey] : -9)));
      const thead = el("tr", null, el("th", { scope: "col" }, "Sector"),
        ...cols.map(([k, lab]) => {
          const th = el("th", { class: "n sortable", scope: "col", "aria-sort": k === sortKey ? (desc ? "descending" : "ascending") : "none" },
            el("button", { class: "th-btn", type: "button" }, lab));
          th.addEventListener("click", () => { if (sortKey === k) desc = !desc; else { sortKey = k; desc = true; } draw(); });
          return th;
        }));
      const cellRow = (r, cls) => el("tr", { class: cls },
        el("td", null, el("span", { class: "co" }, r.name), el("span", { class: "tk" }, r.symbol)),
        ...cols.map(([k, lab, cap]) => {
          if (!ok(r[k])) return el("td", { class: "cell" }, "–");
          const c = heat(r[k], cap);
          return el("td", { class: "cell", style: `background:${c.bg};color:${c.fg}`, title: `${r.name}, ${lab}: ${fmtPct(r[k], 2)}` }, fmtPct(r[k], 1));
        }));
      const body = el("tbody", null, rows.map((r) => cellRow(r)), m.benchmark ? cellRow(Object.assign({}, m.benchmark, { name: "S&P 500 (benchmark)" }), "bench") : null);
      box.replaceChildren(el("table", { class: "heat" }, el("thead", null, thead), body));
    }
    draw();
    redrawers.push(draw);
    const lg = $("#heat-legend");
    const drawLegend = () => {
      lg.replaceChildren(el("span", null, "Loss"),
        el("span", { class: "bar", style: `background:linear-gradient(90deg, ${heat(-1, 1).bg}, ${heat(0, 1).bg}, ${heat(1, 1).bg})`, "aria-hidden": "true" }),
        el("span", null, "Gain"), el("span", null, "Shading is scaled to each column (±2% for 1 day up to ±30% for 1 year). As of " + fmtDate(ts(m.sectors[0].date)) + "."));
    };
    drawLegend(); redrawers.push(drawLegend);
  }

  function renderFirms(f) {
    const tbox = $("#firm-table");
    if (!f || !f.firms || !f.firms.length) { emptyState(tbox); $("#mcap-box").closest(".panel").hidden = true; return; }
    const firms = f.firms;
    const sumTop = (n) => firms.slice(0, n).reduce((s, r) => s + (r.marketCap || 0), 0);
    const pes = firms.map((r) => r.trailingPE).filter((v) => ok(v) && v > 0).sort((a, b) => a - b);
    const median = pes.length ? (pes.length % 2 ? pes[(pes.length - 1) / 2] : (pes[pes.length / 2 - 1] + pes[pes.length / 2]) / 2) : null;
    $("#firm-keyfigs").replaceChildren(
      keyFig("Largest company", firms[0].name, `${fmtCap(firms[0].marketCap)} market value`),
      keyFig("Top 10 combined", fmtCap(sumTop(10)), `${Math.round((100 * sumTop(10)) / sumTop(20))}% of the top 20's value`),
      keyFig("Top 20 combined", fmtCap(sumTop(20)), `Prices as of ${fmtDate(ts(f.asof))}`),
      keyFig("Median P/E, top 20", ok(median) ? fmtNum(median, 1) + "×" : "–", "Trailing 12-month earnings"));

    const pe = (v) => (!ok(v) || v <= 0 ? "n/m" : v > 999 ? ">999" : fmtNum(v, 1));
    const cols = [
      ["rank", "#", (r) => r.rank, "n"],
      ["name", "Company", (r) => el("span", null, el("span", { class: "co" }, r.name), el("span", { class: "tk" }, r.symbol)), ""],
      ["sector", "Sector", (r) => r.sector || "–", "hide-sm"],
      ["price", "Price", (r) => "$" + fmtNum(r.price, 2), "n"],
      ["chg1d", "1 day", (r) => change(r.chg1d, fmtPct(r.chg1d, 2)), "n"],
      ["ytd", "YTD", (r) => change(r.ytd, fmtPct(r.ytd, 1)), "n"],
      ["chg1y", "1 year", (r) => change(r.chg1y, fmtPct(r.chg1y, 1)), "n hide-sm"],
      ["marketCap", "Mkt cap", (r) => fmtCap(r.marketCap), "n"],
      ["trailingPE", "P/E", (r) => pe(r.trailingPE), "n"],
      ["forwardPE", "Fwd P/E", (r) => pe(r.forwardPE), "n hide-sm"],
      ["divYield", "Div. yield", (r) => (ok(r.divYield) ? fmtNum(r.divYield * 100, 2) + "%" : "–"), "n hide-sm"],
      ["range", "52-wk range", (r) => rangeCell(r.low52, r.high52, r.price, 2, "$", true), "hide-sm"],
    ];
    let sortKey = "rank", asc = true;
    function draw() {
      const val = (r) => (sortKey === "range" ? (r.price - r.low52) / (r.high52 - r.low52) : r[sortKey]);
      const rows = firms.slice().sort((a, b) => {
        const x = val(a), y = val(b);
        if (typeof x === "string" || typeof y === "string") return (asc ? 1 : -1) * String(x || "").localeCompare(String(y || ""));
        const xx = ok(x) ? x : asc ? Infinity : -Infinity, yy = ok(y) ? y : asc ? Infinity : -Infinity;
        return asc ? xx - yy : yy - xx;
      });
      const head = el("tr", null, cols.map(([k, lab, , cls]) => {
        const th = el("th", { class: `${cls} sortable`, scope: "col", "aria-sort": k === sortKey ? (asc ? "ascending" : "descending") : "none" }, el("button", { class: "th-btn", type: "button" }, lab));
        th.addEventListener("click", () => { if (sortKey === k) asc = !asc; else { sortKey = k; asc = k === "rank" || k === "name" || k === "sector"; } draw(); });
        return th;
      }));
      const body = el("tbody", null, rows.map((r) => el("tr", null, cols.map(([k, , fn, cls]) => el("td", { class: cls || null }, fn(r))))));
      tbox.replaceChildren(el("table", null, el("thead", null, head), body));
    }
    draw();

    // market-cap bars
    const mbox = $("#mcap-box");
    mbox.style.height = `${firms.length * 24 + 36}px`;
    const build = () => new Chart($("#mcap-chart"), {
      type: "bar",
      data: { labels: firms.map((r) => r.symbol), datasets: [{ label: "Market cap", data: firms.map((r) => r.marketCap / 1e12), backgroundColor: css("--s1"), hoverBackgroundColor: css("--accent-ink"), borderRadius: 3, borderSkipped: "start", maxBarThickness: 16, categoryPercentage: 0.8, barPercentage: 0.9 }] },
      options: {
        indexAxis: "y", animation: false, responsive: true, maintainAspectRatio: false,
        layout: { padding: { right: 56 } },
        scales: {
          x: { beginAtZero: true, grid: { color: css("--grid") }, border: { display: false }, ticks: { callback: (v) => "$" + v + "T", maxTicksLimit: 6 } },
          y: { grid: { display: false }, border: { color: css("--axis") }, ticks: { color: css("--ink-2") } },
        },
        plugins: {
          legend: { display: false }, mmCrosshair: { enabled: false },
          mmBarLabels: { format: (v) => "$" + v.toFixed(2) + "T" },
          tooltip: Object.assign(tooltipStyle(), { usePointStyle: false, displayColors: false, callbacks: { title: (it) => firms[it[0].dataIndex].name, label: (it) => `${fmtCap(firms[it.dataIndex].marketCap)} market cap` } }),
        },
      },
    });
    let chart = build();
    redrawers.push(() => { chart.destroy(); chart = build(); });
  }

  // ---- rates
  function renderRates(mc) {
    const tbox = $("#rate-table");
    if (!mc || !mc.rates || !Object.keys(mc.rates).length) { emptyState(tbox); return; }
    const R = mc.rates;
    const order = [
      ["DFF", "Fed funds rate", "Effective", "rate"], ["DGS3MO", "3-month T-bill", null, "rate"], ["DGS2", "2-year Treasury", null, "rate"],
      ["DGS10", "10-year Treasury", null, "rate"], ["DGS30", "30-year Treasury", null, "rate"],
      ["DFII10", "10-year real yield", "TIPS", "rate"], ["T10YIE", "10-year breakeven", "Expected inflation", "rate"],
      ["T10Y2Y", "10Y − 2Y spread", "Curve slope", "spread"], ["T10Y3M", "10Y − 3M spread", "Curve slope", "spread"],
      ["BAMLH0A0HYM2", "High-yield spread", "Over Treasuries", "bp"], ["MORTGAGE30US", "30-year mortgage", "Weekly survey", "rate"],
    ];
    const th = (t, cls) => el("th", { class: cls || null, scope: "col" }, t);
    const body = el("tbody", null, order.filter(([id]) => R[id]).map(([id, lab, sub, kind]) => {
      const s = R[id];
      const v = kind === "rate" ? fmtRate(s.last) : kind === "spread" ? fmtBp(s.last) : Math.round(s.last * 100) + " bp";
      const weekly = id === "MORTGAGE30US";
      return el("tr", null,
        el("td", null, el("span", { class: "co" }, lab), sub ? el("span", { class: "sub" }, sub) : null),
        el("td", { class: "n strong" }, v),
        el("td", { class: "n" }, weekly ? "–" : change(s.chg1d, fmtBp(s.chg1d))),
        el("td", { class: "n" }, change(s.chg1m, fmtBp(s.chg1m))),
        el("td", { class: "n" }, change(s.chg1y, fmtBp(s.chg1y))));
    }));
    tbox.replaceChildren(el("table", null, el("thead", null, el("tr", null, th("Rate"), th("Latest", "n"), th("1 day", "n"), th("1 month", "n"), th("1 year", "n"))), body));
    if (R.DGS10) $("#rates-sub").textContent = `Daily values as of ${fmtDate(ts(R.DGS10.lastDate))} · mortgage rate is a weekly survey`;

    // yield curve
    if (mc.curve && mc.curve.snapshots && mc.curve.snapshots.length) {
      const cv = mc.curve;
      $("#curve-sub").textContent = `Yield by maturity (maturities evenly spaced) · latest ${fmtDate(ts(cv.snapshots[0].date))}`;
      const colors = () => [css("--s1"), css("--s2"), css("--s3")];
      managed({
        canvasId: "curve-chart", legendId: "curve-legend", csvName: "treasury_yield_curve", yFmtCsv: (v) => fmtNum(v, 2),
        build: () => new Chart($("#curve-chart"), {
          type: "line",
          data: {
            labels: cv.labels,
            datasets: cv.snapshots.map((s, i) => ({
              label: `${s.name} (${fmtDate(ts(s.date))})`, data: s.values, borderColor: colors()[i], backgroundColor: colors()[i],
              borderWidth: 2, pointRadius: 4, pointBorderColor: css("--surface"), pointBorderWidth: 2, pointHoverRadius: 5, tension: 0, spanGaps: true,
            })),
          },
          options: {
            animation: false, responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
            scales: { x: { grid: { display: false }, border: { color: css("--axis") } }, y: { grid: { color: css("--grid") }, border: { display: false }, ticks: { callback: (v) => v.toFixed(1) + "%", maxTicksLimit: 7 } } },
            plugins: {
              legend: { display: false }, mmCrosshair: { enabled: false },
              tooltip: Object.assign(tooltipStyle(), { callbacks: { title: (it) => `${it[0].label} maturity`, label: (it) => ` ${fmtRate(it.raw)}   ${it.dataset.label}`, labelPointStyle: () => ({ pointStyle: "line", rotation: 0 }), labelColor: (it) => ({ borderColor: it.dataset.borderColor, backgroundColor: it.dataset.borderColor, borderWidth: 3 }) } }),
            },
          },
        }),
      });
    }

    const rec = (mc.recessions || []).map(([a, b]) => [ts(a), Date.UTC(+b.slice(0, 4), +b.slice(5, 7), 1) - 1]);
    let range = bindRange("rates", (r) => { range = r; rates.rebuild(); spread.rebuild(); });
    const lastTs = Math.max(...["DFF", "DGS10", "DGS2"].filter((k) => R[k]).map((k) => ts(R[k].lastDate)));
    const isDaily = () => ["1Y", "3Y"].includes(range);
    function build(canvasId, ids, labels, extra) {
      const series = ids.filter((id) => R[id]).map((id, i) => ({ label: labels[i], color: css(`--s${i + 1}`), points: toPoints(isDaily() ? R[id].daily : R[id].monthly) }));
      const first = Math.min(...series.map((s) => (s.points[0] ? s.points[0].x : Infinity)));
      const start = rangeStart(range, lastTs, first);
      series.forEach((s) => (s.points = sliceFrom(s.points, start)));
      return timeLineChart($("#" + canvasId), Object.assign({
        series, min: Math.max(start, first), max: lastTs, grain: isDaily() ? "day" : "month", recessions: rec,
        yFmt: (v) => fmtNum(v, 2) + "%", yTick: (v) => v + "%",
      }, extra));
    }
    const rates = managed({
      canvasId: "rates-chart", legendId: "rates-legend", csvName: "rates", legendExtra: ["Recession"],
      grain: () => (isDaily() ? "day" : "month"),
      build: () => build("rates-chart", ["DFF", "DGS2", "DGS10"], ["Fed funds (effective)", "2-year Treasury", "10-year Treasury"], { suggestedMin: 0 }),
    });
    const spread = managed({
      canvasId: "spread-chart", legendId: "spread-legend", csvName: "yield_curve_spreads", legendExtra: ["Recession"],
      grain: () => (isDaily() ? "day" : "month"),
      build: () => build("spread-chart", ["T10Y2Y", "T10Y3M"], ["10Y − 2Y", "10Y − 3M"], { zeroLine: true, yFmt: (v) => signed(v, 2, " pp"), yTick: (v) => v + "" }),
    });
  }

  // ---- economy
  function renderEconomy(mc) {
    const tbox = $("#macro-table");
    if (!mc || !mc.macro || !Object.keys(mc.macro).length) { emptyState(tbox); return; }
    const M = mc.macro;
    const rows = [
      ["CPIAUCSL", "CPI inflation", "Year over year", (v) => fmtNum(v, 1) + "%"],
      ["CPILFESL", "Core CPI", "Excl. food and energy", (v) => fmtNum(v, 1) + "%"],
      ["PCEPILFE", "Core PCE", "The Fed's target gauge", (v) => fmtNum(v, 1) + "%"],
      ["UNRATE", "Unemployment rate", "Household survey", (v) => fmtNum(v, 1) + "%"],
      ["PAYEMS", "Payroll job gains", "Monthly change", (v) => signed(v, 0, "K")],
      ["A191RL1Q225SBEA", "Real GDP growth", "Annualized, quarterly", (v) => signed(v, 1, "%")],
    ].filter(([id]) => M[id]);
    const th = (t, cls) => el("th", { class: cls || null, scope: "col" }, t);
    tbox.replaceChildren(el("table", null,
      el("thead", null, el("tr", null, th("Indicator"), th("Latest", "n"), th("Prior", "n"), th("Year ago", "n hide-sm"), th("Period", "n"))),
      el("tbody", null, rows.map(([id, lab, sub, f]) => {
        const s = M[id];
        return el("tr", null,
          el("td", null, el("span", { class: "co" }, lab), el("span", { class: "sub" }, sub)),
          el("td", { class: "n strong" }, f(s.last)),
          el("td", { class: "n muted" }, ok(s.prev) ? f(s.prev) : "–"),
          el("td", { class: "n muted hide-sm" }, ok(s.yearAgo) ? f(s.yearAgo) : "–"),
          el("td", { class: "n" }, fmtPeriod(s.period, s.freq)));
      }))));

    const rec = (mc.recessions || []).map(([a, b]) => [ts(a), Date.UTC(+b.slice(0, 4), +b.slice(5, 7), 1) - 1]);
    let range = bindRange("econ", (r) => { range = r; infl.rebuild(); unemp.rebuild(); });
    function build(canvasId, ids, labels, extra) {
      const series = ids.filter((id) => M[id]).map((id, i) => ({ label: labels[i], color: css(`--s${i + 1}`), points: toPoints(M[id].monthly) }));
      const last = Math.max(...series.map((s) => s.points[s.points.length - 1].x));
      const first = Math.min(...series.map((s) => s.points[0].x));
      const start = rangeStart(range, last, first);
      series.forEach((s) => (s.points = sliceFrom(s.points, start)));
      return timeLineChart($("#" + canvasId), Object.assign({ series, min: Math.max(start, first), max: last, grain: "month", recessions: rec, yFmt: (v) => fmtNum(v, 1) + "%", yTick: (v) => v + "%" }, extra));
    }
    const infl = managed({
      canvasId: "infl-chart", legendId: "infl-legend", csvName: "inflation", grain: "month", legendExtra: ["Recession"],
      build: () => build("infl-chart", ["CPIAUCSL", "CPILFESL", "PCEPILFE"], ["CPI", "Core CPI", "Core PCE"], { refLines: [{ y: 2, label: "Fed target: 2% (PCE)" }], zeroLine: true }),
    });
    const unemp = managed({
      canvasId: "unemp-chart", legendId: "unemp-legend", csvName: "unemployment_rate", grain: "month", legendExtra: ["Recession"],
      build: () => build("unemp-chart", ["UNRATE"], ["Unemployment rate"], { suggestedMin: 0 }),
    });
  }

  function renderCross(m) {
    const box = $("#cross-table");
    if (!m || !m.cross || !m.cross.length) { emptyState(box); return; }
    const dec = { "EURUSD=X": 4, "BTC-USD": 0 };
    const draw = () => quoteTable(box, m.cross, (q) => dec[q.symbol] ?? 2, (q) => q.unit);
    draw(); redrawers.push(draw);
  }

  function renderValuation(v, mc) {
    const list = $("#val-stats");
    if (!v || !v.cape || !v.cape.length) { emptyState(list); $("#cape-chart").closest(".panel").hidden = true; return; }
    const ey = 1 / v.capeLatest;
    const tips = mc && mc.rates && mc.rates.DFII10 ? mc.rates.DFII10 : null;
    const capeMonth = fmtPeriod(v.capeDate);
    const since = v.capeStart.slice(0, 4);
    const pct = Math.round(v.capePercentile * 100);
    const rows = [
      statRow("Shiller CAPE", fmtNum(v.capeLatest, 1) + "×", `As of ${capeMonth}`),
      statRow("Long-run average", fmtNum(v.capeMean, 1) + "×", `Since ${since} · median ${fmtNum(v.capeMedian, 1)}×`),
      statRow("Historical percentile", `${pct}th`, `Higher than ${pct}% of months since ${since}`),
      statRow("Earnings yield (1 ÷ CAPE)", fmtNum(ey * 100, 2) + "%", "Real earnings per dollar of price"),
    ];
    if (tips) {
      rows.push(statRow("10-year real yield (TIPS)", fmtRate(tips.last), `As of ${fmtDate(ts(tips.lastDate))}`));
      rows.push(statRow("Excess CAPE yield", signed(ey * 100 - tips.last, 2, " pp"), "Earnings yield minus real yield"));
    }
    if (ok(v.peLatest)) rows.push(statRow("Trailing P/E", fmtNum(v.peLatest, 1) + "×", `As of ${fmtPeriod(v.peDate)}`));
    list.replaceChildren(...rows);
    $("#cape-sub").textContent = `S&P Composite, monthly, ${since}–${capeMonth}`;

    const pts = toPoints(v.cape);
    let range = bindRange("val", (r) => { range = r; chart.rebuild(); });
    const chart = managed({
      canvasId: "cape-chart", legendId: "cape-legend", csvName: "shiller_cape", grain: "month",
      build: () => {
        const last = pts[pts.length - 1].x, start = rangeStart(range, last, pts[0].x);
        return timeLineChart($("#cape-chart"), {
          series: [{ label: "CAPE", color: css("--s1"), points: sliceFrom(pts, start) }],
          min: Math.max(start, pts[0].x), max: last, grain: "month", suggestedMin: 0,
          refLines: [{ y: v.capeMean, label: `Average since ${since}: ${fmtNum(v.capeMean, 1)}` }],
          yFmt: (x) => fmtNum(x, 1) + "×", yTick: (x) => x,
        });
      },
    });
  }

  // ------------------------------------------------------------------ drawdown risk model
  const pct1 = (v) => (ok(v) ? fmtNum(v * 100, 1) + "%" : "–");
  const LEVEL_TEXT = { Low: "Low risk", Normal: "Normal risk", Elevated: "Elevated risk", High: "High risk" };
  const LEVEL_PATHS = { Low: ["M3 8.5l3 3 7-7"], Normal: ["M3 8h10"], Elevated: ["M8 3v6.5", "M8 12.5v.5"], High: ["M8 2.5l6.2 11H1.8z", "M8 6.5v3", "M8 11.8v.4"] };
  function levelChip(level) {
    const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 16 16"); svg.setAttribute("aria-hidden", "true");
    for (const d of LEVEL_PATHS[level] || []) {
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", d); p.setAttribute("fill", "none"); p.setAttribute("stroke", "currentColor");
      p.setAttribute("stroke-width", "2"); p.setAttribute("stroke-linecap", "round"); p.setAttribute("stroke-linejoin", "round");
      svg.append(p);
    }
    return el("span", { class: "chip " + String(level).toLowerCase() }, svg, LEVEL_TEXT[level] || level);
  }
  function meter(p, base) {
    const top = Math.max(0.2, Math.ceil((Math.max(p, base) * 1.25) / 0.1) * 0.1);
    return el("div", null,
      el("div", { class: "meter", role: "img", "aria-label": `Estimate ${pct1(p)}; usual rate ${pct1(base)}` },
        el("span", { class: "fill", style: `width:${Math.min(100, (p / top) * 100).toFixed(1)}%` }),
        el("span", { class: "tick", style: `left:${Math.min(100, (base / top) * 100).toFixed(1)}%`, title: `Usual rate ${pct1(base)}` })),
      el("div", { class: "meter-scale" }, el("span", null, "0%"), el("span", null, "tick mark = usual rate"), el("span", null, Math.round(top * 100) + "%")));
  }

  // tiny Markdown subset for content/commentary.md: "# title", "Updated: date", paragraphs, "- " bullets, **bold**, *italic*, [link](url)
  function inlineMd(text) {
    const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)\s]+\))/g; let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(text.slice(last, m.index));
      const t = m[0];
      if (t.startsWith("**")) out.push(el("strong", null, t.slice(2, -2)));
      else if (t.startsWith("*")) out.push(el("em", null, t.slice(1, -1)));
      else { const mm = t.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/); out.push(/^https?:\/\//.test(mm[2]) ? el("a", { href: mm[2] }, mm[1]) : mm[1]); }
      last = m.index + t.length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }
  function renderCommentary(md) {
    const box = $("#commentary");
    if (!md || !md.trim()) { box.closest(".panel").hidden = true; return; }
    const nodes = []; let para = [], list = null, date = null;
    const flush = () => { if (para.length) { nodes.push(el("p", null, inlineMd(para.join(" ")))); para = []; } };
    for (const raw of md.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line) { flush(); list = null; continue; }
      if (/^updated:/i.test(line)) { date = line.replace(/^updated:\s*/i, ""); continue; }
      if (line.startsWith("# ")) { flush(); list = null; nodes.push(el("h4", null, line.slice(2))); continue; }
      if (/^[-*] /.test(line)) { flush(); if (!list) { list = el("ul"); nodes.push(list); } list.append(el("li", null, inlineMd(line.slice(2)))); continue; }
      list = null; para.push(line);
    }
    flush();
    box.replaceChildren(...nodes);
    if (date) { const d = /^\d{4}-\d{2}-\d{2}$/.test(date) ? fmtDate(ts(date)) : date; $("#commentary-date").textContent = `Updated ${d}`; }
  }

  function renderRisk(mdl, md) {
    renderCommentary(md);
    const box = $("#risk-signal");
    if (!mdl || !mdl.horizons || !mdl.history) {
      emptyState(box, "The model's first run hasn't finished yet. Results appear here after the next automatic update.");
      ["#risk-chart", "#risk-score", "#risk-drivers"].forEach((s) => { const p = $(s) && $(s).closest(".panel"); if (p) p.hidden = true; });
      return;
    }
    const H = mdl.horizons, keys = Object.keys(H), asof = ts(mdl.asof);
    const evalStart = (mdl.model && mdl.model.evalStart) || mdl.history.dates[0];
    let cur = bindRange("riskh", (r) => { cur = r; drawSignals(); chart.rebuild(); drawCalib(); drawDrivers(); }) || keys[0];
    if (mdl.sample) { const n = $("#risk-notice"); n.hidden = false; n.textContent = `Preview: these results come from your sample file, which ends on ${fmtDate(asof)}. On the live site they update every trading day.`; }
    $("#risk-sub").textContent = `Estimated after the close on ${fmtDate(asof)} · S&P 500 at ${fmtNum(mdl.spxClose, 2)}`;

    function drawSignals() {
      box.replaceChildren(...keys.map((k) => {
        const h = H[k];
        const pctl = ok(h.percentile) ? `Higher than ${Math.round(h.percentile * 100)}% of the model's estimates since ${fmtDate(ts(evalStart), "month")}` : "";
        return el("div", { class: "signal", "aria-current": String(k === cur) },
          el("div", { class: "k" }, `Next ${h.days} trading days`),
          el("div", { class: "v num" }, pct1(h.probability)),
          el("div", { class: "n" }, `chance of a ${Math.round(Math.abs(h.threshold) * 100)}% drop · usual rate ${pct1(h.baseRate)}`),
          levelChip(h.level), meter(h.probability, h.baseRate),
          pctl ? el("div", { class: "n muted" }, pctl) : null);
      }));
    }
    drawSignals();
    const m0 = mdl.model || {}, h0 = H[keys[0]];
    const live = (mdl.liveLog || []).filter((e) => e.date);
    $("#risk-foot").replaceChildren(...[
      el("span", null, `Logistic regression on ${(m0.features || []).length} turbulence features from ${m0.universe || "–"} stocks in ${m0.clusters || "–"} clusters`),
      el("span", null, `Trained on ${fmtNum(h0.trainDays, 0)} days since ${fmtDate(ts(m0.featureStart || mdl.history.dates[0]), "month")}`),
      live.length > 1 ? el("span", null, `Live estimates logged daily since ${fmtDate(ts(live[0].date))}`) : null,
      el("span", null, "Levels compare the estimate with the usual rate: below 0.75× low, up to 1.25× normal, up to 2× elevated, above that high.")].filter(Boolean));

    // track record chart
    const dates = mdl.history.dates.map(ts);
    let range = bindRange("riskr", (r) => { range = r; chart.rebuild(); });
    const chart = managed({
      canvasId: "risk-chart", legendId: "risk-legend", csvName: "drawdown_model_track_record", grain: "day",
      legendExtra: [{ label: "A 3%+ drop followed within the horizon", cls: "event" }], yFmtCsv: (v) => fmtNum(v * 100, 2) + "%",
      build: () => {
        const h = H[cur], P = mdl.history[`p_${cur}`] || [], Y = mdl.history[`y_${cur}`] || [];
        const pts = []; const periods = []; let runStart = null;
        dates.forEach((x, i) => {
          if (ok(P[i])) pts.push({ x, y: P[i] });
          if (Y[i] === 1 && runStart === null) runStart = x;
          if (Y[i] !== 1 && runStart !== null) { periods.push([runStart, dates[i - 1] + 864e5]); runStart = null; }
        });
        if (runStart !== null) periods.push([runStart, dates[dates.length - 1] + 864e5]);
        const last = pts.length ? pts[pts.length - 1].x : asof, first = pts.length ? pts[0].x : dates[0];
        const start = rangeStart(range, last, first);
        $("#risk-chart-sub").textContent = `Out-of-sample estimate of a ${Math.round(Math.abs(h.threshold) * 100)}% drop within ${h.days} trading days. Each day's estimate comes from a model trained only on earlier data`;
        return timeLineChart($("#risk-chart"), {
          series: [{ label: `Estimated probability (${h.days}-day)`, color: css("--s1"), points: sliceFrom(pts, start) }],
          min: Math.max(start, first), max: last, grain: "day", suggestedMin: 0,
          recessions: periods, shadeColor: "--event",
          refLines: [{ y: h.baseRate, label: `Usual rate ${pct1(h.baseRate)}` }],
          yFmt: (v) => fmtNum(v * 100, 1) + "%", yTick: (v) => Math.round(v * 100) + "%",
        });
      },
    });

    // scorecard (both horizons)
    const th = (t, cls) => el("th", { class: cls || null, scope: "col" }, t);
    const ev = (k) => (H[k].evaluation || {});
    const rowsDef = [
      ["Days tested", (s) => fmtNum(s.n, 0), null],
      ["Days followed by a drawdown", (s) => `${fmtNum(s.events, 0)} (${pct1(s.baseRate)})`, null],
      ["AUC (0.5 = coin flip)", (s) => fmtNum(s.auc, 2), "auc"],
      ["Precision-recall AUC (usual rate = no skill)", (s) => fmtNum(s.prAuc, 2), "prAuc"],
      ["Brier skill (above 0 beats the usual rate)", (s) => signed(s.brierSkill, 3), "brierSkill"],
      ["Drawdown rate on the riskiest 20% of days", (s) => pct1(s.hitRateTop20), "hitRateTop20"],
    ];
    const anyEval = ev(keys[0]).model || {};
    $("#risk-score").replaceChildren(el("table", null,
      el("thead", null, el("tr", null, th("Measure"), ...keys.flatMap((k) => [th(`${H[k].days}-day model`, "n"), th("Baseline", "n hide-sm")]))),
      el("tbody", null, rowsDef.map(([lab, f, key]) => el("tr", null, el("td", { style: "white-space:normal;min-width:180px" }, lab),
        ...keys.flatMap((k) => {
          const m = ev(k).model || {}, b = ev(k).volBaseline || {};
          const mb = key && ok(m[key]) && ok(b[key]) ? (m[key] > b[key] ? "m" : m[key] < b[key] ? "b" : "") : "";
          return [el("td", { class: "n" + (mb === "m" ? " best" : "") }, ok(m.n) ? f(m) : "–"),
                  el("td", { class: "n hide-sm" + (mb === "b" ? " best" : "") }, ok(b.n) ? f(b) : "–")];
        }))))));
    if (anyEval.start) $("#score-sub").textContent = `Out of sample, ${fmtDate(ts(anyEval.start))} to ${fmtDate(ts(anyEval.end))}, refit monthly. Baseline: a model that uses only the past month's volatility. Bold marks the better of the two.`;

    function drawCalib() {
      const rows = H[cur].calibration || [];
      $("#calib-sub").textContent = `${H[cur].days}-day horizon: when the model's estimate fell in each range, how often a drawdown followed`;
      $("#risk-calib").replaceChildren(el("table", null,
        el("thead", null, el("tr", null, th("Estimate"), th("Days", "n"), th("Avg. estimate", "n"), th("Drawdown followed", "n"))),
        el("tbody", null, rows.map((r) => el("tr", null,
          el("td", null, `${Math.round(r.from * 100)}–${Math.round(r.to * 100)}%`),
          el("td", { class: "n" }, fmtNum(r.days, 0)), el("td", { class: "n" }, pct1(r.predicted)), el("td", { class: "n strong" }, pct1(r.observed)))))));
    }
    drawCalib();

    function drawDrivers() {
      const d = H[cur].drivers || [];
      const max = Math.max(0.05, ...d.map((x) => Math.abs(x.contribution)));
      $("#risk-drivers").replaceChildren(...d.map((x) => {
        const w = (Math.abs(x.contribution) / max) * 50;
        const bar = el("i", { class: x.contribution >= 0 ? "up" : "down", style: x.contribution >= 0 ? `left:50%;width:${w.toFixed(1)}%` : `left:${(50 - w).toFixed(1)}%;width:${w.toFixed(1)}%` });
        return el("div", { class: "drv" },
          el("div", { class: "lab" }, x.label, el("small", null, `today ${signed(x.z, 1)} standard deviations from its average`)),
          el("div", { class: "bar", role: "img", "aria-label": `${x.label} ${x.contribution >= 0 ? "raises" : "lowers"} the estimate by ${fmtNum(Math.abs(x.contribution), 2)} log-odds` }, bar),
          el("div", { class: "val" }, signed(x.contribution, 2)));
      }), el("div", { class: "drivers-key" }, el("span", null, "◀ lowers the estimate"), el("span", null, "raises the estimate ▶")));
    }
    drawDrivers();

    // internals
    const fmtI = (x) => (x.key === "TurbScalar" ? fmtNum(x.value * 100, 2) + "%" : x.format === "pct" ? fmtNum(x.value * 100, 1) + "%" : x.format === "z" ? signed(x.value, 2) : Math.abs(x.value) < 0.1 ? x.value.toFixed(4) : fmtNum(x.value, 2));
    $("#risk-internals").replaceChildren(el("table", null,
      el("thead", null, el("tr", null, th("Measure"), th("Latest", "n"), th("Percentile vs. history"))),
      el("tbody", null, (mdl.internals || []).map((x) => el("tr", null,
        el("td", { style: "white-space:normal" }, x.label),
        el("td", { class: "n strong" }, fmtI(x)),
        el("td", null, ok(x.percentile) ? el("span", { class: "pbar" }, el("span", { class: "track" }, el("i", { style: `width:${(x.percentile * 100).toFixed(0)}%` })), `${Math.round(x.percentile * 100)}th`) : "–"))))));
  }

  const FACTOR_INFO = {
    "Mkt-RF": ["Market", "Stocks minus T-bills"],
    SMB: ["Size (SMB)", "Small minus big"],
    HML: ["Value (HML)", "High minus low book-to-market"],
    RMW: ["Profitability (RMW)", "Robust minus weak"],
    CMA: ["Investment (CMA)", "Conservative minus aggressive"],
    Mom: ["Momentum (Mom)", "Winners minus losers"],
  };
  function renderFactors(fa) {
    const tbox = $("#factor-table");
    if (!fa || !fa.months || !fa.months.length) { emptyState(tbox); $("#factor-chart").closest(".card").hidden = true; return; }
    const lm = fa.lastMonth, year = lm.slice(0, 4), monthName = MONTHS_LONG[+lm.slice(5, 7) - 1];
    $("#factor-lede").textContent = `Fama-French five factors plus momentum, from Kenneth French's Data Library (monthly, U.S. stocks). Latest month: ${monthName} ${year}. The library updates about once a month, with a lag of a month or two.`;
    const keys = Object.keys(FACTOR_INFO).filter((k) => fa.summary[k]);
    const start = fa.summary[keys[0]].start.slice(0, 4);
    const head = el("tr", null, ["Factor", monthName, `YTD ${year}`, "Trailing 12M", "10Y, per year", `Avg. per year since ${start}`, "Volatility per year", "Sharpe ratio"].map((h, i) => el("th", { class: i ? "n" : null, scope: "col" }, h)));
    const row = (k, name, desc, s, muted) => el("tr", null,
      el("td", null, el("span", { class: "co" }, name), el("span", { class: "tk" }, desc)),
      el("td", { class: "n" }, change(s.lastMonth, fmtPct(s.lastMonth, 2))),
      el("td", { class: "n" }, change(s.ytd, fmtPct(s.ytd, 1))),
      el("td", { class: "n" }, change(s.trailing12m, fmtPct(s.trailing12m, 1))),
      el("td", { class: "n" }, change(s.ann10y, fmtPct(s.ann10y, 1))),
      el("td", { class: "n" }, fmtPct(s.annMean, 1)),
      el("td", { class: "n" }, muted ? "–" : fmtNum(s.annVol * 100, 1) + "%"),
      el("td", { class: "n" }, muted ? "–" : fmtNum(s.sharpe, 2)));
    const body = el("tbody", null, keys.map((k) => row(k, FACTOR_INFO[k][0], FACTOR_INFO[k][1], fa.summary[k])),
      fa.summary.RF ? row("RF", "Risk-free rate", "1-month T-bill", fa.summary.RF, true) : null);
    tbox.replaceChildren(el("table", null, el("thead", null, head), body));

    const months = fa.months.map(ts);
    let range = bindRange("fac", (r) => { range = r; chart.rebuild(); });
    const chart = managed({
      canvasId: "factor-chart", legendId: "factor-legend", csvName: "factor_growth_of_1", grain: "month", yFmtCsv: (v) => fmtNum(v, 3),
      build: () => {
        const last = months[months.length - 1];
        const st = rangeStart(range, last, months[0]);
        let i0 = months.findIndex((m) => m > st); if (i0 < 0) i0 = 0;
        const series = keys.map((k, j) => {
          const r = fa.returns[k], pts = [];
          let g = 1;
          const baseX = i0 > 0 ? months[i0 - 1] : Date.UTC(new Date(months[0]).getUTCFullYear(), new Date(months[0]).getUTCMonth() - 1, 1);
          pts.push({ x: baseX, y: 1 });
          for (let i = i0; i < months.length; i++) { if (ok(r[i])) g *= 1 + r[i] / 100; pts.push({ x: months[i], y: g }); }
          return { label: FACTOR_INFO[k][0], color: css(`--s${j + 1}`), points: pts };
        });
        return timeLineChart($("#factor-chart"), {
          series, min: series[0].points[0].x, max: last, grain: "month", logY: true,
          refLines: [{ y: 1, label: "" }],
          yFmt: (v) => "$" + fmtNum(v, v < 10 ? 2 : 1), yTick: (v) => "$" + (v >= 1 ? v.toLocaleString("en-US") : v),
        });
      },
    });
  }

  function renderStatus(st, markets) {
    const list = $("#status-list");
    const names = { markets: "Indices, sectors, cross-asset", firms: "Largest companies", macro: "Rates and economy", valuation: "Shiller CAPE", factors: "Fama-French factors", model: "Drawdown risk model" };
    if (!st || !st.sections) { list.replaceChildren(el("li", null, "No update has run yet.")); }
    else {
      list.replaceChildren(...Object.entries(names).map(([k, lab]) => {
        const s = st.sections[k];
        if (!s) return el("li", null, el("span", null, lab), el("span", { class: "badge err" }, "not run"));
        const when = s.lastSuccess ? new Date(s.lastSuccess).toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) + " ET" : "never";
        return el("li", { title: s.error || "" }, el("span", null, lab),
          el("span", null, el("span", { class: "muted" }, when + " "), el("span", { class: "badge " + (s.ok ? "ok" : "err") }, s.ok ? "OK" : "kept previous data")));
      }));
    }
    // masthead freshness line
    const asof = $("#asof"), notice = $("#notice");
    if (!markets) {
      asof.textContent = "No data yet";
      notice.hidden = false;
      notice.textContent = "No data has been generated yet. If you just set up this site, open the repository's Actions tab and run the “Update market data” workflow once (see README).";
      return;
    }
    const d = ts(markets.asof), ageDays = (Date.now() - d) / 864e5;
    const upd = markets.updated ? new Date(markets.updated).toLocaleString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }) : "";
    const wd = new Date(d).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" });
    asof.replaceChildren(el("span", { class: "dot" + (ageDays > 4.5 ? " stale" : ""), "aria-hidden": "true" }),
      "Prices as of ", el("strong", null, `${wd}, ${fmtDate(d)}`), upd ? ` · updated ${upd} ET` : "");
    const failed = st && st.sections ? Object.entries(st.sections).filter(([, s]) => !s.ok).map(([k]) => names[k] || k) : [];
    if (ageDays > 4.5) {
      notice.hidden = false;
      notice.textContent = `The latest prices are from ${fmtDate(d)}. The automatic update may have stopped. Check the Actions tab of the repository.`;
    } else if (failed.length) {
      notice.hidden = false;
      notice.textContent = `Some sources didn't update on the last run (${failed.join(", ")}), so those sections show the most recent data available. See "About the data".`;
    }
  }

  // ------------------------------------------------------------------ page chrome
  function applySiteConfig() {
    const S = window.SITE || {};
    if (S.title) { $("#site-title").textContent = S.title; document.title = S.title; }
    if (S.tagline) $("#site-tagline").textContent = S.tagline;
    const who = S.maintainer ? (S.maintainerUrl ? el("a", { href: S.maintainerUrl }, S.maintainer) : S.maintainer) : null;
    if (who) {
      $("#maintainer").replaceChildren("Maintained by ", who, S.course ? ` for ${S.course}` : "", ".");
      $("#foot-line").replaceChildren(`${S.title || "Market Monitor"} · maintained by `, who.cloneNode ? who.cloneNode(true) : who, " · for educational use only, not investment advice.");
    }
  }

  function setupTheme() {
    const btn = $("#theme-toggle"), mq = window.matchMedia("(prefers-color-scheme: dark)");
    const effective = () => document.documentElement.getAttribute("data-theme") || (mq.matches ? "dark" : "light");
    const redraw = () => { chartDefaults(); redrawers.forEach((f) => { try { f(); } catch (e) { console.error(e); } }); };
    btn.addEventListener("click", () => {
      const next = effective() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("mm-theme", next); } catch (e) { /* storage unavailable */ }
      redraw();
    });
    mq.addEventListener && mq.addEventListener("change", () => { if (!document.documentElement.getAttribute("data-theme")) redraw(); });
  }

  function setupNav() {
    const links = [...document.querySelectorAll(".secnav a")];
    const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    if (!("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((a) => a.classList.remove("active"));
          const a = byId.get(e.target.id);
          if (a) { a.classList.add("active"); a.scrollIntoView({ block: "nearest", inline: "nearest" }); }
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("main section[id]").forEach((s) => io.observe(s));
  }

  function safe(fn, containerSel, ...args) {
    try { fn(...args); } catch (e) { console.error(e); emptyState($(containerSel), "Something went wrong drawing this section."); }
  }

  async function main() {
    applySiteConfig();
    chartDefaults();
    setupTheme();
    setupNav();
    const [markets, firms, macro, valuation, factors, status, model, commentary] = await Promise.all([
      ...["markets", "firms", "macro", "valuation", "factors", "status", "model"].map(getJSON), getText("content/commentary.md")]);
    safe(renderStatus, "#status-list", status, markets);
    safe(renderStrip, "#key-strip", markets, macro);
    safe(renderOverview, "#index-table", markets);
    safe(renderRisk, "#risk-signal", model, commentary);
    safe(renderSectors, "#sector-table", markets);
    safe(renderFirms, "#firm-table", firms);
    safe(renderRates, "#rate-table", macro);
    safe(renderEconomy, "#macro-table", macro);
    safe(renderCross, "#cross-table", markets);
    safe(renderValuation, "#val-stats", valuation, macro);
    // redraw charts once the web fonts arrive so axis labels use them
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { chartDefaults(); redrawers.forEach((f) => { try { f(); } catch (e) { console.error(e); } }); });
    safe(renderFactors, "#factor-table", factors);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main);
  else main();
})();
