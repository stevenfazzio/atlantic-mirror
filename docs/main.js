/* ============================================================================
   Atlantic Mirror — two linked map panels.

   North America and Europe are drawn as two framed, independently pan/zoomable
   map-cards (side by side on desktop, stacked on mobile). Dots are sized by
   prominence. Select a city (hover on desktop, tap on mobile; nearest-city snap
   via a per-panel d3-quadtree) → it turns vermilion, its three character-twins
   light up teal in the OTHER card, an arc is drawn to each, and a card shows the
   captions. Desktop: the card tracks the selected point and pins on click.
   Mobile: the card is a bottom peek sheet (space reserved so it never covers the
   far map) that expands to reveal the shared descriptions.
   ============================================================================ */
(function () {
  "use strict";

  const DATA_URL = "data/atlantic-mirror.json";
  const WORLD_URL = "data/countries-50m.json";
  const STATES_URL = "data/states-10m.json";   // US state borders (NA-card navigation aid)

  const CFG = {
    na: { group: "North America", title: "North America", rotate: [98, 0], parallels: [20, 60] },
    eu: { group: "Europe", title: "Europe", rotate: [-14, 0], parallels: [43, 62] },
  };
  const BP = 820;                 // desktop / mobile breakpoint (matches CSS)
  const SNAP_PX = 42;             // nearest-city snap radius (screen px)
  const PAD = 0.06;               // projection fit padding within each card
  const PAN_MARGIN = 0.35;        // how far past the fitted view you may pan (fraction of the card)
  const rScale = d3.scaleSqrt().domain([1, 250]).range([4.3, 1.5]);  // dot radius by prominence rank
  const R_SEL = 7.2, R_MATCH = 6.0;

  const hoverCapable = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  const S = {
    cities: {}, blocks: { na: [], eu: [] },
    land: null, borders: null, naStates: null,
    P: {   // per-panel state (independent projection + zoom transform + quadtree)
      na: { proj: null, t: d3.zoomIdentity, quad: null, rect: null, zoom: null, layer: null, group: null },
      eu: { proj: null, t: d3.zoomIdentity, quad: null, rect: null, zoom: null, layer: null, group: null },
    },
    pos: new Map(),   // qid -> [x, y] in its own panel's identity plane
    mode: "side", W: 0, H: 0,
    selected: null, pinned: false,
  };

  const app = document.getElementById("app");
  const svg = d3.select("#map");
  const panel = document.getElementById("panel");
  const panelBody = panel.querySelector(".panel__body");
  const bar = panel.querySelector(".panel__bar");
  const resetBtn = document.querySelector('#controls button[data-zoom="reset"]');
  let gDefs, gArcs, gLabels;

  init();

  async function init() {
    const [data, world, statesTopo] = await Promise.all([d3.json(DATA_URL), d3.json(WORLD_URL), d3.json(STATES_URL)]);
    S.cities = data.cities;
    for (const [qid, c] of Object.entries(S.cities)) { c.qid = qid; S.blocks[panelKey(qid)].push(c); }
    S.land = topojson.feature(world, world.objects.land);
    S.borders = topojson.mesh(world, world.objects.countries, (a, b) => a !== b);         // country borders
    S.naStates = topojson.mesh(statesTopo, statesTopo.objects.states, (a, b) => a !== b);  // US state borders
    document.querySelector(".hint__label").textContent = hoverCapable ? "Hover a city to begin" : "Tap a city to begin";
    buildScaffold();
    bindGlobal();
    render();
    window.addEventListener("resize", debounce(onResize, 160));
  }

  const panelKey = (qid) => (S.cities[qid].group === "Europe" ? "eu" : "na");

  // ---- one-time scaffold -------------------------------------------------
  function buildScaffold() {
    gDefs = svg.append("defs");
    for (const key of ["na", "eu"]) {
      gDefs.append("clipPath").attr("id", `clip-${key}`).append("rect").attr("rx", 12).attr("ry", 12);
      const P = S.P[key];
      P.group = svg.append("g").attr("class", "panel-card");
      P.group.append("rect").attr("class", "panel-bg").attr("rx", 12).attr("ry", 12);
      P.layer = P.group.append("g").attr("clip-path", `url(#clip-${key})`).append("g").attr("class", "zoom-layer");
      P.layer.append("path").attr("class", "land");
      P.layer.append("path").attr("class", "substate").attr("fill", "none");   // US states (NA only)
      P.layer.append("path").attr("class", "borders").attr("fill", "none");    // country borders
      P.layer.append("g").attr("class", "dots");
      P.group.append("rect").attr("class", "panel-frame").attr("rx", 12).attr("ry", 12);
      P.group.append("text").attr("class", "panel-title");

      P.zoom = d3.zoom().scaleExtent([0.6, 16]).on("zoom", (e) => onZoom(key, e));   // 0.6 → can zoom OUT a bit
      P.group.call(P.zoom).on("dblclick.zoom", null);
      if (hoverCapable) {
        P.group.on("pointermove", (e) => {
          if (S.pinned || e.buttons) return;
          const q = pick(key, e);
          if (q) select(q, false); else clearSelection();     // non-sticky: clears when off a city
        });
        P.group.on("pointerleave", () => { if (!S.pinned) clearSelection(); });
      }
      P.group.on("click", (e) => {
        const q = pick(key, e);
        if (q) select(q, true);              // click pins, so you can move onto the card and its links
        else { S.pinned = false; clearSelection(); }
      });
    }
    gArcs = svg.append("g").attr("class", "arcs");
    gLabels = svg.append("g").attr("class", "labels");
  }

  function onZoom(key, e) {
    S.P[key].t = e.transform;
    S.P[key].layer.attr("transform", e.transform);
    sizeDots(key, false);
    updateOverlays();
    updateResetBtn();
  }

  function bindGlobal() {
    panel.querySelector(".panel__close").addEventListener("click", () => { S.pinned = false; clearSelection(); });
    bar.addEventListener("click", () => { panel.classList.toggle("panel--expanded"); updateBar(); });
    window.addEventListener("keydown", (e) => { if (e.key === "Escape") { S.pinned = false; clearSelection(); } });
    d3.selectAll("#controls button").on("click", function () {
      const kind = this.dataset.zoom;                 // buttons act on both cards at once
      for (const key of ["na", "eu"]) {
        const P = S.P[key];
        if (kind === "reset") P.group.transition().duration(450).call(P.zoom.transform, d3.zoomIdentity);
        else P.group.transition().duration(220).call(P.zoom.scaleBy, kind === "in" ? 1.6 : 1 / 1.6);
      }
    });
  }

  const isDefault = (t) => t.k === 1 && t.x === 0 && t.y === 0;
  function updateResetBtn() { resetBtn.disabled = isDefault(S.P.na.t) && isDefault(S.P.eu.t); }

  // ---- layout + render (re-run on resize) --------------------------------
  function computeLayout() {
    const b = svg.node().getBoundingClientRect();     // the SVG is shorter than #stage on mobile (reserved peek)
    const W = Math.round(b.width), H = Math.round(b.height);
    const mode = W < BP ? "stacked" : "side";
    const M = 10;
    let na, eu;
    if (mode === "side") {
      const g = Math.max(14, W * 0.014);
      const w = (W - 2 * M - g) / 2;
      na = { x: M, y: M, w, h: H - 2 * M };
      eu = { x: M + w + g, y: M, w, h: H - 2 * M };
    } else {
      const g = Math.max(12, H * 0.014);
      const h = (H - 2 * M - g) / 2;
      na = { x: M, y: M, w: W - 2 * M, h };
      eu = { x: M, y: M + h + g, w: W - 2 * M, h };
    }
    return { mode, na, eu, W, H };
  }

  function makeProjection(cfg, cities, r) {
    const pts = { type: "MultiPoint", coordinates: cities.map((c) => [c.lon, c.lat]) };
    const px = r.w * PAD, py = r.h * PAD;
    const proj = d3.geoConicEqualArea().rotate(cfg.rotate).parallels(cfg.parallels);
    proj.fitExtent([[r.x + px, r.y + py], [r.x + r.w - px, r.y + r.h - py]], pts);
    proj.clipExtent([[r.x, r.y], [r.x + r.w, r.y + r.h]]);
    return proj;
  }

  function render() {
    const L = computeLayout();
    S.mode = L.mode; S.W = L.W; S.H = L.H;
    svg.attr("viewBox", `0 0 ${L.W} ${L.H}`);
    if (S.mode !== "side") clearCardPos();

    for (const key of ["na", "eu"]) {
      const P = S.P[key], r = L[key];
      P.rect = r;
      setRect(gDefs.select(`#clip-${key} rect`), r);
      setRect(P.group.select("rect.panel-bg"), r);
      setRect(P.group.select("rect.panel-frame"), r);
      P.group.select("text.panel-title").attr("x", r.x + 16).attr("y", r.y + 25).text(CFG[key].title);
      P.proj = makeProjection(CFG[key], S.blocks[key], r);
      const mx = r.w * PAN_MARGIN, my = r.h * PAN_MARGIN;   // room to pan past the fitted view
      P.zoom.extent([[r.x, r.y], [r.x + r.w, r.y + r.h]])
        .translateExtent([[r.x - mx, r.y - my], [r.x + r.w + mx, r.y + r.h + my]]);
      drawPanel(key);
    }
    computePositions();
    buildQuadtrees();
    for (const key of ["na", "eu"]) S.P[key].group.call(S.P[key].zoom.transform, d3.zoomIdentity);
    updateResetBtn();
    reapplySelection();
  }

  function drawPanel(key) {
    const P = S.P[key], path = d3.geoPath(P.proj);
    P.layer.select("path.land").datum(S.land).attr("d", path);
    P.layer.select("path.substate").datum(key === "na" ? S.naStates : null).attr("d", path);
    P.layer.select("path.borders").datum(S.borders).attr("d", path);
    P.layer.select("g.dots").selectAll("circle").data(S.blocks[key], (d) => d.qid).join(
      (enter) => enter.append("circle").attr("class", "dot").style("vector-effect", "non-scaling-stroke")
        .attr("cx", (d) => P.proj([d.lon, d.lat])[0]).attr("cy", (d) => P.proj([d.lon, d.lat])[1]).attr("r", (d) => dotR(key, d)),
      (update) => update.attr("cx", (d) => P.proj([d.lon, d.lat])[0]).attr("cy", (d) => P.proj([d.lon, d.lat])[1]),
      (exit) => exit.remove()
    );
  }

  function computePositions() {
    S.pos.clear();
    for (const key of ["na", "eu"]) for (const c of S.blocks[key]) S.pos.set(c.qid, S.P[key].proj([c.lon, c.lat]));
  }
  function buildQuadtrees() {
    for (const key of ["na", "eu"]) {
      S.P[key].quad = d3.quadtree().x((d) => d[1][0]).y((d) => d[1][1])
        .addAll(S.blocks[key].map((c) => [c.qid, S.pos.get(c.qid)]));
    }
  }

  // ---- interaction -------------------------------------------------------
  function pick(key, event) {
    const P = S.P[key];
    const [ix, iy] = P.t.invert(d3.pointer(event, svg.node()));
    const hit = P.quad.find(ix, iy, SNAP_PX / P.t.k);
    if (!hit) return null;
    return inside(P.t.apply(hit[1]), P.rect) ? hit[0] : null;
  }

  function select(qid, pin) {
    if (pin) S.pinned = true;
    if (qid === S.selected) { if (pin) renderPanel(qid); return; }
    S.selected = qid;
    app.classList.add("has-selection");
    reapplySelection();
    renderPanel(qid);
  }
  function clearSelection() {
    if (S.selected === null) return;
    S.selected = null;
    app.classList.remove("has-selection");
    reapplySelection();
    panel.classList.remove("is-open", "panel--pinned");
    panel.setAttribute("aria-hidden", "true");
  }

  function reapplySelection() {
    const sel = S.selected;
    const matchSet = sel ? new Set(S.cities[sel].matches.map((m) => m.qid)) : null;
    app.classList.toggle("is-selecting", !!sel);
    for (const key of ["na", "eu"]) {
      S.P[key].layer.select("g.dots").selectAll("circle").each(function (d) {
        d.__role = d.qid === sel ? "sel" : (matchSet && matchSet.has(d.qid) ? "match" : null);
        d.__hi = !!d.__role;
        const c = d3.select(this).classed("dot--sel", d.__role === "sel").classed("dot--match", d.__role === "match");
        if (d.__hi) c.raise();
      });
      sizeDots(key, true);
    }
    updateOverlays();
  }

  function dotR(key, d) {
    const r = d && d.__hi ? (d.__role === "sel" ? R_SEL : R_MATCH) : rScale(d.rank);
    return r / S.P[key].t.k;         // counter-scale: constant screen size across zoom
  }
  function sizeDots(key, animate) {
    const sel = S.P[key].layer.select("g.dots").selectAll("circle");
    const f = (d) => dotR(key, d);
    if (animate) sel.transition("r").duration(180).attr("r", f); else sel.interrupt("r").attr("r", f);
  }

  // ---- overlays: arcs + labels (positioned by each panel's own transform) ----
  const screenPos = (qid) => S.P[panelKey(qid)].t.apply(S.pos.get(qid));
  const visible = (qid) => inside(screenPos(qid), S.P[panelKey(qid)].rect);

  function updateOverlays() { drawArcs(); renderLabels(); if (S.mode === "side") positionCard(); }

  function drawArcs() {
    const sel = S.selected, arcs = [];
    if (sel && visible(sel)) {
      const A = screenPos(sel);
      for (const m of S.cities[sel].matches) if (visible(m.qid)) arcs.push({ id: m.qid, d: arcPath(A, screenPos(m.qid)) });
    }
    gArcs.selectAll("path").data(arcs, (d) => d.id).join(
      (e) => e.append("path").attr("class", "arc"), (u) => u, (x) => x.remove()
    ).attr("d", (d) => d.d);
  }
  function arcPath([ax, ay], [bx, by]) {
    const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1;
    let nx = -dy / len, ny = dx / len;
    if (ny > 0) { nx = -nx; ny = -ny; }              // bow upward
    const k = 0.22 * len;
    return `M${ax},${ay}Q${(ax + bx) / 2 + nx * k},${(ay + by) / 2 + ny * k} ${bx},${by}`;
  }

  // Only the selected city and its three twins get on-map labels.
  function renderLabels() {
    const sel = S.selected, cands = [];
    if (sel) {
      // desktop: the info card sits at the dot and serves as its label, so skip it (it was overlapping the card).
      // mobile: the card is a bottom sheet far from the dot, so keep the selected label on the map.
      if (S.mode !== "side") cands.push({ qid: sel, role: "sel", fs: 15 });
      for (const m of S.cities[sel].matches) cands.push({ qid: m.qid, role: "match", fs: 13.5 });
    }
    const placed = [], out = [];
    for (const cand of cands) {
      const key = panelKey(cand.qid);
      const [sx, sy] = S.P[key].t.apply(S.pos.get(cand.qid));
      if (!inside([sx, sy], S.P[key].rect)) continue;
      const name = S.cities[cand.qid].city, hw = name.length * cand.fs * 0.28;
      const box = { x0: sx - hw, y0: sy - cand.fs - 8, x1: sx + hw, y1: sy - 1 };
      if (placed.some((b) => box.x0 < b.x1 && box.x1 > b.x0 && box.y0 < b.y1 && box.y1 > b.y0)) continue;
      placed.push(box);
      out.push({ qid: cand.qid, role: cand.role, name, sx, sy, fs: cand.fs });
    }
    gLabels.selectAll("text").data(out, (d) => d.qid).join(
      (e) => e.append("text").attr("text-anchor", "middle").attr("dy", "-0.72em"), (u) => u, (x) => x.remove()
    ).attr("class", (d) => `city-label city-label--${d.role}`)
      .style("font-size", (d) => `${d.fs}px`)
      .attr("transform", (d) => `translate(${d.sx},${d.sy})`)
      .text((d) => d.name);
  }

  // ---- info card ---------------------------------------------------------
  const wikiURL = (qid) => "https://en.wikipedia.org/wiki/" + encodeURIComponent(S.cities[qid].wiki.replace(/ /g, "_"));
  const wikiLink = (qid, text) => `<a class="p-link" href="${wikiURL(qid)}" target="_blank" rel="noopener">${esc(text)}</a>`;

  function renderPanel(qid) {
    const c = S.cities[qid];
    const shore = c.group === "Europe" ? "North America" : "Europe";
    const rows = c.matches.map((m, i) => {
      const t = S.cities[m.qid];
      return `<div class="p-match"><div class="p-match__rank">${i + 1}</div>` +
        `<div class="p-match__name">${wikiLink(m.qid, t.city)} <span class="p-match__country">${esc(t.country)}</span></div>` +
        `<div class="p-match__cap">${esc(m.caption)}</div></div>`;
    }).join("");
    panelBody.innerHTML =
      `<h2 class="p-city">${wikiLink(qid, c.city)}<span class="p-city__country">${esc(c.country)}</span></h2>` +
      `<p class="p-lead">Character-twins in ${esc(shore)}</p>` +
      `<p class="p-note">Each description fits <em>both</em> ${esc(c.city)} and its twin.</p>` +
      rows;
    panel.classList.add("is-open");
    panel.classList.toggle("panel--pinned", S.pinned);   // only a pinned card is interactive (hover cards never intercept)
    panel.setAttribute("aria-hidden", "false");
    updateBar();
    if (S.mode === "side") positionCard(); else clearCardPos();
  }

  // Desktop: place the card beside the selected dot, on the OUTER side so it never covers the arcs,
  // flipping to stay on-screen.
  function positionCard() {
    if (S.mode !== "side" || !S.selected) return;
    const [sx, sy] = screenPos(S.selected);
    const w = panel.offsetWidth, h = panel.offsetHeight, GAP = 14, M = 8;
    // card on the OUTER side (NA → left, EU → right) so it never crosses the dot to cover the arcs,
    // then just clamp on-screen (stays on that side rather than flipping over the arcs).
    let left = panelKey(S.selected) === "na" ? sx - GAP - w : sx + GAP;
    left = Math.max(M, Math.min(left, S.W - w - M));
    const top = Math.max(M, Math.min(sy - h * 0.38, S.H - h - M));
    panel.style.left = `${left}px`; panel.style.top = `${top}px`;
    panel.style.right = "auto"; panel.style.bottom = "auto";
  }
  function clearCardPos() { panel.style.left = panel.style.top = panel.style.right = panel.style.bottom = ""; }

  function updateBar() {
    bar.textContent = panel.classList.contains("panel--expanded") ? "Show fewer ↑" : "See the other two ↓";
  }

  // ---- resize + helpers --------------------------------------------------
  function onResize() {
    const wasPinned = S.pinned, wasSel = S.selected;
    render();
    S.pinned = wasPinned;
    if (wasSel && S.cities[wasSel]) { S.selected = wasSel; reapplySelection(); renderPanel(wasSel); }
  }
  function setRect(sel, r) { sel.attr("x", r.x).attr("y", r.y).attr("width", r.w).attr("height", r.h); }
  function inside([x, y], r) { return x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h; }
  function esc(s) { return String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch])); }
  function debounce(fn, ms) { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; }
})();
