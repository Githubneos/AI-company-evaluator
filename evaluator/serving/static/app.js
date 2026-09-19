/* AI Company Evaluator — dashboard.
 *
 * Plain JS, no build step: the service stays a single `uvicorn` process.
 * Every number on screen comes from the JSON API; this file only arranges and
 * animates it. It must never soften what the API says about model skill — the
 * verdict text is rendered verbatim.
 */
"use strict";

(() => {
  // ------------------------------------------------------------------ utils

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  const isNum = (x) => typeof x === "number" && Number.isFinite(x);
  const MINUS = "−";
  const signed = (x, d) => (x >= 0 ? "+" : MINUS) + Math.abs(x).toFixed(d);

  const FMT = {
    pct0: (x) => (x * 100).toFixed(0) + "%",
    pct1: (x) => (x * 100).toFixed(1) + "%",
    spct1: (x) => signed(x * 100, 1) + "%",
    spct2: (x) => signed(x * 100, 2) + "%",
    s4: (x) => signed(x, 4),
    s3: (x) => signed(x, 3),
    f3: (x) => x.toFixed(3),
    f2: (x) => x.toFixed(2),
    lift: (x) => x.toFixed(2) + "×",
    int: (x) => Math.round(x).toLocaleString(),
    money: (x) => "$" + x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  };
  const fmt = (x, kind) => (isNum(x) ? FMT[kind](x) : "—");

  /** A number that counts up from zero when its view enters. */
  const num = (x, kind, cls = "") =>
    isNum(x)
      ? `<span class="num ${cls}" data-to="${x}" data-fmt="${kind}">${FMT[kind](x)}</span>`
      : `<span class="num ${cls}">—</span>`;

  const toneOf = (x, eps = 0) => (!isNum(x) ? "" : x > eps ? "pos" : x < -eps ? "neg" : "");

  const TARGETS = ["magnitude_1d", "magnitude_5d", "magnitude_20d", "direction_1d", "direction_5d", "direction_20d"];
  const HORIZONS = [1, 5, 20];
  const REGIMES = [
    ["pre_gfc", "Pre-GFC"], ["gfc", "GFC"], ["recovery", "Recovery"], ["covid_crash", "COVID crash"],
    ["covid_recovery", "COVID recovery"], ["rate_hikes", "Rate hikes"], ["recent", "Recent"],
  ];
  const regimeName = (k) => (REGIMES.find(([key]) => key === k) || [k, k])[1];
  const targetParts = (t) => {
    const [kind, h] = t.split("_");
    return { kind, h: parseInt(h, 10) };
  };
  const targetLabel = (t) => {
    const { kind, h } = targetParts(t);
    return `${kind[0].toUpperCase()}${kind.slice(1)} · ${h}d`;
  };

  const ICON = {
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.1V12a10 10 0 1 1-5.9-9.1"/><path d="m22 4-10 10-3-3"/></svg>',
    alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/></svg>',
    gauge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.3 19a10 10 0 1 1 17.4 0"/></svg>',
    split: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3h5v5M8 3H3v5M21 3l-9 9-9-9M12 12v9"/></svg>',
    bars: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h10M3 12h16M3 18h7"/></svg>',
    news: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8M15 18h-5M10 6h8v4h-8z"/></svg>',
    history: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l4 2"/></svg>',
    target: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>',
    chevron: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>',
    refresh: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5M3 21v-5h5"/></svg>',
    heat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    pulse: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 8-6-16-3 8H2"/></svg>',
    tag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.6 2.6A2 2 0 0 0 11.2 2H4a2 2 0 0 0-2 2v7.2a2 2 0 0 0 .6 1.4l8.7 8.7a2.4 2.4 0 0 0 3.4 0l6.6-6.6a2.4 2.4 0 0 0 0-3.4Z"/><circle cx="7.5" cy="7.5" r="1.5"/></svg>',
    trend: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg>',
    sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>',
  };

  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch (_) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* private mode */ }
    },
  };

  // ------------------------------------------------------------------ api

  let inflight = 0;
  const busy = (delta) => {
    inflight = Math.max(0, inflight + delta);
    $("#progress").classList.toggle("active", inflight > 0);
  };

  async function api(path, { signal, method = "GET", quiet = false } = {}) {
    if (!quiet) busy(1);
    try {
      const res = await fetch(path, { method, signal, headers: { Accept: "application/json" } });
      let body = null;
      try { body = await res.json(); } catch (_) { /* empty or non-JSON body */ }
      if (!res.ok) {
        const detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `${res.status} ${res.statusText}`;
        const err = new Error(detail);
        err.status = res.status;
        throw err;
      }
      return body;
    } finally {
      if (!quiet) busy(-1);
    }
  }

  const state = {
    universe: null,
    byTicker: new Map(),
    validation: null, // target -> validation response, filled in the background
    validationPromise: null,
    controller: null,
    renderToken: 0,
    route: null,
    range: store.get("ace-range", 252),
    driverKind: "magnitude",
    driverH: 1,
    calibClass: "DROP",
    modelTarget: "magnitude_1d",
    sentiment: new Map(),
    ticker: null,
  };

  function loadUniverse() {
    if (!state.universePromise) {
      state.universePromise = api("/universe", { quiet: true })
        .then((u) => {
          state.universe = u;
          for (const row of u.tickers || []) state.byTicker.set(row.ticker, row);
          return u;
        })
        .catch(() => null);
    }
    return state.universePromise;
  }

  function loadValidation() {
    if (!state.validationPromise) {
      state.validationPromise = Promise.all(
        TARGETS.map((t) => api(`/model/${t}/validation`, { quiet: true }).then((v) => [t, v]).catch(() => [t, null])),
      ).then((pairs) => {
        state.validation = Object.fromEntries(pairs.filter(([, v]) => v));
        return state.validation;
      });
    }
    return state.validationPromise;
  }

  /** Regimes in which a target's out-of-sample Brier skill is negative. */
  function negativeRegimes(target) {
    const v = state.validation && state.validation[target];
    const by = v && v.validation && v.validation.by_regime;
    if (!by) return [];
    return Object.entries(by).filter(([, m]) => isNum(m.brier_skill) && m.brier_skill < 0).map(([k]) => k);
  }

  // ------------------------------------------------------------------ motion

  function countUp(el) {
    const to = parseFloat(el.dataset.to);
    const f = FMT[el.dataset.fmt];
    if (!f || !isNum(to)) return;
    if (reducedMotion.matches) { el.textContent = f(to); return; }
    const from = 0;
    const dur = 950;
    const t0 = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - p, 4);
      el.textContent = f(from + (to - from) * e);
      if (p < 1) requestAnimationFrame(step);
    };
    el.textContent = f(from);
    requestAnimationFrame(step);
  }

  /** Stagger children in, count numbers up, and trigger grow/draw transitions. */
  function animateIn(root, { stagger = true } = {}) {
    if (stagger) {
      Array.from(root.children).forEach((el, i) => el.style.setProperty("--i", i));
      root.classList.remove("view-enter");
      void root.offsetWidth; // restart the entrance animation
      root.classList.add("view-enter");
    }
    $$("[data-to]", root).forEach(countUp);
    // Two frames: the first commits the start state, the second transitions away from it.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $$(".stack, .drivers, .fold-spark, .abl", root).forEach((el) => el.classList.add("grown"));
      $$(".ring", root).forEach((ring) => {
        const v = $(".value", ring);
        if (v) v.style.strokeDashoffset = v.dataset.offset;
        ring.classList.add("drawn");
      });
      $$(".senti-knob", root).forEach((k) => { k.style.left = k.dataset.left; });
    }));
    $$(".seg", root).forEach(syncSeg);
  }

  function syncSeg(seg) {
    const thumb = $(".seg-thumb", seg);
    const on = $('button[aria-pressed="true"], button[aria-selected="true"]', seg);
    if (!thumb || !on) return;
    thumb.style.width = on.offsetWidth + "px";
    thumb.style.transform = `translateX(${on.offsetLeft}px)`;
  }

  function seg(options, current, attr = "data-value", label = "") {
    return `<div class="seg" role="group" aria-label="${esc(label)}"><span class="seg-thumb" aria-hidden="true"></span>${options
      .map(([v, text]) => `<button type="button" ${attr}="${esc(v)}" aria-pressed="${String(v) === String(current)}">${esc(text)}</button>`)
      .join("")}</div>`;
  }

  function pressSeg(segEl, btn) {
    $$("button", segEl).forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    syncSeg(segEl);
  }

  function toast(message, tone = "") {
    const el = document.createElement("div");
    el.className = `toast ${tone}`;
    el.innerHTML = message;
    $("#toasts").appendChild(el);
    setTimeout(() => {
      el.classList.add("out");
      el.addEventListener("animationend", () => el.remove(), { once: true });
      setTimeout(() => el.remove(), 600);
    }, 3800);
  }

  function elapsedTimer(el) {
    const t0 = Date.now();
    const id = setInterval(() => {
      if (!el.isConnected) return clearInterval(id);
      el.textContent = `${Math.floor((Date.now() - t0) / 1000)}s`;
    }, 1000);
    return () => clearInterval(id);
  }

  // ------------------------------------------------------------------ shell

  function updateNav(name) {
    const links = $$(".nav a");
    let active = null;
    for (const a of links) {
      const on = a.dataset.route === name;
      if (on) { a.setAttribute("aria-current", "page"); active = a; } else a.removeAttribute("aria-current");
    }
    const ind = $("#nav-indicator");
    if (active) {
      ind.style.width = active.offsetWidth + "px";
      ind.style.height = active.offsetHeight + "px";
      ind.style.transform = `translate(${active.offsetLeft}px, ${active.offsetTop}px)`;
      ind.style.opacity = "1";
    } else ind.style.opacity = "0";
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("ace-theme", theme); } catch (_) { /* ignore */ }
    const btn = $("#theme-toggle");
    btn.innerHTML = `${theme === "light" ? ICON.moon : ICON.sun}<span>${theme === "light" ? "Dark theme" : "Light theme"}</span>`;
  }

  async function checkHealth() {
    const el = $("#api-status");
    try {
      const h = await api("/health", { quiet: true });
      const n = (h.trained_targets || []).length;
      el.innerHTML = `<span class="dot ${n ? "ok" : "bad"}"></span><span>API online · ${n} model${n === 1 ? "" : "s"}</span>`;
    } catch (_) {
      el.innerHTML = '<span class="dot bad"></span><span>API unreachable</span>';
    }
  }

  // ------------------------------------------------------------------ search

  const search = {
    input: null,
    list: null,
    items: [],
    active: -1,

    init() {
      this.input = $("#search-input");
      this.list = $("#search-list");
      this.input.addEventListener("input", () => this.update());
      this.input.addEventListener("focus", () => this.update());
      this.input.addEventListener("keydown", (e) => this.key(e));
      this.input.addEventListener("blur", () => setTimeout(() => this.close(), 120));
      this.list.addEventListener("mousedown", (e) => e.preventDefault()); // keep focus
      this.list.addEventListener("click", (e) => {
        const opt = e.target.closest(".option");
        if (opt) this.choose(opt.dataset.ticker);
      });
      this.list.addEventListener("mousemove", (e) => {
        const opt = e.target.closest(".option");
        if (opt) this.highlight(parseInt(opt.dataset.index, 10), false);
      });
      document.addEventListener("keydown", (e) => {
        const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "");
        if ((e.key === "/" && !typing) || (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey))) {
          e.preventDefault();
          this.input.focus();
          this.input.select();
        }
      });
    },

    rank(q) {
      const rows = state.universe ? state.universe.tickers || [] : [];
      if (!q) {
        const recent = store.get("ace-recent", []);
        const picks = [...recent, "AAPL", "NVDA", "MSFT", "JPM", "XOM", "TSLA", "UNH", "AMZN"];
        const seen = new Set();
        return picks.filter((t) => !seen.has(t) && seen.add(t) && state.byTicker.has(t)).slice(0, 8).map((t) => state.byTicker.get(t));
      }
      const Q = q.toUpperCase();
      const lq = q.toLowerCase();
      const scored = [];
      for (const r of rows) {
        const name = r.name.toLowerCase();
        let s = -1;
        if (r.ticker === Q) s = 0;
        else if (r.ticker.startsWith(Q)) s = 1;
        else if (name.startsWith(lq)) s = 2;
        else if (name.split(/\s+/).some((w) => w.startsWith(lq))) s = 3;
        else if (r.ticker.includes(Q)) s = 4;
        else if (name.includes(lq)) s = 5;
        if (s >= 0) scored.push([s, r]);
      }
      scored.sort((a, b) => a[0] - b[0] || a[1].ticker.localeCompare(b[1].ticker));
      return scored.slice(0, 8).map(([, r]) => r);
    },

    mark(text, q) {
      if (!q) return esc(text);
      const i = text.toLowerCase().indexOf(q.toLowerCase());
      if (i < 0) return esc(text);
      return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + q.length)) + "</mark>" + esc(text.slice(i + q.length));
    },

    update() {
      const q = this.input.value.trim();
      this.items = this.rank(q);
      const looksLikeTicker = /^[A-Za-z.\-]{1,6}$/.test(q) && !this.items.some((r) => r.ticker === q.toUpperCase());
      let html = this.items
        .map(
          (r, i) => `<div class="option" role="option" id="opt-${i}" data-index="${i}" data-ticker="${esc(r.ticker)}" aria-selected="false">
            <span class="option-ticker">${this.mark(r.ticker, q)}</span>
            <span class="option-name">${this.mark(r.name, q)}</span>
            <span class="option-sector">${esc(r.sector)}</span></div>`,
        )
        .join("");
      if (looksLikeTicker) {
        const t = q.toUpperCase();
        this.items.push({ ticker: t, name: "Outside the S&P 500 universe", sector: "" });
        const i = this.items.length - 1;
        html += `<div class="option" role="option" id="opt-${i}" data-index="${i}" data-ticker="${esc(t)}" aria-selected="false">
          <span class="option-ticker">${esc(t)}</span><span class="option-name muted">Try anyway — outside the trained universe</span><span></span></div>`;
      }
      if (!html) html = `<div class="listbox-empty">No company matches “${esc(q)}”.</div>`;
      else if (!q) html = `<div class="eyebrow" style="padding:8px 10px 4px">${store.get("ace-recent", []).length ? "Recent & popular" : "Popular"}</div>` + html;
      this.list.innerHTML = html;
      this.list.hidden = false;
      this.input.setAttribute("aria-expanded", "true");
      this.highlight(this.items.length ? 0 : -1, false);
    },

    highlight(i, scroll = true) {
      this.active = i;
      $$(".option", this.list).forEach((el) => el.setAttribute("aria-selected", String(parseInt(el.dataset.index, 10) === i)));
      const el = $(`#opt-${i}`, this.list);
      if (el) {
        this.input.setAttribute("aria-activedescendant", el.id);
        if (scroll) el.scrollIntoView({ block: "nearest" });
      } else this.input.removeAttribute("aria-activedescendant");
    },

    key(e) {
      if (this.list.hidden && (e.key === "ArrowDown" || e.key === "ArrowUp")) this.update();
      const n = this.items.length;
      if (e.key === "ArrowDown") { e.preventDefault(); if (n) this.highlight((this.active + 1) % n); }
      else if (e.key === "ArrowUp") { e.preventDefault(); if (n) this.highlight((this.active - 1 + n) % n); }
      else if (e.key === "Enter") {
        e.preventDefault();
        const pick = this.items[this.active];
        if (pick) this.choose(pick.ticker);
      } else if (e.key === "Escape") { this.close(); this.input.blur(); }
    },

    choose(ticker) {
      this.close();
      this.input.value = "";
      this.input.blur();
      location.hash = `#/t/${encodeURIComponent(ticker)}`;
    },

    close() {
      this.list.hidden = true;
      this.input.setAttribute("aria-expanded", "false");
      this.input.removeAttribute("aria-activedescendant");
    },
  };

  function rememberTicker(t) {
    const recent = store.get("ace-recent", []).filter((x) => x !== t);
    recent.unshift(t);
    store.set("ace-recent", recent.slice(0, 6));
  }

  // ------------------------------------------------------------------ router

  function parseRoute() {
    const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
    if (parts[0] === "t" && parts[1]) return { name: "evaluate", ticker: parts[1].toUpperCase() };
    if (parts[0] === "models") return { name: "models", target: TARGETS.includes(parts[1]) ? parts[1] : null };
    if (parts[0] === "monitoring") return { name: "monitoring" };
    return { name: "evaluate", ticker: null };
  }

  function onRoute() {
    const route = parseRoute();
    const prev = state.route;
    state.route = route;
    updateNav(route.name);

    // Selecting a model inside the models view is an in-place update, not a new page.
    if (prev && prev.name === "models" && route.name === "models" && $("#model-detail")) {
      selectModel(route.target || state.modelTarget);
      return;
    }

    if (state.controller) state.controller.abort();
    state.controller = new AbortController();
    const token = ++state.renderToken;

    const run = () => {
      const view = $("#view");
      window.scrollTo({ top: 0, behavior: "instant" });
      if (route.name === "models") renderModels(view, token);
      else if (route.name === "monitoring") renderMonitoring(view, token);
      else if (route.ticker) renderTicker(view, route.ticker, token);
      else renderHome(view, token);
    };
    if (prev && document.startViewTransition && !reducedMotion.matches && document.visibilityState === "visible") {
      // A transition interrupted by the next navigation rejects; the content still renders.
      const vt = document.startViewTransition(run);
      vt.ready.catch(() => {});
      vt.finished.catch(() => {});
      if (vt.updateCallbackDone) vt.updateCallbackDone.catch(() => {});
    } else run();
  }

  const stale = (token) => token !== state.renderToken;

  // ------------------------------------------------------------------ home

  async function renderHome(view, token) {
    document.title = "Company Evaluator";
    const recent = store.get("ace-recent", []);
    const picks = recent.length ? recent : ["AAPL", "NVDA", "JPM", "XOM", "TSLA", "UNH"];
    view.innerHTML = `
      <section class="empty">
        <div class="empty-mark" aria-hidden="true">${ICON.trend}</div>
        <h1>How likely is a large move?</h1>
        <p>Pick a company to see the probability of an outsized price move over 1, 5 and 20 days, what drives it,
           and how similar situations played out. Every score is shown next to the base rate it has to beat.</p>
        <div class="quick" aria-label="${recent.length ? "Recent" : "Suggested"} tickers">
          ${picks.map((t) => `<a class="btn" href="#/t/${esc(t)}">${esc(t)}</a>`).join("")}
        </div>
        <p class="xs muted">Press <span class="kbd">/</span> to search from anywhere.</p>
      </section>
      <section class="stats" id="home-stats">
        ${[0, 1, 2, 3].map(() => '<div class="stat"><div class="skel skel-line" style="width:60%"></div><div class="skel" style="height:28px;margin-top:10px;width:45%"></div></div>').join("")}
      </section>`;
    animateIn(view);

    const [u, v] = await Promise.all([loadUniverse(), loadValidation()]);
    if (stale(token)) return;
    const best = Object.entries(v || {}).map(([t, r]) => [t, r.validation.pooled_out_of_sample.brier_skill]).sort((a, b) => b[1] - a[1])[0];
    const folds = v && v.magnitude_1d ? v.magnitude_1d.validation.folds.length : null;
    const el = $("#home-stats");
    el.innerHTML = `
      <div class="stat"><span class="eyebrow">Universe</span>${num(u ? u.count : null, "int")}<span class="xs muted">S&amp;P 500 names</span></div>
      <div class="stat"><span class="eyebrow">Models</span>${num(Object.keys(v || {}).length, "int")}<span class="xs muted">direction &amp; magnitude × 3 horizons</span></div>
      <div class="stat"><span class="eyebrow">Best Brier skill</span>${num(best ? best[1] : null, "s4", best ? toneOf(best[1]) : "")}<span class="xs muted">${best ? esc(targetLabel(best[0])) + " vs base rate" : "no models loaded"}</span></div>
      <div class="stat"><span class="eyebrow">Walk-forward folds</span>${num(folds, "int")}<span class="xs muted">purged, out of sample</span></div>`;
    animateIn(el);
  }

  // ------------------------------------------------------------------ ticker

  function skeletonTicker(ticker) {
    const card = (span, h) => `<div class="card ${span}"><div class="skel skel-line" style="width:35%"></div><div class="skel" style="height:${h}px;margin-top:16px"></div></div>`;
    return `
      <section class="card hero span-12">
        <div class="hero-top">
          <div class="hero-id"><div class="hero-ticker"><h1>${esc(ticker)}</h1></div><div class="skel skel-line" style="width:180px"></div></div>
          <div class="hero-price"><div class="skel" style="height:34px;width:140px"></div></div>
        </div>
        <div class="skel" style="height:220px;margin-top:18px;border-radius:12px"></div>
      </section>
      <div class="grid grid-12">${card("span-7", 190)}${card("span-5", 190)}${card("span-7", 240)}${card("span-5", 240)}</div>`;
  }

  async function renderTicker(view, ticker, token) {
    document.title = `${ticker} · Company Evaluator`;
    state.ticker = ticker;
    view.innerHTML = skeletonTicker(ticker);
    animateIn(view);
    const signal = state.controller.signal;

    const payloadP = api(`/payload/${encodeURIComponent(ticker)}?sentiment=false`, { signal });
    const pricesP = api(`/prices/${encodeURIComponent(ticker)}?days=1260`, { signal, quiet: true }).catch(() => null);
    loadUniverse();
    loadValidation().then(() => { if (!stale(token)) fillRegimeWarnings(view); });

    let payload;
    try {
      payload = await payloadP;
    } catch (err) {
      if (err.name === "AbortError" || stale(token)) return;
      view.innerHTML = errorCard(`Couldn’t evaluate ${ticker}`, err, `#/t/${ticker}`);
      animateIn(view);
      return;
    }
    const prices = await pricesP;
    await loadUniverse();
    if (stale(token)) return;

    rememberTicker(ticker);
    const meta = state.byTicker.get(ticker);
    const targets = (payload.gbm && payload.gbm.targets) || {};

    view.innerHTML = `
      ${heroCard(payload, meta, prices)}
      ${verdictBanner(payload.model_quality)}
      <div class="grid grid-12">
        ${riskCard(payload, targets)}
        ${directionCard(targets)}
        ${driversCard(targets)}
        ${sentimentCard(ticker)}
        ${analogsCard(payload.historical_analogs)}
        ${trackCard(payload.feedback_context)}
        ${evalCard(ticker)}
        ${caveatsCard(payload.data_caveats)}
      </div>`;
    // Stagger the grid's cards as well as the top-level blocks.
    animateIn(view);
    const grid = $(".grid", view);
    Array.from(grid.children).forEach((el, i) => { el.style.animation = `rise var(--t-slow) var(--ease-out) both`; el.style.animationDelay = `${120 + i * 55}ms`; });

    if (prices) mountPriceChart(view, prices);
    wireTicker(view, ticker, targets);
    fillRegimeWarnings(view);
    const cached = state.sentiment.get(ticker);
    if (cached) showSentiment($("#sentiment-body", view), cached);
  }

  function heroCard(p, meta, prices) {
    return `
      <section class="card hero span-12">
        <div class="hero-top">
          <div class="hero-id">
            <div class="hero-ticker"><h1>${esc(p.ticker)}</h1><span class="hero-name">${esc(meta ? meta.name : "Outside the trained universe")}</span></div>
            <div class="hero-meta">
              ${meta ? `<span class="chip">${esc(meta.sector)}</span>` : '<span class="chip chip-warn">Not in S&amp;P 500 universe</span>'}
              <span class="chip">As of ${esc(p.as_of)}</span>
              ${isNum(p.label_scale) ? `<span class="chip" title="Trailing daily volatility: the yardstick a 'large move' is measured against">1σ daily ≈ ${fmt(p.label_scale, "pct1")}</span>` : ""}
            </div>
          </div>
          <div class="hero-price">
            <div class="eyebrow">Last close</div>
            <div class="price">${num(p.last_close, "money")}</div>
            <div class="change" id="range-change"></div>
          </div>
        </div>
        ${prices ? `
          <div class="chart-wrap" id="chart-host"></div>
          <div class="chart-toolbar">
            ${seg([[21, "1M"], [63, "3M"], [126, "6M"], [252, "1Y"], [1260, "5Y"]], state.range, "data-range", "Chart range")}
            <span class="xs muted">Daily closes, split &amp; dividend adjusted · context only, not a model input here</span>
          </div>` : '<p class="small muted" style="margin-top:14px">Price history unavailable.</p>'}
      </section>`;
  }

  function verdictBanner(q) {
    if (!q) return "";
    const good = !!q.any_target_has_skill;
    return `
      <section class="verdict ${good ? "good" : "bad"}" role="note">
        ${good ? ICON.check : ICON.alert}
        <div>
          <strong>${good ? `Best model: ${esc(targetLabel(q.best_target || ""))}` : "No model beats the base rate"}</strong>
          <span class="muted">${esc(q.interpretation || "")}</span>
          ${good ? "" : '<div class="small" style="margin-top:6px">These outputs must not be read as a signal.</div>'}
        </div>
      </section>`;
  }

  function riskCard(p, targets) {
    const R = 50;
    const C = 2 * Math.PI * R;
    const rings = HORIZONS.map((h) => {
      const t = targets[`magnitude_${h}d`];
      if (!t) return `<div class="ring-card"><div class="muted small">${h}-day model not trained</div></div>`;
      const prob = t.probabilities.LARGE_MOVE;
      const base = t.baseline_probabilities.LARGE_MOVE;
      const lift = t.lift_over_baseline.LARGE_MOVE;
      const ang = 2 * Math.PI * base;
      const tick = (r) => `${(60 + r * Math.cos(ang)).toFixed(2)},${(60 + r * Math.sin(ang)).toFixed(2)}`;
      const threshold = isNum(p.label_scale) ? p.label_scale * Math.sqrt(h) * p.threshold_sigmas : null;
      const [chipCls, chipText] =
        lift >= 1.15 ? ["chip-warn", "Elevated"] : lift <= 0.87 ? ["chip-pos", "Calmer"] : ["", "Typical"];
      const pressed = state.driverKind === "magnitude" && state.driverH === h;
      return `
        <button type="button" class="ring-card" data-ring="${h}" aria-pressed="${pressed}"
                aria-label="${h}-day large-move probability ${fmt(prob, "pct0")}, base rate ${fmt(base, "pct0")}. Show drivers.">
          <div class="ring">
            <svg viewBox="0 0 120 120" aria-hidden="true">
              <circle class="track" cx="60" cy="60" r="${R}" fill="none" stroke-width="10"/>
              <circle class="value" cx="60" cy="60" r="${R}" fill="none" stroke-width="10"
                      stroke-dasharray="${C.toFixed(2)}" stroke-dashoffset="${C.toFixed(2)}" data-offset="${(C * (1 - prob)).toFixed(2)}"/>
              <line class="base" x1="${tick(42).split(",")[0]}" y1="${tick(42).split(",")[1]}" x2="${tick(58).split(",")[0]}" y2="${tick(58).split(",")[1]}" stroke-linecap="round"/>
            </svg>
            <div class="ring-label">${num(prob, "pct0")}<span class="xs muted">odds</span></div>
          </div>
          <div>
            <div class="ring-h">${h} day${h > 1 ? "s" : ""}</div>
            <div class="xs muted">${threshold ? `move &gt; ${fmt(threshold, "pct1")}` : "large move"} · base ${fmt(base, "pct0")}</div>
            <div style="margin-top:8px;display:flex;gap:6px;justify-content:center;flex-wrap:wrap">
              <span class="chip ${chipCls}" title="Model probability divided by the historical base rate: ${fmt(lift, "lift")} the usual odds">${fmt(lift, "lift")} · ${chipText}</span>
            </div>
          </div>
        </button>`;
    }).join("");
    return `
      <section class="card span-7">
        <div class="card-head">
          <div>
            <h2 class="card-title">${ICON.gauge} Large-move risk</h2>
            <p class="card-sub">Probability of a move bigger than ${esc(p.threshold_sigmas)}σ, either direction. The tick on each ring marks the base rate.</p>
          </div>
          <span class="chip chip-accent">Magnitude models</span>
        </div>
        <div class="rings">${rings}</div>
      </section>`;
  }

  function directionCard(targets) {
    const rows = HORIZONS.map((h) => {
      const t = targets[`direction_${h}d`];
      if (!t) return "";
      const p = t.probabilities;
      const b = t.baseline_probabilities;
      const part = (cls, key, label) => {
        const w = (p[key] || 0) * 100;
        return `<div class="seg-bar ${cls}" style="width:${w}%;transition-delay:${h === 1 ? 0 : h === 5 ? 90 : 180}ms" title="${label} ${fmt(p[key], "pct1")} (base ${fmt(b[key], "pct1")})">${w >= 11 ? fmt(p[key], "pct0") : ""}</div>`;
      };
      return `
        <div class="dir-row">
          <span class="mono small">${h}d</span>
          <div>
            <div class="stack" role="img" aria-label="${h}-day: drop ${fmt(p.DROP, "pct0")}, neutral ${fmt(p.NEUTRAL, "pct0")}, spike ${fmt(p.SPIKE, "pct0")}">
              ${part("seg-drop", "DROP", "Drop")}${part("seg-neutral", "NEUTRAL", "Neutral")}${part("seg-spike", "SPIKE", "Spike")}
              <i class="base-tick" style="left:calc(${(b.DROP * 100).toFixed(2)}% - 1px)"></i>
              <i class="base-tick" style="left:calc(${((b.DROP + b.NEUTRAL) * 100).toFixed(2)}% - 1px)"></i>
            </div>
            <div class="xs muted" style="margin-top:5px">Predicted: <span class="${t.predicted_class === "DROP" ? "neg" : t.predicted_class === "SPIKE" ? "pos" : ""}">${esc(t.predicted_class)}</span>
              · skill ${fmt(t.skill && t.skill.brier_skill, "s4")}</div>
          </div>
        </div>`;
    }).join("");
    return `
      <section class="card span-5">
        <div class="card-head">
          <div>
            <h2 class="card-title">${ICON.split} Direction</h2>
            <p class="card-sub">Drop / neutral / spike. Ticks mark base rates.</p>
          </div>
          <span data-regime-warn="direction"></span>
        </div>
        ${rows || '<p class="muted small">No direction models trained.</p>'}
        <div class="legend" style="margin-top:12px">
          <span><i style="background:var(--drop)"></i>Drop</span><span><i style="background:var(--neutral)"></i>Neutral</span>
          <span><i style="background:var(--spike)"></i>Spike</span><span><i style="background:var(--text);width:2px"></i>Base rate</span>
        </div>
      </section>`;
  }

  function fillRegimeWarnings(root) {
    if (!state.validation) return;
    $$("[data-regime-warn]", root).forEach((el) => {
      const kind = el.dataset.regimeWarn;
      const bad = new Set();
      TARGETS.filter((t) => t.startsWith(kind)).forEach((t) => negativeRegimes(t).forEach((r) => bad.add(r)));
      el.innerHTML = bad.size
        ? `<span class="chip chip-warn" title="Out-of-sample Brier skill is negative in: ${esc([...bad].map(regimeName).join(", "))}">${ICON.alert} Research only · negative in ${esc([...bad].map(regimeName).join(", "))}</span>`
        : '<span class="chip chip-pos">Positive in every regime</span>';
      el.firstElementChild.style.animation = "pop var(--t-med) var(--ease-spring)";
    });
  }

  function driversCard(targets) {
    return `
      <section class="card span-7" id="drivers-card">
        <div class="card-head">
          <div>
            <h2 class="card-title">${ICON.bars} What drives the score</h2>
            <p class="card-sub" id="drivers-sub"></p>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            ${seg([["magnitude", "Magnitude"], ["direction", "Direction"]], state.driverKind, "data-kind", "Model kind")}
            ${seg(HORIZONS.map((h) => [h, `${h}d`]), state.driverH, "data-h", "Horizon")}
          </div>
        </div>
        <div id="drivers-body">${driversBody(targets)}</div>
        <div class="legend" style="margin-top:10px;justify-content:space-between">
          <span><i style="background:var(--accent)"></i>Lowers the probability</span>
          <span>Raises the probability<i style="background:var(--large)"></i></span>
        </div>
      </section>`;
  }

  function driversBody(targets) {
    const t = targets[`${state.driverKind}_${state.driverH}d`];
    if (!t) return '<p class="muted small">This model is not trained.</p>';
    const feats = t.top_features || [];
    if (!feats.length) return '<p class="muted small">No attributions available.</p>';
    const max = Math.max(...feats.map((f) => Math.abs(f.shap_contribution || 0)), 1e-9);
    return `<div class="drivers">${feats.map((f, i) => {
      const v = f.shap_contribution || 0;
      const w = (Math.abs(v) / max) * 50;
      const up = f.direction ? f.direction === "increases" : v >= 0;
      return `
        <div class="driver" style="--i:${i}">
          <div class="driver-name">${esc(f.description || f.feature)}<br><code>${esc(f.feature)} = ${f.value == null ? "missing" : esc(typeof f.value === "number" ? +f.value.toFixed(4) : f.value)}</code></div>
          <div class="driver-bar" title="SHAP ${signed(v, 4)}">
            <span class="fill ${up ? "up" : "down"}" style="width:${w}%;transition-delay:${i * 60}ms"></span>
          </div>
        </div>`;
    }).join("")}</div>`;
  }

  function driversSub(targets) {
    const t = targets[`${state.driverKind}_${state.driverH}d`];
    if (!t) return "";
    return `TreeSHAP contributions toward <strong>${esc(t.predicted_class)}</strong> for the ${state.driverH}-day ${state.driverKind} model.`;
  }

  function sentimentCard(ticker) {
    return `
      <section class="card span-5" id="sentiment-card">
        <div class="card-head">
          <div>
            <h2 class="card-title">${ICON.news} News sentiment</h2>
            <p class="card-sub">FinBERT over recent headlines and 8-K filings, recency weighted.</p>
          </div>
        </div>
        <div id="sentiment-body">
          <p class="small muted">Not part of the model scores above: free news feeds have no history to train on.
            Scoring can take a couple of minutes the first time.</p>
          <button class="btn" type="button" id="sentiment-run" style="margin-top:14px">${ICON.spark} Score headlines for ${esc(ticker)}</button>
        </div>
      </section>`;
  }

  function showSentiment(body, s) {
    if (!s.available) {
      body.innerHTML = `
        <div class="verdict bad" style="padding:12px 14px">${ICON.alert}<div><strong>No usable sentiment</strong><span class="muted small">${esc(s.reason || "")}</span></div></div>
        <dl class="kv" style="margin-top:14px"><dt>Articles</dt><dd class="num">${esc(s.article_count ?? 0)}</dd>
        ${isNum(s.newest_age_hours) ? `<dt>Newest</dt><dd class="num">${esc(ago(s.newest_age_hours))}</dd>` : ""}</dl>`;
      animateIn(body);
      return;
    }
    const score = s.sentiment_score;
    const tone = score >= 0.15 ? "pos" : score <= -0.15 ? "neg" : "";
    body.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px">
        <div><div class="eyebrow">Score</div><div style="font-size:28px;font-weight:600">${num(score, "s3", tone)}</div></div>
        <dl class="kv small"><dt>Confidence</dt><dd>${num(s.sentiment_confidence, "pct0")}</dd><dt>Articles</dt><dd>${num(s.article_count, "int")}</dd></dl>
      </div>
      <div class="senti-scale" aria-hidden="true"><span class="senti-knob" data-left="${(((score + 1) / 2) * 100).toFixed(1)}%"></span></div>
      <div class="legend" style="justify-content:space-between"><span>Negative</span><span>Neutral</span><span>Positive</span></div>
      <ul class="headlines">${(s.top_headlines || []).map((h, i) => `
        <li style="--i:${i}">
          <span class="pdot" style="background:${h.polarity > 0.15 ? "var(--pos)" : h.polarity < -0.15 ? "var(--neg)" : "var(--neutral)"}" title="Polarity ${esc(h.polarity)}"></span>
          <div>${esc(h.title)}<div class="xs muted">${esc(h.source)} · ${esc(h.kind || "")} · ${esc(timeAgo(h.published))}</div></div>
        </li>`).join("")}</ul>`;
    animateIn(body, { stagger: false });
  }

  const ago = (hours) => (hours < 1 ? "under an hour" : hours < 48 ? `${Math.round(hours)}h ago` : `${Math.round(hours / 24)}d ago`);
  const timeAgo = (iso) => {
    const t = Date.parse(iso);
    return Number.isNaN(t) ? "" : ago((Date.now() - t) / 3.6e6);
  };

  function analogsCard(a) {
    let inner;
    if (!a || !a.available || !(a.matches || []).length) {
      inner = `<p class="muted small">${esc((a && (a.reason || a.note)) || "No analog index available. Run scripts.build_analogs.")}</p>`;
    } else {
      const dmin = Math.min(...a.matches.map((m) => m.distance));
      inner = `
        <div class="table-wrap"><table>
          <thead><tr><th>Company</th><th>Date</th><th>Similarity</th><th class="r">5d</th><th class="r">20d</th><th>Outcome</th></tr></thead>
          <tbody>${a.matches.map((m, i) => `
            <tr style="--i:${i}">
              <td><a class="mono" href="#/t/${esc(m.ticker)}">${esc(m.ticker)}</a></td>
              <td class="mono small">${esc(m.date)}</td>
              <td><span class="sim" title="Feature distance ${esc(m.distance.toFixed(3))} (lower is closer)"><span class="sim-bar"><i style="width:${((dmin / m.distance) * 100).toFixed(0)}%"></i></span></span></td>
              <td class="r mono ${toneOf(m.forward_return_5d, 0.0005)}">${fmt(m.forward_return_5d, "spct1")}</td>
              <td class="r mono ${toneOf(m.forward_return_20d, 0.0005)}">${fmt(m.forward_return_20d, "spct1")}</td>
              <td><span class="chip ${m.outcome_5d === "DROP" ? "chip-neg" : m.outcome_5d === "SPIKE" ? "chip-pos" : ""}">${esc(m.outcome_5d || "—")}</span></td>
            </tr>`).join("")}</tbody>
        </table></div>
        <p class="xs muted" style="margin-top:12px">${esc(a.note || "")}</p>`;
    }
    return `
      <section class="card span-7">
        <div class="card-head"><div><h2 class="card-title">${ICON.history} Historical analogs</h2>
          <p class="card-sub">Closest past situations by feature similarity, and what happened next.</p></div></div>
        ${inner}
      </section>`;
  }

  function trackCard(fc) {
    fc = fc || {};
    const extra = Object.entries(fc).filter(([k, v]) => !["resolved_predictions", "open_predictions", "note"].includes(k) && (isNum(v) || typeof v === "string"));
    return `
      <section class="card span-5">
        <div class="card-head"><div><h2 class="card-title">${ICON.target} Track record</h2>
          <p class="card-sub">How past predictions for this ticker turned out.</p></div></div>
        <div class="stats" style="grid-template-columns:1fr 1fr">
          <div class="stat"><span class="eyebrow">Resolved</span>${num(fc.resolved_predictions ?? 0, "int")}</div>
          <div class="stat"><span class="eyebrow">Open</span>${num(fc.open_predictions ?? 0, "int")}</div>
        </div>
        ${extra.length ? `<dl class="kv" style="margin-top:14px">${extra.map(([k, v]) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd class="num">${esc(isNum(v) ? +v.toFixed(4) : v)}</dd>`).join("")}</dl>` : ""}
        ${fc.note ? `<p class="small muted" style="margin-top:14px">${esc(fc.note)}</p>` : ""}
      </section>`;
  }

  function evalCard(ticker) {
    return `
      <section class="card span-12" id="eval-card">
        <div class="card-head">
          <div><h2 class="card-title">${ICON.spark} Written evaluation</h2>
            <p class="card-sub">An LLM reads everything above and writes it up. It is instructed to defer to the skill verdict, never to soften it.</p></div>
          <span id="eval-meta"></span>
        </div>
        <div id="eval-body">
          <div class="eval-cta">
            <button class="btn btn-primary" type="button" id="eval-run">${ICON.spark} Write evaluation for ${esc(ticker)}</button>
            <label class="check"><input type="checkbox" id="eval-senti"> Include news sentiment (slower)</label>
          </div>
          <p class="xs muted" style="margin-top:8px">Calls the configured LLM provider and logs this prediction so it can be scored once its window closes.</p>
        </div>
      </section>`;
  }

  function caveatsCard(caveats) {
    if (!caveats || !caveats.length) return "";
    return `
      <section class="card span-12">
        <details class="caveats">
          <summary><span style="display:flex;align-items:center;gap:8px"><span class="warn" style="display:inline-flex;width:17px">${ICON.alert}</span> Data caveats (${caveats.length})</span>${ICON.chevron}</summary>
          <ul>${caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>
        </details>
      </section>`;
  }

  /** Minimal, escape-first Markdown: paragraphs, bullets, **bold**, *italic*, # headings. */
  function renderMarkdown(src) {
    const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, "$1<em>$2</em>").replace(/`([^`]+)`/g, "<code class=\"mono\">$1</code>");
    const blocks = [];
    let list = null;
    let para = [];
    const flush = () => {
      if (para.length) { blocks.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
      if (list) { blocks.push(`<ul>${list.map((li) => `<li>${inline(li)}</li>`).join("")}</ul>`); list = null; }
    };
    for (const raw of String(src).split(/\r?\n/)) {
      const line = raw.trim();
      if (!line) { flush(); continue; }
      const bullet = line.match(/^[-*•]\s+(.*)$/);
      const heading = line.match(/^#{1,4}\s+(.*)$/);
      if (heading) { flush(); blocks.push(`<p><strong>${inline(heading[1])}</strong></p>`); }
      else if (bullet) { if (para.length) { blocks.push(`<p>${inline(para.join(" "))}</p>`); para = []; } (list = list || []).push(bullet[1]); }
      else { if (list) flush(); para.push(line); }
    }
    flush();
    return blocks.map((b, i) => b.replace(/^<(\w+)/, `<$1 style="--i:${i}"`)).join("");
  }

  function wireTicker(view, ticker, targets) {
    // Chart range
    const rangeSeg = $("[data-range]", view)?.closest(".seg");
    if (rangeSeg) {
      rangeSeg.addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (!b) return;
        state.range = parseInt(b.dataset.range, 10);
        store.set("ace-range", state.range);
        pressSeg(rangeSeg, b);
        drawPriceChart(true);
      });
    }

    // Drivers: kind + horizon segments, and the ring cards as shortcuts.
    const card = $("#drivers-card", view);
    const refreshDrivers = () => {
      $("#drivers-body", card).innerHTML = driversBody(targets);
      $("#drivers-sub", card).innerHTML = driversSub(targets);
      $$("[data-ring]", view).forEach((r) => r.setAttribute("aria-pressed", String(state.driverKind === "magnitude" && +r.dataset.ring === state.driverH)));
      $$("[data-kind]", card).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.kind === state.driverKind)));
      $$("[data-h]", card).forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.h === state.driverH)));
      $$(".seg", card).forEach(syncSeg);
      requestAnimationFrame(() => requestAnimationFrame(() => $$(".drivers", card).forEach((d) => d.classList.add("grown"))));
    };
    $("#drivers-sub", card).innerHTML = driversSub(targets);
    card.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-kind], button[data-h]");
      if (!b) return;
      if (b.dataset.kind) state.driverKind = b.dataset.kind;
      if (b.dataset.h) state.driverH = parseInt(b.dataset.h, 10);
      refreshDrivers();
    });
    $$("[data-ring]", view).forEach((r) => r.addEventListener("click", () => {
      state.driverKind = "magnitude";
      state.driverH = parseInt(r.dataset.ring, 10);
      refreshDrivers();
      card.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "nearest" });
    }));

    // Sentiment, on demand.
    const sbody = $("#sentiment-body", view);
    sbody.addEventListener("click", async (e) => {
      if (!e.target.closest("#sentiment-run")) return;
      const ctl = new AbortController();
      const outer = state.controller;
      outer.signal.addEventListener("abort", () => ctl.abort(), { once: true });
      sbody.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;gap:12px">
          <span class="small muted">Scoring headlines with FinBERT… <span class="mono" id="senti-timer">0s</span></span>
          <button class="btn btn-ghost" type="button" id="senti-cancel">Cancel</button>
        </div>
        <div class="skel" style="height:10px;margin:18px 0"></div>
        ${[82, 64, 74, 58].map((w) => `<div class="skel skel-line" style="width:${w}%"></div>`).join("")}`;
      const stop = elapsedTimer($("#senti-timer", sbody));
      $("#senti-cancel", sbody).addEventListener("click", () => ctl.abort());
      try {
        const s = await api(`/sentiment/${encodeURIComponent(ticker)}`, { signal: ctl.signal });
        state.sentiment.set(ticker, s);
        if (sbody.isConnected) showSentiment(sbody, s);
      } catch (err) {
        if (!sbody.isConnected) return;
        sbody.innerHTML = `<p class="small ${err.name === "AbortError" ? "muted" : "neg"}">${err.name === "AbortError" ? "Cancelled." : esc(err.message)}</p>
          <button class="btn" type="button" id="sentiment-run" style="margin-top:12px">${ICON.refresh} Try again</button>`;
      } finally {
        stop();
      }
    });

    // Written evaluation.
    const ebody = $("#eval-body", view);
    ebody.addEventListener("click", async (e) => {
      if (!e.target.closest("#eval-run")) return;
      const withSenti = $("#eval-senti", ebody)?.checked;
      ebody.innerHTML = `
        <p class="small muted">Writing the evaluation… <span class="mono" id="eval-timer">0s</span></p>
        ${[96, 88, 92, 70, 84, 60].map((w) => `<div class="skel skel-line" style="width:${w}%;height:13px;margin:12px 0"></div>`).join("")}`;
      const stop = elapsedTimer($("#eval-timer", ebody));
      try {
        const r = await api(`/evaluate/${encodeURIComponent(ticker)}?sentiment=${withSenti ? "true" : "false"}`, { signal: state.controller.signal });
        if (!ebody.isConnected) return;
        $("#eval-meta", view).innerHTML = `<span class="chip">${esc(r.provider || "")} · ${esc(r.model || "")}</span>`;
        if (r.refused || !r.evaluation) {
          ebody.innerHTML = `<div class="verdict bad">${ICON.alert}<div><strong>No evaluation returned</strong><span class="muted">${esc(r.refusal_reason || "The model declined to answer.")}</span></div></div>`;
        } else {
          ebody.innerHTML = `<div class="eval-body">${renderMarkdown(r.evaluation)}</div>
            ${r.prediction_id != null ? `<p class="xs muted" style="margin-top:14px">Logged as prediction #${esc(r.prediction_id)}. It will be scored once its window closes.</p>` : ""}`;
        }
        toast(`${ICON.check.replace("<svg", '<svg style="width:15px;height:15px;vertical-align:-3px;color:var(--pos)"')} Evaluation ready for ${esc(ticker)}`);
      } catch (err) {
        if (err.name === "AbortError" || !ebody.isConnected) return;
        ebody.innerHTML = `<div class="verdict bad">${ICON.alert}<div><strong>Evaluation failed</strong><span class="muted">${esc(err.message)}</span></div></div>
          <div class="eval-cta" style="margin-top:14px"><button class="btn" type="button" id="eval-run">${ICON.refresh} Try again</button>
          <label class="check"><input type="checkbox" id="eval-senti" ${withSenti ? "checked" : ""}> Include news sentiment (slower)</label></div>`;
      } finally {
        stop();
      }
    });
  }

  // ------------------------------------------------------------------ price chart

  let chartData = null;
  let chartObserver = null;

  function mountPriceChart(view, prices) {
    chartData = prices;
    const host = $("#chart-host", view);
    if (!host) return;
    drawPriceChart(true);
    if (chartObserver) chartObserver.disconnect();
    let lastW = host.clientWidth;
    let raf = 0;
    chartObserver = new ResizeObserver(() => {
      if (Math.abs(host.clientWidth - lastW) < 2) return;
      lastW = host.clientWidth;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => drawPriceChart(false));
    });
    chartObserver.observe(host);
  }

  function drawPriceChart(animate) {
    const host = $("#chart-host");
    if (!host || !chartData) return;
    const n0 = chartData.close.length;
    const from = Math.max(0, n0 - state.range);
    const close = chartData.close.slice(from);
    const dates = chartData.dates.slice(from);
    const n = close.length;
    if (n < 2) { host.innerHTML = '<p class="small muted">Not enough history.</p>'; return; }

    const w = Math.max(280, host.clientWidth);
    const h = w < 520 ? 180 : 220;
    const padR = 58, padT = 10, padB = 24;
    let lo = Math.min(...close), hi = Math.max(...close);
    const pad = (hi - lo) * 0.08 || hi * 0.02 || 1;
    lo -= pad; hi += pad;
    const x = (i) => (i / (n - 1)) * (w - padR);
    const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (h - padT - padB);
    const d = close.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    const up = close[n - 1] >= close[0];
    const col = up ? "var(--pos)" : "var(--neg)";

    const change = close[n - 1] / close[0] - 1;
    const label = { 21: "1M", 63: "3M", 126: "6M", 252: "1Y", 1260: "5Y" }[state.range] || "";
    const changeEl = $("#range-change");
    if (changeEl) {
      changeEl.innerHTML = `<span class="${up ? "pos" : "neg"}">${num(change, "spct2")}</span> <span class="muted">${label}</span>`;
      if (animate) $$("[data-to]", changeEl).forEach(countUp);
    }

    const grid = [0.15, 0.5, 0.85].map((f) => {
      const v = hi - (hi - lo) * f;
      const yy = y(v);
      return `<line class="grid-line" x1="0" x2="${w - padR}" y1="${yy.toFixed(1)}" y2="${yy.toFixed(1)}"/>
        <text class="axis" x="${w - padR + 8}" y="${(yy + 3.5).toFixed(1)}">${v.toFixed(v >= 1000 ? 0 : 2)}</text>`;
    }).join("");
    const short = state.range <= 126;
    const fmtDate = (s) => {
      const dt = new Date(s + "T00:00:00");
      return short ? dt.toLocaleDateString(undefined, { month: "short", day: "numeric" }) : dt.toLocaleDateString(undefined, { month: "short", year: "2-digit" });
    };
    const xt = [0, Math.round((n - 1) / 3), Math.round((2 * (n - 1)) / 3), n - 1].map((i, k) =>
      `<text class="axis" x="${x(i).toFixed(1)}" y="${h - 6}" text-anchor="${k === 0 ? "start" : k === 3 ? "end" : "middle"}">${esc(fmtDate(dates[i]))}</text>`).join("");

    host.innerHTML = `
      <svg class="price-chart" viewBox="0 0 ${w} ${h}" style="height:${h}px" role="img"
           aria-label="${esc(state.ticker || "")} closing price, ${label}: ${fmt(change, "spct2")}">
        <defs><linearGradient id="price-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" style="stop-color:${col};stop-opacity:0.28"/><stop offset="1" style="stop-color:${col};stop-opacity:0"/>
        </linearGradient></defs>
        ${grid}${xt}
        <path class="${animate ? "area" : ""}" d="${d}L${x(n - 1).toFixed(1)},${h - padB}L0,${h - padB}Z" fill="url(#price-fill)"/>
        <path class="line ${animate ? "draw" : ""}" d="${d}" style="stroke:${col}"/>
        <g class="hover" style="opacity:0;transition:opacity var(--t-fast)">
          <line class="crosshair" y1="${padT}" y2="${h - padB}"/>
          <circle class="cross-dot" r="5" style="fill:${col}"/>
        </g>
      </svg>
      <div class="tooltip" style="opacity:0"></div>`;

    const svg = $("svg", host);
    const line = $(".line", svg);
    if (animate) line.style.setProperty("--len", line.getTotalLength().toFixed(1));
    const hover = $(".hover", svg);
    const tip = $(".tooltip", host);
    const cross = $(".crosshair", hover);
    const dot = $(".cross-dot", hover);

    const show = (clientX) => {
      const rect = svg.getBoundingClientRect();
      const px = ((clientX - rect.left) / rect.width) * w;
      const i = Math.max(0, Math.min(n - 1, Math.round((px / (w - padR)) * (n - 1))));
      const cx = x(i), cy = y(close[i]);
      cross.setAttribute("x1", cx); cross.setAttribute("x2", cx);
      dot.setAttribute("cx", cx); dot.setAttribute("cy", cy);
      hover.style.opacity = "1";
      const rel = close[i] / close[0] - 1;
      tip.innerHTML = `<div class="mono" style="font-weight:600">${FMT.money(close[i])}</div>
        <div class="xs muted">${esc(new Date(dates[i] + "T00:00:00").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: "numeric" }))}
        · <span class="${rel >= 0 ? "pos" : "neg"}">${FMT.spct2(rel)}</span></div>`;
      const scale = rect.width / w;
      const tw = tip.offsetWidth;
      const left = Math.max(0, Math.min(host.clientWidth - tw, cx * scale - tw / 2));
      const top = Math.max(-8, cy * scale - 62);
      tip.style.transform = `translate(${left}px, ${top}px)`;
      tip.style.opacity = "1";
    };
    const hide = () => { hover.style.opacity = "0"; tip.style.opacity = "0"; };
    svg.addEventListener("pointermove", (e) => show(e.clientX));
    svg.addEventListener("pointerdown", (e) => show(e.clientX));
    svg.addEventListener("pointerleave", hide);
  }

  // ------------------------------------------------------------------ models

  async function renderModels(view, token) {
    document.title = "Models · Company Evaluator";
    view.innerHTML = `
      <header><h1 style="font-size:26px">Models</h1><p class="muted" style="margin-top:4px">Loading validation reports…</p></header>
      <div class="model-cards">${TARGETS.map(() => '<div class="card"><div class="skel skel-line" style="width:50%"></div><div class="skel" style="height:34px;width:40%;margin:12px 0"></div><div class="skel" style="height:42px"></div></div>').join("")}</div>`;
    animateIn(view);

    const v = await loadValidation();
    if (stale(token)) return;
    const present = TARGETS.filter((t) => v[t]);
    if (!present.length) {
      view.innerHTML = errorCard("No validation reports", new Error("No trained models found. Run: python -m scripts.train"), "#/models");
      animateIn(view);
      return;
    }
    const target = state.route.target && v[state.route.target] ? state.route.target : v[state.modelTarget] ? state.modelTarget : present[0];
    state.modelTarget = target;
    const first = v[present[0]];
    const folds = first.validation.folds;
    const span = folds.length ? `${folds[0].test_start.slice(0, 4)}–${folds[folds.length - 1].test_end.slice(0, 4)}` : "";

    view.innerHTML = `
      <header>
        <h1 style="font-size:26px">Models</h1>
        <p class="muted" style="margin-top:4px">${folds.length} purged walk-forward folds · tested ${esc(span)} ·
          every metric is out of sample and measured against a base-rate baseline fitted on that fold’s own training window.</p>
      </header>
      <div class="model-cards">${present.map((t) => modelCard(t, v[t])).join("")}</div>
      ${heatmapCard(present, v)}
      <div class="grid grid-12" id="model-detail"></div>`;
    animateIn(view);

    $(".model-cards", view).addEventListener("click", (e) => {
      const c = e.target.closest("[data-target]");
      if (c) location.hash = `#/models/${c.dataset.target}`;
    });
    selectModel(target);
  }

  function modelCard(t, r) {
    const pooled = r.validation.pooled_out_of_sample;
    const folds = r.validation.folds || [];
    const neg = folds.filter((f) => f.brier_skill < 0).length;
    const max = Math.max(...folds.map((f) => Math.abs(f.brier_skill)), 1e-9);
    const bars = folds.map((f, i) => {
      const hgt = (Math.abs(f.brier_skill) / max) * 50;
      const pos = f.brier_skill >= 0;
      return `<i class="${pos ? "" : "n"}" style="--i:${i};height:${hgt.toFixed(1)}%;background:${pos ? "var(--pos)" : "var(--neg)"};align-self:${pos ? "flex-end" : "flex-start"};margin-${pos ? "bottom" : "top"}:21px" title="Fold ${f.fold}: ${f.test_start} → ${f.test_end}, skill ${signed(f.brier_skill, 4)}"></i>`;
    }).join("");
    const bad = negativeRegimes(t);
    return `
      <button type="button" class="card hoverable model-card" data-target="${t}" aria-pressed="${t === state.modelTarget}">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
          <span class="eyebrow">${esc(targetLabel(t))}</span>
          ${bad.length ? `<span class="chip chip-warn" title="Negative in ${esc(bad.map(regimeName).join(", "))}">${bad.length} weak regime${bad.length > 1 ? "s" : ""}</span>` : '<span class="chip chip-pos">All regimes +</span>'}
        </div>
        <div class="big">${num(pooled.brier_skill, "s4", toneOf(pooled.brier_skill))}</div>
        <div class="xs muted">Brier skill vs base rate</div>
        <dl class="kv small" style="margin-top:12px">
          <dt>AUC</dt><dd class="num">${fmt(pooled.macro_auc, "f3")}</dd>
          <dt>Precision @ top 5%</dt><dd class="num">${fmt(pooled.precision_at_5pct, "pct1")}</dd>
          <dt>Negative folds</dt><dd class="num ${neg ? "warn" : "pos"}">${neg}/${folds.length}</dd>
          ${overSimple(r)}
        </dl>
        <div class="fold-spark-wrap"><div class="fold-spark" aria-hidden="true">${bars}</div></div>
      </button>`;
  }

  /** Card row: skill over the simple-signals baseline, when it has been computed. */
  function overSimple(r) {
    const b = r.baselines;
    const inc = b && b.incremental && b.incremental[`vs_${b.reference}`];
    if (!inc || !isNum(inc.pooled)) return "";
    return `<dt title="Brier skill over the strongest of three small baselines (volatility; earnings cycle; both plus VIX), same out-of-sample rows. Best baseline here: ${esc(b.reference)}">Over best baseline</dt>
      <dd class="num ${inc.ci90 && inc.ci90[0] > 0 ? "pos" : inc.pooled < 0 ? "neg" : "warn"}">${signed(inc.pooled, 4)}</dd>`;
  }

  function baselineCard(r) {
    const b = r.baselines;
    if (!b) {
      return `<section class="card span-12"><div class="card-head"><div><h2 class="card-title">${ICON.split} Versus simple signals</h2>
        <p class="card-sub">Not computed yet. Run <code class="mono">python -m scripts.evaluate_baselines --targets ${esc(r.target)}</code>
        to see how much of this model\u2019s skill a small model on volatility, earnings timing and VIX already gets.</p></div></div></section>`;
    }
    const ref = b.reference;
    const inc = b.incremental[`vs_${ref}`];
    const rows = [["model", "This model (44 features)"], ["vol", "Volatility only"], ["earnings", "Earnings cycle only"], ["simple", "Vol + earnings + VIX"]];
    const skills = rows.map(([k]) => (b.pooled[k] || {}).brier_skill).filter(isNum);
    const max = Math.max(...skills.map(Math.abs), 1e-9);
    const good = inc.ci90[0] > 0;
    const regimes = REGIMES.filter(([k]) => b.by_regime[k]);
    return `
      <section class="card span-12">
        <div class="card-head"><div><h2 class="card-title">${ICON.split} Versus simple signals</h2>
          <p class="card-sub">Brier skill vs class priors on the identical out-of-sample rows (${(b.n_rows || 0).toLocaleString()} predictions, ${inc.n_folds} folds).</p></div>
          <span class="chip ${good ? "chip-pos" : inc.pooled < 0 ? "chip-neg" : "chip-warn"}">${signed(inc.pooled, 4)} over ${esc(ref)} \u00b7 ${inc.fold_wins}/${inc.n_folds} folds</span>
        </div>
        <div class="drivers">${rows.map(([k, label], i) => {
          const v = (b.pooled[k] || {}).brier_skill;
          const w = isNum(v) ? (Math.abs(v) / max) * 100 : 0;
          return `<div class="driver" style="--i:${i};grid-template-columns:minmax(0,230px) 1fr 76px">
            <span class="small">${esc(label)}${k === ref ? ' <span class="chip" style="height:20px;margin-left:4px">best baseline</span>' : ""}</span>
            <div class="driver-bar plain" style="height:14px"><span class="fill up" style="left:0;top:2px;bottom:2px;width:${w.toFixed(1)}%;background:${k === "model" ? "var(--accent)" : v < 0 ? "var(--neg)" : "var(--accent-2)"}"></span></div>
            <span class="mono small ${toneOf(v)}" style="text-align:right">${fmt(v, "s4")}</span></div>`;
        }).join("")}</div>
        <div class="verdict ${good ? "good" : "bad"}" style="margin-top:16px">${good ? ICON.check : ICON.alert}
          <div><strong>Skill over the best baseline (${esc(ref)}): ${signed(inc.pooled, 4)} <span class="muted small">(90% CI ${signed(inc.ci90[0], 4)} to ${signed(inc.ci90[1], 4)}, resampling folds)</span></strong>
          <span class="muted">${esc(b.verdict)}</span></div></div>
        ${regimes.length ? `<div class="table-wrap" style="margin-top:14px"><table class="heat"><thead><tr><th>By regime</th>${regimes.map(([, n]) => `<th class="c">${esc(n)}</th>`).join("")}</tr></thead>
          <tbody><tr><td class="name">over ${esc(ref)}</td>${regimes.map(([k]) => {
            const x = b.by_regime[k].incremental;
            const a = Math.min(1, Math.abs(x) / 0.04);
            return `<td class="cell" style="background:color-mix(in srgb, ${x >= 0 ? "var(--pos)" : "var(--neg)"} ${(8 + a * 42).toFixed(0)}%, transparent)" title="n=${b.by_regime[k].n.toLocaleString()}">${signed(x, 4)}</td>`;
          }).join("")}</tr></tbody></table></div>` : ""}
      </section>`;
  }

  const ABL_VERDICT = {
    "earns its place": "chip-pos",
    "small, inconsistent gain": "chip-warn",
    hurts: "chip-neg",
    "no measurable effect": "",
    reference: "chip-accent",
  };

  function ablationLabel(name, labels) {
    if (name === "vol") return ["Volatility only", "reference"];
    if (name === "all") return ["All features", "small trees"];
    if (name === "vol+macro-vix_level") return ["+ Macro without VIX level", "vol + macro \u2212 vix"];
    const group = name.replace("vol+", "");
    return [`+ ${labels[group] || group}`, `vol + ${group}`];
  }

  function ablationCard(r) {
    const a = r.ablations;
    if (!a) {
      return `<section class="card span-12"><div class="card-head"><div><h2 class="card-title">${ICON.heat} Which feature groups earn their place</h2>
        <p class="card-sub">Not computed yet. Run <code class="mono">python -m scripts.evaluate_ablations --targets ${esc(r.target)}</code>.</p></div></div></section>`;
    }
    const rows = [...Object.entries(a.sets), ["__model__", a.model]];
    const bounds = rows.flatMap(([, x]) => [x.edge_vs_vol.ci[0], x.edge_vs_vol.ci[1], x.edge_vs_vol.pooled]).filter(isNum);
    const m = Math.max(0.004, ...bounds.map(Math.abs)) * 1.08;
    const pct = (v) => ((v + m) / (2 * m)) * 100;
    const ticks = [-m, -m / 2, 0, m / 2, m];
    const body = rows.map(([name, x], i) => {
      const e = x.edge_vs_vol;
      const isModel = name === "__model__";
      const isRef = name === a.reference;
      const [label, sub] = isModel ? ["Production model", "44 features, deep trees"] : ablationLabel(name, a.group_labels || {});
      const v = e.pooled;
      const cls = isRef ? "flat" : v >= 0 ? "up" : "down";
      const left = v >= 0 ? pct(0) : pct(v);
      const width = Math.max(isRef ? 0.6 : 0, Math.abs(pct(v) - pct(0)));
      const verdict = isModel ? (e.ci[0] > 0 ? "earns its place" : e.ci[1] < 0 ? "hurts" : "no measurable effect") : x.verdict;
      return `
        <div class="abl-row ${isRef ? "ref" : ""} ${isModel ? "prod" : ""}" style="--i:${i}">
          <div class="abl-name">${esc(label)}<small>${esc(sub)}${x.n_features ? ` \u00b7 ${x.n_features}` : ""}</small></div>
          <div class="abl-track" style="--zero:${pct(0).toFixed(2)}%"
               title="Edge over volatility: ${signed(v, 4)} Brier (adjusted ${Math.round((a.ci_level || 0.9) * 100)}% CI ${signed(e.ci[0], 4)} to ${signed(e.ci[1], 4)}), better in ${e.fold_wins}/${e.n_folds} folds">
            <span class="abl-bar ${cls}" style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%;transition-delay:${i * 50}ms"></span>
            ${isRef ? "" : `<span class="abl-whisker" style="left:${pct(e.ci[0]).toFixed(2)}%;width:${Math.max(0.4, pct(e.ci[1]) - pct(e.ci[0])).toFixed(2)}%"></span>`}
          </div>
          <div class="abl-stat">
            <span class="mono small ${isRef ? "muted" : toneOf(v, 0.0005)}">${isRef ? "\u00b10" : signed(v, 4)}${isRef ? "" : ` <span class="muted xs">${e.fold_wins}/${e.n_folds}</span>`}</span>
            <span class="chip ${ABL_VERDICT[verdict] ?? ""}" style="height:20px">${esc(verdict)}</span>
          </div>
        </div>`;
    }).join("");
    const earns = a.groups_that_earn_their_place || [];
    return `
      <section class="card span-12">
        <div class="card-head"><div><h2 class="card-title">${ICON.heat} Which feature groups earn their place</h2>
          <p class="card-sub">Brier edge over volatility alone when each group is added, on the model\u2019s own out-of-sample rows.
            Whiskers are ${Math.round((a.ci_level || 0.9) * 1000) / 10}% intervals (Bonferroni-adjusted across groups); \u201cearns its place\u201d also needs \u2265${signed(a.min_edge ?? 0.002, 3)} and 70% of folds.</p></div></div>
        <div class="abl-axis" aria-hidden="true"><span></span><div class="ticks">${ticks.map((t) => `<span style="left:${pct(t).toFixed(2)}%">${signed(t, 3)}</span>`).join("")}</div><span></span></div>
        <div class="abl">${body}</div>
        <div class="verdict ${earns.length ? "good" : "bad"}" style="margin-top:14px">${earns.length ? ICON.check : ICON.alert}
          <div><strong>${earns.length ? `Groups that earn their place: ${esc(earns.map((g) => (a.group_labels || {})[g] || g).join(", "))}` : "No feature group earns its place over volatility"}</strong>
          <span class="muted">Recommended feature set: ${esc((a.recommended_features || []).length)} features \u2014 <span class="mono xs">${esc((a.recommended_features || []).join(", "))}</span></span></div></div>
      </section>`;
  }

  function heatmapCard(present, v) {
    const cols = REGIMES.filter(([k]) => present.some((t) => v[t].validation.by_regime && v[t].validation.by_regime[k]));
    const cell = (x) => {
      if (!isNum(x)) return '<td class="cell muted">—</td>';
      const a = Math.min(1, Math.abs(x) / 0.08);
      const c = x >= 0 ? "var(--pos)" : "var(--neg)";
      return `<td class="cell" style="background:color-mix(in srgb, ${c} ${(8 + a * 42).toFixed(0)}%, transparent)" title="Brier skill ${signed(x, 4)}">${signed(x, 4)}</td>`;
    };
    return `
      <section class="card">
        <div class="card-head"><div><h2 class="card-title">${ICON.heat} Skill by market regime</h2>
          <p class="card-sub">The table that decides deployability: a model that is positive overall can still lose in one regime.</p></div></div>
        <div class="table-wrap"><table class="heat">
          <thead><tr><th>Model</th>${cols.map(([, n]) => `<th class="c">${esc(n)}</th>`).join("")}</tr></thead>
          <tbody>${present.map((t, i) => `<tr style="--i:${i}"><td class="name"><a href="#/models/${t}">${esc(targetLabel(t))}</a></td>
            ${cols.map(([k]) => cell(v[t].validation.by_regime && v[t].validation.by_regime[k] && v[t].validation.by_regime[k].brier_skill)).join("")}</tr>`).join("")}</tbody>
        </table></div>
      </section>`;
  }

  function selectModel(target) {
    const v = state.validation;
    if (!v || !v[target]) return;
    state.modelTarget = target;
    $$(".model-card").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.target === target)));
    const detail = $("#model-detail");
    if (!detail) return;
    const r = v[target];
    const pooled = r.validation.pooled_out_of_sample;
    const { kind } = targetParts(target);
    const classes = Object.keys(pooled.calibration || {});
    const calibOptions = kind === "magnitude" ? ["LARGE_MOVE"] : classes.filter((c) => c !== "NEUTRAL");
    if (!calibOptions.includes(state.calibClass)) state.calibClass = calibOptions[0];

    detail.innerHTML = `
      <section class="card span-7">
        <div class="card-head"><div><h2 class="card-title">${ICON.bars} ${esc(targetLabel(target))} · skill per fold</h2>
          <p class="card-sub">Each bar is one out-of-sample test year. Below the line means worse than the base rate.</p></div></div>
        ${foldChart(r.validation.folds)}
      </section>
      <section class="card span-5">
        <div class="card-head"><div><h2 class="card-title">${ICON.target} Calibration</h2>
          <p class="card-sub">Predicted vs observed frequency. On the diagonal means probabilities can be taken at face value.</p></div>
          ${calibOptions.length > 1 ? seg(calibOptions.map((c) => [c, c[0] + c.slice(1).toLowerCase()]), state.calibClass, "data-cls", "Class") : ""}
        </div>
        <div id="calib-host">${calibChart(pooled.calibration[state.calibClass] || [])}</div>
      </section>
      ${baselineCard(r)}
      ${ablationCard(r)}
      <section class="card span-12">
        <div class="stats stats-3">
          <div class="stat"><span class="eyebrow">Train window</span><span class="num" style="font-size:16px">${esc(r.train_start)} → ${esc(r.train_end)}</span></div>
          <div class="stat"><span class="eyebrow">Train rows</span>${num(r.train_rows, "int")}</div>
          <div class="stat"><span class="eyebrow">Tickers</span>${num(Array.isArray(r.train_tickers) ? r.train_tickers.length : r.train_tickers, "int")}</div>
          <div class="stat"><span class="eyebrow">Log-loss skill</span>${num(pooled.log_loss_skill, "s4", toneOf(pooled.log_loss_skill))}</div>
          <div class="stat"><span class="eyebrow">Calibration error</span>${num(pooled.calibration_error, "f3")}</div>
          <div class="stat"><span class="eyebrow">Accuracy vs baseline</span><span class="num" style="font-size:16px">${fmt(pooled.accuracy, "pct1")} / ${fmt(pooled.accuracy_baseline, "pct1")}</span><span class="xs muted">not the number that matters</span></div>
        </div>
      </section>`;
    animateIn(detail);
    const calSeg = $("[data-cls]", detail)?.closest(".seg");
    if (calSeg) calSeg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      state.calibClass = b.dataset.cls;
      pressSeg(calSeg, b);
      $("#calib-host", detail).innerHTML = calibChart(pooled.calibration[state.calibClass] || []);
    });
  }

  function foldChart(folds) {
    const W = 640, H = 230, padL = 46, padB = 26, padT = 10;
    const vals = folds.map((f) => f.brier_skill);
    const m = Math.max(...vals.map(Math.abs), 0.01) * 1.15;
    const y = (v) => padT + ((m - v) / (2 * m)) * (H - padT - padB);
    const bw = (W - padL) / folds.length;
    const bars = folds.map((f, i) => {
      const v = f.brier_skill;
      const y0 = y(0), y1 = y(v);
      const pos = v >= 0;
      return `<rect class="bar-anim ${pos ? "" : "n"}" style="--i:${i}" x="${(padL + i * bw + bw * 0.18).toFixed(1)}" y="${Math.min(y0, y1).toFixed(1)}" width="${(bw * 0.64).toFixed(1)}" height="${Math.max(1, Math.abs(y1 - y0)).toFixed(1)}" rx="3" fill="${pos ? "var(--pos)" : "var(--neg)"}">
        <title>Fold ${f.fold}: ${f.test_start} → ${f.test_end}\nBrier skill ${signed(v, 4)} · AUC ${fmt(f.macro_auc, "f3")} · n=${(f.n || 0).toLocaleString()}</title></rect>`;
    }).join("");
    const ticks = [m, m / 2, 0, -m / 2, -m].map((t) => `<line class="grid-line" x1="${padL}" x2="${W}" y1="${y(t).toFixed(1)}" y2="${y(t).toFixed(1)}" ${t === 0 ? 'style="stroke:var(--border-strong)"' : 'stroke-dasharray="3 5"'}/>
      <text class="axis" x="${padL - 8}" y="${(y(t) + 3.5).toFixed(1)}" text-anchor="end">${signed(t, 3)}</text>`).join("");
    const every = folds.length > 10 ? 2 : 1;
    const labels = folds.map((f, i) => (i % every ? "" : `<text class="axis" x="${(padL + i * bw + bw / 2).toFixed(1)}" y="${H - 6}" text-anchor="middle">${esc((f.test_start || "").slice(2, 4) ? "’" + f.test_start.slice(2, 4) : "")}</text>`)).join("");
    return `<svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Brier skill per walk-forward fold">${ticks}${bars}${labels}</svg>`;
  }

  function calibChart(bins) {
    const W = 360, H = 300, pad = 40;
    const usable = bins.filter((b) => b.n > 0);
    if (!usable.length) return '<p class="muted small">No calibration data.</p>';
    const maxV = Math.max(...usable.map((b) => Math.max(b.mean_predicted, b.observed_frequency)));
    const top = Math.min(1, Math.max(0.3, Math.ceil(maxV * 10) / 10));
    const sx = (v) => pad + (v / top) * (W - pad - 10);
    const sy = (v) => H - pad - (v / top) * (H - pad - 10);
    const maxN = Math.max(...usable.map((b) => b.n));
    const grid = [0, top / 2, top].map((t) => `
      <line class="grid-line" x1="${sx(0)}" x2="${sx(top)}" y1="${sy(t)}" y2="${sy(t)}" stroke-dasharray="3 5"/>
      <text class="axis" x="${pad - 8}" y="${sy(t) + 3.5}" text-anchor="end">${(t * 100).toFixed(0)}%</text>
      <text class="axis" x="${sx(t)}" y="${H - pad + 16}" text-anchor="middle">${(t * 100).toFixed(0)}%</text>`).join("");
    const pathD = usable.map((b, i) => `${i ? "L" : "M"}${sx(b.mean_predicted).toFixed(1)},${sy(b.observed_frequency).toFixed(1)}`).join("");
    const pts = usable.map((b, i) => `<circle class="pt-anim" style="--i:${i}" cx="${sx(b.mean_predicted).toFixed(1)}" cy="${sy(b.observed_frequency).toFixed(1)}" r="${(3.5 + 7 * Math.sqrt(b.n / maxN)).toFixed(1)}" fill="var(--accent-2)" fill-opacity="0.85" stroke="var(--surface)" stroke-width="2">
      <title>Predicted ${(b.mean_predicted * 100).toFixed(1)}% → observed ${(b.observed_frequency * 100).toFixed(1)}% (n=${b.n.toLocaleString()})</title></circle>`).join("");
    return `<svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration: predicted versus observed frequency">
      ${grid}
      <line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(top)}" y2="${sy(top)}" stroke="var(--muted)" stroke-dasharray="5 5" opacity="0.6"/>
      <path d="${pathD}" fill="none" stroke="var(--accent-2)" stroke-width="2" opacity="0.55"/>
      ${pts}
      <text class="axis" x="${(W + pad) / 2}" y="${H - 4}" text-anchor="middle">predicted</text>
      <text class="axis" x="12" y="${(H - pad) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(H - pad) / 2})">observed</text>
    </svg><p class="xs muted" style="margin-top:6px">Dot size shows how many predictions fall in each bin.</p>`;
  }

  // ------------------------------------------------------------------ monitoring

  async function renderMonitoring(view, token) {
    document.title = "Monitoring · Company Evaluator";
    view.innerHTML = `
      <header><h1 style="font-size:26px">Monitoring</h1><p class="muted" style="margin-top:4px">Feedback-loop health: are predictions being logged, resolved and explained?</p></header>
      <div class="stats">${[0, 1, 2, 3].map(() => '<div class="stat"><div class="skel skel-line" style="width:60%"></div><div class="skel" style="height:28px;margin-top:10px;width:40%"></div></div>').join("")}</div>`;
    animateIn(view);
    let m;
    try {
      m = await api("/monitoring", { signal: state.controller.signal });
    } catch (err) {
      if (err.name === "AbortError" || stale(token)) return;
      view.innerHTML = errorCard("Monitoring unavailable", err, "#/monitoring");
      animateIn(view);
      return;
    }
    if (stale(token)) return;
    const fl = m.feedback_loop || {};
    const pr = m.predictions || {};
    const tags = m.post_mortem_tags || {};
    const known = new Set(["feedback_loop", "predictions", "post_mortem_tags"]);
    const extra = Object.entries(m).filter(([k]) => !known.has(k));

    const shareBars = (obj, colorOf) => {
      const entries = Object.entries(obj || {});
      if (!entries.length) return '<p class="muted small">Nothing logged yet.</p>';
      return `<div class="drivers">${entries.map(([k, val], i) => `
        <div class="driver" style="--i:${i};grid-template-columns:110px 1fr 56px">
          <span class="mono small">${esc(k)}</span>
          <div class="driver-bar plain" style="height:14px"><span class="fill up" style="left:0;top:2px;bottom:2px;width:${(val * 100).toFixed(1)}%;background:${colorOf(k)}"></span></div>
          <span class="mono small" style="text-align:right">${fmt(val, "pct1")}</span>
        </div>`).join("")}</div>`;
    };
    const classColor = (k) => ({ DROP: "var(--drop)", SPIKE: "var(--spike)", NEUTRAL: "var(--neutral)", LARGE_MOVE: "var(--large)", QUIET: "var(--neutral)" })[k] || "var(--accent)";
    const tagEntries = Object.entries(tags).sort((a, b) => b[1] - a[1]);
    const tagMax = Math.max(1, ...tagEntries.map(([, c]) => (isNum(c) ? c : 0)));

    view.innerHTML = `
      <header style="display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap">
        <div><h1 style="font-size:26px">Monitoring</h1><p class="muted" style="margin-top:4px">Feedback-loop health: are predictions being logged, resolved and explained?</p></div>
        <button class="btn" type="button" id="resolve-btn">${ICON.refresh} Resolve elapsed predictions</button>
      </header>
      <section class="verdict ${fl.healthy ? "good" : "bad"}">${fl.healthy ? ICON.check : ICON.alert}
        <div><strong>${fl.healthy ? "Feedback loop healthy" : "Feedback loop not healthy yet"}</strong>
        <span class="muted">${fl.healthy ? "Predictions are being resolved and tagged." : "Too few predictions have been resolved to measure live skill. Resolve elapsed windows, and keep logging evaluations."}</span></div>
      </section>
      <div class="stats">
        <div class="stat"><span class="eyebrow">Predictions</span>${num(fl.predictions_total ?? 0, "int")}</div>
        <div class="stat"><span class="eyebrow">Resolved</span>${num(fl.predictions_resolved ?? 0, "int")}</div>
        <div class="stat"><span class="eyebrow">Resolution rate</span>${num(fl.resolution_rate, "pct0")}</div>
        <div class="stat"><span class="eyebrow">Post-mortem coverage</span>${num(fl.post_mortem_coverage, "pct0")}</div>
        <div class="stat"><span class="eyebrow">Days to resolve</span>${num(fl.mean_days_to_resolution, "f2")}</div>
      </div>
      <div class="grid grid-12">
        <section class="card span-6"><div class="card-head"><div><h2 class="card-title">${ICON.split} Predicted classes</h2>
          <p class="card-sub">Share of logged predictions${isNum(pr.recent_predictions) ? ` · ${pr.recent_predictions} in the last ${pr.recent_window_days} days` : ""}.</p></div></div>
          ${shareBars(pr.class_share, classColor)}</section>
        <section class="card span-6"><div class="card-head"><div><h2 class="card-title">${ICON.gauge} Mean probabilities</h2>
          <p class="card-sub">Average model output across logged predictions.</p></div></div>
          ${shareBars(pr.mean_probabilities, classColor)}</section>
        <section class="card span-12"><div class="card-head"><div><h2 class="card-title">${ICON.tag} Post-mortem tags</h2>
          <p class="card-sub">Why resolved predictions went wrong, tagged by rules first and an LLM second.</p></div></div>
          ${tagEntries.length ? `<div class="drivers">${tagEntries.map(([k, c], i) => `
            <div class="driver" style="--i:${i};grid-template-columns:minmax(0,220px) 1fr 48px">
              <span class="small">${esc(k.replace(/_/g, " "))}</span>
              <div class="driver-bar plain" style="height:14px"><span class="fill up" style="left:0;top:2px;bottom:2px;width:${((isNum(c) ? c : 0) / tagMax * 100).toFixed(1)}%;background:var(--accent-2)"></span></div>
              <span class="mono small" style="text-align:right">${esc(c)}</span></div>`).join("")}</div>`
            : '<p class="muted small">No post-mortems yet. They appear once predictions resolve.</p>'}
        </section>
        ${extra.map(([k, val]) => `<section class="card span-6"><h2 class="card-title" style="margin-bottom:12px">${esc(k.replace(/_/g, " "))}</h2>${kvBlock(val)}</section>`).join("")}
      </div>`;
    animateIn(view);
    const grid = $(".grid", view);
    Array.from(grid.children).forEach((el, i) => { el.style.animation = "rise var(--t-slow) var(--ease-out) both"; el.style.animationDelay = `${200 + i * 60}ms`; });

    $("#resolve-btn", view).addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      btn.innerHTML = `${ICON.refresh} Resolving…`;
      try {
        const r = await api("/feedback/resolve", { method: "POST" });
        const n = typeof r.resolved === "number" ? r.resolved : Array.isArray(r.resolved) ? r.resolved.length : 0;
        toast(`Resolved ${n} prediction${n === 1 ? "" : "s"}.`);
        if (!stale(token)) renderMonitoring(view, token);
      } catch (err) {
        toast(`<span class="neg">Resolve failed:</span> ${esc(err.message)}`);
        btn.disabled = false;
        btn.innerHTML = `${ICON.refresh} Resolve elapsed predictions`;
      }
    });
  }

  function kvBlock(val) {
    if (val == null) return '<p class="muted small">—</p>';
    if (typeof val !== "object") return `<p class="mono">${esc(val)}</p>`;
    const entries = Object.entries(val);
    if (!entries.length) return '<p class="muted small">Empty.</p>';
    return `<dl class="kv small">${entries.map(([k, v]) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd class="mono">${
      v && typeof v === "object" ? esc(JSON.stringify(v)).slice(0, 160) : isNum(v) ? esc(+v.toFixed(4)) : esc(v)}</dd>`).join("")}</dl>`;
  }

  // ------------------------------------------------------------------ errors

  function errorCard(title, err, retryHref) {
    const hint = err.status === 404 || err.status === 400 ? "The ticker may be unknown or have no price history, or the models are not trained yet."
      : err.status === 502 ? "An upstream data source or the LLM provider failed."
      : !err.status ? "The API could not be reached. Is uvicorn running?" : "";
    return `
      <section class="card error-card">
        <div class="card-head"><div><h2 class="card-title">${ICON.alert} ${esc(title)}</h2>
          <p class="card-sub">${esc(hint)}</p></div></div>
        <pre class="mono small" style="white-space:pre-wrap;margin:0;color:var(--text-2)">${esc(err.message)}</pre>
        <div style="display:flex;gap:10px;margin-top:16px;flex-wrap:wrap">
          <button class="btn" type="button" onclick="window.dispatchEvent(new HashChangeEvent('hashchange'))">${ICON.refresh} Retry</button>
          <a class="btn btn-ghost" href="#/">Back to search</a>
        </div>
      </section>`;
  }

  // ------------------------------------------------------------------ boot

  function boot() {
    setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
    $("#theme-toggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
    const topbar = $("#topbar");
    addEventListener("scroll", () => topbar.classList.toggle("scrolled", scrollY > 4), { passive: true });
    addEventListener("resize", () => { updateNav(state.route ? state.route.name : "evaluate"); $$(".seg").forEach(syncSeg); });
    addEventListener("hashchange", () => {
      // The retry button re-dispatches hashchange for the same hash: force a full re-render.
      if (state.route && location.hash === state.lastHash) state.route = null;
      state.lastHash = location.hash;
      onRoute();
    });
    search.init();
    loadUniverse();
    checkHealth();
    state.lastHash = location.hash;
    onRoute();
    // Warm the validation cache so the Models view and regime chips are instant.
    (window.requestIdleCallback || ((f) => setTimeout(f, 300)))(() => loadValidation());
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { updateNav(state.route.name); $$(".seg").forEach(syncSeg); });
  }

  boot();
})();
