/* Landing-page hero: interactive 3D globe of this week's USGS earthquakes (globe.gl),
 * with a Leaflet world-map fallback when WebGL or globe.gl is unavailable.
 *
 * - Data: https://earthquake.usgs.gov/.../2.5_week.geojson, fetched in the browser. No keys, nothing stored.
 * - All feed text is inserted with textContent / DOM APIs or escaped before it goes into tooltip HTML.
 * - Honours prefers-reduced-motion (no auto-rotate, no pulsing rings, no camera animation),
 *   and stops rendering while the tab is hidden or the hero is scrolled off-screen.
 */
(function () {
  "use strict";

  var root = document.getElementById("ndp-globe");
  if (!root) return;

  function $(id) { return document.getElementById(id); }

  var FEED_URL = root.getAttribute("data-feed") || "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson";
  var ESTIMATE_URL = root.getAttribute("data-estimate-url") || "/earthquake";
  var TEX_GLOBE = root.getAttribute("data-tex-globe");
  var TEX_BUMP = root.getAttribute("data-tex-bump");
  var USGS_HOST = "https://earthquake.usgs.gov/";
  var REFRESH_MS = 10 * 60 * 1000;
  var IDLE_RESUME_MS = 4000;

  var hero = $("ndp-hero");
  var wrap = $("ndp-globe-wrap");
  var fallbackEl = $("ndp-fallback-map");
  var fallbackMsg = $("ndp-fallback-msg");
  var loadingEl = $("ndp-globe-loading");
  var rotateBtn = $("ndp-rotate-toggle");
  var card = $("ndp-quake-card");
  var feedText = $("ndp-feed-text");
  var retryBtn = $("ndp-feed-retry");
  var topWrap = $("ndp-top-wrap");
  var topList = $("ndp-top-list");

  var motionQuery = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  function motionOK() { return !(motionQuery && motionQuery.matches); }

  var mode = "none";        // "globe" | "fallback" | "none"
  var globe = null;
  var controls = null;
  var map = null;
  var markerLayer = null;
  var quakes = [];
  var selected = null;
  var userPaused = false;
  var heroVisible = true;
  var idleTimer = null;
  var refreshTimer = null;
  var loading = false;

  /* ------------------------------------------------------------ helpers */

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function setText(id, text) { var el = $(id); if (el) el.textContent = text; }

  function magColor(m) { return m >= 5.5 ? "#ef4444" : m >= 4 ? "#f97316" : "#facc15"; }
  function magRgb(m) { return m >= 5.5 ? "239,68,68" : m >= 4 ? "249,115,22" : "250,204,21"; }
  function fmtMag(m) { return "M " + m.toFixed(1); }

  function fmtTime(t) {
    if (!isFinite(t)) return "time unknown";
    var d = new Date(t), abs;
    try { abs = d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); } catch (e) { abs = d.toUTCString(); }
    var mins = Math.max(0, Math.round((Date.now() - t) / 60000));
    var ago = mins < 60 ? mins + " min ago" : mins < 2880 ? Math.round(mins / 60) + " h ago" : Math.round(mins / 1440) + " days ago";
    return abs + " (" + ago + ")";
  }

  function hasWebGL() {
    try {
      var c = document.createElement("canvas");
      var gl = c.getContext("webgl2") || c.getContext("webgl") || c.getContext("experimental-webgl");
      if (!gl) return false;
      var lose = gl.getExtension && gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
      return true;
    } catch (e) { return false; }
  }

  /* ------------------------------------------------------------ feed */

  function parseFeed(json) {
    var out = [];
    var feats = json && Array.isArray(json.features) ? json.features : null;
    if (!feats) throw new Error("unexpected feed format");
    feats.forEach(function (f) {
      var c = f && f.geometry && f.geometry.coordinates;
      var p = (f && f.properties) || {};
      if (!Array.isArray(c) || c.length < 2) return;
      var lon = Number(c[0]), lat = Number(c[1]), depth = Number(c[2]), mag = Number(p.mag);
      if (!isFinite(lon) || !isFinite(lat) || !isFinite(mag)) return;
      if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return;
      var url = typeof p.url === "string" && p.url.indexOf(USGS_HOST) === 0 ? p.url : "";
      out.push({
        id: String(f.id || out.length),
        lat: lat, lon: lon, depth: isFinite(depth) ? depth : 0, mag: mag,
        place: typeof p.place === "string" && p.place ? p.place : "Unnamed location",
        time: Number(p.time), url: url, color: magColor(mag)
      });
    });
    out.sort(function (a, b) { return b.mag - a.mag; });
    return out;
  }

  function setFeedState(kind, message) {
    if (feedText) feedText.textContent = message;
    if (retryBtn) retryBtn.hidden = kind !== "error";
  }

  function load() {
    if (loading) return;
    loading = true;
    setFeedState("loading", quakes.length ? "Refreshing live earthquake data…" : "Loading live earthquake data from USGS…");
    var ctrl = typeof AbortController === "function" ? new AbortController() : null;
    var to = setTimeout(function () { if (ctrl) ctrl.abort(); }, 15000);
    fetch(FEED_URL, { signal: ctrl ? ctrl.signal : undefined, credentials: "omit", referrerPolicy: "no-referrer", headers: { Accept: "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (json) {
        var list = parseFeed(json);
        quakes = list;
        selected = null;
        hideCard();
        render();
        setFeedState("ok", list.length
          ? "Live: " + list.length.toLocaleString() + " earthquakes of M2.5+ in the past 7 days. Updated " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) + "."
          : "The USGS feed returned no earthquakes of M2.5+ for the past week.");
      })
      .catch(function (err) {
        if (window.console) console.warn("[ndp] earthquake feed unavailable:", err);
        setFeedState("error", quakes.length
          ? "Live feed unavailable; showing the last data loaded."
          : "Live earthquake feed unavailable (USGS could not be reached).");
        if (!quakes.length) {
          root.setAttribute("aria-label", "Interactive 3D globe. The live earthquake feed is currently unavailable.");
        }
      })
      .then(function () { clearTimeout(to); loading = false; });
  }

  /* ------------------------------------------------------------ rendering */

  function render() {
    // counters
    var m5 = quakes.filter(function (q) { return q.mag >= 5; }).length;
    var top = quakes[0];
    setText("ndp-c-total", quakes.length.toLocaleString());
    setText("ndp-c-m5", String(m5));
    setText("ndp-c-max", top ? fmtMag(top.mag) : "–");
    setText("ndp-c-max-place", top ? top.place : "");
    var summary = quakes.length
      ? "Interactive 3D globe of " + quakes.length + " earthquakes of magnitude 2.5 and above in the past week. Strongest: magnitude " +
        top.mag.toFixed(1) + ", " + top.place + ". " + m5 + " of magnitude 5 or more. Use the list of strongest earthquakes below the globe for keyboard access."
      : "Interactive 3D globe of earthquakes in the past week. No earthquakes in the feed.";
    root.setAttribute("aria-label", summary);
    if (fallbackEl) fallbackEl.setAttribute("aria-label", "World map of " + quakes.length + " earthquakes in the past week. " + (top ? "Strongest: magnitude " + top.mag.toFixed(1) + ", " + top.place + "." : ""));

    renderTopList();
    if (mode === "globe") renderGlobe(); else if (mode === "fallback") renderFallback();
  }

  function renderTopList() {
    if (!topList) return;
    topList.textContent = "";
    var list = quakes.slice(0, 6);
    list.forEach(function (q) {
      var li = document.createElement("li");
      var b = document.createElement("button");
      b.type = "button";
      b.className = "ndp-chip flex max-w-full items-center gap-2 rounded-full border border-white/25 bg-white/5 px-3 py-1 text-left text-xs text-slate-100 hover:bg-white/15";
      var dot = document.createElement("span");
      dot.className = "inline-block h-2 w-2 shrink-0 rounded-full";
      dot.style.background = q.color;
      dot.setAttribute("aria-hidden", "true");
      var m = document.createElement("span");
      m.className = "notranslate font-semibold tabular-nums";
      m.textContent = fmtMag(q.mag);
      var pl = document.createElement("span");
      pl.className = "truncate";
      pl.textContent = q.place;
      b.appendChild(dot); b.appendChild(m); b.appendChild(pl);
      b.addEventListener("click", function () { select(q); });
      li.appendChild(b);
      topList.appendChild(li);
    });
    if (topWrap) topWrap.hidden = list.length === 0;
  }

  /* ---- globe ---- */

  function pointRadius(q) {
    var r = 0.28 + (q.mag - 2.5) * 0.14;
    return q === selected ? r * 1.8 + 0.2 : Math.max(0.2, r);
  }
  function pointAlt(q) { return 0.004 + Math.max(0, q.mag - 2.5) * 0.011 + (q === selected ? 0.03 : 0); }

  function tooltipHtml(q) {
    return '<div class="ndp-tip"><b>' + esc(fmtMag(q.mag)) + "</b> &middot; " + esc(q.place) + "<br>" +
      esc(fmtTime(q.time)) + "<br>Depth " + esc(q.depth.toFixed(1)) + " km<br><i>Click for details</i></div>";
  }

  function renderGlobe() {
    if (!globe) return;
    globe.pointsData(quakes.slice())
      .pointRadius(function (q) { return pointRadius(q); })
      .pointAltitude(function (q) { return pointAlt(q); });
    applyRings();
  }

  function applyRings() {
    if (!globe) return;
    var big = motionOK() ? quakes.filter(function (q) { return q.mag >= 5; }).slice(0, 40) : [];
    globe.ringsData(big);
  }

  function initGlobe() {
    if (typeof window.Globe !== "function") throw new Error("globe.gl did not load");
    if (!hasWebGL()) throw new Error("WebGL unavailable");

    var w = root.clientWidth || 600, h = root.clientHeight || 420;
    globe = new window.Globe(root, {
      animateIn: motionOK(),
      rendererConfig: { antialias: true, alpha: true, powerPreference: "high-performance" }
    })
      .width(w).height(h)
      .backgroundColor("rgba(0,0,0,0)")
      .globeImageUrl(TEX_GLOBE)
      .bumpImageUrl(TEX_BUMP)
      .showAtmosphere(true)
      .atmosphereColor("#5b9bff")
      .atmosphereAltitude(0.2)
      .pointLat("lat").pointLng("lon")
      .pointColor(function (q) { return q.color; })
      .pointRadius(function (q) { return pointRadius(q); })
      .pointAltitude(function (q) { return pointAlt(q); })
      .pointResolution(10)
      .pointsMerge(false)
      .pointsTransitionDuration(motionOK() ? 700 : 0)
      .pointLabel(tooltipHtml)
      .onPointHover(function (q) { root.style.cursor = q ? "pointer" : ""; })
      .onPointClick(function (q) { if (q) select(q); })
      .ringLat("lat").ringLng("lon")
      .ringAltitude(0.004)
      .ringMaxRadius(function (q) { return 2.5 + (q.mag - 5) * 2; })
      .ringPropagationSpeed(2)
      .ringRepeatPeriod(1500)
      .ringColor(function (q) {
        var rgb = magRgb(q.mag);
        return function (t) { return "rgba(" + rgb + "," + Math.max(0, 1 - t).toFixed(3) + ")"; };
      })
      .onGlobeReady(function () { hideLoading(); });

    globe.pointOfView({ lat: 20, lng: 78, altitude: root.clientWidth < 500 ? 2.6 : root.clientHeight < 520 ? 2.0 : 2.3 }, 0);

    controls = globe.controls();
    controls.enableZoom = false;       // never hijack page scroll
    controls.enablePan = false;
    controls.autoRotate = motionOK();
    controls.autoRotateSpeed = 0.35;
    controls.addEventListener("start", function () { controls.autoRotate = false; clearTimeout(idleTimer); });
    controls.addEventListener("end", function () { scheduleResume(); });

    var dom = globe.renderer().domElement;
    dom.setAttribute("aria-hidden", "true");
    dom.addEventListener("webglcontextlost", function (ev) {
      ev.preventDefault();
      destroyGlobe();
      startFallback();
    });

    mode = "globe";
    updateRotateButton();
    updateRunState();
    setTimeout(hideLoading, 10000);
  }

  function destroyGlobe() {
    try { if (globe && globe._destructor) globe._destructor(); } catch (e) { /* ignore */ }
    globe = null; controls = null;
    root.textContent = "";
    root.style.cursor = "";
  }

  function hideLoading() { if (loadingEl) loadingEl.hidden = true; }

  /* rotation + run state */

  function canRotate() { return motionOK() && !userPaused && !selected; }
  function scheduleResume() {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(function () { if (controls) controls.autoRotate = canRotate(); }, IDLE_RESUME_MS);
  }
  function updateRotateButton() {
    if (!rotateBtn) return;
    rotateBtn.hidden = !(mode === "globe" && motionOK());
    rotateBtn.textContent = userPaused ? "Resume rotation" : "Pause rotation";
    rotateBtn.setAttribute("aria-pressed", userPaused ? "true" : "false");
  }
  function updateRunState() {
    if (!globe) return;
    var run = heroVisible && !document.hidden;
    if (run) globe.resumeAnimation(); else globe.pauseAnimation();
  }

  /* ---- fallback (Leaflet) ---- */

  function showMessage(text) {
    if (!fallbackMsg) return;
    fallbackMsg.textContent = text;
    fallbackMsg.classList.remove("hidden");
    fallbackMsg.classList.add("flex");
  }

  function startFallback() {
    mode = "fallback";
    hideLoading();
    root.hidden = true;
    if (rotateBtn) rotateBtn.hidden = true;
    var hint = $("ndp-globe-hint");
    if (hint) hint.textContent = "The 3D globe is unavailable on this device, so this is a flat map of the same earthquakes. Click a circle for details.";
    if (hero) hero.classList.add("ndp-is-fallback");
    if (window.L && typeof window.L.map === "function" && fallbackEl) {
      try {
        fallbackEl.hidden = false;
        map = window.L.map(fallbackEl, { worldCopyJump: true, minZoom: 1, scrollWheelZoom: false, zoomControl: true }).setView([20, 10], 2);
        window.L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 18, attribution: "&copy; OpenStreetMap contributors"
        }).addTo(map);
        markerLayer = window.L.layerGroup().addTo(map);
        setTimeout(function () { if (map) map.invalidateSize(); }, 0);
        renderFallback();
        return;
      } catch (e) {
        if (window.console) console.warn("[ndp] fallback map failed:", e);
        fallbackEl.hidden = true;
        map = null;
      }
    }
    showMessage("The interactive globe and map could not be loaded. The live counters and the list of strongest earthquakes below still work.");
  }

  function renderFallback() {
    if (!map || !markerLayer) return;
    markerLayer.clearLayers();
    // weakest first so strong quakes draw on top
    quakes.slice().reverse().forEach(function (q) {
      var m = window.L.circleMarker([q.lat, q.lon], {
        radius: 3 + Math.max(0, q.mag - 2.5) * 2.4,
        color: "#0f172a", weight: 1, fillColor: q.color, fillOpacity: 0.8
      });
      var tip = document.createElement("span");
      tip.textContent = fmtMag(q.mag) + " · " + q.place;
      m.bindTooltip(tip, { direction: "top" });
      m.on("click", function () { select(q); });
      m.addTo(markerLayer);
    });
  }

  /* ------------------------------------------------------------ detail card */

  function select(q) {
    selected = q;
    setText("ndp-q-mag", fmtMag(q.mag));
    setText("ndp-q-place", q.place);
    setText("ndp-q-meta", fmtTime(q.time) + " · depth " + q.depth.toFixed(1) + " km");
    var est = $("ndp-q-estimate");
    if (est) {
      est.href = ESTIMATE_URL + "?lat=" + q.lat.toFixed(4) + "&lon=" + q.lon.toFixed(4) + "&depth=" + Math.max(0, q.depth).toFixed(1);
    }
    var usgs = $("ndp-q-usgs");
    if (usgs) { usgs.hidden = !q.url; if (q.url) usgs.href = q.url; }
    if (card) { card.hidden = false; try { card.focus({ preventScroll: true }); } catch (e) { /* ignore */ } }

    if (mode === "globe" && globe) {
      if (controls) { controls.autoRotate = false; clearTimeout(idleTimer); }
      globe.pointOfView({ lat: q.lat, lng: q.lon, altitude: 1.8 }, motionOK() ? 1000 : 0);
      renderGlobe();
    } else if (mode === "fallback" && map) {
      map.setView([q.lat, q.lon], Math.max(map.getZoom(), 4), { animate: motionOK() });
    }
  }

  function hideCard() {
    if (card) card.hidden = true;
    var had = selected;
    selected = null;
    if (had && mode === "globe") { renderGlobe(); scheduleResume(); }
  }

  /* ------------------------------------------------------------ wiring */

  function bind() {
    var closeBtn = $("ndp-quake-close");
    if (closeBtn) closeBtn.addEventListener("click", hideCard);
    if (card) card.addEventListener("keydown", function (e) { if (e.key === "Escape") hideCard(); });
    if (retryBtn) retryBtn.addEventListener("click", load);
    if (rotateBtn) rotateBtn.addEventListener("click", function () {
      userPaused = !userPaused;
      if (controls) controls.autoRotate = canRotate();
      updateRotateButton();
    });

    document.addEventListener("visibilitychange", updateRunState);
    if ("IntersectionObserver" in window && hero) {
      new IntersectionObserver(function (entries) {
        heroVisible = entries[entries.length - 1].isIntersecting;
        updateRunState();
      }, { threshold: 0.05 }).observe(hero);
    }
    if ("ResizeObserver" in window && wrap) {
      new ResizeObserver(function () {
        if (globe) globe.width(root.clientWidth || 300).height(root.clientHeight || 300);
        if (map) map.invalidateSize();
      }).observe(wrap);
    } else {
      window.addEventListener("resize", function () {
        if (globe) globe.width(root.clientWidth || 300).height(root.clientHeight || 300);
        if (map) map.invalidateSize();
      });
    }
    if (motionQuery) {
      var onMotion = function () {
        if (controls) controls.autoRotate = canRotate();
        applyRings();
        updateRotateButton();
      };
      if (motionQuery.addEventListener) motionQuery.addEventListener("change", onMotion);
      else if (motionQuery.addListener) motionQuery.addListener(onMotion);
    }
    refreshTimer = setInterval(function () { if (!document.hidden && heroVisible) load(); }, REFRESH_MS);
  }

  function start() {
    bind();
    try {
      initGlobe();
    } catch (err) {
      if (window.console) console.warn("[ndp] globe unavailable, using fallback:", err && err.message ? err.message : err);
      destroyGlobe();
      startFallback();
    }
    load();
  }

  start();
})();
