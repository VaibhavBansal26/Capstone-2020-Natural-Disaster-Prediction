/* Leaflet helpers and small progressive enhancements for the NDP templates.
 *
 * Public helpers (also usable from inline scripts):
 *   ndpMap(id, lat, lon, zoom)      -> L.Map with an OpenStreetMap base layer
 *   ndpUsgsLayer(map, url)          -> Promise<L.GeoJSON> of the USGS feed (circle markers scaled by magnitude)
 *   ndpMarkers(map, items)          -> L.FeatureGroup of circle markers for
 *                                      [{lat, lon, title, lines: [text...] (escaped), popup: html (trusted), color, radius}]
 *
 * Declarative initialisation runs on DOMContentLoaded and again on every HTMX swap
 * (`htmx:load`), so partials can simply carry data attributes:
 *   <div data-ndp-map data-lat data-lon data-zoom
 *        data-usgs-url="..."                      live USGS GeoJSON feed
 *        data-markers-src="#json-script-id"       JSON array of marker items
 *        data-geojson-url="/covid/data/centres"   GeoJSON FeatureCollection fetched lazily
 *        data-pick-lat="#lat" data-pick-lon="#lon" draggable marker bound to two inputs
 *        data-status="#element"                   where to print load errors/counts>
 *   <button data-tab-target="#panel">  with sibling panels marked data-tab-panel
 *   <input type="date" data-default-today>
 */
(function () {
  "use strict";

  var OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
  var OSM_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

  function hasLeaflet() {
    return typeof window.L !== "undefined";
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function num(v, fallback) {
    var n = parseFloat(v);
    return isNaN(n) ? fallback : n;
  }

  function setStatus(el, text, isError) {
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("text-rose-600", !!isError);
  }

  /* ---------- public helpers ---------- */

  function ndpMap(id, lat, lon, zoom) {
    var el = typeof id === "string" ? document.getElementById(id) : id;
    if (!el || !hasLeaflet()) return null;
    if (el._ndpMap) return el._ndpMap;
    var map = L.map(el, { scrollWheelZoom: false }).setView([num(lat, 20), num(lon, 0)], num(zoom, 2));
    L.tileLayer(OSM_URL, { maxZoom: 19, attribution: OSM_ATTR }).addTo(map);
    el._ndpMap = map;
    return map;
  }

  function usgsPopup(p, coords) {
    var when = p.time ? new Date(p.time).toUTCString() : "unknown time";
    var depth = coords && coords.length > 2 ? Number(coords[2]).toFixed(1) + " km deep" : "";
    return (
      "<p><strong>M " + esc(Number(p.mag).toFixed(1)) + "</strong> " + esc(p.place || "") + "</p>" +
      "<p>" + esc(when) + (depth ? " &middot; " + esc(depth) : "") + "</p>" +
      (p.url ? '<p><a class="underline" target="_blank" rel="noopener" href="' + esc(p.url) + '">USGS event page</a></p>' : "")
    );
  }

  function magColor(mag) {
    if (mag >= 6) return "#be123c";
    if (mag >= 5) return "#ea580c";
    if (mag >= 4) return "#d97706";
    return "#2563eb";
  }

  function ndpUsgsLayer(map, url) {
    if (!map || !hasLeaflet()) return Promise.resolve(null);
    return fetch(url, { headers: { Accept: "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (geojson) {
        var layer = L.geoJSON(geojson, {
          pointToLayer: function (feature, latlng) {
            var mag = num(feature.properties && feature.properties.mag, 1);
            return L.circleMarker(latlng, {
              radius: Math.max(3, mag * 2.2),
              color: magColor(mag), weight: 1, fillColor: magColor(mag), fillOpacity: 0.45,
            });
          },
          onEachFeature: function (feature, l) {
            l.bindPopup(usgsPopup(feature.properties || {}, feature.geometry && feature.geometry.coordinates));
          },
        }).addTo(map);
        layer.ndpCount = geojson.features ? geojson.features.length : 0;
        return layer;
      });
  }

  function ndpMarkers(map, items) {
    if (!map || !hasLeaflet()) return null;
    var group = L.featureGroup();
    (items || []).forEach(function (it) {
      var lat = num(it.lat, NaN), lon = num(it.lon, NaN);
      if (isNaN(lat) || isNaN(lon)) return;
      var color = it.color || "#2563eb";
      var m = L.circleMarker([lat, lon], {
        radius: num(it.radius, 7), color: color, weight: 2, fillColor: color, fillOpacity: 0.6,
      });
      var lines = (it.lines || []).filter(function (s) { return s != null && String(s) !== ""; })
        .map(function (s) { return "<p>" + esc(s) + "</p>"; }).join("");
      var html = "<p><strong>" + esc(it.title || it.name || "") + "</strong></p>" + lines + (it.popup || "");
      m.bindPopup(html);
      if (it.title || it.name) m.bindTooltip(String(it.title || it.name));
      group.addLayer(m);
    });
    group.addTo(map);
    if (group.getLayers().length) map.fitBounds(group.getBounds().pad(0.2), { maxZoom: 14 });
    return group;
  }

  /* ---------- GeoJSON FeatureCollection (covid layers) ---------- */

  function featurePopup(props) {
    var keys = Object.keys(props || {}).filter(function (k) { return k !== "name" && k !== "color"; });
    var rows = keys.map(function (k) {
      var v = props[k];
      if (v == null || v === "") return "";
      return "<p><span class=\"text-slate-500\">" + esc(k.replace(/_/g, " ")) + ":</span> " + esc(v) + "</p>";
    }).join("");
    return "<p><strong>" + esc(props && props.name) + "</strong></p>" + rows;
  }

  function loadGeoJsonMarkers(map, url, statusEl) {
    setStatus(statusEl, "Loading…");
    return fetch(url, { headers: { Accept: "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (fc) {
        var items = (fc.features || []).map(function (f) {
          var c = (f.geometry && f.geometry.coordinates) || [];
          var p = f.properties || {};
          return { lat: c[1], lon: c[0], title: p.name, popup: featurePopup(p), color: p.color, radius: 6 };
        });
        ndpMarkers(map, items);
        setStatus(statusEl, items.length + " locations");
      })
      .catch(function (err) { setStatus(statusEl, "Could not load data (" + err.message + ")", true); });
  }

  /* ---------- draggable location picker ---------- */

  function bindPicker(map, el) {
    var latInput = document.querySelector(el.getAttribute("data-pick-lat"));
    var lonInput = document.querySelector(el.getAttribute("data-pick-lon"));
    if (!latInput || !lonInput) return;
    var start = [num(latInput.value, map.getCenter().lat), num(lonInput.value, map.getCenter().lng)];
    var marker = L.marker(start, { draggable: true, title: "Drag to set latitude/longitude" }).addTo(map);
    marker.bindTooltip("Drag me, or click the map");

    function write(latlng) {
      latInput.value = latlng.lat.toFixed(4);
      lonInput.value = latlng.lng.toFixed(4);
      latInput.dispatchEvent(new Event("change", { bubbles: true }));
    }
    marker.on("dragend", function () { write(marker.getLatLng()); });
    map.on("click", function (e) { marker.setLatLng(e.latlng); write(e.latlng); });

    function fromInputs() {
      var lat = num(latInput.value, NaN), lon = num(lonInput.value, NaN);
      if (!isNaN(lat) && !isNaN(lon)) marker.setLatLng([lat, lon]);
    }
    latInput.addEventListener("input", fromInputs);
    lonInput.addEventListener("input", fromInputs);
    el._ndpPicker = marker;
  }

  /* ---------- declarative init ---------- */

  function initMapElement(el) {
    if (el._ndpMap) return el._ndpMap;
    var map = ndpMap(el, el.getAttribute("data-lat"), el.getAttribute("data-lon"), el.getAttribute("data-zoom"));
    if (!map) return null;
    var statusEl = el.getAttribute("data-status") ? document.querySelector(el.getAttribute("data-status")) : null;

    var usgs = el.getAttribute("data-usgs-url");
    if (usgs) {
      setStatus(statusEl, "Loading USGS feed…");
      ndpUsgsLayer(map, usgs)
        .then(function (layer) { setStatus(statusEl, layer ? layer.ndpCount + " events in the live feed" : ""); })
        .catch(function (err) { setStatus(statusEl, "USGS feed unavailable (" + err.message + ")", true); });
    }

    var src = el.getAttribute("data-markers-src");
    if (src) {
      var script = document.querySelector(src);
      if (script) {
        try { ndpMarkers(map, JSON.parse(script.textContent || "[]")); }
        catch (err) { setStatus(statusEl, "Bad marker data", true); }
      }
    }

    if (el.getAttribute("data-pick-lat")) bindPicker(map, el);

    var geo = el.getAttribute("data-geojson-url");
    if (geo) {
      el._ndpLoad = function () {
        if (el._ndpLoaded) return;
        el._ndpLoaded = true;
        loadGeoJsonMarkers(map, geo, statusEl);
      };
      if (!el.closest("[data-tab-panel][hidden]")) el._ndpLoad();
    }
    return map;
  }

  function initTabs(root) {
    root.querySelectorAll("[data-tab-target]").forEach(function (btn) {
      if (btn._ndpTab) return;
      btn._ndpTab = true;
      btn.addEventListener("click", function () {
        var target = document.querySelector(btn.getAttribute("data-tab-target"));
        if (!target) return;
        var group = btn.closest("[data-tabs]") || document;
        group.querySelectorAll("[data-tab-panel]").forEach(function (p) { p.hidden = p !== target; });
        group.querySelectorAll("[data-tab-target]").forEach(function (b) {
          var on = b === btn;
          b.setAttribute("aria-selected", on ? "true" : "false");
          b.classList.toggle("bg-brand-600", on);
          b.classList.toggle("text-white", on);
          b.classList.toggle("text-slate-600", !on);
        });
        target.querySelectorAll("[data-ndp-map]").forEach(function (m) {
          if (m._ndpMap) m._ndpMap.invalidateSize();
          if (m._ndpLoad) m._ndpLoad();
        });
      });
    });
  }

  function initDefaults(root) {
    root.querySelectorAll('input[type="date"][data-default-today]').forEach(function (inp) {
      if (!inp.value) inp.value = new Date().toISOString().slice(0, 10);
    });
  }

  function init(root) {
    root = root || document;
    if (root.matches && root.matches("[data-ndp-map]")) initMapElement(root);
    root.querySelectorAll("[data-ndp-map]").forEach(initMapElement);
    initTabs(root);
    initDefaults(root);
  }

  window.ndpMap = ndpMap;
  window.ndpUsgsLayer = ndpUsgsLayer;
  window.ndpMarkers = ndpMarkers;
  window.ndpInit = init;

  document.addEventListener("DOMContentLoaded", function () {
    init(document);
    document.body.addEventListener("htmx:load", function (e) { init(e.detail && e.detail.elt ? e.detail.elt : document); });
  });
})();
