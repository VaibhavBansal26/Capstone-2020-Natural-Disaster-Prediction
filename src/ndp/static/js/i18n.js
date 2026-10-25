/* On-demand page translation.
 *
 * The header (base.html) ships an empty, hidden <select id="ndp-lang">. This script fills it with a short
 * list of languages and un-hides it. Google's translate script (https://translate.google.com/translate_a/element.js)
 * is NOT loaded until somebody picks a language other than English, so visitors who never use translation
 * make no request to Google. The choice is remembered in localStorage and re-applied on later pages;
 * choosing English loads nothing and clears the googtrans cookie.
 */
(function () {
  "use strict";

  var STORE_KEY = "ndp-lang";
  var CALLBACK = "ndpGoogleTranslateInit";
  var SCRIPT_SRC = "https://translate.google.com/translate_a/element.js?cb=" + CALLBACK;
  var HOST_ID = "ndp-gt-host";

  // [code, label in the language's own script]. The first option means "English / original".
  var LANGS = [
    ["en", "Language"],
    ["hi", "हिन्दी"],
    ["ta", "தமிழ்"],
    ["ml", "മലയാളം"],
    ["te", "తెలుగు"],
    ["bn", "বাংলা"],
    ["ne", "नेपाली"],
    ["es", "Español"],
    ["fr", "Français"],
    ["ar", "العربية"],
    ["ja", "日本語"],
    ["zh-CN", "中文"],
    ["id", "Bahasa Indonesia"]
  ];
  var CODES = LANGS.map(function (l) { return l[0]; });

  var select = document.getElementById("ndp-lang");
  if (!select) return;

  var scriptRequested = false;
  var widgetReady = false;
  var pending = null;      // language chosen before the widget finished initialising

  /* ---------- storage / cookie ---------- */

  function readStored() {
    try { var v = window.localStorage.getItem(STORE_KEY); return CODES.indexOf(v) > -1 ? v : "en"; } catch (e) { return "en"; }
  }
  function writeStored(code) {
    try { window.localStorage.setItem(STORE_KEY, code); } catch (e) { /* storage blocked: choice just won't persist */ }
  }
  function cookieSet(code) {
    var v = "googtrans=/en/" + code + "; path=/";
    document.cookie = v;
    try { document.cookie = v + "; domain=" + location.hostname; } catch (e) { /* ignore */ }
  }
  function cookieClear() {
    var past = "googtrans=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
    document.cookie = past;
    try { document.cookie = past + "; domain=" + location.hostname; } catch (e) { /* ignore */ }
    try { document.cookie = past + "; domain=." + location.hostname; } catch (e) { /* ignore */ }
  }
  function hasCookie() { return /(?:^|;\s*)googtrans=/.test(document.cookie); }

  /* ---------- styling: hide Google's injected banner so it never pushes the sticky header down ---------- */

  function injectCss() {
    if (document.getElementById("ndp-i18n-css")) return;
    var st = document.createElement("style");
    st.id = "ndp-i18n-css";
    st.textContent =
      "body > .skiptranslate, iframe.goog-te-banner-frame, .VIpgJd-ZVi9od-ORHb-OEVmcd, .VIpgJd-ZVi9od-aZ2wEe-wOHMyf," +
      "#goog-gt-tt, .goog-te-balloon-frame { display: none !important; visibility: hidden !important; }" +
      "body { top: 0 !important; position: static !important; }" +
      ".goog-text-highlight { background: none !important; box-shadow: none !important; }" +
      "font[style] { vertical-align: baseline !important; }" +
      "#" + HOST_ID + " { position: absolute; left: -9999px; width: 1px; height: 1px; overflow: hidden; }" +
      "#ndp-i18n-note { position: fixed; left: 50%; bottom: 1rem; transform: translateX(-50%); z-index: 60; max-width: 90vw;" +
      " background: #0f172a; color: #fff; font-size: .8125rem; padding: .5rem .875rem; border-radius: .5rem; box-shadow: 0 4px 16px rgba(0,0,0,.3); }";
    document.head.appendChild(st);
  }

  function note(text) {
    var el = document.getElementById("ndp-i18n-note");
    if (!el) {
      el = document.createElement("div");
      el.id = "ndp-i18n-note";
      el.setAttribute("role", "status");
      el.className = "notranslate";
      document.body.appendChild(el);
    }
    el.textContent = text;
    setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 5000);
  }

  /* ---------- google widget ---------- */

  function comboFor(code, tries) {
    var combo = document.querySelector("#" + HOST_ID + " select.goog-te-combo");
    if (combo) {
      if (combo.value !== code) {
        combo.value = code;
        combo.dispatchEvent(new Event("change"));
      }
      return;
    }
    if (tries > 0) setTimeout(function () { comboFor(code, tries - 1); }, 250);
  }

  window[CALLBACK] = function () {
    try {
      var host = document.getElementById(HOST_ID);
      new window.google.translate.TranslateElement({
        pageLanguage: "en",
        includedLanguages: CODES.filter(function (c) { return c !== "en"; }).join(","),
        autoDisplay: false
      }, host);
      widgetReady = true;
      if (pending) comboFor(pending, 20);
    } catch (e) {
      if (window.console) console.warn("[ndp] translate widget failed:", e);
      failed();
    }
  };

  function failed() {
    note("Translation is unavailable right now.");
    pending = null;
    scriptRequested = false;
    writeStored("en");
    cookieClear();
    select.value = "en";
  }

  function loadGoogle() {
    if (scriptRequested) return;
    scriptRequested = true;
    injectCss();
    var host = document.createElement("div");
    host.id = HOST_ID;
    host.className = "notranslate";
    host.setAttribute("aria-hidden", "true");
    document.body.appendChild(host);
    var s = document.createElement("script");
    s.src = SCRIPT_SRC;
    s.async = true;
    s.onerror = function () { var h = document.getElementById(HOST_ID); if (h) h.remove(); s.remove(); failed(); };
    document.head.appendChild(s);
  }

  /* ---------- apply a choice ---------- */

  function apply(code, fromUser) {
    if (code === "en") {
      var wasTranslated = hasCookie() || scriptRequested;
      cookieClear();
      writeStored("en");
      // Google rewrote the DOM in place; the only clean way back to the original text is a reload.
      if (fromUser && wasTranslated) window.location.reload();
      return;
    }
    writeStored(code);
    cookieSet(code);
    document.documentElement.setAttribute("data-ndp-lang", code);
    pending = code;
    if (!scriptRequested) loadGoogle();
    else if (widgetReady) comboFor(code, 20);
  }

  /* ---------- init ---------- */

  LANGS.forEach(function (l, i) {
    var o = document.createElement("option");
    o.value = l[0];
    o.textContent = l[1];
    if (i === 0) o.setAttribute("aria-label", "Language (English)");
    else o.setAttribute("lang", l[0]);
    select.appendChild(o);
  });
  // Keep the sticky header on one line on phones (brand + tools + Menu otherwise overflow ~375px).
  var fit = document.createElement("style");
  fit.textContent = "@media (max-width: 480px) { #ndp-lang { max-width: 4.25rem; text-overflow: ellipsis; padding-left: .375rem; padding-right: .125rem; } }";
  document.head.appendChild(fit);
  select.classList.add("notranslate");
  select.setAttribute("translate", "no");
  select.hidden = false;
  select.removeAttribute("hidden");

  var initial = readStored();
  select.value = initial;
  select.addEventListener("change", function () { apply(select.value, true); });

  if (initial !== "en") apply(initial, false);
  else if (hasCookie()) cookieClear();   // stale cookie from an earlier session: do not translate
})();
