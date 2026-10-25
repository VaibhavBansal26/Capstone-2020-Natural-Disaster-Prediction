/* Voice integration for the NDP pages (no build step, no dependencies).
 *
 * A. Voice output (SpeechSynthesis): reads result partials aloud.
 * B. Voice input (SpeechRecognition): fills form fields, submits, navigates and
 *    triggers buttons from spoken commands.
 *
 * Page conventions (all optional):
 *   data-speak="sentence"            on a result root: the text to read aloud.
 *   data-speak-priority="alert"      also read when the toggle is OFF (only if the user has
 *                                    enabled voice at least once before); slower, higher pitch.
 *   data-speak-region                marks an extra container that gets a "Read aloud" button.
 *                                    #result, #result-* and #text-score get one automatically.
 *   data-voice-command="phrase"      element clicked when the phrase is spoken ("a|b" = alternatives).
 *   data-voice-confirm               on that same element: spoken confirmation required first.
 *
 * Pure functions (interpret, parseNumber, parseDate, chunkText, pickVoice, ...) are exported
 * through `module.exports` for the node tests and through `window.NdpVoice` in the browser.
 */
(function (root) {
  "use strict";

  /* The script element is only available while the script runs. */
  var SELF_SRC = (root.document && root.document.currentScript && root.document.currentScript.src) || "";

  /* =====================================================================
   * 1. Pure helpers: text normalisation, numbers, dates
   * ===================================================================== */

  var UNITS = {
    zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9,
    ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16,
    seventeen: 17, eighteen: 18, nineteen: 19
  };
  var TENS = { twenty: 20, thirty: 30, forty: 40, fourty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 };
  var ORD_UNITS = {
    first: 1, second: 2, third: 3, fourth: 4, fifth: 5, sixth: 6, seventh: 7, eighth: 8, ninth: 9,
    tenth: 10, eleventh: 11, twelfth: 12, thirteenth: 13, fourteenth: 14, fifteenth: 15,
    sixteenth: 16, seventeenth: 17, eighteenth: 18, nineteenth: 19
  };
  var ORD_TENS = { twentieth: 20, thirtieth: 30 };
  var DIGIT_WORDS = { zero: "0", one: "1", two: "2", three: "3", four: "4", five: "5", six: "6", seven: "7", eight: "8", nine: "9" };
  var POINT_WORDS = { point: 1, decimal: 1, dot: 1 };
  var MONTHS = {
    january: 1, jan: 1, february: 2, feb: 2, march: 3, mar: 3, april: 4, apr: 4, may: 5, june: 6, jun: 6,
    july: 7, jul: 7, august: 8, aug: 8, september: 9, sept: 9, sep: 9, october: 10, oct: 10,
    november: 11, nov: 11, december: 12, dec: 12
  };

  var CONFIRM_WINDOW_MS = 8000;

  function has(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj, key);
  }

  /* lowercase, strip punctuation to single spaces: "Wind speed (m/s)" -> "wind speed m s" */
  function norm(s) {
    return String(s == null ? "" : s).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  }

  function stripParens(s) {
    return String(s == null ? "" : s).replace(/\([^)]*\)/g, " ").replace(/\s+/g, " ").trim();
  }

  function words(s) {
    var n = norm(s);
    return n ? n.split(" ") : [];
  }

  /* Split a transcript into tokens {raw, lc}. Digits, signs and dates survive; units and
   * punctuation do not. */
  function tokenize(transcript) {
    var s = String(transcript == null ? "" : transcript);
    s = s.replace(/−/g, "-")
      .replace(/[°%$€£]/g, " ")
      .replace(/(\d),(?=\d{3}(\D|$))/g, "$1")
      .replace(/([A-Za-z])-(?=[A-Za-z])/g, "$1 ");
    var out = [];
    s.split(/\s+/).forEach(function (t) {
      t = t.replace(/^[\s"'“”‘’(\[]+|[\s"'“”‘’.,!?;:)\]]+$/g, "");
      if (t) out.push({ raw: t, lc: t.toLowerCase() });
    });
    return out;
  }

  function isDigitWord(lc) {
    return has(DIGIT_WORDS, lc);
  }

  function isNumberWord(lc) {
    return has(UNITS, lc) || has(TENS, lc) || lc === "hundred" || lc === "thousand" ||
      /^\d+(\.\d+)?$/.test(lc);
  }

  /* Number words -> {value, next, ordinal}. `o.lenient` reads "to"/"too" as 2 and "for" as 4. */
  function parseWords(toks, start, o) {
    var p = start, total = 0, cur = 0, any = false, ordinal = false, sawBig = false;
    var lc;

    /* "one zero one three" -> 1013 (spoken digit strings) */
    if (toks[p] && isDigitWord(toks[p].lc) && toks[p + 1] && isDigitWord(toks[p + 1].lc)) {
      var str = "";
      while (toks[p] && (isDigitWord(toks[p].lc) || (str && toks[p].lc === "oh"))) {
        str += toks[p].lc === "oh" ? "0" : DIGIT_WORDS[toks[p].lc];
        p++;
      }
      return { value: parseInt(str, 10), next: p, ordinal: false };
    }

    for (; p < toks.length; p++) {
      lc = toks[p].lc;
      var v;
      if (lc === "and" && any && sawBig && toks[p + 1] && isNumberWord(toks[p + 1].lc)) continue;
      if (lc === "a" && !any && toks[p + 1] && (toks[p + 1].lc === "hundred" || toks[p + 1].lc === "thousand")) {
        cur = 1; any = true; continue;
      }
      v = has(UNITS, lc) ? UNITS[lc] : undefined;
      if (v === undefined && o && o.lenient) {
        if (lc === "to" || lc === "too") v = 2;
        else if (lc === "for") v = 4;
      }
      var low = cur % 100;
      /* "one fifty one" = 151, "two seventy" = 270 (hundreds elided in speech) */
      if (!(o && o.noElide) && any && p === start + 1 && cur >= 1 && cur <= 9 &&
          (has(TENS, lc) || (v !== undefined && v >= 10))) {
        cur *= 100;
        low = 0;
      }
      if (v !== undefined) {
        if (v === 0) {
          if (any) break;
          cur = 0; any = true; p++; break;
        }
        if (v <= 9 ? (low === 0 || (low >= 20 && low % 10 === 0)) : low === 0) {
          cur += v; any = true; continue;
        }
        break;
      }
      if (has(TENS, lc)) {
        if (low === 0) { cur += TENS[lc]; any = true; continue; }
        break;
      }
      if (o && o.ordinal !== false && has(ORD_UNITS, lc)) {
        var ou = ORD_UNITS[lc];
        if (ou <= 9 ? (low === 0 || (low >= 20 && low % 10 === 0)) : low === 0) {
          cur += ou; any = true; ordinal = true; p++;
        }
        break;
      }
      if (o && o.ordinal !== false && has(ORD_TENS, lc)) {
        if (low === 0) { cur += ORD_TENS[lc]; any = true; ordinal = true; p++; }
        break;
      }
      if (lc === "hundred") {
        if (any && cur > 0 && cur < 100) { cur *= 100; sawBig = true; continue; }
        break;
      }
      if (lc === "thousand") {
        if (any) { total += (cur || 1) * 1000; cur = 0; sawBig = true; continue; }
        break;
      }
      break;
    }
    if (!any) return null;
    return { value: total + cur, next: p, ordinal: ordinal };
  }

  /* Read a (possibly signed, possibly decimal) number starting at toks[i].
   * Returns {value, next, ordinal} or null. */
  function parseNumberAt(toks, i, o) {
    o = o || {};
    var p = i, sign = 1;
    while (toks[p] && (toks[p].lc === "minus" || toks[p].lc === "negative")) { sign = -sign; p++; }
    if (!toks[p]) return null;

    var value = null, ordinal = false, hasDecimalAlready = false, intText = null;
    var lc = toks[p].lc;
    var m = /^([-+]?)(\d+)(\.\d+)?$/.exec(lc);
    var om = /^(\d{1,2})(st|nd|rd|th)$/.exec(lc);
    if (m) {
      if (m[1] === "-") sign = -sign;
      intText = m[2];
      value = parseFloat(m[2] + (m[3] || ""));
      hasDecimalAlready = !!m[3];
      p++;
      if (!hasDecimalAlready && toks[p] && (toks[p].lc === "hundred" || toks[p].lc === "thousand")) {
        value *= toks[p].lc === "hundred" ? 100 : 1000;
        intText = null;
        p++;
      }
    } else if (om && o.ordinal !== false) {
      value = parseInt(om[1], 10); ordinal = true; p++;
    } else {
      var w = parseWords(toks, p, o);
      if (w) { value = w.value; p = w.next; ordinal = w.ordinal; intText = String(w.value); }
    }

    if (!ordinal && !hasDecimalAlready && toks[p] && has(POINT_WORDS, toks[p].lc)) {
      var q = p + 1, frac = "";
      while (toks[q] && (/^\d+$/.test(toks[q].lc) || isDigitWord(toks[q].lc) || toks[q].lc === "oh")) {
        frac += /^\d+$/.test(toks[q].lc) ? toks[q].lc : (toks[q].lc === "oh" ? "0" : DIGIT_WORDS[toks[q].lc]);
        q++;
      }
      if (frac) {
        value = parseFloat((intText !== null ? intText : (value === null ? "0" : String(value))) + "." + frac);
        p = q;
      }
    }
    if (value === null) return null;
    return { value: sign * value, next: p, ordinal: ordinal };
  }

  /* Convenience for tests/other callers: whole string -> number or null. */
  function parseNumber(text, opts) {
    var toks = tokenize(text);
    var r = parseNumberAt(toks, 0, opts || {});
    return r && r.next === toks.length ? r.value : null;
  }

  var NUMBER_FILLERS = {
    to: 1, is: 1, equals: 1, equal: 1, at: 1, of: 1, as: 1, be: 1, the: 1, value: 1, set: 1,
    around: 1, about: 1, approximately: 1, roughly: 1, "=": 1
  };

  /* A number right after a field keyword. Fillers ("to", "is", ...) are skipped; "to"/"two"
   * ambiguity is resolved here and nowhere else: "rainfall to 120" is 120, "depth to" is 2. */
  function readNumberSlot(toks, p) {
    var q = p;
    while (q < toks.length && has(NUMBER_FILLERS, toks[q].lc)) q++;
    var r = parseNumberAt(toks, q, { ordinal: false });
    if (r) return r;
    var back = q > p && (toks[q - 1].lc === "to" || toks[q - 1].lc === "too") ? q - 1 : q;
    return parseNumberAt(toks, back, { lenient: true, ordinal: false });
  }

  function pad2(n) {
    return (n < 10 ? "0" : "") + n;
  }

  function isoOf(y, mo, d) {
    return String(y).padStart(4, "0") + "-" + pad2(mo) + "-" + pad2(d);
  }

  function validYmd(y, mo, d) {
    if (!(y >= 1 && y <= 9999 && mo >= 1 && mo <= 12 && d >= 1)) return false;
    var dt = new Date(Date.UTC(2000, mo - 1, d));
    dt.setUTCFullYear(y);
    return dt.getUTCFullYear() === y && dt.getUTCMonth() === mo - 1 && dt.getUTCDate() === d;
  }

  function localIso(date) {
    return isoOf(date.getFullYear(), date.getMonth() + 1, date.getDate());
  }

  function yearFromParts(parts, now) {
    if (!parts.length) return now.getFullYear();
    if (parts.length === 1) {
      var y = parts[0];
      if (y >= 1000) return y;
      if (y < 100) return 2000 + y;
      return null;
    }
    if (parts.length === 2 && parts[0] >= 10 && parts[0] <= 99 && parts[1] >= 0 && parts[1] <= 99) {
      return parts[0] * 100 + parts[1];
    }
    if (parts.length === 3 && parts[1] === 0 && parts[0] >= 10 && parts[0] <= 99) {
      return parts[0] * 100 + parts[2];
    }
    for (var i = 0; i < parts.length; i++) if (parts[i] >= 1000) return parts[i];
    return null;
  }

  /* Spoken/typed date -> "yyyy-mm-dd" or null. `toks` are tokens (see tokenize) or a string. */
  function parseDate(input, now) {
    now = now instanceof Date ? now : new Date();
    var toks = typeof input === "string" ? tokenize(input) : input;
    var skip = { of: 1, the: 1, on: 1, and: 1, th: 1, st: 1, nd: 1, rd: 1, is: 1, to: 1, set: 1, as: 1, at: 1, date: 1 };
    var ts = [];
    toks.forEach(function (t) {
      if (has(skip, t.lc)) return;
      ts.push(t.lc === "oh" ? { raw: "zero", lc: "zero" } : t);
    });
    if (!ts.length) return null;

    var i, t0;
    for (i = 0; i < ts.length; i++) {
      var rel = { today: 0, tomorrow: 1, yesterday: -1 }[ts[i].lc];
      if (rel !== undefined) {
        var d = new Date(now.getFullYear(), now.getMonth(), now.getDate() + rel);
        return localIso(d);
      }
    }

    var m;
    for (i = 0; i < ts.length; i++) {
      t0 = ts[i].lc;
      if ((m = /^(\d{4})[-\/.](\d{1,2})[-\/.](\d{1,2})$/.exec(t0))) {
        return validYmd(+m[1], +m[2], +m[3]) ? isoOf(+m[1], +m[2], +m[3]) : null;
      }
      if ((m = /^(\d{1,2})[-\/.](\d{1,2})[-\/.](\d{2,4})$/.exec(t0))) {
        var a = +m[1], b = +m[2], yy = +m[3] < 100 ? 2000 + +m[3] : +m[3];
        var mo = a > 12 ? b : a, dd = a > 12 ? a : b;
        return validYmd(yy, mo, dd) ? isoOf(yy, mo, dd) : null;
      }
    }

    var allDigits = ts.length >= 3 && ts.slice(0, 3).every(function (t) { return /^\d+$/.test(t.lc); });
    if (allDigits) {
      var n0 = ts[0].lc, n1 = +ts[1].lc, n2 = ts[2].lc;
      if (n0.length === 4) return validYmd(+n0, n1, +n2) ? isoOf(+n0, n1, +n2) : null;
      var yr = +n2 < 100 ? 2000 + +n2 : +n2;
      var mm = +n0 > 12 ? n1 : +n0, dy = +n0 > 12 ? +n0 : n1;
      return validYmd(yr, mm, dy) ? isoOf(yr, mm, dy) : null;
    }

    /* month-name forms */
    var items = [], p = 0;
    while (p < ts.length) {
      if (has(MONTHS, ts[p].lc)) { items.push({ k: "m", v: MONTHS[ts[p].lc] }); p++; continue; }
      var r = parseNumberAt(ts, p, { lenient: true, noElide: true });
      if (r) { items.push({ k: "n", v: r.value, ord: r.ordinal }); p = r.next; } else p++;
    }
    var mi = -1;
    for (i = 0; i < items.length; i++) if (items[i].k === "m") { mi = i; break; }
    if (mi < 0) return null;
    var before = items.slice(0, mi).filter(function (x) { return x.k === "n"; });
    var after = items.slice(mi + 1).filter(function (x) { return x.k === "n"; });
    var day, parts;
    if (before.length) {
      day = before[before.length - 1].v;
      parts = after.map(function (x) { return x.v; });
    } else {
      if (!after.length) return null;
      day = after[0].v;
      parts = after.slice(1).map(function (x) { return x.v; });
      if (!after[0].ord && after.length === 1 && day > 31) return null;
    }
    var year = yearFromParts(parts, now);
    if (year === null || !validYmd(year, items[mi].v, day)) return null;
    return isoOf(year, items[mi].v, day);
  }

  /* Phone-ish digits: "nine eight 7 6" -> "9876" */
  function readDigitString(toks, p) {
    var out = "", q = p, sawPlus = false;
    if (toks[q] && toks[q].lc === "plus") { sawPlus = true; q++; }
    while (toks[q] && (/^\d+$/.test(toks[q].lc) || isDigitWord(toks[q].lc) || toks[q].lc === "oh")) {
      out += /^\d+$/.test(toks[q].lc) ? toks[q].lc : (toks[q].lc === "oh" ? "0" : DIGIT_WORDS[toks[q].lc]);
      q++;
    }
    return out ? { value: (sawPlus ? "+" : "") + out, next: q } : null;
  }

  /* =====================================================================
   * 2. Field table (names, labels, aliases)
   * ===================================================================== */

  var ALIAS_GROUPS = [
    { id: "lat", phrases: ["lat", "latitude"] },
    { id: "lon", phrases: ["lon", "long", "longitude", "lng"] },
    { id: "depth", phrases: ["depth"] },
    { id: "height", phrases: ["height", "wave height", "max height", "maximum height", "max wave height", "maximum wave height"] },
    { id: "rainfall", phrases: ["rainfall", "cumulative rainfall", "total rainfall", "rain", "cumulative rain"] },
    { id: "current-rainfall", phrases: ["current rainfall", "current rain"] },
    { id: "temperature", phrases: ["temperature", "temp"] },
    { id: "pressure", phrases: ["pressure"] },
    { id: "humidity", phrases: ["humidity"] },
    { id: "wind-direction", phrases: ["wind direction", "direction"] },
    { id: "wind-speed", phrases: ["wind speed", "speed"] },
    { id: "radius", phrases: ["radius", "distance", "range"] },
    { id: "city", phrases: ["city", "town"] },
    { id: "place", phrases: ["place", "location", "address"] },
    { id: "date", phrases: ["date"] },
    { id: "level-now", phrases: ["level now", "current level", "river level", "level"] },
    { id: "level-1h", phrases: ["level 1 hour ago", "level one hour ago", "level 1h ago", "level an hour ago"] },
    { id: "level-3h", phrases: ["level 3 hours ago", "level three hours ago", "level 3h ago"] },
    { id: "rain-3h", phrases: ["rain last 3 hours", "rain last three hours", "rain 3 hours", "rain three hours", "rain 3h"] },
    { id: "rain-6h", phrases: ["rain last 6 hours", "rain last six hours", "rain 6 hours", "rain six hours", "rain 6h"] },
    { id: "rain-24h", phrases: ["rain last 24 hours", "rain last twenty four hours", "rain 24 hours", "rain twenty four hours", "rain 24h", "rain a day"] },
    { id: "message", phrases: ["message", "text"] },
    { id: "phone", phrases: ["phone", "phone number", "mobile"] }
  ];

  function groupsOf(field) {
    var names = [norm(field.name), norm(stripParens(field.label || ""))];
    var out = [];
    ALIAS_GROUPS.forEach(function (g) {
      if (names.some(function (n) { return n && g.phrases.indexOf(n) >= 0; })) out.push(g);
    });
    return out;
  }

  function fieldLabel(f) {
    var l = stripParens(f.label || "") || String(f.name || "").replace(/[_-]+/g, " ");
    return l.replace(/\s+/g, " ").trim();
  }

  function fieldKind(f) {
    var t = String(f.type || "").toLowerCase();
    if (t === "date") return "date";
    if (t === "tel") return "tel";
    if (t === "number" || t === "range") return "number";
    var ids = groupsOf(f).map(function (g) { return g.id; });
    if (t === "text" || t === "search" || t === "textarea" || t === "email" || t === "url") return "text";
    if (ids.indexOf("date") >= 0) return "date";
    if (ids.indexOf("city") >= 0 || ids.indexOf("place") >= 0 || ids.indexOf("message") >= 0) return "text";
    if (ids.indexOf("phone") >= 0) return "tel";
    return "number";
  }

  /* phrase -> {words, field, strong}. Longest-match scanning happens in interpret(). */
  function buildFieldTable(fields) {
    var byPhrase = {};
    function add(phrase, field, strong) {
      phrase = norm(phrase);
      if (!phrase) return;
      var cur = byPhrase[phrase];
      if (cur && (cur.strong || !strong)) return;
      byPhrase[phrase] = { phrase: phrase, words: phrase.split(" "), field: field, strong: strong };
    }
    fields.forEach(function (f) {
      var own = [norm(f.name), norm(stripParens(f.label || ""))];
      own.forEach(function (p) { add(p, f, true); });
      groupsOf(f).forEach(function (g) { g.phrases.forEach(function (p) { add(p, f, true); }); });
      /* single-word tails of multi-word phrases are weak: "speed" for "wind speed" */
      own.concat(groupsOf(f).reduce(function (acc, g) { return acc.concat(g.phrases); }, [])).forEach(function (p) {
        var w = p.split(" ");
        if (w.length > 1) add(w[w.length - 1], f, false);
      });
    });
    return Object.keys(byPhrase).map(function (k) { return byPhrase[k]; });
  }

  /* =====================================================================
   * 3. Command interpreter
   * ===================================================================== */

  var SUBMIT_WORDS = ["submit", "estimate", "calculate", "predict", "compare", "search", "go", "classify", "score"];
  var NAV_VERBS = ["open", "go to", "goto", "show", "show me", "take me to", "navigate to", "visit", "switch to", "load"];
  var NAV_FAIL_VERBS = ["open", "go to", "goto", "take me to", "navigate to", "visit", "switch to"];
  var CONTROLS = [
    ["read", ["read result", "read results", "read the result", "read the results", "read aloud", "read it", "read that", "read out", "read out loud", "read"]],
    ["stop", ["stop", "quiet", "silence", "be quiet", "stop reading", "stop talking", "shut up"]],
    ["repeat", ["repeat", "repeat that", "say again", "say that again", "read again", "again"]],
    ["help", ["help", "what can i say", "voice commands", "commands", "show commands"]],
    ["stop-listening", ["stop listening", "turn off microphone", "mic off", "microphone off"]]
  ];
  var CONFIRM_TOKENS = { confirm: 1, confirmed: 1, yes: 1, yeah: 1, yep: 1, proceed: 1, please: 1, go: 1, ahead: 1, do: 1, it: 1, okay: 1, ok: 1 };
  var CONFIRM_STRONG = { confirm: 1, confirmed: 1, yes: 1, yeah: 1, yep: 1, proceed: 1, okay: 1, ok: 1 };
  var CANCEL_TOKENS = { cancel: 1, no: 1, nope: 1, abort: 1, never: 1, mind: 1, please: 1, stop: 1, dont: 1, "don't": 1 };
  var BARE_FILLERS = { please: 1, the: 1, page: 1, tab: 1, section: 1, screen: 1, now: 1 };
  var TEXT_LEAD = { to: 1, is: 1, equals: 1, equal: 1, as: 1, set: 1, the: 1, named: 1, called: 1, be: 1, "=": 1 };
  var TEXT_TRAIL = { and: 1, then: 1, please: 1 };
  var RANK = { command: 5, nav: 4, control: 3, navfail: 3, submit: 2, field: 1 };

  function allIn(lcs, set) {
    return lcs.length > 0 && lcs.every(function (t) { return has(set, t); });
  }

  function isConfirmUtterance(lcs) {
    if (!allIn(lcs, CONFIRM_TOKENS)) return false;
    var joined = lcs.join(" ");
    return joined === "go ahead" || joined === "do it" || lcs.some(function (t) { return has(CONFIRM_STRONG, t); });
  }

  function titleCase(s) {
    return s.replace(/\b([a-z])/g, function (m) { return m.toUpperCase(); });
  }

  function interpret(transcript, context) {
    context = context || {};
    var fields = context.fields || [];
    var cmds = context.commands || [];
    var navs = context.nav || [];
    var now = context.now instanceof Date ? context.now : new Date();
    var nowMs = typeof context.nowMs === "number" ? context.nowMs : Date.now();
    var toks = tokenize(transcript);
    var n = toks.length;
    var lcs = toks.map(function (t) { return t.lc; });
    var actions = [];
    if (!n) return actions;

    /* -- pending confirmation ------------------------------------------------ */
    var pending = context.pending;
    if (pending) {
      var age = nowMs - (pending.at || 0);
      if (age > CONFIRM_WINDOW_MS) {
        actions.push({ type: "cancel", reason: "timeout" });
      } else if (isConfirmUtterance(lcs)) {
        return [{ type: "confirm", id: pending.id, phrase: pending.phrase }];
      } else if (allIn(lcs, CANCEL_TOKENS) && lcs.join(" ") !== "stop") {
        return [{ type: "cancel", reason: "cancelled" }];
      } else {
        actions.push({ type: "cancel", reason: "not confirmed" });
      }
    }

    /* -- tables ------------------------------------------------------------- */
    var ftable = buildFieldTable(fields);
    var cmdTable = [];
    cmds.forEach(function (c) {
      String(c.phrase || "").split("|").forEach(function (ph) {
        var w = words(ph);
        if (w.length) cmdTable.push({ words: w, id: c.id, phrase: norm(ph), confirm: !!c.confirm });
      });
    });
    var navTable = [];
    navs.forEach(function (e) {
      var base = words(e.label);
      var variants = [base];
      var noDigits = base.filter(function (w) { return !/^\d+$/.test(w); });
      if (noDigits.length && noDigits.length !== base.length) variants.push(noDigits);
      variants.forEach(function (w) {
        if (w.length) navTable.push({ words: w, label: e.label, href: e.href, current: !!e.current });
      });
    });

    function startsWith(i, w) {
      if (i + w.length > n) return false;
      for (var j = 0; j < w.length; j++) if (lcs[i + j] !== w[j]) return false;
      return true;
    }

    /* -- bare nav ("home", "alerts page") ------------------------------------ */
    var bare = lcs.filter(function (t) { return !has(BARE_FILLERS, t); }).join(" ");
    for (var bi = 0; bi < navTable.length; bi++) {
      if (navTable[bi].words.join(" ") === bare && !(bare === "compare" && navTable[bi].current)) {
        actions.push({ type: "navigate", href: navTable[bi].href, label: navTable[bi].label });
        return actions;
      }
    }

    /* -- longest match at position i ----------------------------------------- */
    function matchAt(i) {
      var best = null;
      function consider(kind, w, extra) {
        if (!startsWith(i, w)) return;
        if (!best || w.length > best.len || (w.length === best.len && RANK[kind] > RANK[best.kind])) {
          best = Object.assign({ kind: kind, len: w.length }, extra || {});
        }
      }
      cmdTable.forEach(function (c) { consider("command", c.words, { cmd: c }); });
      navTable.forEach(function (e) {
        NAV_VERBS.forEach(function (v) {
          var vw = v.split(" ");
          consider("nav", vw.concat(e.words), { entry: e });
          consider("nav", vw.concat(["the"], e.words), { entry: e });
        });
      });
      NAV_FAIL_VERBS.forEach(function (v) { consider("navfail", v.split(" ")); });
      CONTROLS.forEach(function (c) { c[1].forEach(function (ph) { consider("control", ph.split(" "), { control: c[0] }); }); });
      SUBMIT_WORDS.forEach(function (w) { consider("submit", [w]); });
      ftable.forEach(function (e) { consider("field", e.words, { entry: e }); });
      return best;
    }

    /* A match ends a free-text/date slot; ambiguous numeric aliases ("long") only do so
     * when a number follows, so "city Long Beach" survives. */
    function endsSlot(m, j, free) {
      if (free) {
        /* free text (messages): only fields, data-voice-commands and a trailing submit word end it */
        if (m.kind === "command") return true;
        if (m.kind === "submit") return j + m.len === n;
        if (m.kind !== "field") return false;
      }
      if (m.kind !== "field") return true;
      var k = fieldKind(m.entry.field);
      if (k === "number") return !!readNumberSlot(toks, j + m.len);
      return true;
    }

    function slotEnd(start, free) {
      var j = start;
      while (j < n) {
        var m = matchAt(j);
        if (m && endsSlot(m, j, free)) break;
        j++;
      }
      return j;
    }

    function outOfRange(f, v, kind) {
      var lo = f.min, hi = f.max;
      if (kind === "number") {
        lo = lo === undefined || lo === null || lo === "" ? null : Number(lo);
        hi = hi === undefined || hi === null || hi === "" ? null : Number(hi);
        if ((lo !== null && !isNaN(lo) && v < lo) || (hi !== null && !isNaN(hi) && v > hi)) {
          return { lo: lo, hi: hi };
        }
      } else if (kind === "date") {
        if ((lo && v < lo) || (hi && v > hi)) return { lo: lo || null, hi: hi || null };
      }
      return null;
    }

    function rangeText(r) {
      if (r.lo !== null && r.hi !== null) return r.lo + " to " + r.hi;
      return r.lo !== null ? "at least " + r.lo : "at most " + r.hi;
    }

    function fillOrReject(f, value, kind, spoken) {
      var label = fieldLabel(f);
      var r = outOfRange(f, value, kind);
      if (r) {
        actions.push({
          type: "reject", name: f.name, label: label, value: value, reason: "out of range",
          message: label + " " + value + " is out of range (" + rangeText(r) + ")"
        });
      } else {
        actions.push({ type: "fill", name: f.name, label: label, value: value, kind: kind, spoken: spoken });
      }
    }

    function fieldError(f, what) {
      actions.push({
        type: "error", name: f.name, label: fieldLabel(f),
        message: "Didn't catch " + what + " for " + fieldLabel(f)
      });
    }

    var dateField = null;
    fields.forEach(function (f) { if (!dateField && fieldKind(f) === "date") dateField = f; });

    /* -- scan ---------------------------------------------------------------- */
    var i = 0;
    while (i < n) {
      var m = matchAt(i);
      if (!m) {
        if (dateField && (lcs[i] === "today" || lcs[i] === "tomorrow" || lcs[i] === "yesterday") &&
            !actions.some(function (a) { return a.name === dateField.name; })) {
          fillOrReject(dateField, parseDate([toks[i]], now), "date");
        }
        i++;
        continue;
      }
      if (m.kind === "command") {
        actions.push({ type: "command", id: m.cmd.id, phrase: m.cmd.phrase, confirm: m.cmd.confirm });
        i += m.len;
      } else if (m.kind === "nav") {
        actions.push({ type: "navigate", href: m.entry.href, label: m.entry.label });
        i += m.len;
        if (lcs[i] === "page") i++;
      } else if (m.kind === "navfail") {
        var rest = toks.slice(i + m.len, i + m.len + 3).map(function (t) { return t.raw; }).join(" ");
        actions.push({ type: "error", message: rest ? "No page named “" + rest + "”" : "Which page?" });
        i = n;
      } else if (m.kind === "control") {
        actions.push({ type: m.control });
        i += m.len;
      } else if (m.kind === "submit") {
        actions.push({ type: "submit", word: lcs[i] });
        i += 1;
      } else {
        var f = m.entry.field, kind = fieldKind(f), start = i + m.len, end, slot, ids;
        if (kind === "date") {
          end = slotEnd(start);
          var iso = parseDate(toks.slice(start, end), now);
          if (iso) fillOrReject(f, iso, "date");
          else fieldError(f, "a date");
          i = Math.max(end, start);
        } else if (kind === "tel") {
          slot = readDigitString(toks, start);
          if (slot) { fillOrReject(f, slot.value, "tel"); i = slot.next; } else { fieldError(f, "digits"); i = start; }
        } else if (kind === "text") {
          var s0 = start;
          while (s0 < n && has(TEXT_LEAD, lcs[s0])) s0++;
          ids = groupsOf(f).map(function (g) { return g.id; });
          end = slotEnd(s0, ids.indexOf("message") >= 0 || f.type === "textarea");
          var e0 = end;
          while (e0 > s0 && has(TEXT_TRAIL, lcs[e0 - 1])) e0--;
          var text = toks.slice(s0, e0).map(function (t) { return t.raw; }).join(" ");
          ids = groupsOf(f).map(function (g) { return g.id; });
          if (text && (ids.indexOf("city") >= 0 || ids.indexOf("place") >= 0) && text === text.toLowerCase()) text = titleCase(text);
          if (text) actions.push({ type: "fill", name: f.name, label: fieldLabel(f), value: text, kind: "text" });
          else fieldError(f, "a value");
          i = Math.max(end, start);
        } else {
          slot = readNumberSlot(toks, start);
          if (slot) {
            var val = slot.value, next = slot.next;
            ids = groupsOf(f).map(function (g) { return g.id; });
            var hemi = toks[next] ? toks[next].lc : "";
            if (ids.indexOf("lat") >= 0 && (hemi === "north" || hemi === "south")) {
              if (hemi === "south") val = -Math.abs(val);
              next++;
            } else if (ids.indexOf("lon") >= 0 && (hemi === "east" || hemi === "west")) {
              if (hemi === "west") val = -Math.abs(val);
              next++;
            }
            fillOrReject(f, val, "number");
            i = next;
          } else {
            fieldError(f, "a number");
            i = start;
          }
        }
      }
    }

    /* -- post-processing ------------------------------------------------------ */
    /* Submit runs after fills; a confirm-gated command is alone (besides fills). */
    var confirmCmd = null;
    actions.forEach(function (a) { if (!confirmCmd && a.type === "command" && a.confirm) confirmCmd = a; });
    var result = actions.filter(function (a) {
      if (confirmCmd) return a === confirmCmd || a.type === "fill" || a.type === "reject" || a.type === "error" || a.type === "cancel";
      return true;
    });
    var submits = result.filter(function (a) { return a.type === "submit"; });
    result = result.filter(function (a) { return a.type !== "submit"; });
    if (submits.length) result.push(submits[0]);
    return result;
  }

  /* =====================================================================
   * 4. Speech helpers (pure)
   * ===================================================================== */

  function cleanForSpeech(text) {
    return String(text == null ? "" : text)
      .replace(/https?:\/\/\S+/g, " ")
      .replace(/≥/g, " at least ")
      .replace(/≤/g, " at most ")
      .replace(/±/g, " plus or minus ")
      .replace(/°/g, " degrees ")
      .replace(/%/g, " percent")
      .replace(/&/g, " and ")
      .replace(/[·•]/g, ", ")
      .replace(/\s+/g, " ")
      .trim();
  }

  /* Split into chunks of at most `max` chars, breaking at sentence boundaries (never inside
   * "3.47"), then at commas, then at spaces. */
  function chunkText(text, max) {
    max = max || 200;
    text = String(text == null ? "" : text).replace(/\s+/g, " ").trim();
    if (!text) return [];
    var sentences = text.replace(/([.!?;:])\s+/g, "$1\u0000").split("\u0000");
    var out = [], cur = "";
    function flush() {
      if (cur) out.push(cur);
      cur = "";
    }
    function pieces(s) {
      if (s.length <= max) return [s];
      var res = [], acc = "";
      s.replace(/,\s+/g, ",\u0000").split("\u0000").forEach(function (part) {
        if (part.length > max) {
          if (acc) { res.push(acc); acc = ""; }
          var line = "";
          part.split(" ").forEach(function (w) {
            while (w.length > max) {
              if (line) { res.push(line); line = ""; }
              res.push(w.slice(0, max));
              w = w.slice(max);
            }
            if ((line ? line.length + 1 : 0) + w.length > max) { res.push(line); line = w; } else line = line ? line + " " + w : w;
          });
          if (line) res.push(line);
        } else if ((acc ? acc.length + 1 : 0) + part.length > max) {
          res.push(acc); acc = part;
        } else acc = acc ? acc + " " + part : part;
      });
      if (acc) res.push(acc);
      return res;
    }
    sentences.forEach(function (s) {
      pieces(s).forEach(function (piece) {
        if (!cur) cur = piece;
        else if (cur.length + 1 + piece.length <= max) cur += " " + piece;
        else { flush(); cur = piece; }
      });
    });
    flush();
    return out;
  }

  function capSummary(text, cap) {
    cap = cap || 280;
    text = String(text || "").replace(/\s+/g, " ").trim();
    if (text.length <= cap) return text;
    var cut = text.slice(0, cap);
    var stop = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf("; "));
    if (stop > cap * 0.5) return cut.slice(0, stop + 1);
    var sp = cut.lastIndexOf(" ");
    return (sp > 0 ? cut.slice(0, sp) : cut).replace(/[,;:\s]+$/, "") + ".";
  }

  function normLang(l) {
    return String(l || "").toLowerCase().replace(/_/g, "-");
  }

  function pickVoice(voices, lang) {
    var want = normLang(lang);
    if (!voices || !voices.length || !want) return null;
    var base = want.split("-")[0], best = null, bestScore = -1;
    for (var i = 0; i < voices.length; i++) {
      var v = voices[i], l = normLang(v.lang), s;
      if (l === want) s = 100;
      else if (l.split("-")[0] === base) s = 60;
      else continue;
      if (v.localService) s += 5;
      if (v.default) s += 3;
      if (s > bestScore) { bestScore = s; best = v; }
    }
    return best;
  }

  var api = {
    interpret: interpret,
    tokenize: tokenize,
    parseNumber: parseNumber,
    parseNumberAt: parseNumberAt,
    parseDate: parseDate,
    buildFieldTable: buildFieldTable,
    chunkText: chunkText,
    capSummary: capSummary,
    cleanForSpeech: cleanForSpeech,
    pickVoice: pickVoice,
    CONFIRM_WINDOW_MS: CONFIRM_WINDOW_MS
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;

  if (!root.document || !root.window) return;

  /* =====================================================================
   * 5. Browser runtime
   * ===================================================================== */

  var doc = root.document;
  var win = root.window;
  var REGION_SEL = '#result, [id^="result-"], #text-score, [data-speak-region]';
  var LS_ON = "ndp.voice.on";
  var LS_EVER = "ndp.voice.ever";

  function lsGet(k) {
    try { return win.localStorage.getItem(k); } catch (e) { return null; }
  }
  function lsSet(k, v) {
    try { win.localStorage.setItem(k, v); } catch (e) { /* storage blocked */ }
  }
  function $(sel, rootEl) {
    return (rootEl || doc).querySelector(sel);
  }
  function $$(sel, rootEl) {
    return Array.prototype.slice.call((rootEl || doc).querySelectorAll(sel));
  }
  function visible(el) {
    if (!el) return false;
    if (el.hidden) return false;
    var rects = el.getClientRects ? el.getClientRects().length : 1;
    if (!rects) return false;
    var cs = win.getComputedStyle ? win.getComputedStyle(el) : null;
    return !cs || (cs.visibility !== "hidden" && cs.display !== "none");
  }

  function currentLang() {
    var sel = $("#ndp-lang");
    var v = sel && sel.value ? String(sel.value) : "";
    if (/^[a-z]{2,3}([-_][A-Za-z0-9]{2,4})?$/.test(v)) return v.replace("_", "-");
    return doc.documentElement.lang || (win.navigator && win.navigator.language) || "en-US";
  }

  /* ---- CSS ---------------------------------------------------------------- */
  function injectCss() {
    if (doc.getElementById("ndp-voice-css") || !doc.head) return;
    var link = doc.createElement("link");
    link.id = "ndp-voice-css";
    link.rel = "stylesheet";
    link.href = SELF_SRC ? SELF_SRC.replace(/voice\.js(?=([?#]|$))/, "voice.css") : "/static/voice.css";
    doc.head.appendChild(link);
  }

  /* ---- toast & help -------------------------------------------------------- */
  var toastEl = null, toastTimer = null;

  function ensureToast() {
    if (toastEl) return toastEl;
    toastEl = doc.createElement("div");
    toastEl.id = "ndp-voice-toast";
    toastEl.className = "ndp-voice-toast";
    toastEl.setAttribute("role", "status");
    toastEl.setAttribute("aria-live", "polite");
    toastEl.hidden = true;
    var text = doc.createElement("span");
    text.className = "ndp-voice-toast-text";
    var help = doc.createElement("button");
    help.type = "button";
    help.className = "ndp-voice-toast-help";
    help.setAttribute("aria-label", "Show voice commands");
    help.textContent = "?";
    help.addEventListener("click", function () { openHelp(); });
    toastEl.appendChild(text);
    toastEl.appendChild(help);
    doc.body.appendChild(toastEl);
    return toastEl;
  }

  function showToast(message, sticky) {
    var el = ensureToast();
    el.firstChild.textContent = message;
    el.hidden = false;
    if (toastTimer) win.clearTimeout(toastTimer);
    toastTimer = null;
    if (!sticky) toastTimer = win.setTimeout(function () {
      if (mic.listening) showToast(listeningText(), true); else hideToast();
    }, 5000);
  }

  function hideToast() {
    if (toastTimer) win.clearTimeout(toastTimer);
    toastTimer = null;
    if (toastEl) toastEl.hidden = true;
  }

  var helpDialog = null;

  function helpContent() {
    var ctx = buildContext();
    var rows = [
      ["Fill fields", "latitude 28.6 longitude 77.2 depth 10 · set rainfall to 120 · temperature forty two pressure 1013"],
      ["Dates", "date today · date twenty fifth of October twenty twenty six · date 2026 10 25"],
      ["Text", "city Kochi · place Vellore"],
      ["Submit", "submit · estimate · calculate · predict · compare · search · go"],
      ["Pages", "open flood · go to earthquake · home · alerts"],
      ["Results", "read result · repeat · stop (or quiet)"],
      ["Mic", "stop listening · help"]
    ];
    if (ctx.fields.length) rows.push(["Fields here", ctx.fields.map(function (f) { return fieldLabel(f); }).join(", ")]);
    if (ctx.commands.length) rows.push(["Buttons here", ctx.commands.map(function (c) { return c.phrase + (c.confirm ? " (asks to confirm)" : ""); }).join(", ")]);
    return rows;
  }

  function openHelp() {
    if (!helpDialog) {
      helpDialog = doc.createElement("dialog");
      helpDialog.className = "ndp-voice-help";
      helpDialog.setAttribute("aria-labelledby", "ndp-voice-help-title");
      helpDialog.addEventListener("click", function (e) { if (e.target === helpDialog && helpDialog.close) helpDialog.close(); });
      doc.body.appendChild(helpDialog);
    }
    helpDialog.textContent = "";
    var h = doc.createElement("h2");
    h.id = "ndp-voice-help-title";
    h.textContent = "Voice commands";
    helpDialog.appendChild(h);
    var dl = doc.createElement("dl");
    helpContent().forEach(function (r) {
      var dt = doc.createElement("dt"), dd = doc.createElement("dd");
      dt.textContent = r[0]; dd.textContent = r[1];
      dl.appendChild(dt); dl.appendChild(dd);
    });
    helpDialog.appendChild(dl);
    var note = doc.createElement("p");
    note.className = "ndp-voice-help-note";
    note.textContent = "Speech recognition is done by your browser's speech service, so it may send audio to it.";
    helpDialog.appendChild(note);
    var close = doc.createElement("button");
    close.type = "button";
    close.className = "ndp-voice-help-close";
    close.textContent = "Close";
    close.addEventListener("click", function () { if (helpDialog.close) helpDialog.close(); else helpDialog.removeAttribute("open"); });
    helpDialog.appendChild(close);
    if (typeof helpDialog.showModal === "function") {
      if (!helpDialog.open) helpDialog.showModal();
    } else helpDialog.setAttribute("open", "");
    close.focus();
  }

  /* ---- voice output --------------------------------------------------------- */
  var synthOk = !!(win.speechSynthesis && win.SpeechSynthesisUtterance);
  var tts = {
    enabled: false, ever: false, speaking: false, gen: 0, last: [], source: null,
    voices: [], utterances: [], blocked: null
  };

  function refreshVoices() {
    try { tts.voices = win.speechSynthesis.getVoices() || []; } catch (e) { tts.voices = []; }
  }

  function setSpeaking(on, source) {
    tts.speaking = on;
    tts.source = on ? source || null : null;
    updateReadButtons();
  }

  /* parts: [{text, alert, source}] spoken in order; cancels anything in progress. */
  function speak(parts, source) {
    if (!synthOk) return false;
    parts = (parts || []).map(function (p) {
      return { text: cleanForSpeech(p.text), alert: !!p.alert };
    }).filter(function (p) { return p.text; });
    if (!parts.length) return false;
    var synth = win.speechSynthesis;
    try {
      synth.cancel();
      if (synth.paused) synth.resume();
    } catch (e) { /* ignore */ }
    tts.last = parts;
    var gen = ++tts.gen;
    if (!tts.voices.length) refreshVoices();
    var lang = currentLang();
    var voice = pickVoice(tts.voices, lang);
    var queue = [];
    parts.forEach(function (p) {
      chunkText(p.text, 200).forEach(function (c) { queue.push({ text: c, alert: p.alert }); });
    });
    var remaining = queue.length;
    tts.utterances = [];
    queue.forEach(function (q) {
      var u = new win.SpeechSynthesisUtterance(q.text);
      u.lang = voice ? voice.lang : lang;
      if (voice) u.voice = voice;
      u.rate = q.alert ? 0.9 : 1;
      u.pitch = q.alert ? 1.15 : 1;
      u.volume = 1;
      u.onstart = function () { if (gen === tts.gen) setSpeaking(true, source); };
      u.onend = function () {
        if (gen !== tts.gen) return;
        remaining--;
        if (remaining <= 0) setSpeaking(false);
      };
      u.onerror = function (ev) {
        if (gen !== tts.gen) return;
        if (ev && ev.error === "not-allowed") {
          tts.blocked = parts;
          setSpeaking(false);
          showToast("Your browser blocked automatic speech. Tap anywhere to hear the result.");
          return;
        }
        remaining--;
        if (remaining <= 0) setSpeaking(false);
      };
      tts.utterances.push(u);
      try { synth.speak(u); } catch (e) { remaining--; }
    });
    setSpeaking(true, source);
    return true;
  }

  function cancelSpeech() {
    tts.gen++;
    tts.blocked = null;
    if (synthOk) { try { win.speechSynthesis.cancel(); } catch (e) { /* ignore */ } }
    setSpeaking(false);
  }

  /* Build the text for a result container: data-speak elements, else a short summary. */
  function speakEls(node) {
    var out = [];
    if (node.matches && node.matches("[data-speak]")) out.push(node);
    return out.concat($$("[data-speak]", node));
  }

  /* textContent, but with a space between sibling elements ("4.82" + "m" -> "4.82 m"). */
  function spacedText(node) {
    if (node.nodeType === 3) return node.nodeValue;
    if (node.nodeType !== 1 || /^(SCRIPT|STYLE)$/.test(node.tagName)) return "";
    var out = [];
    for (var c = node.firstChild; c; c = c.nextSibling) out.push(spacedText(c));
    return out.join(" ");
  }

  function summarizeRegion(region) {
    var alertEl = $('[role="alert"]', region);
    if (alertEl) return capSummary(cleanForSpeech(spacedText(alertEl)));
    var parts = [];
    var h = $("h1, h2, h3, h4", region);
    if (h) parts.push(cleanForSpeech(spacedText(h)));
    var pairs = [];
    $$(".uppercase, [data-speak-label]", region).some(function (labelEl) {
      if (labelEl.closest("dl, table, ul, ol, details, blockquote")) return false;
      var val = labelEl.nextElementSibling;
      if (!val) return false;
      var lt = cleanForSpeech(spacedText(labelEl)).toLowerCase();
      var vt = cleanForSpeech(spacedText(val));
      if (!lt || !vt || vt.length > 40) return false;
      pairs.push(lt + " " + vt);
      return pairs.length >= 3;
    });
    var text = parts.concat(pairs).join(". ");
    return capSummary(text ? text + "." : "");
  }

  function collectParts(container) {
    var els = speakEls(container);
    if (els.length) {
      return els.map(function (el) {
        return { text: el.getAttribute("data-speak"), alert: el.getAttribute("data-speak-priority") === "alert" };
      });
    }
    var s = summarizeRegion(container);
    return s ? [{ text: s }] : [];
  }

  var spokenEls = typeof WeakSet === "function" ? new WeakSet() : null;
  var lastSummary = { text: "", at: 0 };

  function regionsIn(node) {
    var out = [];
    if (node.matches && node.matches(REGION_SEL)) out.push(node);
    else {
      var up = node.closest && node.closest(REGION_SEL);
      if (up) out.push(up);
    }
    return out.concat($$(REGION_SEL, node));
  }

  /* Swaps that happen before the user ever touched the page (hx-trigger="load" lists) must not
   * start talking; browsers refuse speech without user activation anyway. */
  var activated = false;
  function userActive() {
    return activated || !!(win.navigator && win.navigator.userActivation && win.navigator.userActivation.hasBeenActive);
  }
  var MAX_AUTO_SPOKEN = 3;

  /* Called for freshly inserted content (htmx swap, htmx:load, first page load). */
  function handleInserted(node, initial) {
    if (!synthOk || !node || node.nodeType !== 1) return;
    var parts = [];
    var els = speakEls(node);
    var up = node.closest && node.closest("[data-speak]");
    if (up && els.indexOf(up) < 0) els.unshift(up);
    var fresh = els.filter(function (el) { return !spokenEls || !spokenEls.has(el); });
    var gated = !initial && !userActive();
    var plain = 0;
    fresh.forEach(function (el) {
      if (spokenEls) spokenEls.add(el);
      var alert = el.getAttribute("data-speak-priority") === "alert";
      if (gated || (!tts.enabled && !(alert && tts.ever))) return;
      if (!alert && ++plain > MAX_AUTO_SPOKEN) return; /* long lists: use the Read aloud button */
      parts.push({ text: el.getAttribute("data-speak"), alert: alert });
    });
    if (!els.length && tts.enabled && !gated) {
      var seen = [];
      regionsIn(node).forEach(function (region) {
        if (seen.indexOf(region) >= 0 || !region.textContent.trim()) return;
        seen.push(region);
        if ($("[data-speak]", region)) return;
        var s = summarizeRegion(region);
        var t = Date.now();
        if (!s || (s === lastSummary.text && t - lastSummary.at < 800)) return;
        lastSummary = { text: s, at: t };
        parts.push({ text: s });
      });
    }
    parts.sort(function (a, b) { return (b.alert ? 1 : 0) - (a.alert ? 1 : 0); });
    if (parts.length) speak(parts, null);
  }

  /* ---- read aloud buttons ----------------------------------------------------- */
  var readBars = [];

  function ensureReadBars() {
    if (!synthOk) return;
    $$(REGION_SEL).forEach(function (container) {
      if (container._ndpVoiceBar) { updateBar(container); return; }
      var bar = doc.createElement("div");
      bar.className = "ndp-voice-bar";
      var btn = doc.createElement("button");
      btn.type = "button";
      btn.className = "ndp-voice-read";
      btn.addEventListener("click", function () {
        if (tts.speaking && tts.source === container) { cancelSpeech(); return; }
        var parts = collectParts(container);
        if (parts.length) speak(parts, container); else showToast("Nothing to read yet.");
      });
      bar.appendChild(btn);
      container.parentNode.insertBefore(bar, container);
      container._ndpVoiceBar = bar;
      container._ndpVoiceBtn = btn;
      readBars.push(container);
      if (typeof win.MutationObserver === "function") {
        new win.MutationObserver(function () { updateBar(container); }).observe(container, { childList: true });
      }
      updateBar(container);
    });
  }

  function updateBar(container) {
    var bar = container._ndpVoiceBar, btn = container._ndpVoiceBtn;
    if (!bar) return;
    bar.hidden = !container.textContent.trim();
    var active = tts.speaking && tts.source === container;
    btn.textContent = active ? "Stop" : "Read aloud";
    btn.setAttribute("aria-label", active ? "Stop reading the result" : "Read the result aloud");
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  }

  function updateReadButtons() {
    readBars.forEach(function (c) { if (doc.contains(c)) updateBar(c); });
  }

  /* ---- toggle ---------------------------------------------------------------------- */
  function initOutput() {
    if (!synthOk) return;
    tts.enabled = lsGet(LS_ON) === "1";
    tts.ever = lsGet(LS_EVER) === "1" || tts.enabled;
    refreshVoices();
    if (win.speechSynthesis.addEventListener) win.speechSynthesis.addEventListener("voiceschanged", refreshVoices);
    else win.speechSynthesis.onvoiceschanged = refreshVoices;

    var btn = $("#ndp-voice-toggle");
    function paint() {
      if (!btn) return;
      btn.setAttribute("aria-pressed", tts.enabled ? "true" : "false");
      btn.title = tts.enabled ? "Reading results aloud (click to turn off)" : "Read results aloud";
    }
    if (btn) {
      btn.hidden = false;
      paint();
      btn.addEventListener("click", function () {
        tts.enabled = !tts.enabled;
        lsSet(LS_ON, tts.enabled ? "1" : "0");
        if (tts.enabled) {
          tts.ever = true;
          lsSet(LS_EVER, "1");
          speak([{ text: "Voice on." }], null);
        } else {
          cancelSpeech();
        }
        paint();
      });
    }

    doc.addEventListener("htmx:afterSwap", function (e) {
      var t = (e.detail && e.detail.target) || e.target;
      ensureReadBars();
      handleInserted(t);
    });
    doc.addEventListener("htmx:load", function (e) {
      ensureReadBars();
      handleInserted(e.target);
    });
    var gestureRetry = function () {
      if (tts.blocked) { var p = tts.blocked; tts.blocked = null; speak(p, null); }
    };
    doc.addEventListener("pointerdown", gestureRetry, true);
    doc.addEventListener("keydown", gestureRetry, true);
    ["pointerdown", "keydown", "touchstart"].forEach(function (ev) {
      doc.addEventListener(ev, function () { activated = true; }, true);
    });
    win.addEventListener("pagehide", cancelSpeech);
    win.addEventListener("beforeunload", cancelSpeech);

    ensureReadBars();
    if (doc.body) handleInserted(doc.body, true);
  }

  /* ---- form / context ------------------------------------------------------------------- */
  var lastForm = null;

  function labelTextFor(el) {
    var labels = el.labels && el.labels.length ? Array.prototype.slice.call(el.labels) : [];
    if (!labels.length && el.id) {
      var byFor = doc.querySelector('label[for="' + (win.CSS && win.CSS.escape ? win.CSS.escape(el.id) : el.id) + '"]');
      if (byFor) labels = [byFor];
    }
    if (labels.length) {
      var lab = labels[0];
      if (lab.contains(el)) {
        /* wrapping label: only the first piece of text; later spans are help text */
        for (var k = 0; k < lab.childNodes.length; k++) {
          var node = lab.childNodes[k];
          if (node === el || (node.contains && node.contains(el))) continue;
          var tx = (node.textContent || "").replace(/\s+/g, " ").trim();
          if (tx) return tx;
        }
      } else {
        return (lab.textContent || "").replace(/\s+/g, " ").trim();
      }
    }
    return el.getAttribute("aria-label") || el.getAttribute("placeholder") || "";
  }

  function fieldEls(form) {
    return Array.prototype.slice.call(form.elements || []).filter(function (el) {
      if (!el.name) return false;
      var tag = el.tagName;
      if (tag === "TEXTAREA") return !el.disabled && !el.readOnly;
      if (tag !== "INPUT") return false;
      var t = (el.type || "text").toLowerCase();
      if (["hidden", "submit", "button", "reset", "checkbox", "radio", "file", "password", "image", "color"].indexOf(t) >= 0) return false;
      return !el.disabled && !el.readOnly && visible(el);
    });
  }

  function formFields(form) {
    return fieldEls(form).map(function (el) {
      var type = el.tagName === "TEXTAREA" ? "textarea" : (el.type || "text").toLowerCase();
      var f = { name: el.name, label: labelTextFor(el), type: type };
      if (el.getAttribute("min") !== null) f.min = el.getAttribute("min");
      if (el.getAttribute("max") !== null) f.max = el.getAttribute("max");
      return f;
    });
  }

  function candidateForms() {
    return $$("form").filter(function (f) { return !f.closest("header") && fieldEls(f).length && visible(f); });
  }

  function activeForm() {
    var ae = doc.activeElement;
    var f = ae && ae.closest ? ae.closest("form") : null;
    if (f && !f.closest("header")) return f;
    if (lastForm && doc.contains(lastForm) && visible(lastForm)) return lastForm;
    var forms = candidateForms();
    if (!forms.length) return $("form");
    var vh = win.innerHeight || 800, best = forms[0], bestVis = -1;
    forms.forEach(function (form) {
      var r = form.getBoundingClientRect();
      var vis = Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
      if (vis > bestVis + 1) { best = form; bestVis = vis; }
    });
    return best;
  }

  function primarySubmit(form) {
    var btns = $$('button[type="submit"], input[type="submit"], button:not([type])', form).filter(function (b) {
      return !b.disabled && visible(b);
    });
    return btns[0] || null;
  }

  function collectNav() {
    var seen = {}, out = [];
    var current = win.location.pathname;
    function add(label, href) {
      if (!label || !href) return;
      var key = href;
      if (seen[key]) return;
      seen[key] = 1;
      var path;
      try { path = new win.URL(href, win.location.href).pathname; } catch (e) { path = href; }
      out.push({ label: label.replace(/\s+/g, " ").trim(), href: href, current: path === current });
    }
    var brand = $("header a[href]");
    if (brand) add("Home", brand.getAttribute("href"));
    $$("header nav a[href], header details a[href]").forEach(function (a) { add(a.textContent, a.getAttribute("href")); });
    return out;
  }

  var commandEls = [];

  function collectCommands() {
    commandEls = [];
    var out = [];
    $$("[data-voice-command]").forEach(function (el) {
      if (el.disabled || !visible(el)) return;
      var id = String(commandEls.length);
      commandEls.push(el);
      out.push({ phrase: el.getAttribute("data-voice-command"), id: id, confirm: el.hasAttribute("data-voice-confirm") });
    });
    /* Tab buttons (data-tab-target) are voice commands too: "show states", "cities tab". */
    $$("[data-tab-target]:not([data-voice-command])").forEach(function (el) {
      var label = (el.textContent || "").replace(/\s+/g, " ").trim().toLowerCase();
      if (!label || el.disabled || !visible(el)) return;
      var id = String(commandEls.length);
      commandEls.push(el);
      out.push({ phrase: "show " + label + "|" + label + " tab|open " + label, id: id, confirm: false });
    });
    return out;
  }

  var pendingConfirm = null, pendingTimer = null;

  function buildContext() {
    var form = activeForm();
    return {
      fields: form ? formFields(form) : [],
      commands: collectCommands(),
      nav: collectNav(),
      form: form,
      now: new Date(),
      nowMs: Date.now(),
      pending: pendingConfirm ? { id: pendingConfirm.id, phrase: pendingConfirm.phrase, at: pendingConfirm.at } : null
    };
  }

  /* ---- applying actions --------------------------------------------------------------------- */
  function setNativeValue(el, value) {
    var proto = el.tagName === "TEXTAREA" ? win.HTMLTextAreaElement.prototype : win.HTMLInputElement.prototype;
    var desc = Object.getOwnPropertyDescriptor(proto, "value");
    if (desc && desc.set) desc.set.call(el, value); else el.value = value;
  }

  function fillInput(el, value) {
    setNativeValue(el, String(value));
    el.dispatchEvent(new win.Event("input", { bubbles: true }));
    el.dispatchEvent(new win.Event("change", { bubbles: true }));
    el.classList.add("ndp-voice-filled");
    win.setTimeout(function () { el.classList.remove("ndp-voice-filled"); }, 1600);
  }

  function clearPending() {
    pendingConfirm = null;
    if (pendingTimer) win.clearTimeout(pendingTimer);
    pendingTimer = null;
  }

  function clickEl(el) {
    if (typeof el.click === "function") el.click();
  }

  function doSubmit(form, msgs) {
    if (!form) { msgs.push("There is no form on this page to submit."); return; }
    var btn = primarySubmit(form);
    if (typeof form.checkValidity === "function" && !form.checkValidity()) {
      if (typeof form.reportValidity === "function") form.reportValidity();
      msgs.push("Not submitted: some fields are empty or invalid.");
      return;
    }
    if (btn) clickEl(btn);
    else if (typeof form.requestSubmit === "function") form.requestSubmit();
    else form.submit();
    msgs.push("Submitted");
  }

  /* Execute interpret() output. `ctx` is buildContext() (form, commandEls); returns the messages
   * that were also shown to the user. Safe to call from tests/console. */
  function applyActions(actions, ctx) {
    ctx = ctx || buildContext();
    var form = ctx.form || activeForm();
    var msgs = [], blocked = false;
    (actions || []).forEach(function (a) {
      var el, cmdEl;
      switch (a.type) {
        case "fill":
          el = form && $$("input, textarea", form).filter(function (x) { return x.name === a.name; })[0];
          if (el) { fillInput(el, a.value); msgs.push(a.label + " " + a.value); } else { msgs.push("No field " + a.label); blocked = true; }
          break;
        case "reject":
        case "error":
          msgs.push(a.message);
          blocked = true;
          break;
        case "submit":
          if (blocked) msgs.push("Not submitted: fix the values above first.");
          else doSubmit(form, msgs);
          break;
        case "navigate":
          msgs.push("Opening " + a.label);
          (ctx.navigate || function (href) { win.location.href = href; })(a.href);
          break;
        case "command":
          cmdEl = commandEls[Number(a.id)] || (ctx.commandEls && ctx.commandEls[Number(a.id)]);
          if (!cmdEl) { msgs.push("That button is no longer on the page."); break; }
          if (a.confirm) {
            clearPending();
            pendingConfirm = { id: a.id, phrase: a.phrase, at: Date.now(), el: cmdEl };
            pendingTimer = win.setTimeout(function () {
              if (pendingConfirm) { clearPending(); showToast("Confirmation timed out."); }
            }, CONFIRM_WINDOW_MS + 200);
            msgs.push("Say confirm to proceed (or cancel)");
            if ((tts.enabled || mic.listening) && synthOk) speak([{ text: "Say confirm to proceed." }], null);
          } else {
            clickEl(cmdEl);
            msgs.push(a.phrase);
          }
          break;
        case "confirm":
          cmdEl = pendingConfirm && pendingConfirm.el;
          clearPending();
          if (cmdEl && doc.contains(cmdEl) && !cmdEl.disabled) { clickEl(cmdEl); msgs.push("Confirmed"); } else msgs.push("Nothing to confirm.");
          break;
        case "cancel":
          if (pendingConfirm) { clearPending(); msgs.push(a.reason === "timeout" ? "Confirmation timed out" : "Cancelled"); }
          break;
        case "read":
          readCurrentResult(msgs);
          break;
        case "stop":
          cancelSpeech();
          msgs.push("Stopped");
          break;
        case "repeat":
          if (tts.last.length && speak(tts.last, null)) msgs.push("Repeating"); else msgs.push("Nothing to repeat yet.");
          break;
        case "help":
          msgs.push(helpSummary(ctx));
          break;
        case "stop-listening":
          stopListening("command");
          msgs.push("Stopped listening");
          break;
        default:
          break;
      }
    });
    return { messages: msgs };
  }

  function currentResultContainer() {
    var regions = $$(REGION_SEL).filter(function (r) { return r.textContent.trim() && visible(r); });
    var form = activeForm();
    for (var p = form && form.parentElement; p && p !== doc.body; p = p.parentElement) {
      var hit = regions.filter(function (r) { return p.contains(r); })[0];
      if (hit) return hit;
    }
    return regions[0] || null;
  }

  function readCurrentResult(msgs) {
    if (!synthOk) { msgs.push("Speech output is not supported in this browser."); return; }
    var c = currentResultContainer();
    var parts = c ? collectParts(c) : [];
    if (parts.length && speak(parts, c)) msgs.push("Reading result"); else msgs.push("There is no result to read yet.");
  }

  function helpSummary(ctx) {
    var bits = ["Say: " + (ctx.fields.length ? ctx.fields.slice(0, 3).map(function (f) { return fieldLabel(f).toLowerCase() + " …"; }).join(", ") + ", " : "") + "submit, open <page>, read result, stop, repeat"];
    if (ctx.commands.length) bits.push("Buttons: " + ctx.commands.map(function (c) { return c.phrase; }).join(", "));
    return bits.join(". ");
  }

  /* ---- voice input ----------------------------------------------------------------------------- */
  var mic = { rec: null, listening: false, interim: "", noSpeech: 0, supported: false };
  var SR = win.SpeechRecognition || win.webkitSpeechRecognition;

  function listeningText() {
    return "Listening: " + (mic.interim || "…");
  }

  function paintMic() {
    var btn = $("#ndp-mic");
    if (btn) btn.setAttribute("aria-pressed", mic.listening ? "true" : "false");
  }

  function handleTranscript(text) {
    var ctx = buildContext();
    var actions = interpret(text, ctx);
    if (!actions.length) {
      showToast("Didn't understand “" + text + "”. Say “help” for examples.");
      return { actions: actions, messages: [] };
    }
    var res = applyActions(actions, ctx);
    showToast(res.messages.filter(Boolean).join(" · ") || "OK");
    return { actions: actions, messages: res.messages };
  }

  function isStopOnly(text) {
    var t = tokenize(text).map(function (x) { return x.lc; }).join(" ");
    return /^(stop|quiet|silence|be quiet|stop reading|stop talking|shut up)$/.test(t);
  }

  function startRecognition() {
    if (!mic.listening) return;
    var rec = new SR();
    mic.rec = rec;
    rec.lang = currentLang();
    rec.interimResults = true;
    rec.continuous = false;
    rec.maxAlternatives = 1;
    rec.onstart = function () { mic.interim = ""; showToast(listeningText(), true); };
    rec.onresult = function (e) {
      var interim = "", final = "";
      for (var i = e.resultIndex; i < e.results.length; i++) {
        var r = e.results[i];
        if (r.isFinal) final += r[0].transcript + " "; else interim += r[0].transcript;
      }
      mic.noSpeech = 0;
      if (final.trim()) {
        mic.interim = "";
        if (tts.speaking && !isStopOnly(final)) { showToast("Ignored while reading aloud. Say “stop” first."); return; }
        try { handleTranscript(final.trim()); } catch (err) { showToast("Something went wrong with that command."); }
      } else {
        mic.interim = interim.trim();
        showToast(listeningText(), true);
      }
    };
    rec.onerror = function (e) {
      var code = e && e.error;
      if (code === "not-allowed" || code === "service-not-allowed") {
        stopListening("error");
        showToast("Microphone access is blocked. Allow the microphone for this site in your browser settings, then try again.");
      } else if (code === "no-speech") {
        mic.noSpeech++;
        if (mic.noSpeech >= 3) { stopListening("error"); showToast("Stopped listening: no speech heard."); }
      } else if (code === "network") {
        stopListening("error");
        showToast("Speech recognition needs a network connection (your browser uses an online speech service).");
      } else if (code === "audio-capture") {
        stopListening("error");
        showToast("No microphone was found.");
      } else if (code === "language-not-supported") {
        stopListening("error");
        showToast("This language is not supported for voice input.");
      } else if (code !== "aborted") {
        stopListening("error");
        showToast("Voice input stopped (" + code + ").");
      }
    };
    rec.onend = function () {
      if (mic.rec === rec) mic.rec = null;
      if (mic.listening) win.setTimeout(function () { if (mic.listening && !mic.rec) safeStart(); }, 250);
    };
    try { rec.start(); } catch (err) { stopListening("error"); showToast("Could not start the microphone."); }
  }

  function safeStart() {
    try { startRecognition(); } catch (err) { stopListening("error"); }
  }

  function startListening() {
    if (!SR || mic.listening) return;
    cancelSpeech();
    mic.listening = true;
    mic.noSpeech = 0;
    mic.interim = "";
    paintMic();
    showToast(listeningText(), true);
    safeStart();
  }

  function stopListening() {
    mic.listening = false;
    paintMic();
    var rec = mic.rec;
    mic.rec = null;
    if (rec) { try { rec.onend = null; rec.abort(); } catch (e) { /* ignore */ } }
    if (toastEl && !toastEl.hidden && /^Listening/.test(toastEl.firstChild.textContent)) hideToast();
  }

  function initInput() {
    mic.supported = !!SR;
    doc.addEventListener("focusin", function (e) {
      var f = e.target && e.target.closest ? e.target.closest("form") : null;
      if (f && !f.closest("header")) lastForm = f;
    });
    doc.addEventListener("input", function (e) {
      var f = e.target && e.target.closest ? e.target.closest("form") : null;
      if (f && !f.closest("header")) lastForm = f;
    });
    var btn = $("#ndp-mic");
    if (!SR || !btn) return;
    btn.hidden = false;
    var pressTimer = null, longPressed = false;
    btn.addEventListener("pointerdown", function () {
      longPressed = false;
      pressTimer = win.setTimeout(function () { longPressed = true; openHelp(); }, 650);
    });
    ["pointerup", "pointerleave", "pointercancel"].forEach(function (ev) {
      btn.addEventListener(ev, function () { if (pressTimer) win.clearTimeout(pressTimer); pressTimer = null; });
    });
    btn.addEventListener("click", function (e) {
      if (longPressed) { longPressed = false; e.preventDefault(); return; }
      if (mic.listening) stopListening(); else startListening();
    });
    btn.addEventListener("contextmenu", function (e) { e.preventDefault(); openHelp(); });
    btn.addEventListener("keydown", function (e) { if (e.key === "?") { e.preventDefault(); openHelp(); } });
    doc.addEventListener("visibilitychange", function () { if (doc.hidden && mic.listening) stopListening(); });
    win.addEventListener("pagehide", function () { stopListening(); });
    var langSel = $("#ndp-lang");
    if (langSel) langSel.addEventListener("change", function () {
      if (mic.listening && mic.rec) { var r = mic.rec; try { r.abort(); } catch (e) { /* ignore */ } }
    });
  }

  /* ---- boot --------------------------------------------------------------------------------------- */
  api.speak = speak;
  api.cancelSpeech = cancelSpeech;
  api.buildContext = buildContext;
  api.applyActions = applyActions;
  api.handleTranscript = handleTranscript;
  api.summarizeRegion = summarizeRegion;
  api.collectParts = collectParts;
  api.state = { tts: tts, mic: mic };
  win.NdpVoice = api;

  injectCss();

  function boot() {
    initOutput();
    initInput();
  }
  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", boot); else boot();
})(typeof window !== "undefined" ? window : globalThis);
