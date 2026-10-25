// Pure-function tests for src/ndp/static/js/voice.js (no DOM, no microphone).
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const V = require(path.join(here, "..", "..", "src", "ndp", "static", "js", "voice.js"));
const { interpret, parseNumber, parseDate, chunkText, pickVoice, cleanForSpeech, capSummary } = V;

const NOW = new Date(2026, 9, 25, 12, 0, 0); // 25 Oct 2026, local time

const earthquake = {
  now: NOW,
  fields: [
    { name: "lat", label: "Latitude", type: "number", min: "-90", max: "90" },
    { name: "lon", label: "Longitude", type: "number", min: "-180", max: "180" },
    { name: "depth", label: "Depth (km)", type: "number", min: "0", max: "800" },
    { name: "date", label: "Date", type: "date" },
  ],
};
const storm = {
  now: NOW,
  fields: [
    { name: "temperature", label: "Temperature (°F)", type: "number", min: "-100", max: "150" },
    { name: "pressure", label: "Pressure (hPa)", type: "number", min: "800", max: "1100" },
    { name: "humidity", label: "Humidity (%)", type: "number", min: "0", max: "100" },
    { name: "wind_direction", label: "Wind direction (°)", type: "number", min: "0", max: "360" },
    { name: "wind_speed", label: "Wind speed", type: "number", min: "0" },
  ],
};
const weather = {
  now: NOW,
  fields: [
    { name: "city", label: "City", type: "text" },
    { name: "lat", label: "Latitude", type: "number", min: "-90", max: "90" },
    { name: "lon", label: "Longitude", type: "number", min: "-180", max: "180" },
  ],
};
const alerts = {
  now: NOW,
  fields: [
    { name: "place", label: "Place", type: "text" },
    { name: "radius_km", label: "Radius (km)", type: "number", min: "1", max: "100" },
    { name: "message", label: "Message", type: "textarea" },
  ],
};
const flood = { now: NOW, fields: [{ name: "cumulative_rainfall", label: "Cumulative rainfall (mm)", type: "number", min: "0" }] };
const compareFlood = {
  now: NOW,
  fields: [
    { name: "current_rainfall", label: "Current rainfall (mm)", type: "number" },
    { name: "cumulative_rainfall", label: "Cumulative rainfall (mm)", type: "number" },
  ],
};
const nav = [
  { label: "Home", href: "/" },
  { label: "Earthquake", href: "/earthquake" },
  { label: "Tsunami", href: "/tsunami" },
  { label: "Flood", href: "/flood" },
  { label: "Alerts", href: "/alerts" },
  { label: "Sentiment", href: "/sentiment" },
  { label: "Compare", href: "/compare" },
  { label: "COVID-19", href: "/covid" },
];

const fills = (actions) => Object.fromEntries(actions.filter((a) => a.type === "fill").map((a) => [a.name, a.value]));
const types = (actions) => actions.map((a) => a.type);

test("number words", () => {
  const cases = {
    zero: 0, seven: 7, twelve: 12, nineteen: 19, twenty: 20, "twenty eight": 28, "forty two": 42,
    "ninety three": 93, "one hundred": 100, "a hundred": 100, "one hundred and five": 105,
    "two hundred forty two": 242, "nine hundred ninety nine": 999, "one thousand": 1000,
    "one thousand and one": 1001, "two thousand twenty six": 2026, "ten thousand": 10000,
    "twelve hundred": 1200, "one zero one three": 1013, "seven seven": 77, "one eight zero": 180,
    "one fifty one": 151, "two seventy": 270, "one eleven": 111, "five twenty five": 525,
  };
  for (const [text, want] of Object.entries(cases)) assert.equal(parseNumber(text), want, text);
  assert.equal(parseNumber("twenty-eight"), 28);
  assert.equal(parseNumber("1,013"), 1013);
  assert.equal(parseNumber("banana"), null);
});

test("negative and decimal numbers", () => {
  assert.equal(parseNumber("minus 5"), -5);
  assert.equal(parseNumber("negative twelve"), -12);
  assert.equal(parseNumber("-33.9"), -33.9);
  assert.equal(parseNumber("28.6"), 28.6);
  assert.equal(parseNumber("twenty eight point six"), 28.6);
  assert.equal(parseNumber("28 point 6"), 28.6);
  assert.equal(parseNumber("seventy seven point two"), 77.2);
  assert.equal(parseNumber("zero point five"), 0.5);
  assert.equal(parseNumber("point five"), 0.5);
  assert.equal(parseNumber("minus point five"), -0.5);
  assert.equal(parseNumber("three point one four"), 3.14);
  assert.equal(parseNumber("ten decimal zero five"), 10.05);
  assert.equal(parseNumber("5 hundred"), 500);
});

test("earthquake form: lat, lon, depth and date in one utterance", () => {
  const a = interpret("latitude 28.6 longitude 77.2 depth 10 date 2026 10 25", earthquake);
  assert.deepEqual(fills(a), { lat: 28.6, lon: 77.2, depth: 10, date: "2026-10-25" });
  assert.deepEqual(types(a), ["fill", "fill", "fill", "fill"]);
});

test("spoken numbers, negatives and hemispheres", () => {
  const a = interpret("latitude twenty eight point six longitude seventy seven point two depth ten", earthquake);
  assert.deepEqual(fills(a), { lat: 28.6, lon: 77.2, depth: 10 });
  assert.deepEqual(fills(interpret("latitude minus thirty three point nine longitude 151.2", earthquake)), { lat: -33.9, lon: 151.2 });
  assert.deepEqual(fills(interpret("latitude 33.9 south longitude 70 west", earthquake)), { lat: -33.9, lon: -70 });
  assert.deepEqual(fills(interpret("longitude one fifty one point two", earthquake)), { lon: 151.2 });
  assert.deepEqual(fills(interpret("lat 12 north long 77 east", earthquake)), { lat: 12, lon: 77 });
});

test("storm form: five fields, spoken and digits mixed", () => {
  const a = interpret(
    "temperature forty two pressure 1013 humidity ninety three wind direction one eight zero wind speed five point five",
    storm
  );
  assert.deepEqual(fills(a), { temperature: 42, pressure: 1013, humidity: 93, wind_direction: 180, wind_speed: 5.5 });
  assert.equal(a.length, 5);
  // single-word tails are weak aliases
  assert.deepEqual(fills(interpret("direction 90 speed 3", storm)), { wind_direction: 90, wind_speed: 3 });
  // "wind" alone is not a keyword, longest match wins between direction and speed
  assert.deepEqual(fills(interpret("wind speed 12 wind direction 270", storm)), { wind_speed: 12, wind_direction: 270 });
});

test("filler words and the to/two ambiguity only inside numeric slots", () => {
  assert.deepEqual(fills(interpret("set rainfall to 120", flood)), { cumulative_rainfall: 120 });
  assert.deepEqual(fills(interpret("set the cumulative rainfall to one hundred and twenty", flood)), { cumulative_rainfall: 120 });
  assert.deepEqual(fills(interpret("rainfall is 80", flood)), { cumulative_rainfall: 80 });
  // "depth to" / "depth too" is the digit two; "for" is four
  assert.deepEqual(fills(interpret("depth to", earthquake)), { depth: 2 });
  assert.deepEqual(fills(interpret("depth for", earthquake)), { depth: 4 });
  assert.deepEqual(fills(interpret("latitude to longitude for", earthquake)), { lat: 2, lon: 4 });
  // outside a numeric slot "to" is just a word
  assert.deepEqual(fills(interpret("go to flood", { ...earthquake, nav })), {});
});

test("label vs alias disambiguation (lat / latitude / long / longitude / lng)", () => {
  assert.deepEqual(fills(interpret("lat 10 lon 20", earthquake)), { lat: 10, lon: 20 });
  assert.deepEqual(fills(interpret("latitude 10 longitude 20", earthquake)), { lat: 10, lon: 20 });
  assert.deepEqual(fills(interpret("long 20 lat 10", earthquake)), { lat: 10, lon: 20 });
  assert.deepEqual(fills(interpret("lng 20", earthquake)), { lon: 20 });
  // a label must not steal the number that belongs to the next label
  const a = interpret("latitude longitude 77", earthquake);
  assert.deepEqual(fills(a), { lon: 77 });
  assert.equal(a[0].type, "error");
  assert.match(a[0].message, /Latitude/);
  // two similar fields: bare "rainfall" goes to the cumulative field, "current rainfall" to the other
  assert.deepEqual(fills(interpret("rainfall 200 current rainfall 20", compareFlood)), { cumulative_rainfall: 200, current_rainfall: 20 });
  assert.deepEqual(fills(interpret("current rainfall 5", compareFlood)), { current_rainfall: 5 });
});

test("text capture: city, place and message stop at the next keyword or command", () => {
  assert.deepEqual(fills(interpret("city Kochi", weather)), { city: "Kochi" });
  assert.deepEqual(fills(interpret("city new delhi latitude 28.6 longitude 77.2", weather)), { city: "New Delhi", lat: 28.6, lon: 77.2 });
  // "Long" in a name is not the longitude keyword unless a number follows
  assert.deepEqual(fills(interpret("city Long Beach latitude 33.8", weather)), { city: "Long Beach", lat: 33.8 });
  assert.deepEqual(fills(interpret("city Kochi longitude 76.3", weather)), { city: "Kochi", lon: 76.3 });
  const a = interpret("city Kochi search", weather);
  assert.deepEqual(types(a), ["fill", "submit"]);
  assert.equal(a[0].value, "Kochi");
  assert.deepEqual(fills(interpret("place Vellore Tamil Nadu radius 25", alerts)), { place: "Vellore Tamil Nadu", radius_km: 25 });
  // leading "is"/"to" and trailing "and" are not part of the value
  assert.deepEqual(fills(interpret("city is Chennai and latitude 13", weather)), { city: "Chennai", lat: 13 });
  // free text keeps words that are commands elsewhere ("help", "go to")
  assert.deepEqual(fills(interpret("message send help to the school and go to the shelter", alerts)), { message: "send help to the school and go to the shelter" });
  const b = interpret("message flood at the bridge submit", alerts);
  assert.deepEqual(types(b), ["fill", "submit"]);
  assert.equal(b[0].value, "flood at the bridge");
  assert.equal(interpret("city", weather)[0].type, "error");
});

test("dates", () => {
  const d = (s) => parseDate(s, NOW);
  assert.equal(d("today"), "2026-10-25");
  assert.equal(d("tomorrow"), "2026-10-26");
  assert.equal(d("yesterday"), "2026-10-24");
  assert.equal(d("2026-10-25"), "2026-10-25");
  assert.equal(d("2026 10 25"), "2026-10-25");
  assert.equal(d("10/25/2026"), "2026-10-25");
  assert.equal(d("25/10/2026"), "2026-10-25");
  assert.equal(d("twenty fifth of October twenty twenty six"), "2026-10-25");
  assert.equal(d("October 25 2026"), "2026-10-25");
  assert.equal(d("25th of october 2026"), "2026-10-25");
  assert.equal(d("october twenty fifth two thousand twenty six"), "2026-10-25");
  assert.equal(d("first of march nineteen ninety nine"), "1999-03-01");
  assert.equal(d("fifth of july twenty oh five"), "2005-07-05");
  assert.equal(d("seven march"), "2026-03-07"); // year defaults to the current one
  assert.equal(d("november five twenty twenty six"), "2026-11-05"); // no hundreds elision in dates
  assert.equal(d("30 february 2026"), null); // not a real date
  assert.equal(d("blah"), null);
  assert.deepEqual(fills(interpret("date today", earthquake)), { date: "2026-10-25" });
  assert.deepEqual(fills(interpret("today", earthquake)), { date: "2026-10-25" });
  assert.deepEqual(
    fills(interpret("date twenty fifth of October twenty twenty six depth 10 submit", earthquake)),
    { date: "2026-10-25", depth: 10 }
  );
  const bad = interpret("date the thirty first of february", earthquake);
  assert.equal(bad[0].type, "error");
});

test("out of range values are rejected, not filled", () => {
  const a = interpret("latitude 120 longitude 77", earthquake);
  assert.deepEqual(types(a), ["reject", "fill"]);
  assert.equal(a[0].reason, "out of range");
  assert.match(a[0].message, /Latitude 120 is out of range \(-90 to 90\)/);
  assert.deepEqual(fills(a), { lon: 77 });
  assert.equal(interpret("pressure 500", storm)[0].type, "reject");
  assert.equal(interpret("wind speed minus 3", storm)[0].type, "reject");
  assert.match(interpret("wind speed minus 3", storm)[0].message, /at least 0/);
  // out of range blocks nothing but a rejected action carries no value to fill
  const d = interpret("date 2026 10 25", { ...earthquake, fields: [{ name: "date", label: "Date", type: "date", max: "2026-01-01" }] });
  assert.equal(d[0].type, "reject");
});

test("submit words run after the fills, once", () => {
  for (const w of ["submit", "estimate", "calculate", "predict", "search", "go", "classify"]) {
    const a = interpret(`${w} depth 5`, earthquake);
    assert.deepEqual(types(a), ["fill", "submit"], w);
  }
  assert.deepEqual(types(interpret("depth 5 compare", earthquake)), ["fill", "submit"]);
  assert.deepEqual(types(interpret("submit submit", earthquake)), ["submit"]);
});

test("navigation by header link text", () => {
  const ctx = { ...earthquake, nav };
  const href = (s) => interpret(s, ctx).find((a) => a.type === "navigate")?.href;
  assert.equal(href("open earthquake"), "/earthquake");
  assert.equal(href("go to flood"), "/flood");
  assert.equal(href("go to the flood page"), "/flood");
  assert.equal(href("take me to sentiment"), "/sentiment");
  assert.equal(href("home"), "/");
  assert.equal(href("alerts"), "/alerts");
  assert.equal(href("open covid"), "/covid");
  assert.equal(href("covid 19"), "/covid");
  assert.equal(href("compare"), "/compare");
  // "go to X" for an unknown page never degrades into the "go" submit word
  const unknown = interpret("go to mars", ctx);
  assert.deepEqual(types(unknown), ["error"]);
  assert.match(unknown[0].message, /mars/);
  // on the compare page itself, "compare" submits
  const onCompare = { ...earthquake, nav: nav.map((n) => ({ ...n, current: n.label === "Compare" })) };
  assert.deepEqual(types(interpret("compare", onCompare)), ["submit"]);
  assert.equal(interpret("open compare", onCompare).find((a) => a.type === "navigate").href, "/compare");
});

test("read, stop, repeat, help, stop listening", () => {
  const one = (s) => types(interpret(s, earthquake));
  assert.deepEqual(one("read result"), ["read"]);
  assert.deepEqual(one("read the result"), ["read"]);
  assert.deepEqual(one("read"), ["read"]);
  assert.deepEqual(one("stop"), ["stop"]);
  assert.deepEqual(one("quiet"), ["stop"]);
  assert.deepEqual(one("repeat"), ["repeat"]);
  assert.deepEqual(one("read again"), ["repeat"]);
  assert.deepEqual(one("help"), ["help"]);
  assert.deepEqual(one("what can I say"), ["help"]);
  assert.deepEqual(one("stop listening"), ["stop-listening"]);
  assert.deepEqual(one("tell me a joke"), []);
  assert.deepEqual(one(""), []);
});

const commands = [
  { phrase: "find nearby help", id: "0" },
  { phrase: "send alert", id: "1", confirm: true },
  { phrase: "show map|open map", id: "2" },
];

test("data-voice-command phrases click their element", () => {
  const ctx = { ...alerts, commands, nav };
  assert.deepEqual(interpret("find nearby help", ctx), [{ type: "command", id: "0", phrase: "find nearby help", confirm: false }]);
  assert.equal(interpret("please find nearby help now", ctx)[0].id, "0");
  assert.equal(interpret("open map", ctx)[0].id, "2");
  assert.equal(interpret("show map", ctx)[0].id, "2");
  // fills happen before the button is pressed
  const a = interpret("place Kochi radius 10 find nearby help", ctx);
  assert.deepEqual(types(a), ["fill", "fill", "command"]);
  // command phrases end a text slot
  assert.equal(a[0].value, "Kochi");
  // a command phrase beats a shorter built-in word it contains
  assert.deepEqual(types(interpret("find nearby help", { ...ctx, commands: [{ phrase: "help me", id: "9" }, ...commands] })), ["command"]);
  assert.equal(interpret("help me", { ...ctx, commands: [{ phrase: "help me", id: "9" }] })[0].id, "9");
});

test("confirmation flow for data-voice-confirm commands", () => {
  const ctx = { ...alerts, commands, nav, nowMs: 1_000_000 };
  const ask = interpret("send alert", ctx);
  assert.deepEqual(ask, [{ type: "command", id: "1", phrase: "send alert", confirm: true }]);
  // only the confirm-gated command survives alongside fills
  const mixed = interpret("message flood at the bridge send alert find nearby help submit", ctx);
  assert.deepEqual(types(mixed), ["fill", "command"]);
  assert.equal(mixed[1].confirm, true);

  const pending = { id: "1", phrase: "send alert", at: 1_000_000 };
  for (const yes of ["confirm", "yes", "yes please", "yeah", "go ahead", "do it", "okay"]) {
    assert.deepEqual(interpret(yes, { ...ctx, pending, nowMs: 1_003_000 }), [{ type: "confirm", id: "1", phrase: "send alert" }], yes);
  }
  for (const no of ["cancel", "no", "never mind", "abort"]) {
    assert.deepEqual(interpret(no, { ...ctx, pending, nowMs: 1_003_000 }), [{ type: "cancel", reason: "cancelled" }], no);
  }
  // anything else cancels and is then interpreted normally
  const other = interpret("find nearby help", { ...ctx, pending, nowMs: 1_003_000 });
  assert.deepEqual(types(other), ["cancel", "command"]);
  assert.equal(other[0].reason, "not confirmed");
  // saying "stop" cancels and stops speech
  assert.deepEqual(types(interpret("stop", { ...ctx, pending, nowMs: 1_003_000 })), ["cancel", "stop"]);
  // after 8 seconds a late "yes" is NOT a confirmation
  const late = interpret("yes", { ...ctx, pending, nowMs: 1_008_001 });
  assert.deepEqual(types(late), ["cancel"]);
  assert.equal(late[0].reason, "timeout");
  assert.deepEqual(interpret("confirm", { ...ctx, pending, nowMs: 1_008_000 }), [{ type: "confirm", id: "1", phrase: "send alert" }]);
  // "confirm" with nothing pending does nothing
  assert.deepEqual(interpret("confirm", ctx), []);
});

test("tel fields read digit strings", () => {
  const ctx = { fields: [{ name: "phone", label: "Phone", type: "tel" }] };
  assert.deepEqual(fills(interpret("phone nine eight 7 6 five", ctx)), { phone: "98765" });
  assert.deepEqual(fills(interpret("phone plus 9 1 9 8", ctx)), { phone: "+9198" });
});

test("text helpers: chunking, cleaning, voice choice, summaries", () => {
  const long = Array.from({ length: 30 }, (_, i) => `Sentence number ${i} says that the level is 3.47 metres.`).join(" ");
  const chunks = chunkText(long, 200);
  assert.ok(chunks.length > 1);
  for (const c of chunks) {
    assert.ok(c.length <= 200, c.length);
    assert.ok(/[.]$/.test(c), "chunk should end at a sentence boundary: " + c.slice(-20));
  }
  assert.equal(chunks.join(" "), long);
  // decimals are never split
  assert.deepEqual(chunkText("Estimated river level 3.47 metres. Flood likely."), ["Estimated river level 3.47 metres. Flood likely."]);
  // no sentence punctuation: fall back to commas, then words
  const run = Array.from({ length: 120 }, (_, i) => "w" + i).join(" ");
  for (const c of chunkText(run, 50)) assert.ok(c.length <= 50);
  assert.equal(chunkText("x".repeat(450), 200).length, 3);
  assert.deepEqual(chunkText("   "), []);

  assert.equal(cleanForSpeech("Humidity 93% at 12° see https://example.com/a?b=1 now"), "Humidity 93 percent at 12 degrees see now");
  assert.equal(cleanForSpeech("Wind ≥ 64 kt"), "Wind at least 64 kt");

  const voices = [
    { name: "a", lang: "en-GB" },
    { name: "b", lang: "en-US", localService: true },
    { name: "c", lang: "hi-IN" },
    { name: "d", lang: "en_US", default: true, localService: false },
  ];
  assert.equal(pickVoice(voices, "en-US").name, "b");
  assert.equal(pickVoice(voices, "hi").name, "c");
  assert.equal(pickVoice(voices, "en").name, "b");
  assert.equal(pickVoice(voices, "ja-JP"), null);
  assert.equal(pickVoice([], "en"), null);

  const s = capSummary("A".repeat(10) + ". " + "word ".repeat(100), 280);
  assert.ok(s.length <= 281);
  assert.equal(capSummary("short text"), "short text");
});

const floodForecast = {
  now: NOW,
  fields: [
    { name: "level_now", label: "Level now (m)", type: "number", min: "0", max: "15" },
    { name: "level_1h_ago", label: "Level 1 hour ago (m)", type: "number", min: "0", max: "15" },
    { name: "level_3h_ago", label: "Level 3 hours ago (m)", type: "number", min: "0", max: "15" },
    { name: "rain_3h", label: "Rain last 3 hours (mm)", type: "number", min: "0", max: "500" },
    { name: "rain_6h", label: "Rain last 6 hours (mm)", type: "number", min: "0", max: "500" },
    { name: "rain_24h", label: "Rain last 24 hours (mm)", type: "number", min: "0", max: "500" },
  ],
};

test("flood forecast form: full labels and short spoken forms resolve to the right fields", () => {
  assert.deepEqual(
    fills(interpret("level now 1.2 level 1 hour ago 1.1 level 3 hours ago 1.0", floodForecast)),
    { level_now: 1.2, level_1h_ago: 1.1, level_3h_ago: 1.0 }
  );
  assert.deepEqual(
    fills(interpret("rain 3 hours 4 rain 6 hours 8 rain 24 hours 20", floodForecast)),
    { rain_3h: 4, rain_6h: 8, rain_24h: 20 }
  );
  assert.deepEqual(fills(interpret("level three hours ago 1.4", floodForecast)), { level_3h_ago: 1.4 });
  assert.deepEqual(fills(interpret("rain twenty four hours 35", floodForecast)), { rain_24h: 35 });
  assert.deepEqual(fills(interpret("level 1.2", floodForecast)), { level_now: 1.2 });
});

test("navigation: a label with a generic tail word is reachable without it", () => {
  const ctx = {
    now: NOW,
    fields: [],
    nav: [
      { label: "Compare models", href: "/compare" },
      { label: "Flood", href: "/flood" },
    ],
  };
  const nav = (text) => interpret(text, ctx).filter((a) => a.type === "navigate").map((a) => a.href);
  assert.deepEqual(nav("open compare"), ["/compare"]);
  assert.deepEqual(nav("open compare models"), ["/compare"]);
  assert.deepEqual(nav("go to flood"), ["/flood"]);
  assert.deepEqual(nav("compare"), ["/compare"]);
});
