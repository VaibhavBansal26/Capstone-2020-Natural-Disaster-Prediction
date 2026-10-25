# Frontend reference

This is the detailed reference for everything that runs in or is rendered for the browser: Jinja2
templates, HTMX partials, the four scripts in `static/js/`, the stylesheets, the Tailwind build and
the Node tests. For the one-screen summary see [Frontend conventions in architecture.md](architecture.md#frontend-conventions).
For the server side of each page (routers, services, form parsing) see
[code-walkthrough.md](code-walkthrough.md); for what the models behind the result partials compute
see [models.md](models.md). Numbers about file sizes come from the generated
[statistics.md](statistics.md).

Contents: [1 Stack](#1-stack-and-philosophy) | [2 Templates](#2-template-layout) |
[3 Page and partial pattern](#3-the-page-and-partial-pattern) | [4 app.js](#4-appjs-maps-tabs-and-defaults) |
[5 voice.js](#5-voicejs) | [6 globe](#6-globejs-and-globecss) | [7 i18n.js](#7-i18njs) |
[8 Accessibility](#8-accessibility-and-responsiveness) | [9 Static assets](#9-static-assets) |
[10 Recipes](#10-how-to-recipes) | [11 Troubleshooting](#11-troubleshooting)

## 1. Stack and philosophy

The UI is server-rendered HTML. Every page is a Jinja2 template; interactivity is added by small
progressive enhancements, and every form still works with JavaScript disabled because it is a real
`<form method="post">` that the server answers with a full page. There is no bundler, no
`package.json`, no `node_modules` and no client-side framework.

| Piece | Version and host | Loaded by | Notes |
|---|---|---|---|
| HTMX | 2.0.4, `cdnjs.cloudflare.com` | [base.html](../src/ndp/templates/base.html), `defer` | Partial swaps. |
| Leaflet | 1.9.4, `cdnjs.cloudflare.com` (CSS and JS) | [base.html](../src/ndp/templates/base.html), `defer` | Every page; maps are created only where a `data-ndp-map` element exists. |
| globe.gl | 2.35.0, `cdn.jsdelivr.net/npm` | [index.html](../src/ndp/templates/index.html) `{% block head %}` | Home page only. |
| three-globe textures | 2.31.0, `cdn.jsdelivr.net/npm` | `data-tex-globe` and `data-tex-bump` in index.html, `night-sky.png` in [globe.css](../src/ndp/static/css/globe.css) | Images only. |
| Tailwind CSS | 3.4.17, compiled by `npx` | [frontend/build.sh](../frontend/build.sh) | Output is the committed file `static/css/tailwind.css`. |
| Google Translate widget | `translate.google.com/translate_a/element.js` | [i18n.js](../src/ndp/static/js/i18n.js), on demand | See [section 7](#7-i18njs). |
| Map tiles | `tile.openstreetmap.org` | [app.js](../src/ndp/static/js/app.js) and [globe.js](../src/ndp/static/js/globe.js) | Attribution is set in code. |
| USGS earthquake feed | `earthquake.usgs.gov/.../2.5_week.geojson` | [globe.js](../src/ndp/static/js/globe.js) (always), app.js (when `data-usgs-url` is set) | Fetched by the browser, not the server. |

None of the CDN tags has an `integrity` attribute, and the application sets no
`Content-Security-Policy` header. The middleware in [main.py](../src/ndp/main.py) sets
`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` and
`Permissions-Policy: microphone=(self), camera=(), geolocation=(), payment=()`. The microphone
entry is what lets `voice.js` use speech recognition on this origin.

Versions are pinned in the markup, so a CDN upgrade is a deliberate edit to `base.html` or
`index.html`. All scripts are loaded with `defer`, which preserves document order: HTMX, then
Leaflet, then `app.js`, `voice.js`, `i18n.js`, then anything a page adds in `{% block head %}`
(on the home page: `globe.gl`, then `globe.js`).

### Why there is a compiled stylesheet

The comment in `build.sh` explains the choice: pages should not ship the 300 KB in-browser
compiler. Instead the utilities that the templates use are compiled once into [static/css/tailwind.css](../src/ndp/static/css/tailwind.css), which is
committed to the repository (32 KB, 7 KB gzipped, per [statistics.md](statistics.md)).

### Rebuilding the CSS

```sh
frontend/build.sh
```

The script `cd`s to the repository root and runs

```sh
npx --yes tailwindcss@3.4.17 \
  -c frontend/tailwind.config.js \
  -i frontend/tailwind.input.css \
  -o src/ndp/static/css/tailwind.css \
  --minify
```

It needs Node and, on the first run, network access so `npx` can fetch the compiler. Inputs:

- [frontend/tailwind.input.css](../frontend/tailwind.input.css) contains the three `@tailwind`
  directives plus a few hand-written rules that must be in the compiled file: `.htmx-indicator`
  (opacity 0, becomes 1 inside `.htmx-request`), `.leaflet-container { font: inherit; }`, a global
  `:focus-visible` outline, and a `prefers-reduced-motion` block that shortens all animations and
  transitions to 0.01 ms.
- [frontend/tailwind.config.js](../frontend/tailwind.config.js) sets the content globs and one
  extension: `colors.brand` with exactly the keys `50`, `600`, `700` and `900`.

The content globs, relative to the repository root, are:

```js
content: [
  "./src/ndp/templates/**/*.html",
  "./src/ndp/static/js/*.js",
  // Some markup is built in Python (the assistant renders answers to HTML).
  "./src/ndp/services/assistant.py",
],
```

**When a rebuild is required.** Rebuild after you add or change a Tailwind utility class in any
`*.html` under `templates/`, any `*.js` directly under `static/js/`, or `services/assistant.py`.
Tailwind only emits CSS for class names it finds as literal strings in those files.

**What happens if you forget.** Nothing fails. There is no build step in the Python app and CI
(`uv run pytest -q` in `.github/workflows/ci.yml`) does not check freshness. The new class simply
has no rule, so the element renders unstyled. As of this writing the committed `tailwind.css` is
byte-identical to a fresh build when I checked (before the assistant.py glob was added), so
rebuild and diff to confirm.

Three gotchas that follow from the config:

1. **Only one Python file is scanned.** [services/assistant.py](../src/ndp/services/assistant.py)
   builds HTML with Tailwind classes (`safe_markdown`, `_inline`), so it is listed in the globs and
   its classes (including `text-[0.85em]`, `first:mt-0` and `break-all`) are in the compiled CSS.
   Any other Python file that emits class names must be added to `content` in
   `frontend/tailwind.config.js` (or the classes put in a template), then rebuild.
2. **Class names must be literal.** `'bg-' ~ tone` produces nothing. The macros in
   `_form_bits.html` therefore spell out full class strings in a dict.
3. **Only four brand shades exist.** `bg-brand-500` or `text-brand-100` compile to nothing and raise
   no error.

`app.css`, `globe.css` and `voice.css` are hand-written and are not touched by the build. There is
no dark mode (no `dark:` variants and no `prefers-color-scheme` rules).

## 2. Template layout

### How templates are rendered

[web.py](../src/ndp/web.py) owns one `Jinja2Templates` instance (`templates`, rooted at
`src/ndp/templates`) and three helpers:

| Helper | Behaviour |
|---|---|
| `render(request, name, *, status_code=200, **context)` | Adds `settings` (the settings object) to the context with `setdefault`, then returns `templates.TemplateResponse`. Starlette adds `request` and `url_for`. |
| `is_htmx(request)` | True when the header `HX-Request` equals `"true"`. |
| `page_or_partial(request, page, partial, page_context, **result)` | HTMX request: render only `partial` with `result`. Otherwise call `page_context()`, merge `result` into it and render the full `page`. |

`render` takes a keyword-only `status_code` (default 200); everything else in `**context` goes to
the template.

`url_for` inside templates returns an absolute URL object (scheme and host included), for example
`http://testserver/static/css/tailwind.css` under the test client. Two consequences. Scheme and host
come from the request, so behind a TLS-terminating proxy uvicorn must trust the forwarded headers
(`deploy/entrypoint.sh` passes `--proxy-headers` and `--forwarded-allow-ips`), or assets render as
`http://` on an `https://` page and are blocked as mixed content. And code that needs only the path
uses `.path`: `base.html` compares `path == url_for(endpoint).path` for the active link, and
`index.html` writes `url_for(...).path` for its local script, stylesheet and links.

Custom Jinja filters `assistant_md` and `assistant_plain` are registered on this shared environment
by [routers/assistant.py](../src/ndp/routers/assistant.py).

### base.html

[base.html](../src/ndp/templates/base.html) is the only layout; all 16 page templates
(`extends "base.html"`) share it.

| Block | Where | Used for |
|---|---|---|
| `title` | `<title>` | Page title; default "Natural Disaster Prediction". |
| `description` | `<meta name="description">` | Default text about estimates, not warnings. No page overrides it today. |
| `head` | end of `<head>`, after the shared scripts | Extra `<link>` and `<script>` tags. Only `index.html` uses it (globe CSS, globe.gl, globe.js, a `preconnect` to jsdelivr). |
| `content` | inside `<main id="main">` | The page. |
| `scripts` | end of `<body>`, after the footer and the menu script | Inline page scripts. Used by `assistant/page.html` and `sentiment/page.html`. They are not deferred, so they run at parse end with the DOM already present. |

Other metadata: `<html lang="en">`, `theme-color` `#1e3a8a`, an `og:title` and `og:type`, and an inline
SVG data URI favicon. There is no manifest and no per-page Open Graph description.

**Skip link.** The first element in `<body>` is `<a href="#main" class="sr-only focus:not-sr-only ...">Skip to content</a>`
pointing at `<main id="main">`.

**Navigation.** Three lists are defined at the top of the body with `{% set %}`:

- `primary`: Earthquake, Flood, Live, Assistant, Alerts (tuples of `(endpoint name, label)`).
- `more`: Tsunami, Hurricane, Storm, Weather, Sentiment, Compare models, and COVID-19 appended only
  when `settings.enable_covid` is true (`NDP_ENABLE_COVID`, default true). When it is false the
  covid router is not mounted either (`main.py`), which is why the link must be guarded.
- `nav = primary + more` for the mobile menu.

Three renderings of those lists:

| Rendering | Markup | Visible |
|---|---|---|
| Desktop bar | `<nav aria-label="Main">` with `primary` links and a `<details data-menu>` "More" menu | `lg` and up (`hidden lg:flex`) |
| Mobile menu | `<details class="lg:hidden">` with summary "Menu", a Home link, then all of `nav` | below `lg` |
| Brand link | Left-most `<a>` to `index`; text is "Disaster Prediction" from 420 px up, "Disasters" below (`min-[420px]:` utilities) | always |

**Active state.** For each link, `path == url_for(endpoint).path` (where `path = request.url.path`)
adds `aria-current="page"` and the classes `bg-slate-100 font-medium`. The comparison is exact, so
a page served at a sub-path (for example `/earthquake/graphs`) does not highlight "Earthquake".

An inline script at the bottom of `base.html` closes `details[data-menu][open]` on any outside
click and on Escape. Only the "More" menu carries `data-menu`; the mobile "Menu" does not close that way.

**Header tools** (`<div id="ndp-tools">`). All three controls ship hidden or empty and are enabled
by scripts only when the browser can support them:

| Element | Enabled by | Purpose |
|---|---|---|
| `button#ndp-voice-toggle` (`aria-pressed`, `hidden`) | `voice.js` `initOutput()` when speech synthesis exists | Turns reading results aloud on or off. |
| `button#ndp-mic` (`aria-pressed`, `hidden`) | `voice.js` `initInput()` when `SpeechRecognition` or `webkitSpeechRecognition` exists | Starts and stops voice input. |
| `select#ndp-lang` (`hidden`, empty) plus a `sr-only` label | `i18n.js` | Language list for the translate widget; also the language for speech. |

The `aria-pressed:bg-brand-600` style on the two buttons uses Tailwind's `aria-pressed:` variant.

### Directory map of templates/

46 files, 2,520 non-blank lines in total (statistics.md). The convention is `<area>/page.html` for the
full page and `_name.html` for a fragment that HTMX swaps in.

| Folder | Files | Notes |
|---|---|---|
| (root) | `base.html`, `index.html` | `index.html` is the home page: globe hero, "What you can do", module cards from `modules` (built in `routers/pages.py`), credits. |
| `partials/` | `_form_bits.html`, `_model_card.html`, `_plots.html` | Shared, see below. |
| `earthquake/` | `page.html`, `_result.html`, `_events.html`, `graphs.html` | `_events.html` is loaded by `hx-get ... hx-trigger="load"`; `graphs.html` is a second full page. |
| `tsunami/`, `hurricane/`, `storm/`, `flood/`, `weather/` | `page.html`, `_result.html` | The standard form-and-result shape. |
| `compare/` | `page.html`, `_result.html` | One form and one `#result-<key>` per hazard. |
| `alerts/` | `page.html`, `_results.html`, `_translations.html`, `_sent.html` | Three chained swaps: search, translate preview, send. |
| `live/` | `page.html`, `_gdacs.html`, `_forecast.html`, `_watcher.html`, `_error.html` | All fragments load on page open; the watcher polls every 30 s and deliberately carries no `data-speak`, so the refresh is not read aloud again. |
| `sentiment/` | `page.html`, `_score.html`, `_text_score.html`, `_triage.html`, `_triage_bits.html`, `_triage_sample.html` | `_triage_bits.html` holds macros only. |
| `assistant/` | `page.html`, `_turn.html`, `_welcome.html` | `_turn.html` is appended to `#transcript` with `hx-swap="beforeend"`. |
| `covid/` | `page.html`, `maps.html`, `_banner.html`, `_result.html`, `_stats.html`, `_centres.html` | Whole folder is unreachable when `NDP_ENABLE_COVID` is false. |
| `errors/` | `not_trained.html`, `_not_trained.html` | Rendered by the `ModelNotTrained` handler in `main.py`: the full page (status 503) for normal requests, the small partial (status 200) for HTMX requests. |

### partials/_form_bits.html (macros)

Import with `{% from "partials/_form_bits.html" import field, submit %}`. All macros return markup;
none sets `id`s unless noted.

| Macro | Parameters | Use it for |
|---|---|---|
| `field` | `name, label, type='number', value='', step='any', placeholder='', required=True, min=None, max=None, help=None` | A labelled `<input>` wrapped in a `<label>`. `step` is emitted only for `type='number'`. `min` and `max` are also what `voice.js` uses to reject out-of-range speech. There is no `id` parameter; write the input by hand if a script needs an id (the earthquake page does this for the map picker). |
| `textarea` | `name, label, value='', rows=4, placeholder='', required=True, help=None, id=None` | Multi-line input. The only field macro with `id`. |
| `submit` | `label='Estimate'` | Submit button containing a `.htmx-indicator` spinner. |
| `card` | `title, subtitle=None` | White section with heading. Use with `{% call card(...) %}...{% endcall %}`; the body is `caller()`. |
| `metric` | `label, value, unit=''` | A small stat tile. |
| `badge` | `text, tone='muted'` | Pill. Tones: `ok`, `warn`, `danger`, `info`, `muted`. |
| `status_pill` | `status` | Badge for module status: `ready`, `needs training`, `not configured`, `archived`. Used by `index.html`. |
| `error_box` | `message` | `role="alert"` box titled "Could not estimate." Use for form errors in `_result.html`. |
| `notice` | `message, tone='info'` | Callout; tones `info`, `warn`, `danger`, `muted`. Pass a `{% call %}` body or `message`. |
| `fmt` | `value, decimals=2` | Formats numbers with fixed decimals; non-numbers and booleans pass through. |
| `inputs_list` | `inputs, labels={}` | A `<dl>` echoing submitted values; `labels` maps field name to display label. |
| `bar` | `pct, tone='bg-brand-600', label=None` | Horizontal bar clamped to 0 to 100, with `role="img"` and an `aria-label`. |
| `tab` | `target, label, active=False` | Tab button with `data-tab-target`; see `app.js` tabs. |

### partials/_model_card.html and partials/_plots.html

[_model_card.html](../src/ndp/templates/partials/_model_card.html) is included with
`{% include "partials/_model_card.html" %}` and reads context variables `meta` (the registry
metadata dict or `None`), `trained` (bool) and optionally `model_label`. It shows a ready or "needs
training" badge, a verdict computed from `meta.skill` (below 0.05 "no better than the baseline",
below 0.25 "weak", below 0.5 "moderate", otherwise "useful") or from accuracy against the baseline
accuracy, a tile for every scalar key not listed in its `_skip` list (new keys appear without
editing the template), a per-horizon table when metadata has horizon dicts, and the back-test and
notes when present. The verdict `<p>` carries `data-model-verdict="ok|warn|danger|info"`. See
[models.md](models.md) for the metadata keys.

[_plots.html](../src/ndp/templates/partials/_plots.html) reads `plots`, a list of
`(title, path)` tuples with paths relative to `/static` (for example
`generated/flood/f1_level_history.png`), and optionally `plots_title`. Each plot is a lazy-loaded
`<img>` with `alt` set to the title, wrapped in a link that opens the image in a new tab. With an empty list it shows
a "No plots yet" message.

## 3. The page and partial pattern

Every hazard page follows one shape. The server-side half is in
[routers/flood.py](../src/ndp/routers/flood.py):

```python
@router.get("", name="flood_page")
def flood_page(request: Request):
    return render(request, "flood/page.html", **_page_context())

    # in flood_predict(), after the Form(...) parameters are collected into `raw`:
    try:
        values = {
            name: parse_float(raw[name], label, 0, LEVEL_MAX_M if kind == "level" else RAIN_MAX_MM)
            for name, label, kind in FIELDS
        }
    except FormError as exc:
        return page_or_partial(
            request,
            "flood/page.html",
            "flood/_result.html",
            _page_context,
            inputs=raw,
            error=str(exc),
        )
```

The template half, from [flood/page.html](../src/ndp/templates/flood/page.html) (the other five
`field(...)` calls are omitted):

```html
{% call card('Current conditions', 'Levels in metres at the gauge; rainfall totals in millimetres. The defaults describe a quiet day.') %}
  <form method="post" action="{{ url_for('flood_predict') }}"
        hx-post="{{ url_for('flood_predict') }}" hx-target="#result" hx-swap="innerHTML" hx-indicator="find .htmx-indicator"
        class="space-y-4">
    {{ field('level_now', 'Level now (m)', value=v.get('level_now', d.get('level_now', '1.42')), step='0.01', min=0, max=15, help='Typical range 1.3 to 1.6 m at this gauge.') }}
    {# five more field(...) calls #}
    {{ submit('Forecast river level') }}
  </form>
{% endcall %}
<div id="result" aria-live="polite">
  {% if result is defined or error is defined %}{% include "flood/_result.html" %}{% endif %}
</div>
```

How the pieces fit:

| Piece | What it does |
|---|---|
| `method="post" action=...` | The no-JavaScript path. The browser posts and receives the full page, because `page_or_partial` merges the result into `page_context()`. |
| `hx-post` (same URL) | With HTMX the same URL is posted in the background. HTMX adds the request header `HX-Request: true`, so the handler returns only `_result.html`. |
| `hx-target="#result"` and `hx-swap="innerHTML"` | Where the partial lands. |
| `hx-indicator="find .htmx-indicator"` | HTMX adds the class `htmx-request` to the spinner inside the form while the request runs; `tailwind.input.css` turns its opacity from 0 to 1. |
| `<div id="result" aria-live="polite">` | The swap target is an `aria-live` region so assistive technology announces the new result. The same container is included on first render, which is why the page template does `{% include "flood/_result.html" %}` when `result` or `error` is defined. |
| `v`, `d` | `v` is `inputs` from the failed or previous submission, `d` is the router's `defaults`; so the form keeps what the user typed. |

**Errors.** Validation problems raise `FormError` in the router, which passes `error=str(exc)` and
the raw `inputs` to the partial. Every `_result.html` starts with
`{% if error is defined and error %}{{ error_box(error) }}...`. The response status is 200, because
HTMX does not swap 4xx and 5xx responses by default. Pages that need an error status swapped (the
assistant) handle `htmx:beforeSwap` themselves (`assistant/page.html`, `shouldSwap = true`). Other
fragments have a retry-able error partial: `live/_error.html` takes `title`, `retry_endpoint`,
`retry_target`.

**Result partials.** The first element of a result partial usually carries `data-speak` (and
sometimes `data-speak-priority="alert"`); see the voice sections. Echo the submitted inputs with
`inputs_list`.

**Other HTMX attributes in use:**

| Attribute | Where | Meaning |
|---|---|---|
| `hx-get` with `hx-trigger="load"` | `earthquake/page.html` (`#events`), `live/page.html` (`#gdacs`), `covid/page.html` (`#live-stats`) | Load a slow fragment after the page is shown. |
| `hx-trigger="load, every 30s"` | `live/page.html` (`#watcher`) | Poll the watcher status. |
| `hx-trigger="load, submit"` | `live/page.html` forecast form | Run once on open, then on submit. |
| `hx-swap="beforeend"` | `assistant/page.html` form | Append a conversation turn. |
| `hx-disabled-elt` | assistant, sentiment, alerts | Disable the button while the request is in flight. |
| `hx-on::after-request` | assistant form | Reset the form when the request succeeded. |
| `hx-confirm` | `alerts/_results.html` | Native `confirm()` before sending an alert, see below. |

**Conditional fields on the alert form.** The send form shows a `passcode` field
(`type="password"`, required) only when the template context has `passcode_required` (server setting
`NDP_ALERTS_PASSCODE` set) and a `phone` field only when `custom_phone_allowed` is true
(`NDP_ALERTS_ALLOW_CUSTOM_PHONE=true`); both flags come from `routers/alerts.py`. Password inputs are
excluded from voice filling.

**`hx-confirm` on alert sending.** The send form in
[alerts/_results.html](../src/ndp/templates/alerts/_results.html) has
`hx-confirm="Send this alert to the configured contacts?"`. HTMX inherits this attribute, so the
"Preview translations" button inside the same form sets `hx-confirm="unset"` to opt out. The send
button also has `data-voice-command="send alert"` and `data-voice-confirm`, which adds a spoken
confirmation on top of the native one (see [section 5](#5-voicejs)). The no-JavaScript fallback
form posts without any confirmation prompt, because `hx-confirm` is an HTMX feature.

**Model not trained.** `ModelNotTrained` is handled globally in `main.py` and the response depends
on the caller (verified with the test client): a normal request gets `errors/not_trained.html` as a
full page with status **503**; an HTMX request (`HX-Request: true`) gets the small
`errors/_not_trained.html` notice (`role="alert"`) with status **200**, because HTMX does not swap
5xx responses, so the notice lands in the target like any result; paths under `/api/` get JSON with
status 503. Run `uv run ndp-fetch-data` and `uv run ndp-train` to clear it.

## 4. app.js: maps, tabs and defaults

[app.js](../src/ndp/static/js/app.js) (11 KB, 4 KB gzipped) is an IIFE that exposes four globals and
initialises declaratively.

| Global | Signature | Result |
|---|---|---|
| `ndpMap` | `ndpMap(idOrElement, lat, lon, zoom)` | An `L.Map` with an OpenStreetMap layer (`scrollWheelZoom: false`, `maxZoom: 19`). Cached on `el._ndpMap`, so calling twice returns the same map. Returns `null` if the element or Leaflet is missing. Defaults: lat 20, lon 0, zoom 2. |
| `ndpUsgsLayer` | `ndpUsgsLayer(map, url)` | `Promise` of an `L.geoJSON` layer of circle markers: radius `max(3, mag * 2.2)`, colour by magnitude (6+ `#be123c`, 5+ `#ea580c`, 4+ `#d97706`, else `#2563eb`), popup with magnitude, UTC time, depth and USGS link. `layer.ndpCount` is the feature count. |
| `ndpMarkers` | `ndpMarkers(map, items)` | An `L.featureGroup` of circle markers; fits the map to them (`pad(0.2)`, `maxZoom: 14`). |
| `ndpInit` | `ndpInit(rootElement)` | Runs the declarative setup below on `root` (default `document`). |

### Declarative data attributes

Put these on a block element with a height class (Leaflet needs a sized container).

| Attribute | On | Meaning |
|---|---|---|
| `data-ndp-map` | map element | Marks the element for initialisation. |
| `data-lat`, `data-lon`, `data-zoom` | map element | Initial view; parsed as floats, fall back to 20, 0, 2. |
| `data-usgs-url` | map element | URL of a USGS GeoJSON feed; adds `ndpUsgsLayer`. The earthquake page passes `usgs_feed_url` from the router. An empty value does nothing. |
| `data-markers-src` | map element | CSS selector of a `<script type="application/json">` holding an array of marker objects (shape below). Invalid JSON prints "Bad marker data" to the status element. |
| `data-geojson-url` | map element | URL of a GeoJSON `FeatureCollection` of points, fetched lazily; used by `covid/maps.html` via `covid_geojson`. Property `name` becomes the title, `color` the colour; every other property (for the COVID layers `address`, `city`, `active`, `confirmed`, ...) is printed in the popup as escaped `key: value` text. |
| `data-pick-lat`, `data-pick-lon` | map element | CSS selectors of two inputs. Adds a draggable marker; dragging or clicking the map writes 4-decimal values to both inputs and dispatches `change` on the latitude input; typing in either input moves the marker (`input` listeners). |
| `data-status` | map element | CSS selector of an element that receives "Loading...", counts or error text; errors get `text-rose-600`. |
| `data-tabs` | tab group container | Scopes the tab buttons and panels. Falls back to `document`. |
| `data-tab-target="#panel"` | button | Click shows that panel, hides the other `data-tab-panel` elements in the group, updates `aria-selected` and swaps the classes `bg-brand-600 text-white` and `text-slate-600`. Buttons are created by the `tab` macro. |
| `data-tab-panel` | panel | Hidden through the `hidden` attribute; on activation the maps inside call `invalidateSize()` and start their lazy GeoJSON load. |
| `data-default-today` | `<input type="date">` | If the input is empty, set it to today's date in ISO format (`toISOString().slice(0, 10)`, so UTC). Used by `hurricane/page.html` and `compare/page.html`. |

### Marker JSON shape

Illustrative values:

```json
[
  {"lat": 12.9, "lon": 79.1, "title": "CMC Hospital", "color": "#be123c", "radius": 9,
   "lines": ["Hospital", "Address", "Phone"], "popup": "<a href=\"...\">trusted HTML</a>"}
]
```

| Key | Type | Notes |
|---|---|---|
| `lat`, `lon` | number or numeric string | Items with a non-numeric value are skipped. |
| `title` (or `name`) | string | Bold first line of the popup and the hover tooltip. Escaped. |
| `lines` | array of strings | One `<p>` each; empty values dropped; **escaped**. |
| `popup` | string | Appended as **raw HTML**; only pass trusted markup. |
| `color` | CSS colour | Default `#2563eb`. |
| `radius` | number | Default 7. |

Real producers: `alerts/_results.html` builds `_items` in Jinja and writes
`<script type="application/json" id="alerts-markers">{{ _items|tojson }}</script>`;
`live/_gdacs.html` and `covid/_centres.html` do the same with `markers`/`_items`.

### Re-initialisation

`init(root)` runs once on `DOMContentLoaded` and again on every `htmx:load` event, using
`e.detail.elt` as the root, so a map inside a swapped partial is created as soon as it is inserted.
Each map element is initialised once (guarded by `el._ndpMap`), tab buttons once (`btn._ndpTab`).
`init` handles both the root itself and its descendants. For lazy GeoJSON maps inside a hidden tab
panel the fetch waits until the tab is shown (`el._ndpLoad`).

## 5. voice.js

[voice.js](../src/ndp/static/js/voice.js) is 67 KB (19 KB gzipped), the largest script by far
(statistics.md). It has no dependencies and two halves:

1. A **pure layer** (sections 1 to 4 in the file): tokenising, number and date parsing, the field
   table, the command interpreter, text chunking. It touches no DOM and can be loaded in Node.
2. A **browser runtime** (section 5): speech synthesis, speech recognition, DOM scanning, toggle and
   buttons. It starts only if `window` and `document` exist.

Styles for its UI live in [voice.css](../src/ndp/static/css/voice.css), which `voice.js` injects
itself as a `<link>` (`injectCss`, derived from `document.currentScript.src` by replacing
`js/voice.js` with `css/voice.css`). `base.html` does not reference `voice.css`.

### Exports

In Node, `module.exports` is the `api` object; in the browser the same object is `window.NdpVoice`.

| Name | Layer | Purpose |
|---|---|---|
| `interpret(transcript, context)` | pure | Transcript to a list of action objects. |
| `tokenize(text)` | pure | `[{raw, lc}]`; keeps digits, signs and dates; drops units and punctuation. |
| `parseNumber(text, opts)` | pure | Whole string to a number or `null`. |
| `parseNumberAt(tokens, i, opts)` | pure | `{value, next, ordinal}` or `null`. |
| `parseDate(input, now)` | pure | Spoken or typed date to `yyyy-mm-dd` or `null`. |
| `buildFieldTable(fields)` | pure | Phrase table used for matching. |
| `chunkText(text, max)` | pure | Splits text into pieces of at most `max` characters (default 200). |
| `capSummary(text, cap)` | pure | Truncates a summary (default 280 characters) at a sentence or word boundary. |
| `cleanForSpeech(text)` | pure | Replaces symbols for the synthesiser. |
| `pickVoice(voices, lang)` | pure | Chooses a voice. |
| `CONFIRM_WINDOW_MS` | pure | 8000. |
| `speak`, `cancelSpeech`, `buildContext`, `applyActions`, `handleTranscript`, `summarizeRegion`, `collectParts`, `state` | browser only | Handy from the console; `state` is `{tts, mic}`. |

### Numbers, decimals, negatives, hemispheres, dates

Verified by calling the functions from Node:

| Input | Output |
|---|---|
| `parseNumber("two hundred forty two")` | `242` |
| `parseNumber("minus three point five")` | `-3.5` |
| `parseNumber("one zero one three")` | `1013` (spoken digit strings) |
| `parseNumber("twelve hundred")` | `1200` |
| `parseNumber("1,013")` | `1013` (thousands commas removed in `tokenize`) |
| `parseNumber("nine point oh five")` | `9.05` |
| `parseDate("25 October 2026")`, `"october twenty fifth"`, `"2026-10-25"` | `"2026-10-25"` (year defaults to the current year) |
| `parseDate("12/31/2026")` and `"31/12/2026"` | `"2026-12-31"` (when the first part is above 12 it is the day) |
| `parseDate("february thirtieth")` | `null` (`validYmd` rejects it) |

Rules worth knowing: `minus` and `negative` flip the sign; decimals use `point`, `decimal` or `dot`
followed by digit words (`oh` means 0); "one fifty one" is 151 (hundreds elided); `to`, `too` and
`for` are read as 2 and 4 only in lenient contexts (the number slot after a field name, dates);
`today`, `tomorrow` and `yesterday` are relative to `context.now`. After a latitude field the
words `north` or `south` follow the number, after a longitude `east` or `west`; `south` and `west`
negate it.

### The field table and how a form field is matched

`buildFieldTable(fields)` takes the fields of the active form as `{name, label, type, min, max}`
(built at runtime by `formFields`) and creates phrases:

1. **Own phrases (strong):** the normalised `name` (`wind_speed` becomes "wind speed") and the
   normalised `label` with parenthesised units removed ("Wind speed (m/s)" becomes "wind speed").
   Normalisation is lowercase with punctuation turned into spaces.
2. **Alias groups (strong):** `ALIAS_GROUPS` is a list of `{id, phrases}`. A group applies to a field when the
   field's normalised name or label equals one of the group's phrases; then every phrase of that group
   becomes a strong phrase for the field.
3. **Weak tails:** the last word of any multi-word phrase ("speed" for "wind speed"). A strong phrase
   is never replaced by a weak one.

| Group id | Phrases |
|---|---|
| `lat` | lat, latitude |
| `lon` | lon, long, longitude, lng |
| `depth` | depth |
| `height` | height, wave height, max height, maximum height, max wave height, maximum wave height |
| `rainfall` | rainfall, cumulative rainfall, total rainfall, rain, cumulative rain |
| `current-rainfall` | current rainfall, current rain |
| `temperature` | temperature, temp |
| `pressure`, `humidity` | pressure; humidity |
| `wind-direction` | wind direction, direction |
| `wind-speed` | wind speed, speed |
| `radius` | radius, distance, range |
| `city` | city, town |
| `place` | place, location, address |
| `date` | date |
| `level-now` | level now, current level, river level, level |
| `level-1h`, `level-3h` | level 1 hour ago, level one hour ago, level 1h ago, level an hour ago; level 3 hours ago, level three hours ago, level 3h ago |
| `rain-3h`, `rain-6h`, `rain-24h` | "rain last 3 hours", "rain 3 hours", "rain 3h" and their spelled-out forms; 6 and 24 likewise ("rain a day" for 24) |
| `message` | message, text |
| `phone` | phone, phone number, mobile |

**Matching.** `interpret` scans the transcript left to right. At each position `matchAt(i)` considers
every candidate that starts there: data-voice commands, navigation phrases (a verb plus a link label),
failed-navigation verbs, controls (read, stop, ...), submit words and fields. The **longest** match
wins; for equal length the rank order is command 5, nav 4, control 3, nav failure 3, submit 2,
field 1. That is why "level 1 hour ago" beats "level", and "rain 24 hours" beats "rain".

**Field kind** decides how the value after the phrase is read (`fieldKind`): `type="date"` is a
date, `tel` a digit string, `number` or `range` a number, `text`, `search`, `textarea`, `email` and
`url` free text. Fields of other types are classified by their alias group (date, city, place,
message, phone) and default to number.

**Slots.** A number slot skips fillers (`to`, `is`, `equals`, `at`, `of`, `the`, `value`, `set`,
`around`, `about`, ...). A date slot runs until the next recognised keyword. A text slot skips
leading `to/is/as/set/the/named/called` and ends at the next field or command; message-like fields
(`message`, any `textarea`) end only at a field, a data-voice command or a trailing submit word, so
ordinary words in a message are kept. Place and city values spoken in lower case are title-cased.
Numbers are range-checked against the input's `min` and `max` attributes (and dates by `min`/`max`
strings): an out-of-range value gives a `reject` action instead of a `fill`.

### Command interpreter: action types

`interpret` returns a list of these objects (the browser applies them in `applyActions`):

| `type` | Produced by | Effect in the browser |
|---|---|---|
| `fill` | field phrase plus a value | Sets the input with the native setter, fires `input` and `change`, adds the class `ndp-voice-filled` for 1.6 s. |
| `reject`, `error` | out-of-range value, unparsable value, unknown page ("open the moon") | Shows the message; blocks any `submit` in the same utterance. |
| `submit` | one of submit, estimate, calculate, predict, compare, search, go, classify, score | Only one is kept and it is moved after the fills. Checks `form.checkValidity()` (calls `reportValidity()` if invalid), then clicks the primary submit button, so HTMX handles it. |
| `navigate` | "open/go to/goto/show/show me/take me to/navigate to/visit/switch to/load" plus a label, or the bare label | `window.location.href = href`. Labels come from the header links (`collectNav`), plus "Home" for the brand link. A digit-free variant makes "open covid" match "COVID-19". |
| `command` | text of a `data-voice-command` element | Clicks the element, or starts a confirmation if it has `data-voice-confirm`. |
| `confirm`, `cancel` | reply while a confirmation is pending | Clicks the pending element, or drops it. |
| `read`, `stop`, `repeat`, `help`, `stop-listening` | control phrases | Read the current result, cancel speech, replay the last speech, show a one-line summary toast, stop the microphone. |

Control phrases include "read result", "read aloud", "stop", "quiet", "be quiet", "repeat",
"say again", "help", "what can i say", "stop listening", "mic off".

**Generic `data-voice-command`.** Any visible, enabled element with `data-voice-command="phrase"`
becomes speakable; `a|b` gives alternatives. Examples in the templates: `find nearby help`
(earthquake result), `preview translations`, `send alert`, `triage post`, `triage sample posts`,
`score <disaster>` (sentiment). Tab buttons (`data-tab-target`) without a command attribute get
`show <label>`, `<label> tab` and `open <label>` automatically.

**Confirmation flow (`data-voice-confirm`).**

1. "send alert" produces a `command` action with `confirm: true`. Alongside it, only fills, rejects
   and errors from the same utterance are kept.
2. `applyActions` stores `pendingConfirm = {id, phrase, at: Date.now(), el}`, shows "Say confirm to
   proceed (or cancel)", and speaks "Say confirm to proceed." when speech output is on or the mic is
   listening. A timer clears it after `CONFIRM_WINDOW_MS + 200` ms (8.2 s) with "Confirmation timed out."
3. The next utterance is interpreted with `context.pending`. Within 8000 ms: "confirm", "yes", "ok",
   "proceed", "go ahead", "do it" (and combinations with "please") give `confirm`; "cancel", "no",
   "abort", "never mind" give `cancel` (reason `cancelled`); anything else cancels with reason
   `not confirmed` and the rest of the utterance is still interpreted. After 8000 ms the result is
   `cancel` with reason `timeout`.
4. On `confirm` the element is clicked and `voiceConfirmedAt` is set. An `htmx:confirm` listener
   in `initOutput` lets a click made within 2 seconds of that skip the native `hx-confirm` dialog
   (`issueRequest(true)`), so a voice user answers once. Other `hx-confirm` prompts are unaffected.
   This path is covered by reading the code, not by a browser test.

### Output side: speech synthesis

- `data-speak="text"` on an element is the text to read. Keep it a plain sentence stream: Jinja
  autoescapes it, `cleanForSpeech` removes URLs and turns `≥`, `≤`, `±`, `°`, `%`, `&` and `·`
  into words or pauses.
- `data-speak-priority="alert"` marks urgent text. It is spoken even when the toggle is off, but only
  if the user has turned voice on at least once (`ndp.voice.ever`), at rate 0.9 and pitch 1.15 and
  ahead of non-priority parts.
- `data-speak-region` marks an extra container that gets a "Read aloud" button. Containers matching
  `#result`, `[id^="result-"]` and `#text-score` get one automatically (`REGION_SEL`). Containers
  with other ids (`#events`, `#watcher`, `#transcript`, `#triage-result`) get no button.
- **Chunking.** `speak()` cancels current speech, then splits each part with `chunkText(text, 200)`
  at sentence ends (never inside "3.47"), then commas, then spaces, and queues one
  `SpeechSynthesisUtterance` per chunk.
- **Voice choice.** `pickVoice(voices, lang)` scores a voice 100 for an exact language match, 60 for
  the same base language, +5 for `localService`, +3 for `default`; no match returns `null` and the browser
  default is used with `utterance.lang` set. `lang` comes from `currentLang()`, which is the value of
  `#ndp-lang` (so "hi" after the user picks Hindi), else `<html lang>`, else the browser language.
- **When it speaks automatically.** `handleInserted(node)` runs on `htmx:afterSwap`, on `htmx:load` and once
  for `document.body` at boot. It collects `[data-speak]` elements that have not been spoken
  before (tracked in a `WeakSet`). Rules: nothing is spoken if the toggle is off (unless priority
  alert and `ever`); at most `MAX_AUTO_SPOKEN = 3` non-priority elements are read automatically, so
  long lists use the button; swaps that happen before any user gesture (`hx-trigger="load"` content)
  are ignored (`userActive()` uses `navigator.userActivation.hasBeenActive` or a flag set on
  pointerdown, keydown or touchstart). If a swapped result has no `data-speak` at all and the toggle is
  on, `summarizeRegion` builds a short summary (the `role="alert"` text, else the first heading
  plus up to three label/value pairs), capped by `capSummary` at 280 characters.
- **Blocked speech.** If the browser refuses (`error === "not-allowed"`), the parts are kept in
  `tts.blocked`, a toast says "Tap anywhere to hear the result", and the next `pointerdown` or `keydown` plays them.
- **Read aloud buttons** (`ensureReadBars`): a `.ndp-voice-bar` with a `.ndp-voice-read` button is inserted
  before each region. It is hidden while the region is empty (a `MutationObserver` on `childList`),
  shows "Read aloud" or "Stop", and sets `aria-pressed` and an `aria-label`. In `voice.css` the bar
  uses `align-items: flex-start` and `align-self: start` so the button keeps its natural height when
  the region is a grid or flex child (it had stretched to 188 px on the Live page).
- Speech is cancelled on `pagehide` and `beforeunload`, and when the mic starts.

### Input side: speech recognition

- `startListening()` cancels speech, sets `mic.listening` and calls `startRecognition()`, which creates a
  `SpeechRecognition` with `lang = currentLang()`, `interimResults = true`, `continuous = false`,
  `maxAlternatives = 1`. A toast shows "Listening: <interim text>".
- Because `continuous` is false, each `onend` restarts recognition after 250 ms while `mic.listening`
  is true. Final results go to `handleTranscript`, which builds a fresh context
  (`buildContext`), runs `interpret`, applies the actions and shows the messages joined with " · ".
  No actions gives "Didn't understand ... Say help".
- **Echo suppression.** While `tts.speaking`, a final transcript is ignored with the toast "Ignored
  while reading aloud. Say stop first." unless it is a stop-only phrase (`isStopOnly`: stop, quiet,
  silence, be quiet, stop reading, stop talking, shut up). This stops the page hearing its own voice.
- **Errors** (`onerror`): `not-allowed` and `service-not-allowed` (microphone blocked, message
  tells the user to allow it), `no-speech` (counted; three in a row stops listening), `network` (the
  browser's speech service needs a connection), `audio-capture` (no microphone), `language-not-supported`,
  anything else except `aborted` stops with the error code. A failed `start()` stops with "Could not start the microphone."
- The mic also stops when the tab is hidden and on `pagehide`. Changing `#ndp-lang` aborts the
  current recognition, which then restarts with the new language.
- The mic button opens the help dialog on a 650 ms long press, a right click or the `?` key. The
  dialog (`<dialog class="ndp-voice-help">`, `showModal()`) lists example phrases, the fields visible
  on the page and the voice commands present.
- **Which form is active** (`activeForm`): the form containing the focused element, else the last
  form the user touched, else the form most visible in the viewport. Forms inside `<header>` are
  ignored. Fields are inputs and textareas that are visible, enabled and not read-only; hidden,
  submit, button, reset, checkbox, radio, file, password, image and color inputs and **`<select>`
  elements are excluded**, so a `<select>` cannot be set by voice. Labels are read from the wrapping
  `<label>` (first text child, so the help text after it is ignored), `label[for]`, then `aria-label`, then `placeholder`.

### Persistence

| Storage | Key | Value | Written by |
|---|---|---|---|
| `localStorage` | `ndp.voice.on` | `"1"` or `"0"`: reading aloud is on | toggle click |
| `localStorage` | `ndp.voice.ever` | `"1"` once voice has ever been turned on; permits priority alerts when off | first time the toggle is turned on |
| `localStorage` | `ndp-lang` | language code | [i18n.js](#7-i18njs) |
| cookie | `googtrans` | `/en/<code>` | i18n.js |

All `localStorage` access is in `try/catch`, so blocked storage just means the settings do not persist.

### Browser support and degradation

There are no polyfills. Speech synthesis is required for the toggle, the buttons and automatic
reading; if `speechSynthesis` or `SpeechSynthesisUtterance` is missing, `initOutput` returns and the
toggle stays `hidden`. Recognition needs `SpeechRecognition` or `webkitSpeechRecognition`; without it
the mic button stays hidden (and the toggle can still work). With neither, no voice UI appears and
the pages are unaffected. The help dialog falls back to the `open` attribute when `showModal` is
missing. The help dialog warns that recognition is done by the browser's speech service and may send audio
to it. I did not test any real browser for this document.

### Tests

[tests/js/voice.test.mjs](../tests/js/voice.test.mjs) has 19 `node:test` cases (statistics.md counts
19) that `require` voice.js and cover number words, negatives and decimals, the earthquake, storm,
weather, alerts, flood and compare form contexts, the to/two ambiguity, alias disambiguation (lat,
latitude, long, longitude, lng), text capture, dates, range rejection, submit ordering, navigation,
controls, `data-voice-command`, the confirmation flow, tel fields, chunking, `cleanForSpeech`,
`pickVoice`, `capSummary` and navigation by label, including labels with a generic tail word. They use fixed form definitions, not the real DOM: the browser runtime (recognition,
synthesis, read buttons, DOM scanning) has no automated test.

```sh
node --test tests/js/*.test.mjs        # pass the files explicitly; a directory is not expanded on newer Node
node --check src/ndp/static/js/voice.js
uv run pytest tests/js                  # the same suite, run from pytest (skipped if node is missing)
```

`tests/js/test_voice_js.py` is the pytest wrapper that runs `node --check` and then the suite.
As of this writing all 19 pass on Node 24.11.

### Example phrases

Each row was produced by running `interpret` from Node with the form fields shown. Fields are the
real labels and `min`/`max` of the page; `Flood` is `flood/page.html`, `Earthquake` is
`earthquake/page.html` (lat, lon, radius), `Hurricane` is date, lat, lon, `Alerts` is place and
radius. The clock was fixed at 25 October 2026.

| Context | Phrase | Actions |
|---|---|---|
| Earthquake | "latitude 28.6 longitude 77.2 radius 300" | fill lat 28.6, lon 77.2, radius_km 300 |
| Earthquake | "latitude twenty eight point six south longitude seventy seven point two west" | fill lat -28.6, lon -77.2 |
| Earthquake | "lat minus thirty three longitude one eighteen east submit" | fill lat -33, lon 118, then submit |
| Earthquake | "latitude 95" | reject: "Latitude 95 is out of range (-90 to 90)" |
| Earthquake | "radius to 300 estimate" | fill radius_km 300, submit |
| Earthquake | "latitude" | error: "Didn't catch a number for Latitude" |
| Flood | "level now one point four two rain last three hours two" | fill level_now 1.42, rain_3h 2 |
| Flood | "rain 24 hours one twenty" | fill rain_24h 120 |
| Hurricane | "date twenty fifth of October twenty twenty six" | fill date 2026-10-25 |
| Hurricane | "date tomorrow" | fill date 2026-10-26 |
| Hurricane | "date 2026 10 25 latitude 25 longitude 80 predict" | fill date, lat 25, lon 80, then submit |
| Alerts | "place Long Beach radius fifteen" | fill place "Long Beach", radius_km 15 |
| Alerts | "place kochi" | fill place "Kochi" (title-cased) |
| any (nav) | "open flood", "go to compare models", "alerts page" | navigate to /flood, /compare, /alerts |
| any (nav) | "open compare" (label "Compare models") | navigate to Compare models |
| any (nav) | "open the moon" | error: No page named "the moon" |
| any | "read result", "stop", "repeat that", "help", "stop listening" | read, stop, repeat, help, stop-listening |
| commands present | "find nearby help" | command (no confirmation) |
| commands present | "send alert" | command with `confirm: true` |
| pending confirm, 3 s later | "confirm" / "cancel" | confirm / cancel (`cancelled`) |
| pending confirm, 9 s later | "confirm" | cancel (`timeout`) |
| any | "hello there" | `[]` (the page shows "Didn't understand") |

One of the verified outputs, pasted verbatim:

```json
[{"type":"fill","name":"lat","label":"Latitude","value":-33,"kind":"number"},
 {"type":"fill","name":"lon","label":"Longitude","value":118,"kind":"number"},
 {"type":"submit","word":"submit"}]
```

Navigation phrases are built from the link text. A label ending in a generic word (`models`,
`page`, `analysis`) is also registered without it, so with the real header labels both "open compare
models" and "open compare" navigate to Compare models. The bare word "compare" navigates too, except
on the Compare page itself (the entry is then `current`), where it falls through to the submit word.
The test `navigation: a label with a generic tail word is reachable without it` covers this.

## 6. globe.js and globe.css

The home page hero ([index.html](../src/ndp/templates/index.html)) shows an interactive 3D globe of
the past week's earthquakes. [globe.js](../src/ndp/static/js/globe.js) (18 KB, 6 KB gzipped) and
[globe.css](../src/ndp/static/css/globe.css) (2 KB) load on this page only, together with globe.gl 2.35.0.
The script exits immediately unless `#ndp-globe` exists.

**Configuration through data attributes** on `div#ndp-globe`: `data-feed` (default the USGS
`2.5_week.geojson` feed), `data-estimate-url` (the earthquake page URL), `data-tex-globe` and
`data-tex-bump` (texture image URLs).

**Data.** `load()` fetches the feed in the browser with `credentials: "omit"`, `referrerPolicy:
"no-referrer"` and a 15 s abort timeout; nothing is stored. `parseFeed` keeps features with finite
`lat`, `lon` and `mag` in range, accepts a USGS `url` only if it starts with `https://earthquake.usgs.gov/`,
collapses duplicate events (below) and sorts by magnitude descending. It reloads every 10 minutes while the tab and hero are visible.

**Duplicate events.** USGS can list one earthquake once per contributing network. `parseFeed` passes
its list through `collapseDuplicates`, which treats two events as one when they are within 90 s, 50 km
(`distanceKm`, a haversine that handles the antimeridian) and 0.5 magnitude of each other. Candidates
are ranked so an id starting with `us` (the core network) wins, then the stronger magnitude. It mirrors
`services/usgs.collapse_duplicates` on the server, so the counters and the strongest-six chips do not
double count.

**Points and rings.**

| Quantity | Rule |
|---|---|
| Colour | magnitude 5.5 and up `#ef4444`, 4 and up `#f97316`, otherwise `#facc15` (the legend in index.html shows the same bands) |
| Point radius | `0.28 + (mag - 2.5) * 0.14`, minimum 0.2; the selected quake uses `r * 1.8 + 0.2` |
| Point altitude | `0.004 + max(0, mag - 2.5) * 0.011`, plus 0.03 when selected |
| Rings | only quakes of magnitude 5 and up, the 40 strongest; max radius `2.5 + (mag - 5) * 2`, repeat period 1500 ms |
| Counters | total, strongest (value and place), and count of magnitude 5 and up |

The "strongest six" appear as chip buttons in `#ndp-top-list`, which is also the keyboard route to
the same data.

**Interaction.** Dragging rotates; zoom and pan are disabled (`enableZoom = false`, `enablePan = false`)
so the globe never captures page scroll. On coarse pointers `globe.css` sets `touch-action: pan-y` on the canvas so vertical swipes scroll
the page. Hover shows a tooltip (`.ndp-tip`, built with escaped text). Click or chip press calls
`select(q)`, which fills `#ndp-quake-card` (magnitude, place, time, depth), moves the camera, and sets
the "Estimate for this location" link to `/earthquake?lat=..&lon=..&depth=..` (the page prefills lat and
lon from the query; depth is accepted and ignored) and the USGS link. The card takes focus and Escape closes it.
Auto-rotate pauses on interaction, resumes 4 s later, and a "Pause rotation" button (`aria-pressed`) toggles it.

**Reduced motion and visibility.** If `prefers-reduced-motion: reduce` matches: no auto-rotation, no
pulsing rings, no `animateIn`, no point transitions, camera moves are instant, and the pause button is
hidden; the media query is observed, so changing the preference applies live. Rendering is paused
when the tab is hidden (`visibilitychange`) or the hero is scrolled out of view (`IntersectionObserver`,
threshold 0.05) via `pauseAnimation()` and `resumeAnimation()`. `globe.css` also removes the spinner
animation and hero transitions in reduced motion.

**Leaflet fallback.** If `window.Globe` is not a function (CDN failed), WebGL is unavailable
(`hasWebGL` probes `webgl2`, `webgl`, `experimental-webgl`), or the WebGL context is lost later
(`webglcontextlost`), `startFallback()` hides the globe, adds `ndp-is-fallback` to the hero, changes the hint
text and creates a Leaflet map in `#ndp-fallback-map` with OSM tiles and circle markers (weakest
first so strong ones draw on top). `globe.css` darkens the tiles with a CSS filter. If Leaflet is also missing,
`#ndp-fallback-msg` explains that the map could not be loaded and that the counters and list still work.

**Failure and retry.** A feed error sets `#ndp-feed-text` to "Live earthquake feed unavailable ..."
(or "showing the last data loaded" if data exists), unhides `#ndp-feed-retry`, and rewrites the globe's
`aria-label`. The Retry button calls `load()` again. A warning is written to the console. The loading
overlay hides on `onGlobeReady` or after 10 s.

**Accessibility attributes.** `#ndp-globe` is `role="img"` with an `aria-label` rewritten with live counts
and the strongest quake; the WebGL canvas is `aria-hidden="true"`. `#ndp-feed-status` is
`role="status" aria-live="polite"`. The quake card is a labelled region with `tabindex="-1"` and a labelled close button;
the USGS link announces "(opens in a new tab)". The legend group has an `aria-label`. Elements with
numbers carry `notranslate` so the translate widget leaves them alone.

## 7. i18n.js

[i18n.js](../src/ndp/static/js/i18n.js) (7 KB) fills `select#ndp-lang` and un-hides it. Languages
(code, label in its own script): `en` (shown as "Language", the original), `hi`, `ta`, `ml`, `te`,
`bn`, `ne`, `es`, `fr`, `ar`, `ja`, `zh-CN`, `id`.

**On-demand loading.** The Google script
`https://translate.google.com/translate_a/element.js?cb=ndpGoogleTranslateInit` is added to `<head>`
only by `loadGoogle()`, which `apply(code)` calls for a language other than `en`. A visitor who never
selects a language makes no request to Google. The widget is created with `pageLanguage: "en"`,
`includedLanguages` of the other twelve codes and `autoDisplay: false`, hosted in a hidden off-screen
`#ndp-gt-host`; the script then sets Google's own combo box to the chosen code. The script injects CSS
that hides Google's banner and balloon and keeps `body` from being shifted down.

**Storage.** The choice is stored in `localStorage` key `ndp-lang` (validated against the list) and in
the cookie `googtrans=/en/<code>; path=/` (also set with `domain=<hostname>`), which is what the Google
widget reads on the next page. The cookie has no expiry, so it lasts for the browser session.

**Returning visitors.** On load, `initial = readStored()`; if it is not `en`, `apply(initial, false)` runs
immediately and loads the Google script on every page view. A stale cookie with a stored `en` is cleared.

**Restoring English.** Choosing `en` clears the cookie (also for `.hostname`) and stores `en`. Because
the widget has rewritten the DOM in place, the only clean way back is a reload, so if a translation was
active in this page load (`hasCookie() || scriptRequested`) the page calls `window.location.reload()`.
Selecting English when nothing was translated loads nothing.

**Failure.** A script load error, or an exception creating the widget, calls `failed()`: shows the
note "Translation is unavailable right now." for 5 s, resets the stored choice to `en`, clears the cookie and resets the select.

**Privacy.** Translation is not private: once a non-English language is active, the browser
requests a Google script and Google's widget translates the visible page text, so text on the screen,
including results and anything the user typed that appears on the page, goes to Google's service, along with the user's IP address
and the request headers. The consent step is the user picking a language, and because the choice is
remembered the request recurs automatically on later visits until the user picks English. The app sets no
Content-Security-Policy, so nothing blocks this. Pages add `translate="no"` or `notranslate` to code
names, numbers and the select itself.

## 8. Accessibility and responsiveness

**Implemented (verified in the code):**

- Skip link to `#main`; `<main id="main">`; a labelled `<nav aria-label="Main">`; `aria-current="page"`
  on the active link.
- Every form input is inside a `<label>` (the `field` and `textarea` macros) and the language select has an
  `sr-only` label; icon-only buttons have `sr-only` text or `aria-label`.
- A global `:focus-visible` outline (`app.css`), lighter outlines on the dark hero (`globe.css`), outlines on the voice UI (`voice.css`) and focus rings on inputs.
- Result containers are `aria-live="polite"`; errors use `role="alert"` (`error_box`, assistant, live
  errors); status text uses `role="status"`; the assistant transcript is `role="log"` with `aria-relevant="additions"`.
- State on toggles: `aria-pressed` for the voice toggle, mic, read-aloud, rotate buttons; `aria-selected` for tabs.
- Keyboard: native `<details>` menus, buttons and links; Escape closes the "More" menu, the quake card and the help `<dialog>` (native modal).
  Leaflet maps rely on Leaflet's default keyboard handling once focused (not tested here).
- Motion: a global `prefers-reduced-motion` rule in `tailwind.input.css`, plus globe, voice-highlight and card-hover handling.
- Layout rules for 375 px: the header swaps the brand text below 420 px; the language select is narrowed at 480 px;
  tables sit in `overflow-x-auto` wrappers with negative margins; grid children use `min-w-0`; long text uses
  `break-words`; the voice toast is limited to `calc(100vw - 2rem)`; the globe stage switches to a stacked layout below `lg`.
- Two pages have `<noscript>` notes for their maps; every form works without JavaScript.

**Not verified or known weak (read from the code, not tested in a browser):**

- **No browser or screen-reader testing was done for this document**, including a 375 px check.
  There is no automated accessibility or viewport test in the repository (no axe, Lighthouse, Playwright);
  CI runs `uv run pytest -q` only. Colour contrast has not been audited.
- `<html lang="en">` never changes. `i18n.js` only sets `data-ndp-lang` on the root element, so a screen
  reader keeps pronouncing translated text with the English voice (whether Google changes `lang` was not checked).
  Likewise `data-speak` text is written in English by the server, while `voice.js` chooses a voice for the
  selected language, so spoken output after translating is probably English text in a foreign voice.
- Tabs (`tab` macro and `initTabs`) have `role="tab"` and `aria-selected` but no `aria-controls`, no
  arrow-key navigation and no roving `tabindex`. Only `covid/maps.html` uses them.
- Leaflet `L.circleMarker` layers are SVG paths and are not keyboard focusable (Leaflet behaviour); the
  map content is reachable by keyboard only where the page also lists it (alerts, GDACS, centres, the home page chip list).
  The COVID state and city maps have no list.
- The mobile "Menu" `<details>` does not close on Escape or outside click, and neither menu moves focus.
- The read-aloud buttons exist only for `#result`, `#result-*`, `#text-score` and `[data-speak-region]`.

## 9. Static assets

```
src/ndp/static/
  css/   tailwind.css (compiled, committed)  app.css  globe.css  voice.css
  js/    app.js  voice.js  i18n.js  globe.js
  img/disasters/   japan.jpg  live.jpg  chennai.jpg  Kerala.jpeg  indonesia.jpg  phailin.jpg
  generated/       training plots, git-ignored
```

| Folder | Contents and owner |
|---|---|
| `css/` | `tailwind.css` is produced by `frontend/build.sh`. `app.css` is loaded by `base.html` (indicator, `[hidden]` override, Leaflet z-index isolation, popup spacing). `globe.css` is loaded by `index.html` only. `voice.css` is injected by `voice.js`. |
| `js/` | Hand-written, unminified, served as-is, loaded by `base.html` (`app.js`, `voice.js`, `i18n.js`) or `index.html` (`globe.js`). |
| `img/disasters/` | Photos for the sentiment table, referenced from `Disaster.image` in [ml/sentiment.py](../src/ndp/ml/sentiment.py) and rendered by `sentiment/page.html`. Not referenced anywhere else. |
| `generated/` | Created by `ndp-train`. `ml/plots.py` (`save_current`, `bar_comparison`) writes PNGs to `<generated_dir>/<sub>/<file>` (`generated_dir` is `static/generated`, created at startup in `main.py`). Routers list them as `generated/<area>/<file>.png` and `_plots.html` links them through `url_for('static', ...)`. The folder is in `.gitignore`; a fresh clone shows "No plots yet" until training has run. |

Front-end payload from [statistics.md](statistics.md):

| File | Size | Gzipped |
|---|---:|---:|
| css/app.css | 591 B | 370 B |
| css/globe.css | 2 KB | 1 KB |
| css/tailwind.css | 32 KB | 7 KB |
| css/voice.css | 3 KB | 1 KB |
| js/app.js | 11 KB | 4 KB |
| js/globe.js | 18 KB | 6 KB |
| js/i18n.js | 7 KB | 3 KB |
| js/voice.js | 67 KB | 19 KB |

From the Repository table: Jinja templates 46 files and 2,520 non-blank lines; front-end JavaScript
4 files and 2,368 lines; hand-written CSS 3 files and 144 lines; JavaScript tests 1 file and 375 lines
with 19 test functions. Leaflet, HTMX and globe.gl come from a CDN, and the globe library only on the home page.

## 10. How-to recipes

### Add a new form page

Example: a `drought` page with one field. These snippets were run against the real app with a
scratch template directory and behaved as described (HTMX request returns the partial, plain POST
returns the full page, invalid input shows the `role="alert"` box).

1. Router `src/ndp/routers/drought.py`:

```python
from fastapi import APIRouter, Form, Request
from ndp.forms import FormError, parse_float
from ndp.web import page_or_partial, render

router = APIRouter(prefix="/drought")


def _page_context() -> dict:
    return {}


@router.get("", name="drought_page")
def drought_page(request: Request):
    return render(request, "drought/page.html", **_page_context())


@router.post("/predict", name="drought_predict")
def drought_predict(request: Request, rain_mm: str = Form("")):
    try:
        value = parse_float(rain_mm, "Rain, 30 days (mm)", 0, 500)
    except FormError as exc:
        return page_or_partial(
            request,
            "drought/page.html",
            "drought/_result.html",
            _page_context,
            inputs={"rain_mm": rain_mm},
            error=str(exc),
        )
    return page_or_partial(
        request,
        "drought/page.html",
        "drought/_result.html",
        _page_context,
        inputs={"rain_mm": value},
        result={"index": value / 10},
    )
```

2. Register it in the router tuple in [main.py](../src/ndp/main.py) (`create_app`, both the import and the loop).
3. `templates/drought/page.html` (the grid wrapper and model card of the real pages are omitted):

```html
{% extends "base.html" %}
{% from "partials/_form_bits.html" import field, submit, card %}
{% block title %}Drought index{% endblock %}
{% block content %}
{% set v = inputs if inputs is defined and inputs else {} %}
<h1 class="text-2xl font-semibold tracking-tight text-slate-900">Drought index</h1>
{% call card('Conditions', 'Rain over the last 30 days in millimetres.') %}
  <form method="post" action="{{ url_for('drought_predict') }}"
        hx-post="{{ url_for('drought_predict') }}" hx-target="#result" hx-swap="innerHTML" hx-indicator="find .htmx-indicator"
        class="space-y-4">
    {{ field('rain_mm', 'Rain, 30 days (mm)', value=v.get('rain_mm', '40'), min=0, max=500, step='0.1') }}
    {{ submit('Compute index') }}
  </form>
{% endcall %}
<div id="result" aria-live="polite">
  {% if result is defined or error is defined %}{% include "drought/_result.html" %}{% endif %}
</div>
{% endblock %}
```

4. `templates/drought/_result.html`:

```html
{% from "partials/_form_bits.html" import metric, error_box, inputs_list %}
{% if error is defined and error %}
  {{ error_box(error) }}
{% elif result is defined and result %}
  <div class="rounded-xl border border-slate-200 bg-white p-5"
       data-speak="Drought index {{ '%.1f'|format(result.index) }} for {{ '%.0f'|format(inputs.rain_mm) }} millimetres of rain.">
    {{ metric('Index', '%.1f'|format(result.index)) }}
    {{ inputs_list(inputs, {'rain_mm': 'Rain, 30 days (mm)'}) }}
  </div>
{% endif %}
```

5. Add a nav entry (below) and rebuild the CSS if you used new classes. Add a model card with
   `{% include "partials/_model_card.html" %}` if `meta` and `trained` are in the page context.

### Add a new field and make it speakable

Add the input with `field(...)` in `page.html`, pass `min` and `max` so out-of-range speech is
rejected, and read the value in the router. Voice matching is automatic: the field is addressable by its
`name` and its label without units, and by the last word of either. Check with `interpret`:

```js
const V = require("./src/ndp/static/js/voice.js");
const f = [{ name: "soil_moisture", label: "Soil moisture (%)", type: "number", min: "0", max: "100" }];
V.interpret("moisture 40", { fields: f });
// [{"type":"fill","name":"soil_moisture","label":"Soil moisture","value":40,"kind":"number"}]
```

If the natural spoken word is not part of the name or label (for example a field `rain_mm` labelled
"Rain, 30 days (mm)"), "rain 45" is not understood. Add an entry to `ALIAS_GROUPS` in `voice.js`
whose phrases include the field's normalised name or label; then all its phrases apply:

```js
{ id: "rain-30d", phrases: ["rain mm", "rain 30 days", "rain"] }
```

With that group, `interpret("rain 45", ...)` returns `fill rain_mm 45` (verified on a patched copy).
Then add a case to `tests/js/voice.test.mjs` and run `node --test tests/js/*.test.mjs`. `<select>`
elements and checkboxes cannot be filled by voice.

### Add a map to a page

```html
<div id="my-map" class="mt-3 h-72 sm:h-96 w-full rounded-lg bg-slate-100" role="application"
     aria-label="Map of ..." data-ndp-map data-lat="20" data-lon="10" data-zoom="2"
     data-markers-src="#my-markers" data-status="#my-map-status"></div>
<script type="application/json" id="my-markers">{{ markers|tojson }}</script>
```

Build `markers` in the router (or with `{% set _items = [] %}` and `.append`, as `alerts/_results.html` does)
as the list described in [Marker JSON shape](#marker-json-shape). Use unique ids. The map initialises
itself on page load and again when the partial is swapped in, so no script is needed. Add
`<noscript>` text if the map is the only way to see the data, and also render a list (see accessibility).

### Add a result that is read aloud

Put `data-speak="..."` on the root element of the result partial, written as plain sentences with units
spelled out ("metres", "percent"). Use `data-speak-priority="alert"` only when the user should hear it even with voice off
(the earthquake partial does so when `result.elevated`). Pre-format numbers with `'%.1f'|format(x)`; use
`{% set text %}...{% endset %}` for conditional wording. Make the container `id="result"` or
`id="result-<name>"` to get the "Read aloud" button; otherwise add `data-speak-region`. Do not put HTML in `data-speak`. Do not put `data-speak` on a fragment that is polled or refreshed on a timer (as `live/_watcher.html` used to), because every refresh creates new elements that are read again.

### Add a nav entry

1. Give the route a name: `@router.get("", name="drought_page")`.
2. Add `('drought_page', 'Drought')` to `primary` or `more` in `base.html`. `nav` is built from those lists, so
   the mobile menu follows. Keep labels one or two words: voice navigation says "open <label>", and
   a label ending in a generic word such as "models" or "page" can also be spoken without it.
3. For a home-page card add a dict to `modules` in `routers/pages.py` (`name`, `blurb`, `endpoint`,
   `status`) and an icon in the `icons` dict in `index.html`; without it the card falls back to a default icon.
4. If the page should be optional, follow the COVID pattern: guard the link in `base.html`, the card in
   `pages.py` and the router in `main.py` with the same setting.

### Rebuild the CSS

Run `frontend/build.sh` from anywhere, commit `src/ndp/static/css/tailwind.css` with your template
change, and check `git diff --stat` shows only that file changed in `static/css`. See
[section 1](#rebuilding-the-css) for the globs and pitfalls.

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| New Tailwind classes have no effect | `tailwind.css` was not rebuilt, or the class is built from string pieces, or it lives in a `.py` file other than `services/assistant.py` | Run `frontend/build.sh`; write full class names in a template or JS file; for other Python-emitted classes add the file to `content` in `tailwind.config.js`. |
| `bg-brand-500` or similar does nothing | Only `brand-50`, `600`, `700`, `900` are defined | Use one of those or extend `tailwind.config.js`, then rebuild. |
| Map not showing in a swapped partial | Missing `data-ndp-map`; element has no height; duplicate `id`; Leaflet not loaded; the swap target is outside the DOM at `htmx:load` time | Check the attribute and a height class (`h-72`). `htmx:load` fires for swapped content, so no script is needed. In a console run `ndpInit(document)` to retry. A map inside a hidden container needs `invalidateSize()` after showing (use the tab pattern). |
| Map shows grey tiles or is cut off | Container was hidden or resized when created | Use `data-tab-panel` inside `data-tabs`, or call `el._ndpMap.invalidateSize()`. |
| Marker missing from a map | Non-numeric `lat` or `lon` (skipped silently) | Fix the JSON; check the script tag's `id` matches `data-markers-src`. "Bad marker data" in the status element means invalid JSON. |
| Voice button missing | The browser lacks `speechSynthesis` (toggle) or `SpeechRecognition` (mic); `voice.js` failed to load; the buttons are `hidden` in `base.html` until the script runs | Check `window.NdpVoice` and the console; in an unsupported browser this is expected. |
| Speech does not start on page load or after a timed swap | Browsers require a user gesture before speaking; `voice.js` also ignores swaps before the first gesture | Click anywhere once. After a `not-allowed` error the pending text plays on the next `pointerdown` or `keydown`. Use the "Read aloud" button. |
| Result is never read aloud | The element has no `data-speak`; the toggle is off (and it is not a priority alert); more than three non-priority `data-speak` elements in one swap | Add `data-speak`; turn voice on; use the button for long lists. |
| No "Read aloud" button | The container id is not `result`, `result-*` or `text-score`, and has no `data-speak-region` | Rename the id or add `data-speak-region`. |
| Spoken command ignored with "Ignored while reading aloud" | The page was speaking when the transcript arrived | Say "stop" first, or wait. |
| Microphone does nothing or is blocked | Permission denied, no mic, or the `Permissions-Policy` header was changed | Allow the microphone for the site; keep `microphone=(self)` in `main.py`; recognition also needs a network connection. |
| Language select is empty or absent | `i18n.js` did not run (script blocked, JS error before it) | The `<select id="ndp-lang">` is empty in HTML on purpose; check the console and that `static/js/i18n.js` returns 200. |
| Choosing a language does nothing | The Google script is blocked (offline, content blocker, firewall) | Expect the note "Translation is unavailable right now."; the choice resets to English. |
| Page still translated after choosing English | The `googtrans` cookie survived (it is cleared for the host and `.host` only) | Choose English again, then clear the site's cookies if it persists. |
| Globe is blank, hero shows a flat map | WebGL unavailable, globe.gl CDN blocked, or context lost | Expected fallback; check the console for "[ndp] globe unavailable". |
| Globe area is empty with a message | Both globe.gl and Leaflet failed | The text in `#ndp-fallback-msg` says so; the counters and chip list still work if the USGS feed loaded. |
| "Live earthquake feed unavailable" | The USGS request failed or timed out (15 s) | Press Retry. The feed is fetched by the browser, so a firewall on the client blocks it, not on the server. |
| Page unstyled after deploy behind a proxy | Absolute `url_for` URLs use `http://` (mixed content) | Run uvicorn with `--proxy-headers` and a correct `--forwarded-allow-ips` (see `deploy/entrypoint.sh`). |
| `#result` shows "Model ... is not trained yet" | `ModelNotTrained`: HTMX requests get the `errors/_not_trained.html` partial (status 200); a normal navigation gets a 503 page | Run `uv run ndp-fetch-data` then `uv run ndp-train`. |
| Assistant errors do not appear | HTMX does not swap 4xx and 5xx responses | The assistant page overrides this with `htmx:beforeSwap`; copy that for new endpoints that return error statuses. |
| Lists stuck on "Loading..." | JavaScript off or HTMX failed to load (`hx-trigger="load"` regions) | Check that the HTMX CDN request succeeded. |

### Known issues found while writing this

Observations, not fixes. Items 1 to 8 were noted from reading the code; none was tested in a browser.

1. `<html lang>` is always `en`. `i18n.js` only sets `data-ndp-lang`, and `voice.js` picks a speech voice for the selected language while `data-speak` text is English.
2. Tabs (`tab` macro, `initTabs`) have no `aria-controls` and no arrow-key navigation.
3. Leaflet circle markers are not keyboard focusable, and the COVID state and city maps have no list alternative.
4. The mobile "Menu" `<details>` does not close on Escape or outside click.
5. `<select>` elements cannot be filled by voice.
6. `data-default-today` uses the UTC date (`toISOString`), which can differ from the local date near midnight.
7. There are no automated accessibility or viewport tests, and CI does not check that `tailwind.css` is current.
8. The i18n language choice persists in `localStorage`, so the Google script loads automatically on later visits until English is chosen.
9. The home-page globe takes about ten seconds before its textures appear and the hero stays dark until then (reported from testing, not reproduced here; the code only shows a 10 s fallback timer that hides the loading overlay).
