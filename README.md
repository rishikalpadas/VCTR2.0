# Image to Vector — POC

Upload raster artwork (PNG / JPG / WebP), get back an SVG containing **real
vector path geometry** — not a bitmap wrapped in an `<svg>` tag.

```
Browser (vanilla JS)
      │  multipart/form-data
      ▼
Node + Express  :5000        ← validation, temp files, error mapping. No image processing.
      │  multipart/form-data
      ▼
Python + FastAPI :8000       ← the entire pipeline lives here
      │
      ├─ load & validate      (magic bytes, EXIF, size limits)
      ├─ analyze              (colour stats, flat-background test, kind heuristic)
      ├─ preprocess           (resize → background → alpha → denoise → quantize → binarize)
      ├─ vectorize            (VTracer, behind a swappable engine interface)
      └─ cleanup & validate   (viewBox, artifact culling, scour, anti-raster check)
      │
      ▼  { success, svg, meta }
Browser → preview + download
```

**Verified working** on Windows 10, Node 22.16, Python 3.12, VTracer 0.6.15,
CPU only. No CUDA, no GPU, no Docker, no database.

---

## 1. Quick start (Windows)

Two terminals. Do the Python one first — Node reports the engine as offline
until it is up.

### Terminal 1 — Python engine

```bat
cd python-engine
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Expect: `Uvicorn running on http://127.0.0.1:8000`

### Terminal 2 — Node API + frontend

```bat
cd server
npm install
npm start
```

Expect: `Image-to-Vector API  http://127.0.0.1:5000`

### Then

Open <http://127.0.0.1:5000> and upload an image. The badge under the title
turns green when the engine is reachable.

> **Use Python 3.12 or 3.13, not 3.14.** VTracer publishes wheels up to
> CPython 3.13. On 3.14 pip tries to build the Rust core from source and fails
> unless you have a Rust toolchain. Check with `py -0p`.

---

## 2. Project layout

```
image-to-vector-poc/
├── frontend/                   Static page served by Express
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── server/                     Node API — thin proxy, zero image logic
│   ├── server.js               app wiring, static hosting, shutdown
│   ├── config.js               env-driven config
│   ├── routes/vectorize.js
│   ├── controllers/vectorizeController.js
│   ├── services/pythonService.js   ← only module that knows Python exists
│   ├── middleware/
│   │   ├── upload.js           multer + magic-byte verification + cleanup
│   │   └── errorHandler.js     log detail, return safe messages
│   ├── utils/{logger,ApiError}.js
│   ├── test/api.test.js        node:test suite (14 tests)
│   ├── uploads/                transient, wiped after every request
│   └── outputs/                only used when PERSIST_OUTPUTS=true
│
├── python-engine/              The actual vectorization service
│   ├── app.py                  FastAPI endpoints + error translation
│   ├── vectorizer.py           pipeline orchestration
│   ├── analysis.py             image statistics + kind heuristic
│   ├── preprocessing.py        resize / denoise / quantize / binarize
│   ├── background.py           border flood fill + enclosed-region removal
│   ├── presets.py              the tuning table
│   ├── svg_optimizer.py        viewBox, artifact culling, scour, validation
│   ├── image_io.py             safe decoding
│   ├── config.py, errors.py, logging_config.py, schemas.py
│   ├── engines/
│   │   ├── base.py             BaseVectorizer + registry
│   │   ├── vtracer_engine.py   the working engine
│   │   ├── photo_extractor.py  V2 garment-photo pipeline (stubbed, documented)
│   │   └── future_engines.py   StarVector / Adobe placeholders
│   ├── tests/
│   │   ├── test_pipeline.py    32 tests incl. precision + background regressions
│   │   ├── make_samples.py     synthetic fixtures
│   │   ├── make_logo_repro.py  dark-field logo repro case
│   │   └── quality_harness.py  scores output against a lossless reference
│   └── requirements.txt
│
├── samples/                    generated fixtures (npm run samples)
├── package.json                convenience scripts
└── .gitignore
```

---

## 3. API

### `GET /api/health`

```json
{
  "status": "ok",
  "service": "image-to-vector-api",
  "uptime_s": 42,
  "engine": { "status": "ok", "version": "1.0.0", "engines": [ ... ] }
}
```

Returns **200 even when the engine is down**, with `status: "degraded"` — the
Node API itself is healthy, and the frontend uses this to drive the badge.

### `POST /api/vectorize`

`multipart/form-data`:

| Field | Required | Description |
|---|---|---|
| `image` | yes | PNG, JPG/JPEG or WebP, ≤ 15 MB |
| `preset` | no | `auto` (default), `standard`, `logo`, `flat_art`, `typography`, `line_art`, `detailed` |
| `background` | no | `auto`, `always`, `never` |
| `remove_enclosed_background` | no | `true` to knock out letter counters |
| `max_dimension` | no | working resolution of the artwork |
| `supersample` | no | `1`–`3`. Trace at this multiple of the working resolution, then scale back via the viewBox. The main lever on edge smoothness. |

Success:

```json
{
  "success": true,
  "svg": "<?xml version=\"1.0\"...<svg ...><path d=\"...\"/></svg>",
  "meta": {
    "preset_requested": "auto",
    "preset_used": "flat_art",
    "engine": "vtracer",
    "path_count": 3,
    "svg_bytes": 2871,
    "size_reduction_pct": 44.4,
    "processing_ms": 357,
    "original_width": 600, "original_height": 600,
    "processed_width": 600, "processed_height": 600,
    "preprocess_steps": ["background_removed(31.2%)", "alpha_hardened", "bilateral(d=7)", "quantize(k=5)"],
    "background": { "applied": true, "removed_ratio": 0.312, "reason": "..." },
    "analysis": { "kind": "flat_graphic", "kind_confidence": 0.7, "...": "..." },
    "warnings": []
  }
}
```

Failure — always this shape, never a stack trace:

```json
{ "success": false, "error": { "code": "CORRUPT_IMAGE", "message": "The image could not be decoded. It may be corrupted or truncated." } }
```

| Code | Status | Cause |
|---|---|---|
| `NO_FILE` | 400 | no `image` field |
| `UNSUPPORTED_FORMAT` | 415 | bad mime **or** bytes that don't match the claimed type |
| `FILE_TOO_LARGE` | 413 | over the upload limit |
| `CORRUPT_IMAGE` | 400 | decode failed |
| `IMAGE_TOO_LARGE` | 413 | over 40 megapixels |
| `UNKNOWN_PRESET` | 400 | bad preset name |
| `ENGINE_UNAVAILABLE` | 503 | Python service not running |
| `ENGINE_TIMEOUT` | 504 | exceeded `PYTHON_TIMEOUT_MS` |
| `INVALID_SVG` | 500 | output failed validation |
| `ENGINE_NOT_IMPLEMENTED` | 501 | a planned engine was selected |

The Python service also exposes `GET /health`, `GET /presets`, `GET /engines`
and `POST /vectorize` directly on `:8000`, plus interactive docs at
<http://127.0.0.1:8000/docs>.

---

## 4. Presets

`auto` runs a heuristic classifier and routes to one of these. It is a rule-based
heuristic over colour/edge statistics, not a trained model — `meta.analysis`
always reports what it decided and how confident it was.

| Preset | For | What it does differently |
|---|---|---|
| `standard` | mixed clean artwork | balanced; light bilateral denoise, no quantization |
| `logo` | badges, emblems, lockups | flat colours *and* crisp lettering at once: 2x supersample, `corner_threshold=40`, `layer_difference=12` so illustration detail survives |
| `flat_art` | stickers, flat vector-style graphics | k-means quantization collapses anti-aliasing bands → far fewer paths |
| `typography` | lettering, logos, outlined type | **no blurring at all**; `corner_threshold=45` keeps letterform corners sharp; upscales small input |
| `line_art` | doodles, outlines, icons | adaptive-threshold binarization + 1-bit tracing |
| `detailed` | dense / textured artwork | `filter_speckle=2`, `layer_difference=8` — keeps detail, much bigger file |

Add a preset by appending one entry to `PRESETS` in `presets.py`. Nothing else
changes.

---

## 5. Background handling

Removal is a **flood fill inward from the image border**, not a global
"delete every pixel matching the background colour".

That distinction matters for your samples. On pink-canvas lettering, the
counters inside the letters are also pink. A global colour match would punch
holes through the artwork; a flood fill only removes pink that is *connected to
the canvas edge*, so enclosed counters survive.

In `auto` mode it refuses to act unless the border is genuinely flat
(≥ 85% uniform) and the region it would remove is between 4% and 95% of the
canvas. Every decision, including a refusal, is reported in
`meta.background.reason`.

**"Also knock out enclosed background areas"** (off by default) additionally
removes small enclosed regions matching the background colour — turning letter
counters into true transparent cutouts. It is off by default because it is a
genuine judgement call: an enclosed same-colour region might be a letter
counter or a real dark area of the artwork. Only regions under 10% of the
canvas are eligible. Turn it on for lettering; leave it off for something like
a dark logo on a dark field.

---

## 6. What "real vector" means here, and how it's enforced

`svg_optimizer.validate_svg()` **rejects** output that contains an `<image>`
element or a `data:image` URI. Faking vectorization fails the pipeline rather
than shipping a file that merely has a `.svg` extension.

The output also:

- carries `viewBox` in traced-pixel space with `width`/`height` in the original
  image's units, so it scales cleanly and reports the size you uploaded;
- uses spline (Bézier) fitting rather than polygon staircases;
- preserves holes — VTracer emits the inner contour wound opposite to the
  outer one, so the default `nonzero` fill rule renders counters and cutouts
  correctly (asserted in the test suite);
- drops paths whose bounding-box diagonal is under a fraction of the canvas
  diagonal, which removes single-pixel tracing noise.

Verify it yourself: click **Download SVG**, open it in Illustrator or Inkscape,
and you'll get selectable, editable paths. Or open it in a text editor — it's
`<path d="M… C…"/>` all the way down.

---

## 7. Honest limitations

**This is classical raster tracing.** It is genuinely good at clean digital
artwork — flat graphics, lettering, line work, moderately complex colourful
illustration. It is doing region segmentation and curve fitting, not
*understanding* the artwork.

What it will **not** do:

- **Photographs of printed T-shirts.** This is the big one, and it is not a
  tuning problem. In a garment photo the artwork is entangled with fabric weave,
  folds warping the print, a lighting gradient, cast shadows, and the garment
  colour showing through halftones. Tracing it directly produces thousands of
  blobby paths that follow the *lighting*, not the design. The required
  information has to be recovered before tracing. `auto` detects photographic
  statistics and returns an explicit warning in `meta.warnings` instead of
  pretending. The planned pipeline is laid out stage by stage in
  `engines/photo_extractor.py`; selecting the `photo_extract` engine returns a
  clean 501.
- **Recover fonts.** Lettering becomes outlined paths, not editable live text.
- **Reconstruct gradients or soft shadows.** These become discrete colour
  bands. Raising `color_precision` adds bands; it does not produce a gradient
  fill.
- **Vectorize halftone dots or glitter/sequin texture meaningfully.** Each dot
  becomes its own path. Path counts explode and the result is not print-ready.
  `filter_speckle` suppresses them instead, which loses the texture.
- **Separate semantic layers.** You get colour regions, not "the character",
  "the headline", "the border".

Rough expectations, from the fixtures in `samples/`:

| Input | Paths | Size | Time |
|---|---|---|---|
| Flat shapes, 600×600 | 3–5 | ~3 KB | ~0.3 s |
| Lettering on flat background | 8–15 | ~6–9 KB | ~0.3 s |
| Line art, 800×600 | 8 | ~6 KB | ~0.15 s |
| Noisy gradient photo, 700×500 | ~3,600 | ~860 KB | ~6 s |

That last row is the honest illustration of the limitation.

---

## 8. Swapping the engine later

The engine contract is one method:

```python
class BaseVectorizer(abc.ABC):
    name: str
    @abc.abstractmethod
    def vectorize(self, rgba: np.ndarray, preset: Preset) -> EngineResult: ...
```

To add StarVector, an Adobe API, or anything else:

1. Subclass `BaseVectorizer` in `python-engine/engines/`.
2. Call `registry.register(YourEngine())`.
3. Set `engine="your_engine"` on a preset.

Node, the frontend, and the pipeline are untouched — `pythonService.js` is the
only Node module aware the engine exists at all, and it only speaks HTTP.
`future_engines.py` contains working skeletons for both, including notes on
the GPU and credential constraints each brings.

---

## 9. Testing

### Python (32 tests)

```bat
cd python-engine
venv\Scripts\activate
python tests\make_samples.py
python -m unittest discover -s tests -t . -v
```

Covers format rejection, truncated files, output-is-real-vector assertions,
hole/cutout geometry, every preset, coordinate-precision regressions,
background-removal safety, supersampling behaviour, and the line-art
binarization bug - all described under Implementation notes.

### Node (14 tests)

```bat
cd server
npm test
```

Covers health, static hosting, 404 shape, all upload gates (including a file
that claims `image/png` but isn't), temp-file cleanup, and the full round trip.
Engine-dependent tests **skip automatically** if Python isn't running, so
`npm test` is always meaningful.

### Manual checks

```bat
curl http://127.0.0.1:5000/api/health
curl http://127.0.0.1:8000/health
```

```bat
curl -X POST http://127.0.0.1:5000/api/vectorize ^
  -F "image=@samples/sample_flat_art.png" ^
  -F "preset=flat_art" ^
  -o result.json
```

PowerShell:

```powershell
$form = @{ image = Get-Item .\samples\sample_flat_art.png; preset = 'flat_art' }
$r = Invoke-RestMethod -Uri http://127.0.0.1:5000/api/vectorize -Method Post -Form $form
$r.meta.path_count
$r.svg | Out-File -Encoding utf8 result.svg
```

---

## 10. Troubleshooting

**Badge says "engine offline"**
The Python service isn't running or crashed. Check terminal 1; confirm with
`curl http://127.0.0.1:8000/health`.

**`pip install vtracer` fails to build / "Cargo, the Rust package manager, is not installed"**
You're on Python 3.14 (or another version without a wheel). Recreate the venv
with 3.12: `py -0p` to list versions, then
`rmdir /s /q venv && py -3.12 -m venv venv`.

**`ModuleNotFoundError: No module named 'cv2'`**
The venv isn't active, or `pip install` ran against the wrong interpreter.
`venv\Scripts\activate` first; the prompt should show `(venv)`.

**`ENGINE_TIMEOUT` on large images**
Detailed tracing on CPU is slow. Lower **Max dimension** in Advanced, switch to
`flat_art`, or raise `PYTHON_TIMEOUT_MS` in `server/.env`.

**Result has thousands of paths and is megabytes**
Expected for photographic or textured input — see Limitations. Use `flat_art`,
or lower Max dimension.

**Background wasn't removed**
`auto` refused because the border wasn't flat enough. `meta.background.reason`
says exactly why. Force it with the Background dropdown set to *Always remove*.

**Letter counters are filled in with the background colour**
That's the conservative default. Tick **"Also knock out enclosed background
areas"**.

**`EADDRINUSE :5000`**
`netstat -ano | findstr :5000`, then `taskkill /PID <pid> /F` — or set `PORT` in
`server/.env`.

**`npm audit` reports a `qs` advisory**
`server/package.json` pins a patched `qs` via `overrides` because Express 5.2.1
still ships the vulnerable range. Remove the override once Express bumps it.

---

## 11. Production-ready vs POC-level

**Solid enough to carry into the MERN app:**

- the Node↔Python split, and `pythonService.js` as the single integration seam;
- the engine abstraction and registry;
- error taxonomy — typed codes, safe client messages, detail only in logs, no
  Python tracebacks reaching the browser;
- upload handling — magic-byte verification, random filenames, size caps,
  decompression-bomb guard, cleanup in a `finally` so it survives failures;
- the preset/tuning structure;
- SVG validation, including the anti-embedded-raster check.

**Deliberately POC-level — change before production:**

- **Synchronous request/response.** A 6-second trace holds an HTTP connection.
  Move to a job queue with polling or webhooks before real traffic.
- **No auth, no rate limiting, no quotas.** Anyone who can reach the port can
  burn CPU. Add both at the Node layer.
- **No CORS config.** Same-origin only; you'll need a policy when the frontend
  is served separately.
- **SVG returned inline in JSON.** Fine at a few hundred KB, wasteful at
  megabytes. Write to object storage and return a URL.
- **Single-process, unbounded concurrency.** Uvicorn with one worker and no
  semaphore; N simultaneous large images will saturate the CPU. Add a worker
  pool and a concurrency limit.
- **The `auto` classifier is a heuristic**, tuned against a handful of fixtures.
  Treat its output as a hint, keep the manual preset override.
- **No persistence or audit trail** by default (`PERSIST_OUTPUTS=false`).
- **`server/uploads/` is local disk.** Won't survive multiple instances.

---

## 12. Implementation notes

### Output quality: what actually moved the needle

A reported case — a 1017x880 JPEG badge logo, dark field, white serif wordmark,
thin olive rules, small camera illustration — came back with visibly jagged
letter edges, lumpy strokes and smeared illustration detail. Measured against a
lossless reference with `tests/quality_harness.py`:

| Change | RMSE | Edge RMSE |
|---|---|---|
| As reported | 41.48 | 89.49 |
| + coordinate-precision fix | 15.16 | 40.96 |
| + JPEG artifact cleanup | 11.50 | 33.60 |
| + supersampling and the `logo` preset | **9.39** | **26.92** |

4.4x less error overall. Two thirds of it was one line of configuration.

### The bugs behind it

1. **The optimizer was destroying the geometry it optimized.** scour's
   precision option counts **significant digits**, not decimal places. It was
   set to 2. On a 1017px canvas that rounds every number in the file to two
   significant figures: the root `width` became `1e3` — literally 1000, a 17px
   error — and every Bezier control point was snapped to the integer pixel
   grid. VTracer was emitting good curves and the cleanup stage was quantizing
   them into a staircase. This is the jaggedness. Now 5 significant digits
   (scour's own default); `OptimizeParams.significant_digits` documents the
   trap. Regression tests: `TestCoordinatePrecision`.

2. **Quantization was spending its colour budget on compression noise.** `k`
   was derived from `unique_colors`, which counts every JPEG ringing artifact
   and anti-aliasing step as a distinct colour — a six-colour logo reported 437
   and got `k=11`. The surplus clusters land on the halos hugging every
   high-contrast edge, turning each into its own thin sliver path. `k` now
   comes from `analysis.significant_colors` (buckets covering >=0.4% of the
   image), which reports 9 for the same file.

3. **The background flood fill ate dark artwork on a dark field.** A fixed
   tolerance of 18 is wider than the gap between a near-black illustration
   (28,27,24) and a near-black canvas (43,41,39), so the fill walked across the
   anti-aliased boundary and erased the artwork. `analysis.border_color_margin`
   now measures the distance to the nearest significant non-background colour
   and the tolerance is clamped to half of it — 18 -> 7 on that logo, and
   removal drops from 86% of the canvas to 77%. Regression tests:
   `TestBackgroundSafety`.

4. **Supersampling needs its engine parameters scaled with it.** Tracing at 2x
   without touching VTracer's thresholds makes `filter_speckle` (an *area*)
   suppress a quarter of what it should and doubles the node count for no gain.
   `presets.scale_engine_params` scales areas by the square of the factor and
   lengths linearly. Until the precision bug above was fixed, supersampling
   appeared to make output *worse* — at 2x the viewBox `2034` was being
   mangled to `2e3`, a 1.7% scale error that outweighed the smoothing.

5. **Line-art preset traced one canvas-sized black rectangle.** VTracer's
   `colormode="binary"` thresholds on *luminance* and ignores alpha. Feeding it
   black ink on a transparent canvas reads as an all-black image. Binarization
   must output black-on-**white** and fully opaque; the tracer then emits only
   the dark shapes, and the SVG background is transparent because nothing was
   drawn there.

6. **Every SVG was malformed** (first build). Setting `xmlns` manually on a
   root element already in the SVG namespace makes ElementTree emit the
   declaration twice: `duplicate attribute`, unparseable document.

7. **Rejected uploads leaked onto disk.** Cleanup lived in the controller's
   `finally`, but a multer or magic-byte rejection short-circuits to the error
   handler and the controller never runs. Now registered on
   `res.once('close')`, which fires on success, error and client abort alike.

8. **`quantize_colors: null` could not turn quantization off.** `apply_overrides`
   skipped any `None` value as "not supplied", so the option silently did
   nothing. Explicit nulls are now distinguished from absent keys.

### Measuring quality changes

Do not tune this by eye. Regenerate the fixtures and score the output:

```bat
cd python-engine
venv\Scripts\activate
python tests\make_samples.py
python tests\make_logo_repro.py
python tests\quality_harness.py
```

Then open <http://127.0.0.1:5000/_quality.html>. It rasterizes each generated
SVG back to a canvas and scores it against the lossless reference:

* **RMSE** — overall pixel error.
* **EDGE RMSE** — error restricted to reference edge pixels, which is where
  jaggedness, bumps and rounded corners live. This is the number to watch.

Edit `DEFAULT_RUNS` to compare presets or override combinations. The generated
page is gitignored; delete it when you are done.

One caveat learned the hard way: compare at the reference's **native**
resolution. Drawing into a rounded-off box makes the SVG
(`preserveAspectRatio="xMidYMid meet"`) letterbox by a fraction of a pixel, and
that constant offset swamps the real tracing error.

### Where the remaining error is

At RMSE 9.4 the residual is mostly colour quantization — flat regions snapped
to the nearest of ~8 clusters — plus genuine detail loss in the smallest
illustration features. `detailed` scores better (6.95) but at 207 paths and
139 KB versus 70 paths and 73 KB. That trade is deliberate: for a logo you
want few, editable paths.

Verified environment: Windows 10 Pro · Node 22.16.0 · npm 11.4.1 ·
Python 3.12 · vtracer 0.6.15 · opencv-python-headless 5.0.0 · numpy 2.5.3 ·
scour 0.38.2 · Express 5.2.1 · Multer 2.3.0.
