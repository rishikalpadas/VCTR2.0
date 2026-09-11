'use strict';

/**
 * API tests for the Node layer.
 *
 * Uses the built-in node:test runner - no test framework dependency.
 *
 *   npm test
 *
 * Tests that need the Python engine are skipped automatically when it is not
 * running, so `npm test` is always meaningful on its own.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const app = require('../server');
const config = require('../config');

const SAMPLES = path.join(__dirname, '..', '..', 'samples');

let baseUrl;
let server;
let engineUp = false;

test.before(async () => {
  await new Promise((resolve) => {
    server = app.listen(0, '127.0.0.1', resolve);
  });
  baseUrl = `http://127.0.0.1:${server.address().port}`;

  try {
    const res = await fetch(`${config.python.baseUrl}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    engineUp = res.ok;
  } catch {
    engineUp = false;
  }

  if (!engineUp) {
    console.warn(
      '\n  ! Python engine not reachable at '
      + `${config.python.baseUrl} - engine-dependent tests will be skipped.\n`,
    );
  }
});

test.after(async () => {
  await new Promise((resolve) => server.close(resolve));
});

/** Minimal valid 1x1 PNG, used to prove format gates without touching disk. */
const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

function formWith(buffer, filename, type, fields = {}) {
  const form = new FormData();
  form.append('image', new Blob([buffer], { type }), filename);
  for (const [key, value] of Object.entries(fields)) form.append(key, value);
  return form;
}

function readSample(name) {
  const file = path.join(SAMPLES, name);
  return fs.existsSync(file) ? fs.readFileSync(file) : null;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

test('GET /api/health returns ok even when the engine is down', async () => {
  const res = await fetch(`${baseUrl}/api/health`);
  assert.equal(res.status, 200);

  const body = await res.json();
  assert.ok(['ok', 'degraded'].includes(body.status));
  assert.equal(body.service, 'image-to-vector-api');
  assert.ok(body.engine, 'health should report engine state');
});

test('GET / serves the frontend', async () => {
  const res = await fetch(`${baseUrl}/`);
  assert.equal(res.status, 200);
  const html = await res.text();
  assert.match(html, /Image.*Vector/i);
});

test('unknown routes return a structured 404', async () => {
  const res = await fetch(`${baseUrl}/api/nope`);
  assert.equal(res.status, 404);
  const body = await res.json();
  assert.equal(body.success, false);
  assert.equal(body.error.code, 'NOT_FOUND');
});

// ---------------------------------------------------------------------------
// Upload validation (no engine required)
// ---------------------------------------------------------------------------

test('POST /api/vectorize rejects a request with no file', async () => {
  const form = new FormData();
  form.append('preset', 'standard');

  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });
  assert.equal(res.status, 400);

  const body = await res.json();
  assert.equal(body.success, false);
  assert.equal(body.error.code, 'NO_FILE');
});

test('POST /api/vectorize rejects an unsupported mime type', async () => {
  const form = formWith(Buffer.from('hello'), 'notes.txt', 'text/plain');

  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });
  assert.equal(res.status, 415);

  const body = await res.json();
  assert.equal(body.error.code, 'UNSUPPORTED_FORMAT');
});

test('POST /api/vectorize rejects a file whose bytes do not match its mime type', async () => {
  // Claims to be a PNG, is actually text. The magic-byte check must catch it.
  const form = formWith(Buffer.from('definitely not a png'), 'fake.png', 'image/png');

  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });
  assert.equal(res.status, 415);

  const body = await res.json();
  assert.equal(body.error.code, 'UNSUPPORTED_FORMAT');
});

test('POST /api/vectorize rejects an unknown preset', async () => {
  const form = formWith(TINY_PNG, 'tiny.png', 'image/png', { preset: 'nonsense' });

  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });
  assert.equal(res.status, 400);

  const body = await res.json();
  assert.equal(body.error.code, 'UNKNOWN_PRESET');
});

test('POST /api/vectorize rejects a file over the size limit', async () => {
  // Valid PNG header followed by padding past the limit.
  const oversized = Buffer.concat([
    TINY_PNG,
    Buffer.alloc(config.upload.maxBytes + 1024, 0),
  ]);
  const form = formWith(oversized, 'huge.png', 'image/png');

  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });
  assert.equal(res.status, 413);

  const body = await res.json();
  assert.equal(body.error.code, 'FILE_TOO_LARGE');
});

/**
 * Cleanup is registered on the response lifecycle, which fires just after the
 * body is sent - so poll briefly rather than asserting on the same tick.
 */
async function waitForEmptyUploads(timeoutMs = 2000) {
  const deadline = Date.now() + timeoutMs;
  let leftovers = [];
  while (Date.now() < deadline) {
    leftovers = fs.readdirSync(config.upload.tempDir).filter((n) => n !== '.gitkeep');
    if (leftovers.length === 0) return leftovers;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return leftovers;
}

test('temp uploads are cleaned up when the preset is rejected', async () => {
  const form = formWith(TINY_PNG, 'tiny.png', 'image/png', { preset: 'nonsense' });
  await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });

  const leftovers = await waitForEmptyUploads();
  assert.deepEqual(leftovers, [], 'upload temp dir should be empty');
});

/**
 * Regression: cleanup used to live in the controller's `finally`, so a
 * rejection from the magic-byte middleware short-circuited to the error
 * handler and orphaned the file that multer had already written to disk.
 */
test('temp uploads are cleaned up when the magic-byte check rejects the file', async () => {
  const form = formWith(Buffer.from('definitely not a png'), 'fake.png', 'image/png');
  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });
  assert.equal(res.status, 415);

  const leftovers = await waitForEmptyUploads();
  assert.deepEqual(leftovers, [], 'a rejected upload must not be left on disk');
});

test('temp uploads are cleaned up after a successful request', async (t) => {
  if (!engineUp) return t.skip('Python engine not running');

  const sample = readSample('sample_flat_art.png');
  if (!sample) return t.skip('sample images not generated');

  const form = formWith(sample, 'sample.png', 'image/png', { preset: 'flat_art' });
  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });
  assert.equal(res.status, 200);

  const leftovers = await waitForEmptyUploads();
  assert.deepEqual(leftovers, [], 'a successful upload must not be left on disk');
  return undefined;
});

// ---------------------------------------------------------------------------
// Full round trip (engine required)
// ---------------------------------------------------------------------------

test('POST /api/vectorize returns real vector paths', async (t) => {
  if (!engineUp) return t.skip('Python engine not running');

  const sample = readSample('sample_flat_art.png');
  if (!sample) {
    return t.skip('run `python tests/make_samples.py` in python-engine first');
  }

  const form = formWith(sample, 'sample.png', 'image/png', { preset: 'flat_art' });
  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });

  assert.equal(res.status, 200);
  const body = await res.json();

  assert.equal(body.success, true);
  assert.match(body.svg, /^<\?xml|^<svg/, 'should be an SVG document');
  assert.ok(body.svg.includes('<path'), 'SVG must contain path geometry');

  // The whole point: no raster smuggled inside the "vector".
  assert.ok(!body.svg.includes('<image'), 'SVG must not embed a raster image');
  assert.ok(!body.svg.includes('data:image'), 'SVG must not embed a data URI');

  assert.ok(body.meta.path_count > 0, 'meta should report paths');
  assert.match(body.svg, /viewBox="0 0 \d+ \d+"/, 'SVG must carry a viewBox');
  return undefined;
});

test('the auto preset resolves to a concrete preset', async (t) => {
  if (!engineUp) return t.skip('Python engine not running');

  const sample = readSample('sample_line_art.png');
  if (!sample) return t.skip('sample images not generated');

  const form = formWith(sample, 'sample.png', 'image/png', { preset: 'auto' });
  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });

  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.meta.preset_requested, 'auto');
  assert.notEqual(body.meta.preset_used, 'auto');
  return undefined;
});

test('corrupt image data produces a clean 400, not a stack trace', async (t) => {
  if (!engineUp) return t.skip('Python engine not running');

  // Real PNG magic bytes, garbage payload: passes the Node sniff, fails decode.
  const corrupt = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    Buffer.from('not actually image data at all'),
  ]);
  const form = formWith(corrupt, 'broken.png', 'image/png');

  const res = await fetch(`${baseUrl}/api/vectorize`, { method: 'POST', body: form });
  assert.equal(res.status, 400);

  const body = await res.json();
  assert.equal(body.error.code, 'CORRUPT_IMAGE');
  const serialized = JSON.stringify(body);
  assert.ok(!serialized.includes('Traceback'), 'must not leak a Python traceback');
  assert.ok(!serialized.includes('.py'), 'must not leak Python file paths');
  return undefined;
});
