'use strict';

const fsp = require('fs/promises');
const path = require('path');

const config = require('../config');
const ApiError = require('../utils/ApiError');
const logger = require('../utils/logger');

/**
 * Client for the Python vectorization service.
 *
 * This is the ONLY module in the Node app that knows the engine exists. Node
 * does no image processing of its own - it validates, forwards, and translates
 * engine errors into HTTP. That separation is what lets the Python service move
 * to a different (bigger) host later without touching the MERN app.
 *
 * Uses the built-in fetch/FormData/Blob from Node 18+, so no HTTP client
 * dependency is required.
 */

const MIME_BY_TYPE = {
  png: 'image/png',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
};

/** fetch with an AbortController timeout, mapped onto ApiError. */
async function fetchWithTimeout(url, options, timeoutMs, context) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw ApiError.gatewayTimeout(
        `${context} timed out after ${Math.round(timeoutMs / 1000)}s. `
        + 'Try a smaller image or a lighter preset.',
        `timeout calling ${url}`,
      );
    }
    // ECONNREFUSED / ENOTFOUND / socket hang up all land here.
    throw ApiError.serviceUnavailable(
      'The vectorization engine is not reachable. Make sure the Python service is running.',
      `${err.name}: ${err.message} (${url})`,
    );
  } finally {
    clearTimeout(timer);
  }
}

/** Parse a JSON body defensively - the engine may die mid-response. */
async function readJson(response, context) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    throw ApiError.internal(
      'The vectorization engine returned an unreadable response.',
      `${context}: non-JSON body (${response.status}): ${text.slice(0, 500)}`,
    );
  }
}

/**
 * Translate a Python error payload into an ApiError.
 * Python owns the code and the user-facing message; we only pick the status.
 */
function toApiError(status, body) {
  const error = (body && body.error) || {};
  const code = error.code || 'ENGINE_FAILED';
  const message = error.message || 'Vectorization failed.';
  // Anything 5xx from the engine is our problem, not the caller's.
  const httpStatus = status >= 400 && status < 600 ? status : 500;
  return new ApiError(httpStatus, code, message, `python responded ${status}`);
}

async function checkHealth() {
  const url = `${config.python.baseUrl}/health`;
  const response = await fetchWithTimeout(
    url,
    { method: 'GET' },
    config.python.healthTimeoutMs,
    'Engine health check',
  );

  if (!response.ok) {
    throw ApiError.serviceUnavailable(
      'The vectorization engine reported an unhealthy state.',
      `health check returned ${response.status}`,
    );
  }
  return readJson(response, 'health');
}

async function listPresets() {
  const url = `${config.python.baseUrl}/presets`;
  const response = await fetchWithTimeout(
    url,
    { method: 'GET' },
    config.python.healthTimeoutMs,
    'Preset lookup',
  );
  if (!response.ok) {
    throw toApiError(response.status, await readJson(response, 'presets'));
  }
  return readJson(response, 'presets');
}

/**
 * Send one image to the engine.
 *
 * @param {object} params
 * @param {string} params.filePath   Absolute path to the temp upload.
 * @param {string} params.detectedType  'png' | 'jpeg' | 'webp' (sniffed, not claimed).
 * @param {string} params.preset
 * @param {object} [params.options]  Narrow set of per-request overrides.
 * @returns {Promise<{svg: string, meta: object}>}
 */
async function vectorize({ filePath, detectedType, preset, options }) {
  const buffer = await fsp.readFile(filePath);

  const form = new FormData();
  form.append(
    'file',
    new Blob([buffer], { type: MIME_BY_TYPE[detectedType] || 'application/octet-stream' }),
    // A neutral name: the engine sniffs the bytes and ignores this anyway.
    `upload.${detectedType}`,
  );
  form.append('preset', preset);
  if (options && Object.keys(options).length > 0) {
    form.append('options', JSON.stringify(options));
  }

  const url = `${config.python.baseUrl}/vectorize`;
  const startedAt = Date.now();
  logger.info(
    `-> engine ${path.basename(filePath)} (${detectedType}, ${(buffer.length / 1024).toFixed(1)} KB) preset=${preset}`,
  );

  const response = await fetchWithTimeout(
    url,
    { method: 'POST', body: form },
    config.python.timeoutMs,
    'Vectorization',
  );

  const body = await readJson(response, 'vectorize');

  if (!response.ok || body.success === false) {
    throw toApiError(response.status, body);
  }
  if (typeof body.svg !== 'string' || !body.svg.includes('<svg')) {
    throw ApiError.internal(
      'The vectorization engine returned an invalid SVG.',
      `missing or malformed svg field (${response.status})`,
    );
  }

  logger.info(
    `<- engine ok in ${Date.now() - startedAt}ms `
    + `(${body.meta?.path_count ?? '?'} paths, ${((body.meta?.svg_bytes ?? 0) / 1024).toFixed(1)} KB)`,
  );

  return { svg: body.svg, meta: body.meta || {} };
}

module.exports = { checkHealth, listPresets, vectorize };
