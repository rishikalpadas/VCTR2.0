'use strict';

const crypto = require('crypto');
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');

const config = require('../config');
const pythonService = require('../services/pythonService');
const ApiError = require('../utils/ApiError');
const logger = require('../utils/logger');

const VALID_PRESETS = new Set([
  'auto',
  'standard',
  'logo',
  'flat_art',
  'typography',
  'line_art',
  'detailed',
]);

const BOOLEAN_OPTIONS = ['remove_enclosed_background'];
const NUMBER_OPTIONS = [
  'max_dimension', 'background_tolerance', 'filter_speckle', 'color_precision',
  'turdsize',
];
const FLOAT_OPTIONS = ['supersample', 'boundary_smooth_sigma', 'alphamax', 'opttolerance'];
const STRING_OPTIONS = ['background', 'turnpolicy'];
const VALID_BACKGROUND_MODES = new Set(['auto', 'always', 'never']);
const VALID_TURN_POLICIES = new Set(['black', 'white', 'left', 'right', 'minority', 'majority', 'random']);

// Forces a specific tracing backend onto whatever preset was chosen, holding
// every other preprocessing option equal - used by the engine-comparison
// test page. Validated the same way `preset` is: reject rather than forward
// anything not on the list.
const VALID_ENGINES = new Set(['vtracer', 'potrace']);

/**
 * Build the override object from form fields.
 *
 * Allow-list only. Unknown fields are dropped rather than forwarded, so the
 * browser cannot reach engine parameters we have not thought about.
 */
function parseOptions(body) {
  const options = {};

  for (const key of BOOLEAN_OPTIONS) {
    if (body[key] !== undefined) {
      options[key] = body[key] === 'true' || body[key] === true;
    }
  }

  for (const key of NUMBER_OPTIONS) {
    if (body[key] === undefined || body[key] === '') continue;
    const value = Number.parseInt(body[key], 10);
    if (Number.isFinite(value)) options[key] = value;
  }

  for (const key of FLOAT_OPTIONS) {
    if (body[key] === undefined || body[key] === '') continue;
    const value = Number.parseFloat(body[key]);
    if (Number.isFinite(value)) options[key] = value;
  }

  for (const key of STRING_OPTIONS) {
    if (body[key] === undefined || body[key] === '') continue;
    const value = String(body[key]);
    if (key === 'background' && !VALID_BACKGROUND_MODES.has(value)) continue;
    if (key === 'turnpolicy' && !VALID_TURN_POLICIES.has(value)) continue;
    options[key] = value;
  }

  if (body.engine !== undefined && body.engine !== '') {
    const engine = String(body.engine);
    if (VALID_ENGINES.has(engine)) options.engine = engine;
  }

  return options;
}

async function persistOutput(svg, preset) {
  fs.mkdirSync(config.upload.outputDir, { recursive: true });
  const name = `${Date.now()}-${crypto.randomBytes(6).toString('hex')}-${preset}.svg`;
  const target = path.join(config.upload.outputDir, name);
  await fsp.writeFile(target, svg, 'utf8');
  return name;
}

/** POST /api/vectorize
 *
 * Temp-file cleanup is NOT done here. It is registered against the response
 * lifecycle in middleware/upload.js, because errors from multer or the
 * magic-byte check never reach this function.
 */
async function vectorize(req, res, next) {
  try {
    if (!req.file) {
      throw ApiError.badRequest(
        'NO_FILE',
        'No image was uploaded. Choose a PNG, JPG/JPEG or WebP file.',
      );
    }

    const preset = String(req.body.preset || 'auto');
    if (!VALID_PRESETS.has(preset)) {
      throw ApiError.badRequest(
        'UNKNOWN_PRESET',
        `Unknown preset "${preset}". Valid options: ${[...VALID_PRESETS].join(', ')}.`,
      );
    }

    const options = parseOptions(req.body);

    const { svg, meta } = await pythonService.vectorize({
      filePath: req.file.path,
      detectedType: req.file.detectedType,
      preset,
      options,
    });

    const payload = {
      success: true,
      svg,
      meta: {
        ...meta,
        original_filename: req.file.originalname,
        original_bytes: req.file.size,
      },
    };

    if (config.upload.persistOutputs) {
      payload.meta.saved_as = await persistOutput(svg, meta.preset_used || preset);
    }

    res.json(payload);
  } catch (err) {
    next(err);
  }
}

/** POST /api/export/pdf - body is the SVG text, response is the PDF. */
async function exportPdf(req, res, next) {
  try {
    const svg = typeof req.body === 'string' ? req.body : '';

    if (!svg.trim()) {
      throw ApiError.badRequest(
        'NO_SVG',
        'No SVG was supplied. Vectorize an image first, then export it.',
      );
    }
    if (Buffer.byteLength(svg, 'utf8') > config.export.maxSvgBytes) {
      throw ApiError.payloadTooLarge(
        `That SVG is too large to export. The limit is `
        + `${Math.round(config.export.maxSvgBytes / (1024 * 1024))} MB.`,
      );
    }
    if (!svg.includes('<svg')) {
      throw ApiError.badRequest('INVALID_SVG', 'That does not look like an SVG document.');
    }

    const pdf = await pythonService.exportPdf(svg);
    const name = sanitizeFilename(req.query.name) || 'artwork';

    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Length', pdf.length);
    res.setHeader('Content-Disposition', `attachment; filename="${name}.pdf"`);
    res.send(pdf);
  } catch (err) {
    next(err);
  }
}

/**
 * Filenames reach us from the browser, and this one goes straight into a
 * Content-Disposition header. Strip anything that could break out of the
 * quoted string or inject a second header.
 */
function sanitizeFilename(value) {
  if (typeof value !== 'string') return '';
  return value
    .replace(/\.[^.]+$/, '')
    // Spaces become hyphens so the PDF name matches the SVG one the frontend
    // builds, rather than the two downloads differing for the same artwork.
    .replace(/[^a-zA-Z0-9_-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 64);
}

/** GET /api/health */
async function health(req, res) {
  const payload = {
    status: 'ok',
    service: 'image-to-vector-api',
    uptime_s: Math.round(process.uptime()),
  };

  try {
    payload.engine = await pythonService.checkHealth();
  } catch (err) {
    // The Node API itself is fine; report the engine as down rather than 500.
    payload.status = 'degraded';
    payload.engine = {
      status: 'unavailable',
      message: err.message,
    };
    logger.warn('Engine health check failed', err.detail || err.message);
  }

  res.status(200).json(payload);
}

/** GET /api/presets - proxied so the frontend has one source of truth. */
async function presets(req, res, next) {
  try {
    res.json(await pythonService.listPresets());
  } catch (err) {
    next(err);
  }
}

module.exports = { vectorize, exportPdf, health, presets };
