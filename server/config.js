'use strict';

const path = require('path');
require('dotenv').config();

const toInt = (value, fallback) => {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const config = {
  port: toInt(process.env.PORT, 5000),
  host: process.env.HOST || '127.0.0.1',
  nodeEnv: process.env.NODE_ENV || 'development',

  // The Python vectorization service.
  python: {
    baseUrl: process.env.PYTHON_SERVICE_URL || 'http://127.0.0.1:8000',
    // Tracing a large detailed image is genuinely slow on CPU. This must be
    // comfortably above the worst realistic case or valid work gets killed.
    timeoutMs: toInt(process.env.PYTHON_TIMEOUT_MS, 120000),
    healthTimeoutMs: toInt(process.env.PYTHON_HEALTH_TIMEOUT_MS, 4000),
    // Format conversion is pure geometry - no tracing - so it is fast.
    exportTimeoutMs: toInt(process.env.PYTHON_EXPORT_TIMEOUT_MS, 30000),
  },

  upload: {
    maxBytes: toInt(process.env.MAX_UPLOAD_BYTES, 15 * 1024 * 1024),
    // Enforced against the sniffed magic bytes, not the client-supplied name.
    allowedMimeTypes: ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'],
    allowedExtensions: ['.png', '.jpg', '.jpeg', '.webp'],
    tempDir: path.join(__dirname, 'uploads'),
    outputDir: path.join(__dirname, 'outputs'),
    // Keep a copy of each generated SVG on disk. Off by default: the POC
    // returns the SVG inline and has no reason to retain user uploads.
    persistOutputs: process.env.PERSIST_OUTPUTS === 'true',
  },

  export: {
    // Generated SVGs run 50-150 KB; this leaves generous headroom without
    // letting a client post something that could exhaust the renderer.
    maxSvgBytes: toInt(process.env.MAX_EXPORT_SVG_BYTES, 8 * 1024 * 1024),
  },

  frontendDir: path.join(__dirname, '..', 'frontend'),
};

module.exports = config;
