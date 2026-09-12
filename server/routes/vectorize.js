'use strict';

const express = require('express');

const config = require('../config');
const controller = require('../controllers/vectorizeController');
const upload = require('../middleware/upload');

const router = express.Router();

router.get('/health', controller.health);
router.get('/presets', controller.presets);

router.post(
  '/vectorize',
  upload.single('image'),      // multipart -> temp file, size/type gate
  upload.verifyMagicBytes,     // confirm the bytes match the claimed type
  controller.vectorize,
);

// The SVG arrives as a raw text body. Parsed here rather than globally so the
// generous size limit applies only to this route.
router.post(
  '/export/pdf',
  express.text({
    type: ['image/svg+xml', 'text/plain', 'application/xml'],
    limit: config.export.maxSvgBytes,
  }),
  controller.exportPdf,
);

module.exports = router;
