'use strict';

const express = require('express');

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

module.exports = router;
