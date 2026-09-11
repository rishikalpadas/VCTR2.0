'use strict';

const crypto = require('crypto');
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const multer = require('multer');

const config = require('../config');
const ApiError = require('../utils/ApiError');
const logger = require('../utils/logger');

fs.mkdirSync(config.upload.tempDir, { recursive: true });

/**
 * Filenames from a browser are attacker-controlled. We never reuse one:
 * the stored name is random and the extension is derived from the
 * *sniffed* type, so `evil.php.png` and `../../etc/passwd` are both inert.
 */
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, config.upload.tempDir),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase();
    const safeExt = config.upload.allowedExtensions.includes(ext) ? ext : '.bin';
    cb(null, `${Date.now()}-${crypto.randomBytes(12).toString('hex')}${safeExt}`);
  },
});

function fileFilter(req, file, cb) {
  const mimetype = (file.mimetype || '').toLowerCase();
  if (!config.upload.allowedMimeTypes.includes(mimetype)) {
    cb(
      ApiError.unsupportedMedia(
        'Unsupported file type. Upload a PNG, JPG/JPEG or WebP image.',
        `rejected mimetype=${mimetype} name=${file.originalname}`,
      ),
    );
    return;
  }
  cb(null, true);
}

const multerUpload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: config.upload.maxBytes,
    files: 1,
    fields: 8,
  },
});

/**
 * Register temp-file cleanup against the response lifecycle.
 *
 * Doing this in the controller alone is not enough: a rejection from multer or
 * from the magic-byte check short-circuits straight to the error handler and
 * the controller never runs, orphaning the file that was already written.
 * `close` fires on every exit - success, error, and client abort - so this is
 * the one hook that cannot be bypassed.
 */
function scheduleCleanup(req, res) {
  if (!req.file || !req.file.path || req.file.cleanupScheduled) return;
  req.file.cleanupScheduled = true;
  res.once('close', () => { cleanupFile(req.file.path); });
}

/** Wrap multer so its own errors become ApiError instances. */
const single = (fieldName) => (req, res, next) => {
  multerUpload.single(fieldName)(req, res, (err) => {
    // Register before any branch below: the file may already exist on disk
    // even when multer reports an error.
    scheduleCleanup(req, res);

    if (!err) return next();

    if (err instanceof ApiError) return next(err);

    if (err instanceof multer.MulterError) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        const limitMb = Math.round(config.upload.maxBytes / (1024 * 1024));
        return next(
          ApiError.payloadTooLarge(
            `That file is too large. The limit is ${limitMb} MB.`,
            err.message,
          ),
        );
      }
      if (err.code === 'LIMIT_UNEXPECTED_FILE') {
        return next(
          ApiError.badRequest(
            'UNEXPECTED_FIELD',
            `Unexpected upload field. Send the image in the "${fieldName}" field.`,
            err.message,
          ),
        );
      }
      return next(
        ApiError.badRequest('UPLOAD_FAILED', 'The upload could not be processed.', err.message),
      );
    }

    return next(ApiError.internal('The upload could not be processed.', err.message));
  });
};

/**
 * Verify the file really is what it claims by reading its leading bytes.
 * The Content-Type header is trivially spoofed; the magic number is not.
 */
const MAGIC = [
  { ext: 'png', bytes: [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], offset: 0 },
  { ext: 'jpeg', bytes: [0xff, 0xd8, 0xff], offset: 0 },
];

async function sniffImageType(filePath) {
  const handle = await fsp.open(filePath, 'r');
  try {
    const buffer = Buffer.alloc(16);
    const { bytesRead } = await handle.read(buffer, 0, 16, 0);
    if (bytesRead < 12) return null;

    for (const signature of MAGIC) {
      const matches = signature.bytes.every(
        (byte, index) => buffer[signature.offset + index] === byte,
      );
      if (matches) return signature.ext;
    }

    // WebP: "RIFF" ???? "WEBP"
    if (buffer.subarray(0, 4).toString('ascii') === 'RIFF'
      && buffer.subarray(8, 12).toString('ascii') === 'WEBP') {
      return 'webp';
    }
    return null;
  } finally {
    await handle.close();
  }
}

async function verifyMagicBytes(req, res, next) {
  if (!req.file) return next();
  try {
    const detected = await sniffImageType(req.file.path);
    if (!detected) {
      return next(
        ApiError.unsupportedMedia(
          'That file is not a valid PNG, JPG/JPEG or WebP image.',
          `magic-byte sniff failed for ${req.file.path}`,
        ),
      );
    }
    req.file.detectedType = detected;
    return next();
  } catch (err) {
    return next(ApiError.internal('The uploaded file could not be read.', err.message));
  }
}

/** Best-effort temp cleanup. Never let it fail a request. */
async function cleanupFile(filePath) {
  if (!filePath) return;
  try {
    await fsp.unlink(filePath);
    logger.debug(`Removed temp upload ${path.basename(filePath)}`);
  } catch (err) {
    if (err.code !== 'ENOENT') {
      logger.warn(`Could not remove temp upload ${filePath}`, err.message);
    }
  }
}

module.exports = {
  single,
  verifyMagicBytes,
  cleanupFile,
  scheduleCleanup,
  sniffImageType,
};
