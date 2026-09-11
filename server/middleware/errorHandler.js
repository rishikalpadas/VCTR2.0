'use strict';

const ApiError = require('../utils/ApiError');
const logger = require('../utils/logger');

function notFound(req, res) {
  res.status(404).json({
    success: false,
    error: { code: 'NOT_FOUND', message: `No route matches ${req.method} ${req.originalUrl}` },
  });
}

/**
 * Terminal error handler.
 *
 * Two audiences, deliberately separated:
 *   - the log gets the stack trace and any Python-side detail;
 *   - the client gets a code and a sentence it can act on, never a trace.
 */
// eslint-disable-next-line no-unused-vars -- Express identifies this by arity.
function errorHandler(err, req, res, next) {
  const apiError = err instanceof ApiError
    ? err
    : ApiError.internal(
      'Something went wrong while processing the image.',
      err && err.stack ? err.stack : String(err),
    );

  const logLine = `${req.method} ${req.originalUrl} -> ${apiError.status} ${apiError.code}: ${apiError.message}`;
  if (apiError.status >= 500) {
    logger.error(logLine, apiError.detail || '');
  } else {
    logger.warn(logLine, apiError.detail || '');
  }

  if (res.headersSent) {
    return res.end();
  }
  return res.status(apiError.status).json(apiError.toJSON());
}

module.exports = { notFound, errorHandler };
