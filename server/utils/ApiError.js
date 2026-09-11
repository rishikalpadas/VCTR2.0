'use strict';

/**
 * Error carrying an HTTP status and a stable machine-readable code.
 *
 * `message` is user-facing and safe to display. `detail` is for the server log
 * only and is never serialized into a response.
 */
class ApiError extends Error {
  constructor(status, code, message, detail) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.detail = detail;
  }

  static badRequest(code, message, detail) {
    return new ApiError(400, code, message, detail);
  }

  static unsupportedMedia(message, detail) {
    return new ApiError(415, 'UNSUPPORTED_FORMAT', message, detail);
  }

  static payloadTooLarge(message, detail) {
    return new ApiError(413, 'FILE_TOO_LARGE', message, detail);
  }

  static serviceUnavailable(message, detail) {
    return new ApiError(503, 'ENGINE_UNAVAILABLE', message, detail);
  }

  static gatewayTimeout(message, detail) {
    return new ApiError(504, 'ENGINE_TIMEOUT', message, detail);
  }

  static internal(message, detail) {
    return new ApiError(500, 'INTERNAL_ERROR', message, detail);
  }

  toJSON() {
    return { success: false, error: { code: this.code, message: this.message } };
  }
}

module.exports = ApiError;
