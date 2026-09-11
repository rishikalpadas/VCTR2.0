'use strict';

/**
 * Minimal structured logger.
 *
 * Detail (stack traces, Python diagnostics) goes here and only here. What the
 * browser receives is built separately in the error handler.
 */

const LEVELS = { error: 0, warn: 1, info: 2, debug: 3 };
const activeLevel = LEVELS[process.env.LOG_LEVEL] ?? LEVELS.info;

const timestamp = () => new Date().toISOString().slice(11, 23);

function emit(level, stream, message, meta) {
  if (LEVELS[level] > activeLevel) return;
  const line = `${timestamp()} ${level.toUpperCase().padEnd(5)} ${message}`;
  if (meta !== undefined) {
    stream(line, meta);
  } else {
    stream(line);
  }
}

const logger = {
  error: (message, meta) => emit('error', console.error, message, meta),
  warn: (message, meta) => emit('warn', console.warn, message, meta),
  info: (message, meta) => emit('info', console.log, message, meta),
  debug: (message, meta) => emit('debug', console.log, message, meta),
};

module.exports = logger;
