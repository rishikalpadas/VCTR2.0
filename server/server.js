'use strict';

const express = require('express');

const config = require('./config');
const vectorizeRoutes = require('./routes/vectorize');
const { notFound, errorHandler } = require('./middleware/errorHandler');
const logger = require('./utils/logger');

const app = express();

// JSON bodies are small here (the image arrives as multipart), so keep the
// limit tight.
app.use(express.json({ limit: '256kb' }));
app.use(express.urlencoded({ extended: false, limit: '256kb' }));

app.use((req, res, next) => {
  const startedAt = Date.now();
  res.on('finish', () => {
    logger.info(`${req.method} ${req.originalUrl} ${res.statusCode} ${Date.now() - startedAt}ms`);
  });
  next();
});

// The POC frontend is served from the same origin as the API, so there is no
// CORS setup to get wrong.
app.use(express.static(config.frontendDir, { extensions: ['html'] }));

app.use('/api', vectorizeRoutes);

app.use(notFound);
app.use(errorHandler);

// Fail loudly instead of dying silently mid-request.
process.on('unhandledRejection', (reason) => {
  logger.error('Unhandled promise rejection', reason);
});
process.on('uncaughtException', (err) => {
  logger.error('Uncaught exception', err.stack || err.message);
  process.exit(1);
});

if (require.main === module) {
  const server = app.listen(config.port, config.host, () => {
    logger.info('-----------------------------------------------------------');
    logger.info(`  Image-to-Vector API  http://${config.host}:${config.port}`);
    logger.info(`  Frontend             http://${config.host}:${config.port}/`);
    logger.info(`  Python engine        ${config.python.baseUrl}`);
    logger.info('-----------------------------------------------------------');
  });

  const shutdown = (signal) => {
    logger.info(`${signal} received, shutting down`);
    server.close(() => process.exit(0));
    // Do not hang forever on a stuck connection.
    setTimeout(() => process.exit(1), 5000).unref();
  };
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
}

module.exports = app;
