/* ==========================================================================
   Image to Vector - frontend
   Vanilla JS, no framework, no build step. Talks only to the Node API.
   ========================================================================== */

(function () {
  'use strict';

  var API = {
    health: '/api/health',
    vectorize: '/api/vectorize',
  };

  var MAX_BYTES = 15 * 1024 * 1024;
  var ACCEPTED = ['image/png', 'image/jpeg', 'image/webp'];

  var PRESET_HELP = {
    auto: 'Inspects the image and picks a preset for you.',
    standard: 'Balanced colour tracing. Safe default for most clean artwork.',
    logo: 'Badges and emblems: flat colours with crisp lettering, thin rules and a small illustration.',
    flat_art: 'Quantizes colours first - fewer, cleaner paths for flat graphics and stickers.',
    typography: 'Sharp corner handling for lettering and logos. No blurring at all.',
    line_art: 'Binarizes to 1-bit and traces strokes. For doodles, outlines and icons.',
    detailed: 'Keeps fine detail in dense or textured artwork. Slower, much larger file.',
  };

  // --- element handles ----------------------------------------------------

  var el = {};
  ['dropZone', 'browseBtn', 'fileInput', 'fileMeta', 'thumbPreview', 'fileName',
    'fileSize', 'fileDims', 'fileType', 'clearBtn', 'presetSelect', 'presetHelp',
    'backgroundSelect', 'enclosedCheck', 'maxDimInput', 'supersampleInput', 'smoothInput',
    'vectorizeBtn', 'progress',
    'progressDetail', 'errorBox', 'errorTitle', 'errorMessage', 'errorCode',
    'warningBox', 'warningList', 'resultPanel', 'originalPreview', 'svgPreview',
    'statsGrid', 'downloadBtn', 'openBtn', 'copyBtn', 'metaDump', 'engineStatus',
  ].forEach(function (id) { el[id] = document.getElementById(id); });

  // --- state --------------------------------------------------------------

  var state = {
    file: null,
    objectUrl: null,   // preview of the uploaded raster
    svgUrl: null,      // blob URL of the generated SVG
    svgText: null,
    meta: null,
    busy: false,
  };

  // --- helpers ------------------------------------------------------------

  function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return '—';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  function formatMs(ms) {
    if (ms == null) return '—';
    return ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(2) + ' s';
  }

  function revoke(url) {
    if (url) { try { URL.revokeObjectURL(url); } catch (e) { /* noop */ } }
  }

  function show(node) { if (node) node.hidden = false; }
  function hide(node) { if (node) node.hidden = true; }

  function showError(message, code) {
    el.errorMessage.textContent = message;
    el.errorCode.textContent = code ? 'code: ' + code : '';
    show(el.errorBox);
    el.errorBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function clearError() {
    hide(el.errorBox);
    el.errorCode.textContent = '';
  }

  // --- engine status ------------------------------------------------------

  function refreshEngineStatus() {
    fetch(API.health)
      .then(function (res) { return res.json(); })
      .then(function (data) {
        var engineUp = data.engine && data.engine.status === 'ok';
        el.engineStatus.className = 'badge ' + (engineUp ? 'badge--ok' : 'badge--down');
        el.engineStatus.textContent = engineUp
          ? 'engine ready · ' + (data.engine.version || '')
          : 'engine offline — start the Python service';
      })
      .catch(function () {
        el.engineStatus.className = 'badge badge--down';
        el.engineStatus.textContent = 'API unreachable';
      });
  }

  // --- file selection -----------------------------------------------------

  function acceptFile(file) {
    clearError();

    if (!file) return;

    if (ACCEPTED.indexOf(file.type) === -1) {
      showError(
        'Unsupported file type "' + (file.type || 'unknown')
        + '". Choose a PNG, JPG/JPEG or WebP image.',
        'UNSUPPORTED_FORMAT',
      );
      return;
    }

    if (file.size > MAX_BYTES) {
      showError(
        'That file is ' + formatBytes(file.size) + '. The limit is '
        + formatBytes(MAX_BYTES) + '.',
        'FILE_TOO_LARGE',
      );
      return;
    }

    revoke(state.objectUrl);
    state.file = file;
    state.objectUrl = URL.createObjectURL(file);

    el.thumbPreview.src = state.objectUrl;
    el.originalPreview.src = state.objectUrl;
    el.fileName.textContent = file.name;
    el.fileName.title = file.name;
    el.fileSize.textContent = formatBytes(file.size);
    el.fileType.textContent = file.type.replace('image/', '').toUpperCase();
    el.fileDims.textContent = 'reading…';

    // Read the real pixel dimensions rather than trusting anything.
    var probe = new Image();
    probe.onload = function () {
      el.fileDims.textContent = probe.naturalWidth + ' × ' + probe.naturalHeight;
    };
    probe.onerror = function () {
      el.fileDims.textContent = 'unreadable';
      showError('That image could not be decoded by the browser. It may be corrupted.',
        'CORRUPT_IMAGE');
    };
    probe.src = state.objectUrl;

    show(el.fileMeta);
    el.vectorizeBtn.disabled = false;
  }

  function clearFile() {
    revoke(state.objectUrl);
    revoke(state.svgUrl);
    state.file = null;
    state.objectUrl = null;
    state.svgUrl = null;
    state.svgText = null;
    state.meta = null;

    el.fileInput.value = '';
    el.thumbPreview.removeAttribute('src');
    hide(el.fileMeta);
    hide(el.resultPanel);
    hide(el.warningBox);
    clearError();
    el.vectorizeBtn.disabled = true;
  }

  // --- vectorize ----------------------------------------------------------

  function setBusy(busy) {
    state.busy = busy;
    el.vectorizeBtn.disabled = busy || !state.file;
    el.vectorizeBtn.textContent = busy ? 'Vectorizing…' : 'Vectorize';
    if (busy) { show(el.progress); } else { hide(el.progress); }
  }

  function buildFormData() {
    var form = new FormData();
    form.append('image', state.file);
    form.append('preset', el.presetSelect.value);

    if (el.backgroundSelect.value) {
      form.append('background', el.backgroundSelect.value);
    }
    if (el.enclosedCheck.checked) {
      form.append('remove_enclosed_background', 'true');
    }
    if (el.maxDimInput.value) {
      form.append('max_dimension', el.maxDimInput.value);
    }
    if (el.supersampleInput.value) {
      form.append('supersample', el.supersampleInput.value);
    }
    if (el.smoothInput.value !== '') {
      form.append('boundary_smooth_sigma', el.smoothInput.value);
    }
    return form;
  }

  function vectorize() {
    if (!state.file || state.busy) return;

    clearError();
    hide(el.warningBox);
    setBusy(true);
    el.progressDetail.textContent = 'Uploading and analysing the image';

    var tick = setTimeout(function () {
      el.progressDetail.textContent =
        'Tracing paths — detailed artwork can take 30 seconds or more';
    }, 2500);

    fetch(API.vectorize, { method: 'POST', body: buildFormData() })
      .then(function (res) {
        return res.json()
          .catch(function () {
            throw new Error('The server returned an unreadable response (HTTP ' + res.status + ').');
          })
          .then(function (body) {
            if (!res.ok || body.success === false) {
              var err = new Error((body.error && body.error.message) || 'Vectorization failed.');
              err.code = body.error && body.error.code;
              throw err;
            }
            return body;
          });
      })
      .then(renderResult)
      .catch(function (err) {
        showError(
          err.message || 'Could not reach the server. Is the Node API running?',
          err.code || 'NETWORK_ERROR',
        );
        hide(el.resultPanel);
      })
      .finally(function () {
        clearTimeout(tick);
        setBusy(false);
        refreshEngineStatus();
      });
  }

  // --- result rendering ---------------------------------------------------

  function renderResult(body) {
    state.svgText = body.svg;
    state.meta = body.meta || {};

    revoke(state.svgUrl);
    // Rendering through an <img> with a blob URL rather than innerHTML: the
    // SVG is treated as an image, so any script inside it can never execute.
    var blob = new Blob([body.svg], { type: 'image/svg+xml;charset=utf-8' });
    state.svgUrl = URL.createObjectURL(blob);
    el.svgPreview.src = state.svgUrl;

    renderStats(state.meta);
    renderWarnings(state.meta.warnings);

    el.metaDump.textContent = JSON.stringify(state.meta, null, 2);

    show(el.resultPanel);
    el.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderStats(meta) {
    var rows = [
      ['Paths', (meta.path_count || 0).toLocaleString()],
      ['SVG size', formatBytes(meta.svg_bytes)],
      ['Preset used', meta.preset_used || '—'],
      ['Detected as', meta.analysis ? meta.analysis.kind : '—'],
      ['Engine', meta.engine || '—'],
      ['Time', formatMs(meta.processing_ms)],
      ['Source', (meta.original_width || '?') + ' × ' + (meta.original_height || '?')],
      ['Traced at', (meta.processed_width || '?') + ' × ' + (meta.processed_height || '?')],
      ['Background', meta.background && meta.background.applied ? 'removed' : 'kept'],
      ['Optimizer', (meta.size_reduction_pct != null ? meta.size_reduction_pct + '% smaller' : '—')],
      ['Supersample', meta.supersample ? meta.supersample + '×' : '—'],
    ];

    el.statsGrid.innerHTML = '';
    rows.forEach(function (row) {
      var wrap = document.createElement('div');
      var dt = document.createElement('dt');
      var dd = document.createElement('dd');
      dt.textContent = row[0];
      dd.textContent = row[1];
      wrap.appendChild(dt);
      wrap.appendChild(dd);
      el.statsGrid.appendChild(wrap);
    });
  }

  function renderWarnings(warnings) {
    if (!warnings || warnings.length === 0) {
      hide(el.warningBox);
      return;
    }
    el.warningList.innerHTML = '';
    warnings.forEach(function (text) {
      var li = document.createElement('li');
      li.textContent = text;
      el.warningList.appendChild(li);
    });
    show(el.warningBox);
  }

  // --- output actions -----------------------------------------------------

  function downloadSvg() {
    if (!state.svgUrl) return;
    var base = (state.file && state.file.name ? state.file.name : 'artwork')
      .replace(/\.[^.]+$/, '')
      .replace(/[^a-z0-9_-]+/gi, '-');
    var link = document.createElement('a');
    link.href = state.svgUrl;
    link.download = base + '.svg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function copySvg() {
    if (!state.svgText) return;
    var done = function () {
      var original = el.copyBtn.textContent;
      el.copyBtn.textContent = 'Copied';
      setTimeout(function () { el.copyBtn.textContent = original; }, 1400);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(state.svgText).then(done, function () {
        showError('Could not copy to the clipboard. Use Download instead.', 'CLIPBOARD_DENIED');
      });
    } else {
      showError('This browser does not expose the clipboard API. Use Download instead.',
        'CLIPBOARD_UNSUPPORTED');
    }
  }

  // --- wiring -------------------------------------------------------------

  el.browseBtn.addEventListener('click', function (event) {
    event.stopPropagation();
    el.fileInput.click();
  });

  el.dropZone.addEventListener('click', function () { el.fileInput.click(); });

  el.dropZone.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      el.fileInput.click();
    }
  });

  el.fileInput.addEventListener('change', function (event) {
    if (event.target.files && event.target.files[0]) acceptFile(event.target.files[0]);
  });

  ['dragenter', 'dragover'].forEach(function (type) {
    el.dropZone.addEventListener(type, function (event) {
      event.preventDefault();
      el.dropZone.classList.add('is-dragging');
    });
  });

  ['dragleave', 'drop'].forEach(function (type) {
    el.dropZone.addEventListener(type, function (event) {
      event.preventDefault();
      el.dropZone.classList.remove('is-dragging');
    });
  });

  el.dropZone.addEventListener('drop', function (event) {
    var files = event.dataTransfer && event.dataTransfer.files;
    if (files && files[0]) acceptFile(files[0]);
  });

  // Dropping anywhere else must not navigate away from the page.
  window.addEventListener('dragover', function (e) { e.preventDefault(); });
  window.addEventListener('drop', function (e) { e.preventDefault(); });

  el.clearBtn.addEventListener('click', clearFile);
  el.vectorizeBtn.addEventListener('click', vectorize);
  el.downloadBtn.addEventListener('click', downloadSvg);
  el.copyBtn.addEventListener('click', copySvg);

  el.openBtn.addEventListener('click', function () {
    if (state.svgUrl) window.open(state.svgUrl, '_blank', 'noopener');
  });

  el.presetSelect.addEventListener('change', function () {
    el.presetHelp.textContent = PRESET_HELP[el.presetSelect.value] || '';
  });

  // Backdrop switcher: white artwork on a light checkerboard is invisible, so
  // the preview surface has to be switchable to judge the result.
  var backdropChips = document.querySelectorAll('.backdrop-toggle .chip');
  Array.prototype.forEach.call(backdropChips, function (chip) {
    chip.addEventListener('click', function () {
      var choice = chip.getAttribute('data-backdrop');

      Array.prototype.forEach.call(backdropChips, function (other) {
        other.classList.toggle('is-active', other === chip);
      });

      Array.prototype.forEach.call(document.querySelectorAll('.backdrop'), function (surface) {
        surface.classList.remove('backdrop--checker', 'backdrop--light', 'backdrop--dark');
        surface.classList.add('backdrop--' + choice);
      });
    });
  });

  window.addEventListener('beforeunload', function () {
    revoke(state.objectUrl);
    revoke(state.svgUrl);
  });

  refreshEngineStatus();
  setInterval(refreshEngineStatus, 15000);
})();
