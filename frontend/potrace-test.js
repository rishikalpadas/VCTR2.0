/* ==========================================================================
   VTracer vs Potrace - engine comparison test page.
   Vanilla JS, no framework. Calls the same /api/vectorize endpoint twice per
   run - once with engine=vtracer, once with engine=potrace - holding every
   other preprocessing option identical, so the only variable is the tracer.
   ========================================================================== */

(function () {
  'use strict';

  var API = { health: '/api/health', vectorize: '/api/vectorize', exportPdf: '/api/export/pdf' };

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

  var el = {};
  [
    'dropZone', 'browseBtn', 'fileInput', 'fileMeta', 'thumbPreview', 'fileName',
    'fileSize', 'fileDims', 'fileType', 'clearBtn',
    'presetSelect', 'presetHelp', 'backgroundSelect', 'maxDimInput',
    'supersampleInput', 'smoothInput', 'enclosedCheck',
    'turnpolicySelect', 'turdsizeInput', 'alphamaxInput', 'opttoleranceInput',
    'runBtn', 'progress', 'progressDetail',
    'errorBox', 'errorMessage', 'errorCode',
    'resultPanel', 'vtracerPreview', 'potracePreview',
    'vtracerStats', 'potraceStats', 'vtracerDownload', 'potraceDownload',
    'vtracerDownloadPdf', 'potraceDownloadPdf',
    'sideVtracer', 'sidePotrace', 'pickVtracer', 'pickPotrace', 'pickClear',
    'engineStatusVtracer', 'engineStatusPotrace',
  ].forEach(function (id) { el[id] = document.getElementById(id); });

  var state = {
    file: null,
    objectUrl: null,
    busy: false,
    results: { vtracer: null, potrace: null }, // { svgText, svgUrl, meta }
    exporting: { vtracer: false, potrace: false },
  };

  // --- helpers --------------------------------------------------------------

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

  function clearError() { hide(el.errorBox); el.errorCode.textContent = ''; }

  // --- engine status ----------------------------------------------------

  function setEngineBadge(node, label, available) {
    node.className = 'badge ' + (available ? 'badge--ok' : 'badge--down');
    node.textContent = label + (available ? ' ready' : ' unavailable');
  }

  function refreshEngineStatus() {
    fetch(API.health)
      .then(function (res) { return res.json(); })
      .then(function (data) {
        var engines = (data.engine && data.engine.engines) || [];
        var byName = {};
        engines.forEach(function (e) { byName[e.name] = e; });

        setEngineBadge(el.engineStatusVtracer, 'vtracer', !!(byName.vtracer && byName.vtracer.available));
        setEngineBadge(el.engineStatusPotrace, 'potrace', !!(byName.potrace && byName.potrace.available));
      })
      .catch(function () {
        setEngineBadge(el.engineStatusVtracer, 'vtracer', false);
        setEngineBadge(el.engineStatusPotrace, 'potrace', false);
      });
  }

  // --- file selection -----------------------------------------------------

  function acceptFile(file) {
    clearError();
    if (!file) return;

    if (ACCEPTED.indexOf(file.type) === -1) {
      showError('Unsupported file type "' + (file.type || 'unknown')
        + '". Choose a PNG, JPG/JPEG or WebP image.', 'UNSUPPORTED_FORMAT');
      return;
    }
    if (file.size > MAX_BYTES) {
      showError('That file is ' + formatBytes(file.size) + '. The limit is '
        + formatBytes(MAX_BYTES) + '.', 'FILE_TOO_LARGE');
      return;
    }

    revoke(state.objectUrl);
    state.file = file;
    state.objectUrl = URL.createObjectURL(file);

    el.thumbPreview.src = state.objectUrl;
    el.fileName.textContent = file.name;
    el.fileName.title = file.name;
    el.fileSize.textContent = formatBytes(file.size);
    el.fileType.textContent = file.type.replace('image/', '').toUpperCase();
    el.fileDims.textContent = 'reading…';

    var probe = new Image();
    probe.onload = function () {
      el.fileDims.textContent = probe.naturalWidth + ' × ' + probe.naturalHeight;
    };
    probe.onerror = function () {
      el.fileDims.textContent = 'unreadable';
      showError('That image could not be decoded by the browser. It may be corrupted.', 'CORRUPT_IMAGE');
    };
    probe.src = state.objectUrl;

    show(el.fileMeta);
    el.runBtn.disabled = false;
  }

  function clearFile() {
    revoke(state.objectUrl);
    revoke(state.results.vtracer && state.results.vtracer.svgUrl);
    revoke(state.results.potrace && state.results.potrace.svgUrl);
    state.file = null;
    state.objectUrl = null;
    state.results = { vtracer: null, potrace: null };

    el.fileInput.value = '';
    el.thumbPreview.removeAttribute('src');
    hide(el.fileMeta);
    hide(el.resultPanel);
    clearError();
    el.runBtn.disabled = true;
  }

  // --- request building ---------------------------------------------------

  function buildSharedFields(form) {
    form.append('preset', el.presetSelect.value);
    if (el.backgroundSelect.value) form.append('background', el.backgroundSelect.value);
    if (el.enclosedCheck.checked) form.append('remove_enclosed_background', 'true');
    if (el.maxDimInput.value) form.append('max_dimension', el.maxDimInput.value);
    if (el.supersampleInput.value) form.append('supersample', el.supersampleInput.value);
    if (el.smoothInput.value !== '') form.append('boundary_smooth_sigma', el.smoothInput.value);
  }

  function buildFormData(engine) {
    var form = new FormData();
    form.append('image', state.file);
    form.append('engine', engine);
    buildSharedFields(form);

    if (engine === 'potrace') {
      form.append('turnpolicy', el.turnpolicySelect.value);
      form.append('turdsize', el.turdsizeInput.value || '2');
      form.append('alphamax', el.alphamaxInput.value || '1.0');
      form.append('opttolerance', el.opttoleranceInput.value || '0.2');
    }
    return form;
  }

  function runOne(engine) {
    return fetch(API.vectorize, { method: 'POST', body: buildFormData(engine) })
      .then(function (res) {
        return res.json()
          .catch(function () {
            throw new Error('The server returned an unreadable response (HTTP ' + res.status + ').');
          })
          .then(function (body) {
            if (!res.ok || body.success === false) {
              var err = new Error((body.error && body.error.message) || (engine + ' vectorization failed.'));
              err.code = body.error && body.error.code;
              throw err;
            }
            return body;
          });
      });
  }

  // --- run ------------------------------------------------------------------

  function setBusy(busy) {
    state.busy = busy;
    el.runBtn.disabled = busy || !state.file;
    el.runBtn.textContent = busy ? 'Comparing…' : 'Compare';
    if (busy) show(el.progress); else hide(el.progress);
  }

  function run() {
    if (!state.file || state.busy) return;

    clearError();
    setBusy(true);
    el.progressDetail.textContent = 'Uploading and tracing with both engines';

    Promise.all([runOne('vtracer'), runOne('potrace')])
      .then(function (bodies) {
        renderResult('vtracer', bodies[0]);
        renderResult('potrace', bodies[1]);
        show(el.resultPanel);
        el.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      })
      .catch(function (err) {
        showError(err.message || 'Could not reach the server. Is the Node API running?',
          err.code || 'NETWORK_ERROR');
        hide(el.resultPanel);
      })
      .finally(function () {
        setBusy(false);
        refreshEngineStatus();
      });
  }

  // --- result rendering -------------------------------------------------

  function renderResult(engine, body) {
    var prevUrl = state.results[engine] && state.results[engine].svgUrl;
    revoke(prevUrl);

    var blob = new Blob([body.svg], { type: 'image/svg+xml;charset=utf-8' });
    var svgUrl = URL.createObjectURL(blob);
    state.results[engine] = { svgText: body.svg, svgUrl: svgUrl, meta: body.meta || {} };

    var img = engine === 'vtracer' ? el.vtracerPreview : el.potracePreview;
    var statsNode = engine === 'vtracer' ? el.vtracerStats : el.potraceStats;
    img.src = svgUrl;
    renderStats(statsNode, body.meta || {});
  }

  function renderStats(node, meta) {
    var engineMeta = meta.engine_meta || {};
    var rows = [
      ['Paths', (meta.path_count || 0).toLocaleString()],
      ['SVG size', formatBytes(meta.svg_bytes)],
      ['Engine time', formatMs(engineMeta.engine_ms)],
      ['Total time', formatMs(meta.processing_ms)],
      ['Traced at', (meta.processed_width || '?') + ' × ' + (meta.processed_height || '?')],
      ['Layers', engineMeta.layer_count != null ? engineMeta.layer_count : '—'],
    ];

    node.innerHTML = '';
    rows.forEach(function (row) {
      var wrap = document.createElement('div');
      var dt = document.createElement('dt');
      var dd = document.createElement('dd');
      dt.textContent = row[0];
      dd.textContent = row[1];
      wrap.appendChild(dt);
      wrap.appendChild(dd);
      node.appendChild(wrap);
    });
  }

  // --- downloads ------------------------------------------------------------

  function baseName() {
    return (state.file && state.file.name ? state.file.name : 'artwork')
      .replace(/\.[^.]+$/, '')
      .replace(/[^a-z0-9_-]+/gi, '-') || 'artwork';
  }

  function triggerDownload(href, filename) {
    var link = document.createElement('a');
    link.href = href;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function downloadResult(engine) {
    var result = state.results[engine];
    if (!result) return;
    triggerDownload(result.svgUrl, baseName() + '-' + engine + '.svg');
  }

  /**
   * PDF is produced server-side from the SVG already on hand, so it matches
   * the preview exactly and costs no re-trace - same as the main app.
   */
  function downloadPdf(engine) {
    var result = state.results[engine];
    if (!result || state.exporting[engine]) return;

    var button = engine === 'vtracer' ? el.vtracerDownloadPdf : el.potraceDownloadPdf;
    var original = button.textContent;
    state.exporting[engine] = true;
    button.disabled = true;
    button.textContent = 'Preparing PDF…';
    clearError();

    fetch(API.exportPdf + '?name=' + encodeURIComponent(baseName() + '-' + engine), {
      method: 'POST',
      headers: { 'Content-Type': 'image/svg+xml' },
      body: result.svgText,
    })
      .then(function (res) {
        if (res.ok) return res.blob();
        return res.json()
          .catch(function () {
            throw new Error('PDF export failed (HTTP ' + res.status + ').');
          })
          .then(function (body) {
            var err = new Error((body.error && body.error.message) || 'PDF export failed.');
            err.code = body.error && body.error.code;
            throw err;
          });
      })
      .then(function (blob) {
        var href = URL.createObjectURL(blob);
        triggerDownload(href, baseName() + '-' + engine + '.pdf');
        setTimeout(function () { revoke(href); }, 10000);
      })
      .catch(function (err) {
        showError(err.message || 'Could not export the PDF.', err.code || 'PDF_EXPORT_FAILED');
      })
      .finally(function () {
        state.exporting[engine] = false;
        button.disabled = false;
        button.textContent = original;
      });
  }

  // --- wiring -----------------------------------------------------------

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
  window.addEventListener('dragover', function (e) { e.preventDefault(); });
  window.addEventListener('drop', function (e) { e.preventDefault(); });

  el.clearBtn.addEventListener('click', clearFile);
  el.runBtn.addEventListener('click', run);
  el.vtracerDownload.addEventListener('click', function () { downloadResult('vtracer'); });
  el.potraceDownload.addEventListener('click', function () { downloadResult('potrace'); });
  el.vtracerDownloadPdf.addEventListener('click', function () { downloadPdf('vtracer'); });
  el.potraceDownloadPdf.addEventListener('click', function () { downloadPdf('potrace'); });

  el.presetSelect.addEventListener('change', function () {
    el.presetHelp.textContent = PRESET_HELP[el.presetSelect.value] || '';
  });
  el.presetHelp.textContent = PRESET_HELP[el.presetSelect.value] || '';

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

  // Purely a note-to-self toggle for the person comparing - no data is sent
  // anywhere, it just highlights a side visually while you decide.
  function pick(side) {
    el.sideVtracer.classList.toggle('is-winner', side === 'vtracer');
    el.sidePotrace.classList.toggle('is-winner', side === 'potrace');
  }
  el.pickVtracer.addEventListener('click', function () { pick('vtracer'); });
  el.pickPotrace.addEventListener('click', function () { pick('potrace'); });
  el.pickClear.addEventListener('click', function () { pick(null); });

  window.addEventListener('beforeunload', function () {
    revoke(state.objectUrl);
    revoke(state.results.vtracer && state.results.vtracer.svgUrl);
    revoke(state.results.potrace && state.results.potrace.svgUrl);
  });

  refreshEngineStatus();
  setInterval(refreshEngineStatus, 15000);
})();
