"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const token = document.querySelector('meta[name="hum-session-token"]').content;
  const audio = [$('input-audio'), $('output-audio')];
  const modes = {fixed: 'Fixed filter', tracked: 'Limited tracking', guarded: 'Guarded tracking', harmonics: 'Harmonic processing', bands: 'Per-band control', compatible: 'Standard WAV', long: 'Long WAV', precision: 'High-resolution WAV', compressed: 'MP3·M4A → WAV', profile: 'Experimental · noise-profile band guard'};
  const help = {fixed: 'Applies a fixed filter at the selected frequency.', tracked: 'Tracks slow drift within ±1 Hz, at up to 0.05 Hz/s. Initial observation takes 2 seconds.', guarded: 'Activates gradually after two qualified tracking observations. Release is also delayed, and initial attenuation may be lower.', harmonics: 'Observes each selected fundamental or harmonic band independently and gradually processes qualified bands. It does not determine whether components are true harmonics or noise.', bands: 'Set attenuation independently for the fundamental, second and third orders. A 0 dB band is off; protecting wanted bass requires listening and manual adjustment.', compatible: 'Processes bands in 44.1/48 kHz mono or stereo PCM16 WAV. Read the resampling notes below and manually enable only the bands you need.', long: 'Processes 44.1/48 kHz mono or stereo PCM16 WAV up to 10 minutes. When any band is enabled, 44.1 kHz uses round-trip FIR resampling; this is not real-time processing.', precision: 'Processes selected bands while preserving the original PCM16, PCM24 or Float32 format, up to 10 minutes and 128 MiB. Check the Float32 range and resampling notes.', compressed: 'Locally decodes MP3 or AAC M4A to Float32 WAV, then processes selected bands. Compressed input is limited to 32 MiB; decoded audio is limited to 10 minutes or 128 MiB, whichever comes first. Output is WAV only.'};
  help.profile = 'Declare at least 2 seconds containing only noise. Only supported bands among orders 1–32 are processed. This offline experiment withholds an entire band if amplitude later grows too large.';
  modes.persistent = 'Persistent hum · offline';
  help.persistent = 'Explicitly opt in to whole-file analysis of stable 50/60 Hz harmonic families. At least four consistent persistent harmonic lines are required; insufficient evidence leaves the validated audio payload unchanged.';
  let selectedFile = null;
  let busy = false;
  let urls = [];
  let lastWaveforms = null;
  let lastChannels = 1;
  let compressedAvailable = false;
  let persistentAvailable = false;
  let desktopAvailable = false;
  let stopping = false;
  let stopRequested = false;
  let lastReport = null;
  let lastReviewContext = null;
  let exampleIntervalOnly = false;
  let profileConfirmedSource = null;
  let fileCheckSerial = 0;
  let fileChecking = false;
  let fileMetadata = null;
  let fileCheckTimer = null;
  let capabilitiesKnown = false;
  const interactionLocked = () => busy || stopping || stopRequested;
  const abReview = window.HumABReview?.create({players: audio, getElement: $, isLocked: interactionLocked});

  function updateDesktopControls() {
    $('desktop-controls').hidden = !desktopAvailable;
    $('desktop-stop').disabled = interactionLocked();
    abReview?.refresh();
    updateFilePreflight();
  }

  function dismissStopConfirmation() {
    $('desktop-stop-confirm').hidden = true;
    $('desktop-stop').setAttribute('aria-expanded', 'false');
  }

  async function loadCapabilities() {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), 10000);
    try {
      const response = await fetch('/api/capabilities', {
        method: 'GET', credentials: 'same-origin', cache: 'no-store', signal: abort.signal
      });
      const value = await response.json();
      if (!response.ok || value.schema_version !== 1 || value.version !== '0.0.15'
        || typeof value.desktop?.available !== 'boolean' || typeof value.desktop?.portable_wav_only !== 'boolean'
        || typeof value.compressed?.available !== 'boolean' || value.compressed.required_version !== '>=19,<20'
        || (value.desktop.portable_wav_only && value.compressed.available)) throw new Error('capabilities');
      compressedAvailable = value.compressed.available;
      // Older processors do not advertise this candidate. Never infer support
      // from a successful generic capability response or switch modes for users.
      persistentAvailable = value.persistent?.available === true;
      desktopAvailable = value.desktop.available;
      const notice = $('capability-notice');
      notice.hidden = compressedAvailable;
      notice.textContent = value.desktop.portable_wav_only
        ? 'This distribution supports WAV only. MP3/M4A processing is not included.'
        : 'No supported MP3/M4A decoder is available in this environment. WAV processing remains available.';
    } catch {
      compressedAvailable = false;
      persistentAvailable = false;
      desktopAvailable = false;
      $('capability-notice').hidden = false;
      $('capability-notice').textContent = 'Could not verify this environment. WAV processing remains available; MP3/M4A processing is disabled.';
    } finally {
      clearTimeout(timer);
      capabilitiesKnown = true;
      $('mode').querySelector('option[value="compressed"]').disabled = !compressedAvailable;
      $('persistent-option').disabled = !persistentAvailable;
      $('persistent-capability-notice').hidden = persistentAvailable;
      $('persistent-capability-notice').textContent = 'Persistent hum is disabled because this processor has not confirmed support. Other available WAV modes remain unchanged.';
      updateDesktopControls();
    }
  }

  $('desktop-stop').addEventListener('click', () => {
    if (interactionLocked() || !desktopAvailable) return;
    $('desktop-stop-confirm').hidden = false;
    $('desktop-stop').setAttribute('aria-expanded', 'true');
    $('desktop-stop-cancel').focus();
  });
  $('desktop-stop-cancel').addEventListener('click', () => {
    if (interactionLocked()) return;
    dismissStopConfirmation();
    $('desktop-stop').focus();
  });
  $('desktop-stop-confirm').addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !interactionLocked()) {
      dismissStopConfirmation();
      $('desktop-stop').focus();
    }
  });
  $('desktop-stop-confirm-button').addEventListener('click', async () => {
    if (interactionLocked() || !desktopAvailable) return;
    stopping = true;
    dismissStopConfirmation();
    updateDesktopControls();
    $('settings-fields').disabled = true;
    for (const player of audio) { player.pause(); player.controls = false; }
    abReview?.suspend();
    for (const id of ['download-wav', 'download-zip']) $(id).setAttribute('aria-disabled', 'true');
    $('desktop-stop-status').hidden = false;
    $('desktop-stop-status').textContent = 'Requesting shutdown. Processing controls and playback are temporarily paused.';
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), 10000);
    try {
      const response = await fetch('/api/desktop/stop', {
        method: 'POST', headers: {'Content-Type': 'application/json', 'X-Hum-Token': token},
        body: new Uint8Array(0), credentials: 'same-origin', cache: 'no-store', signal: abort.signal
      });
      const result = await response.json();
      if (!response.ok || result.status !== 'stopping' || result.temporary_cleanup_pending !== true) throw new Error('stop');
      stopRequested = true;
      $('desktop-stop-status').textContent = 'Shutdown requested. Temporary copies are cleaned up after a normal shutdown. You may close this tab.';
      $('status').textContent = 'Processor shutdown requested. This page does not confirm that temporary-copy cleanup has completed.';
      $('result-state').textContent = 'Shutdown requested';
    } catch {
      $('desktop-stop-status').textContent = 'Could not confirm the shutdown response. Check the launcher window. A normal shutdown and temporary-copy cleanup have not been confirmed.';
      $('settings-fields').disabled = false;
      for (const player of audio) player.controls = true;
      for (const id of ['download-wav', 'download-zip']) $(id).removeAttribute('aria-disabled');
    } finally {
      clearTimeout(timer);
      stopping = false;
      updateDesktopControls();
    }
  });
  for (const id of ['download-wav', 'download-zip']) {
    $(id).addEventListener('click', (event) => { if (stopping || stopRequested) event.preventDefault(); });
  }

  const usesStoredFiles = (mode) => mode === 'long' || mode === 'precision' || mode === 'compressed' || mode === 'persistent';
  const usesBandControls = (mode) => mode === 'bands' || mode === 'compatible' || (usesStoredFiles(mode) && mode !== 'persistent');
  const fileLimitMiB = (mode) => mode === 'compressed' ? 32 : (usesStoredFiles(mode) ? 128 : 32);
  const formatRate = (rate) => `${rate / 1000}kHz`;
  const formatChannels = (channels) => channels === 2 ? 'stereo · 2 channels' : channels === 1 ? 'mono · 1 channel' : `${channels} channels (unsupported)`;
  function formatSamples(metadata) {
    const name = metadata.sample_format;
    if (name === 'signed_PCM24_little_endian') return 'PCM24';
    if (name === 'IEEE_FLOAT32_little_endian') return 'Float32';
    if (name === undefined || name === 'signed_PCM16_little_endian') return 'PCM16';
    throw new Error('Invalid audio sample-format response.');
  }

  for (const player of audio) {
    player.volume = 0.25;
    player.addEventListener('play', () => {
      for (const other of audio) if (other !== player) other.pause();
    });
    player.addEventListener('volumechange', () => {
      for (const other of audio) {
        if (other !== player && other.volume !== player.volume) other.volume = player.volume;
      }
    });
  }

  function clearResult() {
    abReview?.reset();
    for (const player of audio) { player.pause(); player.removeAttribute('src'); player.load(); }
    for (const url of urls) URL.revokeObjectURL(url);
    urls = [];
    lastWaveforms = null;
    lastChannels = 1;
    lastReport = null;
    lastReviewContext = null;
    $('profile-result').hidden = true;
    $('persistent-result').hidden = true;
    $('persistent-result-detail').textContent = '';
    $('profile-band-list').replaceChildren();
    resetListeningReview();
    $('results').hidden = true;
    $('empty-state').hidden = false;
    $('error').hidden = true;
    $('report-json').textContent = '';
    $('format-stat').textContent = '';
    for (const id of ['download-wav', 'download-zip']) $(id).removeAttribute('href');
    $('result-state').textContent = 'Ready';
    $('result-state').className = 'state-label';
  }

  function setFile(file) {
    if (interactionLocked()) return;
    clearResult();
    resetProfileDeclaration();
    fileCheckSerial += 1;
    if (fileCheckTimer !== null) clearTimeout(fileCheckTimer);
    fileCheckTimer = null;
    fileMetadata = null;
    fileChecking = false;
    selectedFile = file || null;
    $('file-name').textContent = file ? file.name : 'No file selected';
    $('process-button').disabled = !selectedFile;
    $('status').textContent = file ? 'Check the settings, then process the selected file.' : 'Choose a file or start with a synthetic example.';
    const validCandidate = !file || (Number.isSafeInteger(file.size) && file.size > 0
      && ((/\.wav$/i.test(file.name) && file.size <= 128*1024*1024)
        || (/\.(mp3|m4a)$/i.test(file.name) && file.size <= 32*1024*1024)));
    const validationError = validCandidate ? null : 'Choose a nonempty WAV up to 128 MiB, or MP3 / AAC M4A up to 32 MiB. The selected mode may have lower limits.';
    if (validationError) {
      selectedFile = null;
      $('process-button').disabled = true;
      showError(validationError);
    }
    if (selectedFile) void checkFileHeader(selectedFile);
    else updateFilePreflight();
  }

  function fallbackMetadata(file, message) {
    const compressed = /\.(mp3|m4a)$/i.test(file.name);
    return {status:compressed?'deferred':'unchecked',kind:compressed?'compressed':'wav',
      container_bytes:file.size,header_only:true,sample_values_checked:false,message};
  }

  function fileAssessment() {
    if (!selectedFile) return null;
    if (fileChecking || ((fileMetadata?.kind === 'compressed' || $('mode').value === 'persistent') && !capabilitiesKnown)) return {allowed:false,level:'checking',message:'Checking the file header and available input modes. No audio is decoded or played.',recommended_mode:null};
    if (window.HumPreflight && fileMetadata) return window.HumPreflight.assess(fileMetadata,$('mode').value,{compressedAvailable,persistentAvailable});
    const error = validateFile(selectedFile,$('mode').value);
    return {allowed:!error,level:error?'error':'warning',message:error || 'Header checker unavailable. No format is confirmed here; the local processor still performs final input validation.',recommended_mode:null};
  }

  function updateFilePreflight() {
    const assessment = fileAssessment();
    $('preflight-panel').hidden = !selectedFile;
    $('preflight-suggest').hidden = true;
    if (assessment) {
      $('preflight-panel').setAttribute('data-state',assessment.level);
      $('preflight-summary').textContent = assessment.level === 'checking' ? 'Checking header…'
        : fileMetadata?.status === 'metadata'
          ? `${formatRate(fileMetadata.sample_rate_hz)} · ${formatChannels(fileMetadata.channels)} · ${fileMetadata.sample_format} · ${fileMetadata.duration_seconds.toFixed(2)} s`
          : 'Metadata not confirmed';
      $('preflight-detail').textContent = assessment.message;
      const suggested = assessment.recommended_mode;
      if (!assessment.allowed && suggested && suggested !== $('mode').value && modes[suggested]) {
        $('preflight-suggest').hidden = false;
        $('preflight-suggest').disabled = interactionLocked() || fileChecking;
        $('preflight-suggest').textContent = `Use ${modes[suggested]}`;
      }
    }
    $('process-button').disabled = interactionLocked() || !selectedFile || !assessment?.allowed;
  }

  async function checkFileHeader(file) {
    const serial = fileCheckSerial;
    const current = () => serial === fileCheckSerial && selectedFile === file;
    if (!window.HumPreflight) {
      fileMetadata = fallbackMetadata(file,'Header checker unavailable; final server validation is required.');
      updateFilePreflight(); return;
    }
    fileChecking = true;
    updateFilePreflight();
    fileCheckTimer = setTimeout(() => {
      if (!current()) return;
      fileCheckSerial += 1;
      fileChecking = false; fileCheckTimer = null;
      fileMetadata = fallbackMetadata(file,'Header check timed out. No format was confirmed. The selected mode still performs final local validation.');
      updateFilePreflight();
    },11000);
    try {
      const info = await window.HumPreflight.inspect(file,{isCancelled:() => !current()});
      if (!current()) return;
      fileMetadata = info;
    } catch {
      if (!current()) return;
      fileMetadata = fallbackMetadata(file,'Could not read this header. No format was confirmed; final server validation is still required.');
    } finally {
      if (current()) {
        if (fileCheckTimer !== null) clearTimeout(fileCheckTimer);
        fileCheckTimer = null; fileChecking = false;
        updateFilePreflight();
      }
    }
  }

  $('preflight-suggest').addEventListener('click', () => {
    if (interactionLocked() || fileChecking) return;
    const suggested = fileAssessment()?.recommended_mode;
    if (!suggested || !modes[suggested] || (suggested === 'compressed' && !compressedAvailable)) return;
    $('mode').value = suggested;
    $('mode').dispatchEvent(new Event('input',{bubbles:true}));
  });

  function validateFile(file, mode) {
    if (mode === 'persistent' && !persistentAvailable) return 'Persistent hum is unavailable in this processor. Select another available WAV mode.';
    if (mode === 'compressed') {
      if (!compressedAvailable) return 'MP3/M4A processing is unavailable in this environment. Choose a WAV file and a WAV processing mode.';
      if (/\.(mp3|m4a)$/i.test(file.name) && file.size > 0 && file.size <= fileLimitMiB(mode) * 1024 * 1024) return null;
      return 'Choose a nonempty MP3 or AAC M4A file no larger than 32 MiB. Decoded audio must be at most 10 minutes, 44.1/48 kHz, mono or stereo. ALAC M4A is not supported.';
    }
    if (/\.wav$/i.test(file.name) && file.size > 0 && file.size <= fileLimitMiB(mode) * 1024 * 1024) return null;
    const format = mode === 'persistent'
      ? 'Persistent hum supports 44.1/48 kHz mono/stereo PCM16, PCM24 or Float32 WAV, up to 60 seconds and 128 MiB. MP3/M4A is not accepted.'
      : mode === 'precision'
      ? 'High-resolution WAV supports up to 10 minutes of 44.1/48 kHz mono or stereo PCM16, PCM24 or Float32 audio.'
      : mode === 'long'
      ? 'Long WAV supports only 44.1/48 kHz mono or stereo PCM16 audio, up to 10 minutes.'
      : (mode === 'compatible' ? 'Standard WAV supports only 44.1/48 kHz mono or stereo PCM16 audio.' : 'This mode accepts only 48 kHz mono PCM16. For stereo or 44.1 kHz files, choose Standard WAV.');
    return `Choose a nonempty WAV file no larger than ${fileLimitMiB(mode)} MiB. ${format}`;
  }

  function serverErrorMessage(message, status, mode) {
    // Keep API/error schemas unchanged. These are English recovery instructions,
    // not purported exact translations of untranslated backend diagnostics.
    if (typeof message === 'string' && !/[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]/.test(message)) return message;
    if (status === 403) return 'The local session was rejected. Reload this local page and try again.';
    const context = `${modes[mode]}. ${help[mode]}`;
    if (status === 413) return `The file or decoded audio exceeds the current mode limits. Check the file requirements. ${context}`;
    if (status === 503) return `The processor may be busy or an optional decoder may be unavailable. Wait for the current job to finish and check the mode requirements. ${context}`;
    if (status === 400) return `This file or setting is not supported by the selected mode. Check the file format and settings, then try again. ${context}`;
    return `The local processor could not complete the request. Check the file format and settings, then try again. ${context}`;
  }

  function showError(message) {
    $('error').textContent = message;
    $('error').hidden = false;
    $('result-state').textContent = 'Needs attention';
    $('result-state').className = 'state-label failed';
    $('status').textContent = 'No result was created. Check the input file and settings.';
  }

  $('audio-file').addEventListener('change', (event) => setFile(event.target.files[0]));
  const zone = $('dropzone');
  for (const eventName of ['dragenter', 'dragover']) {
    zone.addEventListener(eventName, (event) => { event.preventDefault(); if (!interactionLocked()) zone.classList.add('dragover'); });
  }
  for (const eventName of ['dragleave', 'drop']) {
    zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.remove('dragover'); });
  }
  zone.addEventListener('drop', (event) => {
    if (interactionLocked()) return;
    const files = event.dataTransfer.files;
    $('audio-file').value = '';
    if (files.length !== 1) { setFile(null); showError('Choose one audio file at a time.'); return; }
    setFile(files[0]);
  });
  // A dropped file must never navigate the browser away from this local tool.
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => event.preventDefault());

  function updateSettingsControls() {
    const mode = $('mode').value;
    const bandControls = usesBandControls(mode);
    $('mode-help').textContent = help[mode];
    $('harmonic-options').hidden = mode !== 'harmonics';
    $('band-options').hidden = !bandControls;
    $('attenuation-options').hidden = bandControls || mode === 'profile' || mode === 'persistent';
    $('profile-options').hidden = mode !== 'profile';
    $('persistent-options').hidden = mode !== 'persistent';
    $('example-button').disabled = interactionLocked() || (mode === 'persistent' && !persistentAvailable);
    $('compatible-notice').hidden = mode !== 'compatible';
    $('long-notice').hidden = mode !== 'long';
    $('precision-notice').hidden = mode !== 'precision';
    $('compressed-notice').hidden = mode !== 'compressed';
    $('persistent-notice').hidden = mode !== 'persistent';
    $('audio-file').accept = mode === 'compressed' ? '.mp3,.m4a,audio/mpeg,audio/mp4,audio/x-m4a' : '.wav,audio/wav,audio/x-wav';
    $('file-type-label').textContent = mode === 'compressed' ? 'Choose an MP3/M4A file' : 'Choose a WAV file';
    $('input-contract').textContent = mode === 'compressed'
      ? 'MP3/AAC M4A · up to 32 MiB. After decoding: 44.1/48 kHz · mono/stereo · up to 10 minutes or 128 MiB, whichever comes first. Output is Float32 WAV, not re-encoded MP3/M4A.'
      : mode === 'persistent'
      ? '44.1/48 kHz · mono/stereo · PCM16/PCM24/Float32 WAV · up to 60 seconds and 128 MiB. Native sample format, rate, channels and frame count are preserved; no resampling.'
      : mode === 'precision'
      ? '44.1/48 kHz · mono/stereo · PCM16/PCM24/Float32 WAV · up to 10 minutes or 128 MiB. Original sample format and length are preserved; high-resolution stereo may reach the size limit first.'
      : mode === 'long'
      ? '44.1/48 kHz · mono/stereo · PCM16 WAV · up to 10 minutes · 128 MiB. Sample count, sample rate and channel count are preserved. Read the 44.1 kHz resampling notes.'
      : mode === 'compatible'
      ? '44.1/48 kHz · mono/stereo · PCM16 WAV · up to 60 seconds · 32 MiB. Resampling follows the notes below; channel count is preserved.'
      : '48 kHz · mono · PCM16 · up to 60 seconds · 32 MiB. Other formats are not converted automatically.';
    for (let n = 1; n <= 3; n++) {
      const label = `${Number($('hum-hz').value) * n} Hz · ${n === 1 ? 'fundamental' : 'order ' + n}`;
      $('harmonic-label-' + n).textContent = label;
      $('band-label-' + n).textContent = label;
      $('band-depth-value-' + n).textContent = `${$('band-depth-' + n).value} dB`;
    }
    $('attenuation-value').textContent = `${$('attenuation').value} dB`;
    $('profile-depth-value').textContent = `${$('profile-depth').value} dB`;
    $('persistent-depth-value').textContent = `${$('persistent-depth').value} dB`;
    const profileBase = Number($('hum-hz').value);
    $('profile-range').textContent = `${profileBase} Hz × orders 1–32: ${profileBase}–${(profileBase * 32).toLocaleString('en-US')} Hz · ±1 Hz per band`;
    $('band-depth-help').textContent = mode === 'compressed'
      ? 'Defaults are 6/3/0 dB. With all bands at 0 dB, decoded input WAV sample values are preserved. This does not preserve the compressed file or restore the pre-compression signal; settings are not measurements of actual noise reduction.'
      : 'Defaults are 6/3/0 dB. With all bands at 0 dB, the input is unchanged. Settings are not measurements of actual noise reduction.';
    $('example-scope').textContent = mode === 'profile'
      ? 'The experimental example is an 8-second, 48 kHz mono PCM16 synthetic signal. The first 0–2 seconds contain generated noise only; wanted synthetic content starts at 2 seconds. Enter 0–2 seconds and explicitly confirm the example interval. This does not verify a noise-only interval in your own file.'
      : mode === 'persistent'
      ? 'An 8-second, 48 kHz stereo PCM16 WAV with generated wanted content and four persistent harmonic lines. It is a constructed example, not evidence of quality in your own recording. Attenuation settings do not change the synthetic input.'
      : mode === 'compressed'
      ? 'A 6-second, 48 kHz stereo synthetic signal is locally encoded to MP3 and decoded for this example. It requires PyAV 19.x, which is not installed automatically. Attenuation settings do not change the synthetic input; changing the frequency does.'
      : mode === 'precision'
      ? 'The high-resolution example is a generated 6-second, 48 kHz stereo PCM24 signal, not a Float32 or 44.1 kHz test. Attenuation settings do not change the synthetic input; changing the frequency does.'
      : mode === 'long'
      ? 'The Long WAV example is a 65-second, 48 kHz stereo signal with no resampling. To test 44.1 kHz resampling, select a matching WAV file. Attenuation settings do not change the synthetic input; changing the frequency does.'
      : bandControls
      ? `${mode === 'compatible' ? 'The Standard WAV example is 48 kHz stereo. ' : ''}Attenuation settings do not change the same 6-second synthetic input. Changing the frequency changes the example.`
      : 'The example is generated for the selected mode and bands.';
  }

  for (const id of ['mode', 'hum-hz', 'attenuation', 'harmonic-1', 'harmonic-2', 'harmonic-3', 'band-depth-1', 'band-depth-2', 'band-depth-3', 'profile-start', 'profile-end', 'profile-orders', 'profile-depth', 'profile-confirm', 'persistent-depth']) {
    $(id).addEventListener('input', () => {
      if (interactionLocked()) return;
      clearResult();
      if (['mode', 'profile-start', 'profile-end'].includes(id)) {
        $('profile-confirm').checked = false;
        profileConfirmedSource = null;
      }
      if (['profile-start', 'profile-end'].includes(id)) exampleIntervalOnly = false;
      if (id === 'profile-confirm') {
        profileConfirmedSource = $('profile-confirm').checked
          ? (exampleIntervalOnly || !selectedFile ? 'example' : 'user_file') : null;
      }
      updateSettingsControls();
      $('status').textContent = 'Settings changed. Process again with the new settings.';
      updateFilePreflight();
    });
  }
  $('band-bypass').addEventListener('click', () => {
    if (interactionLocked()) return;
    clearResult();
    for (let n = 1; n <= 3; n++) $('band-depth-' + n).value = '0';
    updateSettingsControls();
    $('status').textContent = $('mode').value === 'compressed'
      ? 'All bands are now at 0 dB. Process again to compare unchanged decoded-input WAV samples. This does not restore the pre-compression signal.'
      : 'All bands are now at 0 dB. Process again to compare with the unchanged input.';
  });

  function resetProfileDeclaration() {
    $('profile-start').value = '';
    $('profile-end').value = '';
    $('profile-confirm').checked = false;
    exampleIntervalOnly = false;
    profileConfirmedSource = null;
  }

  $('profile-example-interval').addEventListener('click', () => {
    if (interactionLocked()) return;
    clearResult();
    $('profile-start').value = '0';
    $('profile-end').value = '2';
    $('profile-confirm').checked = false;
    exampleIntervalOnly = true;
    profileConfirmedSource = null;
    $('status').textContent = 'Filled 0–2 seconds for the synthetic example only. No noise-only interval in your file has been verified. Check the confirmation yourself before running the example.';
  });

  function profileParameters() {
    const start = $('profile-start').value.trim();
    const end = $('profile-end').value.trim();
    function sampleBoundary(value) {
      if (!/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(value) || value.length > 32) return null;
      const [whole, fraction = ''] = value.split('.');
      const scale = 10n ** BigInt(fraction.length);
      const numerator = (BigInt(whole) * scale + BigInt(fraction || '0')) * 48000n;
      return numerator % scale === 0n ? numerator / scale : null;
    }
    const startSample = sampleBoundary(start);
    const endSample = sampleBoundary(end);
    if (startSample === null || endSample === null || startSample < 0n || endSample > 2880000n || endSample - startSample < 96000n) {
      throw new Error('Enter noise-only start and end in seconds within the same input. Use at least 2 seconds, ending no later than 60 seconds, aligned to 48 kHz sample boundaries. The processor checks the actual file length.');
    }
    if (!$('profile-confirm').checked) throw new Error('Check that the interval contains no wanted sound, then confirm it. The example and your own file require separate confirmations.');
    const orders = $('profile-orders').value.trim();
    if (orders !== 'all') {
      const parts = orders.split(',');
      if (!parts.length || parts.length > 32 || !parts.every(part => /^(?:[1-9]|[12]\d|3[0-2])$/.test(part))
        || !parts.every((part, index) => index === 0 || Number(part) > Number(parts[index - 1]))) {
        throw new Error('Enter all or unique ascending comma-separated harmonic orders from 1 to 32, for example 1,6,21,24.');
      }
    }
    const depth = Number($('profile-depth').value);
    if (!Number.isInteger(depth) || depth < 0 || depth > 12) throw new Error('Set experimental attenuation to an integer from 0 to 12 dB.');
    return {profile_start_seconds: start, profile_end_seconds: end, confirm_noise_only: 'true', orders, attenuation_db: String(depth)};
  }

  function renderProfileSummary(report) {
    const summary = report.profile_guard;
    if (!summary || !Array.isArray(summary.approved_orders) || !Array.isArray(summary.rejected_orders)
      || !Array.isArray(summary.unsupported_orders) || !Array.isArray(summary.bands)
      || summary.experimental !== true || summary.same_input_interval !== true
      || summary.noise_only_interval_user_asserted !== true || summary.noise_only_interval_machine_verified !== false
      || summary.offline_whole_file_future_lookahead !== true || summary.real_time !== false
      || typeof summary.processing_bypassed !== 'boolean'
      || summary.input_wav_sha256 !== report.input.sha256
      || !Number.isInteger(report.config.profile_start_sample) || !Number.isInteger(report.config.profile_end_sample)
      || summary.profile_start_seconds !== report.config.profile_start_sample / 48000
      || summary.profile_end_seconds !== report.config.profile_end_sample / 48000) {
      throw new Error('Invalid experimental band-guard evidence response.');
    }
    $('profile-result').hidden = false;
    $('profile-result-state').textContent = summary.processing_bypassed ? 'Unchanged · bypassed' : `${summary.approved_orders.length} eligible bands`;
    $('profile-result-interval').textContent = `User-declared noise-only interval ${summary.profile_start_seconds}–${summary.profile_end_seconds} s · the noise-only assumption has not been machine-verified.`;
    const reasons = {
      profile_unsupported: 'Component not supported by the declared interval',
      whole_file_amplitude_exceeds_profile: 'Withheld for the whole file because amplitude increased',
      profile_supported_amplitude_consistent: 'Profile and amplitude conditions met · no quality guarantee',
    };
    const list = $('profile-band-list');
    list.replaceChildren();
    if (!summary.bands.length) {
      const item = document.createElement('p');
      item.textContent = summary.bypass_reason === 'zero_depth' ? '0 dB: analysis and filtering bypassed; input unchanged.' : 'No eligible bands; input unchanged.';
      list.append(item);
    }
    for (const band of summary.bands) {
      if (!Number.isInteger(band.order) || band.order < 1 || band.order > 32 || typeof band.reason !== 'string') throw new Error('Invalid experimental band response.');
      const row = document.createElement('div');
      row.className = 'profile-band-row';
      const label = document.createElement('strong');
      label.textContent = `Order ${band.order} · ${report.config.nominal_hum_hz * band.order} Hz`;
      const state = document.createElement('span');
      state.textContent = band.approved ? 'Candidate applied' : (band.amplitude_guard_rejected ? 'Withheld by guard' : 'Component unsupported');
      state.className = band.approved ? 'profile-band-approved' : 'profile-band-withheld';
      const reason = document.createElement('small');
      reason.textContent = reasons[band.reason] || `See the detailed report: ${band.reason}`;
      row.append(label, state, reason);
      list.append(row);
    }
  }

  function renderPersistentSummary(report) {
    const algorithm = report.algorithm;
    const orders = algorithm?.accepted_orders;
    if (!Array.isArray(orders) || !orders.every((order, index) => Number.isInteger(order)
      && order >= 1 && order <= 32 && (index === 0 || order > orders[index - 1]))
      || (orders.length > 0 && orders.length < 4)
      || typeof algorithm.reason !== 'string' || typeof report.processing_bypassed !== 'boolean'
      || !Number.isFinite(report.config?.max_attenuation_db) || report.config.max_attenuation_db < 0
      || report.config.max_attenuation_db > 18 || ![50,60].includes(report.config.nominal_hum_hz)) {
      throw new Error('Invalid persistent-hum evidence response.');
    }
    const reasons = {
      zero_depth: '0 dB: analysis and filtering bypassed.',
      insufficient_analysis_windows: 'Not enough analysis windows for persistent-family evidence.',
      no_consistent_persistent_family: 'No sufficiently consistent persistent harmonic family was found.',
      persistent_family_mask: 'A consistent persistent harmonic family was found.',
    };
    $('persistent-result').hidden = false;
    $('persistent-result-state').textContent = report.processing_bypassed ? 'Unchanged · bypassed' : `${orders.length} accepted harmonic lines`;
    $('persistent-result-detail').textContent = `${reasons[algorithm.reason] || `Analysis reason: ${algorithm.reason}.`} ${orders.length ? `Accepted orders: ${orders.join(', ')}.` : 'No accepted orders.'} ${report.processing_bypassed ? 'The validated audio payload is unchanged. This is not successful noise removal.' : 'Accepted lines are processing candidates, not a determination that they are unwanted noise.'}`;
  }

  function resetListeningReview() {
    for (const id of ['review-hum', 'review-wanted', 'review-artifacts', 'review-notes']) $(id).value = '';
    $('review-confirm').checked = false;
    $('review-export').disabled = true;
    $('review-status').textContent = 'Complete all three observations, then check your confirmation. Your assessment is not stored on the server.';
  }

  function updateListeningReview() {
    const complete = ['review-hum', 'review-wanted', 'review-artifacts'].every(id => $(id).value !== '');
    $('review-export').disabled = !lastReport || !complete || !$('review-confirm').checked || interactionLocked();
  }
  for (const id of ['review-hum', 'review-wanted', 'review-artifacts', 'review-notes']) {
    $(id).addEventListener('input', () => {
      $('review-confirm').checked = false;
      updateListeningReview();
      $('review-status').textContent = 'Observations changed. Review them and check your confirmation again.';
    });
  }
  $('review-confirm').addEventListener('input', updateListeningReview);
  $('review-export').addEventListener('click', () => {
    updateListeningReview();
    if ($('review-export').disabled) return;
    const record = {schema_version: 1, product: 'Hum DSP', exported_at: new Date().toISOString(),
      kind: 'personal_listening_self_report', verification_status: 'unverified_self_report',
      user_asserted_listened: true, automatic_quality_pass: false, commercial_approval: false,
      source_context: lastReviewContext, report: lastReport,
      observations: {residual_hum: $('review-hum').value, wanted_sound_loss: $('review-wanted').value,
        new_artifacts: $('review-artifacts').value, notes: $('review-notes').value}};
    const link = document.createElement('a');
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], {type: 'application/json'}));
    link.href = url;
    link.download = 'hum-dsp-personal-listening.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    $('review-status').textContent = 'Personal listening JSON download requested. This is not machine verification, an audio-quality pass or commercial approval.';
  });

  function makeBlobURL(encoded, type) {
    if (typeof encoded !== 'string') throw new Error('A required file is missing from the processing response.');
    const binary = atob(encoded);
    const buffer = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) buffer[i] = binary.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([buffer], {type}));
    urls.push(url);
    return url;
  }

  function localAssetURL(value, filename) {
    if (typeof value !== 'string' || !value.startsWith('/api/long/assets/')) throw new Error('Invalid local result-file address.');
    const url = new URL(value, window.location.origin);
    const expectedPath = new RegExp('^/api/long/assets/[a-f0-9]{48}/' + filename.replace('.', '\\.') + '$');
    if (url.origin !== window.location.origin || url.search || url.hash || !expectedPath.test(url.pathname)) throw new Error('Invalid local result-file address.');
    return url.href;
  }

  function paintWaveform(canvas, bins, bound, color) {
    if (!Array.isArray(bins) || !bins.length) throw new Error('Invalid waveform response format.');
    const width = Math.max(1, canvas.getBoundingClientRect().width);
    const height = 100;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = height * ratio;
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    ctx.strokeStyle = '#d8e1d2';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, width / bins.length * 0.7);
    ctx.beginPath();
    bins.forEach((range, i) => {
      if (!Array.isArray(range) || range.length !== 2 || !range.every(Number.isFinite)) throw new Error('Invalid waveform sample values.');
      const x = (i + 0.5) * width / bins.length;
      ctx.moveTo(x, height / 2 - range[0] / bound * height * 0.43);
      ctx.lineTo(x, height / 2 - range[1] / bound * height * 0.43);
    });
    ctx.stroke();
  }

  function drawWaveforms() {
    if (!lastWaveforms || $('results').hidden) return;
    const {input, output} = lastWaveforms;
    const max = Math.max(0.001, ...input.flat().map(Math.abs), ...output.flat().map(Math.abs));
    paintWaveform($('input-waveform'), input, max, '#91a68f');
    paintWaveform($('output-waveform'), output, max, '#347350');
    $('waveform-scale').textContent = `${lastChannels === 2 ? 'Shared L/R envelope scale · per-channel minima/maxima, not a sum' : 'Shared amplitude scale for both waveforms'} ±${max.toFixed(4)} · no playback level normalization`;
  }
  window.addEventListener('resize', drawWaveforms);

  async function run(isExample) {
    if (interactionLocked() || (!isExample && !selectedFile)) return;
    if (!isExample && !fileAssessment()?.allowed) {
      clearResult(); showError(fileAssessment()?.message || 'Wait for the file check or choose a supported mode.'); return;
    }
    const mode = $('mode').value;
    if (mode === 'persistent' && !persistentAvailable) {
      clearResult(); showError('Persistent hum is unavailable in this processor. Select another available WAV mode.'); return;
    }
    if (mode === 'compressed' && !compressedAvailable) {
      clearResult(); showError('MP3/M4A processing is unavailable in this environment. Choose a WAV processing mode.'); return;
    }
    const isLong = mode === 'long';
    const isCompressed = mode === 'compressed';
    const isProfile = mode === 'profile';
    const isPersistent = mode === 'persistent';
    const isStoredFile = usesStoredFiles(mode);
    let profileConfig = null;
    const persistentDepth = Number($('persistent-depth').value);
    if (isPersistent && (!Number.isInteger(persistentDepth) || persistentDepth < 0 || persistentDepth > 18)) {
      clearResult(); showError('Set persistent-hum attenuation to an integer from 0 to 18 dB.'); return;
    }
    if (isProfile) {
      const requestedSource = isExample ? 'example' : 'user_file';
      if ($('profile-confirm').checked && profileConfirmedSource !== requestedSource) {
        clearResult();
        $('profile-confirm').checked = false;
        profileConfirmedSource = null;
        showError(isExample
          ? 'Your current confirmation is for your own file. To run the synthetic example, use the 0–2 s example interval button and confirm the example separately.'
          : 'Your current confirmation is for the synthetic example. Enter the noise-only start and end you checked in your own file, then confirm again.');
        return;
      }
      if (!isExample && exampleIntervalOnly) {
        clearResult(); $('profile-confirm').checked = false;
        profileConfirmedSource = null;
        showError('The example interval cannot verify noise in your own file. Enter the start and end of an interval you checked in your own file.'); return;
      }
      try { profileConfig = profileParameters(); }
      catch (error) { clearResult(); showError(error.message); return; }
    }
    const orders = [1,2,3].filter(n => $('harmonic-' + n).checked);
    if ($('mode').value === 'harmonics' && !orders.length) {
      clearResult(); showError('Select at least one fundamental or harmonic band to process.'); return;
    }
    const bandDepths = [1,2,3].map(n => Number($('band-depth-' + n).value));
    if (usesBandControls($('mode').value) && !bandDepths.every(value => Number.isInteger(value) && value >= 0 && value <= 12)) {
      clearResult(); showError('Set each band attenuation to an integer from 0 to 12 dB. All bands may be set to 0 dB.'); return;
    }
    clearResult();
    busy = true;
    dismissStopConfirmation();
    updateDesktopControls();
    $('settings-fields').disabled = true;
    $('empty-state').hidden = true;
    $('busy-state').hidden = false;
    $('result-state').textContent = 'Processing';
    $('result-state').className = 'state-label busy';
    $('status').textContent = isStoredFile
      ? `Processing ${modes[mode]} on this computer. File length and resampling may require several minutes; this is not real-time processing.`
      : 'Processing on this computer. Files are not sent elsewhere.';
    $('busy-copy').textContent = isPersistent
      ? 'Offline whole-file analysis and FFT processing use memory. There is no progress percentage, automatic playback or real-time microphone processing.'
      : isStoredFile
      ? '44.1 kHz round-trip FIR resampling may take longer. No progress percentage is available, and playback does not start automatically when processing finishes.'
      : 'There is no automatic playback. Processing time depends on file length and mode.';
    const parameters = new URLSearchParams(isStoredFile
      ? {hum_hz: $('hum-hz').value}
      : {mode, hum_hz: $('hum-hz').value});
    if (isPersistent) parameters.set('attenuation_db', String(persistentDepth));
    else if (isProfile) {
      for (const [key, value] of Object.entries(profileConfig)) parameters.set(key, value);
    } else if (usesBandControls($('mode').value)) parameters.set('band_depths_db', bandDepths.join(','));
    else {
      parameters.set('attenuation_db', $('attenuation').value);
      if ($('mode').value === 'harmonics') parameters.set('orders', orders.join(','));
    }
    const sourceName = isExample ? `Generated ${isLong ? '65' : ((isProfile || isPersistent) ? '8' : '6')}-second synthetic example` : selectedFile.name;
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), isStoredFile ? 900000 : 120000);
    try {
      const requestHeaders = {'Content-Type': isCompressed ? 'application/octet-stream' : 'audio/wav', 'X-Hum-Token': token};
      if (isCompressed && !isExample) requestHeaders['X-Hum-Source-Format'] = /\.mp3$/i.test(selectedFile.name) ? 'mp3' : 'm4a';
      const response = await fetch(`/api/${isStoredFile ? mode + '/' : ''}${isExample ? 'example' : 'process'}?${parameters}`, {
        method: 'POST', headers: requestHeaders,
        body: isExample ? new Uint8Array(0) : selectedFile,
        signal: abort.signal, credentials: 'same-origin', cache: 'no-store'
      });
      const result = await response.json();
      if (!response.ok) throw new Error(serverErrorMessage(typeof result.error === 'string' ? result.error : 'Processing failed.', response.status, mode));
      const waveforms = result.waveforms || (isPersistent ? result.report?.waveforms : null);
      if (!result.report || !waveforms || !modes[result.mode] || result.mode !== mode) throw new Error('Invalid processing response format.');
      if (isStoredFile && (!Number.isFinite(result.duration_seconds) || result.duration_seconds <= 0 || result.duration_seconds > (isPersistent ? 60 : 600))) throw new Error('Invalid decoded-audio/WAV duration response.');
      const inputURL = isStoredFile ? localAssetURL(result.input_wav_url, 'input.wav') : makeBlobURL(result.input_wav_base64, 'audio/wav');
      const outputURL = isStoredFile ? localAssetURL(result.output_wav_url, 'processed.wav') : makeBlobURL(result.output_wav_base64, 'audio/wav');
      const zipURL = isStoredFile ? localAssetURL(result.download_zip_url, 'result.zip') : makeBlobURL(result.download_zip_base64, 'application/zip');
      audio[0].src = inputURL;
      audio[1].src = outputURL;
      $('download-wav').href = outputURL;
      $('download-zip').href = zipURL;
      const report = result.report;
      if (isProfile) renderProfileSummary(report);
      if (isPersistent) renderPersistentSummary(report);
      $('source-name').textContent = sourceName;
      $('result-settings').classList.toggle('band-settings', usesBandControls(result.mode));
      if (usesBandControls(result.mode)) {
        const depths = report.config.band_depths_db;
        if (!Array.isArray(depths) || depths.length !== 3 || !depths.every(value => Number.isFinite(value) && value >= 0 && value <= 12)) throw new Error('Invalid per-band processing settings response.');
        $('result-settings').textContent = `${modes[result.mode]} · ` + depths.map((depth, index) => `${report.config.nominal_hum_hz * (index + 1)}Hz ${depth}dB${depth === 0 ? ' (off)' : ''}`).join(' · ');
      } else {
        const orderLabel = result.mode === 'harmonics' ? ` · orders ${report.config.orders.join('·')}` : (isProfile ? ` · ${report.config.orders.length} candidate bands` : '');
        $('result-settings').textContent = `${modes[result.mode]} · ${report.config.nominal_hum_hz}Hz${orderLabel} · setting ${report.config.max_attenuation_db}dB`;
      }
      const inputRate = report.input.sample_rate_hz;
      const outputRate = report.output.sample_rate_hz;
      const inputChannels = report.input.channels ?? 1;
      const outputChannels = report.output.channels ?? 1;
      if (![inputRate, outputRate].every(value => Number.isInteger(value) && value > 0)
        || ![inputChannels, outputChannels].every(value => value === 1 || value === 2)) throw new Error('Invalid sample-rate or channel response.');
      if (isPersistent && (![44100,48000].includes(inputRate) || outputRate !== inputRate
        || outputChannels !== inputChannels || report.output.frames !== report.input.frames
        || !['signed_PCM16_little_endian','signed_PCM24_little_endian','IEEE_FLOAT32_little_endian'].includes(report.input.sample_format)
        || report.output.sample_format !== report.input.sample_format)) throw new Error('Persistent input and output native formats or frame counts do not match.');
      if (isCompressed && (report.input.sample_format !== 'IEEE_FLOAT32_little_endian'
        || report.output.sample_format !== 'IEEE_FLOAT32_little_endian'
        || ![44100, 48000].includes(inputRate) || outputRate !== inputRate
        || outputChannels !== inputChannels || report.output.frames !== report.input.frames)) throw new Error('Decoded input and processed WAV formats or lengths do not match.');
      $('input-track-title').textContent = isCompressed ? 'Decoded input' : 'Input';
      $('ab-a').textContent = isCompressed ? 'A · Decoded input' : 'A · input';
      $('input-track-description').textContent = isCompressed ? 'Float32 WAV decoded from the compressed file' : 'Unmodified input';
      $('output-track-title').textContent = isCompressed ? 'Processed WAV' : 'Output';
      $('input-peak-label').textContent = isCompressed ? 'Decoded input peak' : 'Input peak';
      $('input-waveform').setAttribute('aria-label', isCompressed ? 'Decoded input WAV waveform' : 'Input audio waveform');
      audio[0].setAttribute('aria-label', isCompressed ? 'Play decoded input WAV manually' : 'Play input manually');
      $('listen-note').textContent = isCompressed
        ? 'Start playback yourself at a low listening level. A is the decoded input Float32 WAV, not the compressed file; B is also Float32 WAV. If the browser cannot play them, save the files and use an audio editor. Playing A or B stops the other track.'
        : 'Start playback yourself at a low listening level. Playing A or B stops the other track. Browser playback depends on the output device and is separate from numerical evaluation of the source files.';
      $('duration-stat').textContent = `${(report.input.frames / inputRate).toFixed(2)} s`;
      $('format-stat').textContent = `${isCompressed ? 'Decoded input' : 'Input'} ${formatRate(inputRate)} · ${formatChannels(inputChannels)} · ${formatSamples(report.input)} → ${isCompressed ? 'Processed WAV' : 'Output'} ${formatRate(outputRate)} · ${formatChannels(outputChannels)} · ${formatSamples(report.output)}`;
      $('input-peak').textContent = report.input.peak_abs.toFixed(4);
      $('output-peak').textContent = report.output.peak_abs.toFixed(4);
      $('elapsed-stat').textContent = `${report.timing.file_processing_seconds.toFixed(2)} s`;
      $('report-json').textContent = JSON.stringify(report, null, 2);
      lastReport = report;
      lastReviewContext = {name: sourceName, example: isExample, mode, content_identity: 'see_embedded_processing_report_not_a_listening_attestation'};
      lastWaveforms = waveforms;
      lastChannels = inputChannels;
      $('results').hidden = false;
      abReview?.setPair({input: report.input, output: report.output});
      drawWaveforms();
      $('result-state').textContent = 'Complete';
      $('result-state').className = 'state-label';
      $('status').textContent = isStoredFile
        ? `${modes[mode]} processing complete. Save the files you need before starting another stored-file job (Long WAV, High-resolution WAV, MP3/M4A or Persistent hum), or before a normal processor shutdown. Temporary copies may remain after forced termination. This is not an audio-quality assessment.`
        : (isProfile ? 'Experimental result created. Inspect applied/withheld bands and your declared noise-only interval, then compare by listening. This is not an automatic quality assessment.' : 'A new processed result is ready. Listen and compare, or save the files. This is not an audio-quality assessment.');
      if (isProfile && isExample) {
        // An assertion about a known synthetic fixture never carries over to a user's upload.
        resetProfileDeclaration();
        $('status').textContent += ' The example interval and confirmation were cleared so they cannot carry over to your own file.';
      }
    } catch (error) {
      clearResult();
      showError(error.name === 'AbortError'
        ? 'The response timed out. Server processing may still be running. Wait a moment, then try a shorter file.'
        : (error instanceof TypeError ? 'Could not connect to the local processor. Check that its launcher window is still open.' : (error.message || 'Could not read the processing result. Try again.')));
    } finally {
      clearTimeout(timer);
      busy = false;
      updateSettingsControls();
      updateDesktopControls();
      $('settings-fields').disabled = false;
      $('process-button').disabled = !selectedFile;
      updateFilePreflight();
      $('busy-state').hidden = true;
      if (isProfile && isExample) resetProfileDeclaration();
      updateListeningReview();
    }
  }
  $('processor-form').addEventListener('submit', (event) => { event.preventDefault(); void run(false); });
  $('example-button').addEventListener('click', () => { void run(true); });
  window.addEventListener('beforeunload', () => { fileCheckSerial += 1; if (fileCheckTimer !== null) clearTimeout(fileCheckTimer); abReview?.suspend(); for (const url of urls) URL.revokeObjectURL(url); });
  updateSettingsControls();
  void loadCapabilities();
})();
