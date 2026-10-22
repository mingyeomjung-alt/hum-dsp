"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const token = document.querySelector('meta[name="hum-session-token"]').content;
  const audio = [$('input-audio'), $('output-audio')];
  const modes = {fixed: '고정 필터', tracked: '제한 추적', guarded: '보호 추적', harmonics: '고조파 처리', bands: '대역별 조절', compatible: '일반 WAV', long: '긴 WAV', precision: '고해상도 WAV', compressed: 'MP3·M4A → WAV', profile: '실험 · 잡음 구간 기준 대역 보호'};
  const help = {fixed: '선택한 주파수를 고정해서 처리합니다.', tracked: '±1Hz 안에서 최대 0.05Hz/초의 느린 변화를 추적합니다. 최초 관측은 2초입니다.', guarded: '두 번의 추적 확인 후 서서히 적용합니다. 복귀도 지연되며, 초기 감쇠 성능은 낮아질 수 있습니다.', harmonics: '선택한 기본·배수 대역을 각각 관측하고 확인된 대역만 서서히 처리합니다. 실제 고조파 관계나 잡음 여부를 판별하지는 않습니다.', bands: '기본·2배·3배 대역의 감쇠량을 따로 정합니다. 0 dB 대역은 끄며, 원하는 저음 보호는 직접 듣고 조절해야 합니다.', compatible: '44.1·48kHz의 모노·스테레오 PCM16 WAV를 대역별로 처리합니다. 아래 변환 안내를 확인하고 필요한 대역만 직접 조절하세요.', long: '44.1·48kHz의 모노·스테레오 PCM16 WAV를 최대 10분까지 연속 처리합니다. 44.1kHz는 선택한 대역이 있을 때 왕복 FIR 변환하며 실시간 처리는 아닙니다.', precision: 'PCM16·PCM24·Float32의 원본 형식을 유지해 대역별로 처리합니다. 최대 10분·128MiB이며 Float32 입력 범위와 변환 안내를 확인해 주세요.', compressed: 'MP3·AAC M4A를 로컬에서 Float32 WAV로 디코딩한 뒤 대역별로 처리합니다. 압축 파일은 최대 32MiB, 디코딩 후 최대 10분 또는 128MiB 중 먼저 닿는 한도입니다. 출력은 WAV만 제공합니다.'};
  help.profile = '잡음만 있는 최소 2초를 직접 지정하면, 1~32배 중 확인된 대역에 한해 처리합니다. 이후 소리가 커지면 대역 전체를 보류하는 오프라인 실험 기능입니다.';
  let selectedFile = null;
  let busy = false;
  let urls = [];
  let lastWaveforms = null;
  let lastChannels = 1;
  let compressedAvailable = false;
  let desktopAvailable = false;
  let stopping = false;
  let stopRequested = false;
  let lastReport = null;
  let lastReviewContext = null;
  let exampleIntervalOnly = false;
  let profileConfirmedSource = null;
  const interactionLocked = () => busy || stopping || stopRequested;

  function updateDesktopControls() {
    $('desktop-controls').hidden = !desktopAvailable;
    $('desktop-stop').disabled = interactionLocked();
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
      if (!response.ok || value.schema_version !== 1 || value.version !== '0.0.14'
        || typeof value.desktop?.available !== 'boolean' || typeof value.desktop?.portable_wav_only !== 'boolean'
        || typeof value.compressed?.available !== 'boolean' || value.compressed.required_version !== '>=19,<20'
        || (value.desktop.portable_wav_only && value.compressed.available)) throw new Error('capabilities');
      compressedAvailable = value.compressed.available;
      desktopAvailable = value.desktop.available;
      const notice = $('capability-notice');
      notice.hidden = compressedAvailable;
      notice.textContent = value.desktop.portable_wav_only
        ? '이 배포판은 WAV 전용입니다. MP3·M4A 기능은 포함하지 않았습니다.'
        : '현재 환경에 지원되는 MP3·M4A 디코더가 없습니다. WAV는 계속 사용할 수 있습니다.';
    } catch {
      compressedAvailable = false;
      desktopAvailable = false;
      $('capability-notice').hidden = false;
      $('capability-notice').textContent = '실행 환경을 확인하지 못했습니다. WAV는 계속 사용할 수 있습니다. MP3·M4A 기능은 비활성화했습니다.';
    } finally {
      clearTimeout(timer);
      $('mode').querySelector('option[value="compressed"]').disabled = !compressedAvailable;
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
    for (const id of ['download-wav', 'download-zip']) $(id).setAttribute('aria-disabled', 'true');
    $('desktop-stop-status').hidden = false;
    $('desktop-stop-status').textContent = '종료를 요청하고 있습니다. 처리 설정과 재생을 잠시 멈췄습니다.';
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
      $('desktop-stop-status').textContent = '종료를 요청했습니다. 정상 종료 후 임시 사본이 정리됩니다. 이 창을 닫아도 됩니다.';
      $('status').textContent = '프로그램 종료를 요청했습니다. 임시 사본의 정리 완료를 이 화면에서 확인하지는 않습니다.';
      $('result-state').textContent = '종료 요청';
    } catch {
      $('desktop-stop-status').textContent = '종료 응답을 확인하지 못했습니다. 실행 창을 확인해 주세요. 정상 종료 여부와 임시 사본의 정리는 확인되지 않았습니다.';
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

  const usesStoredFiles = (mode) => mode === 'long' || mode === 'precision' || mode === 'compressed';
  const usesBandControls = (mode) => mode === 'bands' || mode === 'compatible' || usesStoredFiles(mode);
  const fileLimitMiB = (mode) => mode === 'compressed' ? 32 : (usesStoredFiles(mode) ? 128 : 32);
  const formatRate = (rate) => `${rate / 1000}kHz`;
  const formatChannels = (channels) => channels === 2 ? '스테레오 · 2채널' : '모노 · 1채널';
  function formatSamples(metadata) {
    const name = metadata.sample_format;
    if (name === 'signed_PCM24_little_endian') return 'PCM24';
    if (name === 'IEEE_FLOAT32_little_endian') return 'Float32';
    if (name === undefined || name === 'signed_PCM16_little_endian') return 'PCM16';
    throw new Error('오디오 샘플 형식 응답이 올바르지 않습니다.');
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
    for (const player of audio) { player.pause(); player.removeAttribute('src'); player.load(); }
    for (const url of urls) URL.revokeObjectURL(url);
    urls = [];
    lastWaveforms = null;
    lastChannels = 1;
    lastReport = null;
    lastReviewContext = null;
    $('profile-result').hidden = true;
    $('profile-band-list').replaceChildren();
    resetListeningReview();
    $('results').hidden = true;
    $('empty-state').hidden = false;
    $('error').hidden = true;
    $('report-json').textContent = '';
    $('format-stat').textContent = '';
    for (const id of ['download-wav', 'download-zip']) $(id).removeAttribute('href');
    $('result-state').textContent = '준비';
    $('result-state').className = 'state-label';
  }

  function setFile(file) {
    if (interactionLocked()) return;
    clearResult();
    resetProfileDeclaration();
    selectedFile = file || null;
    $('file-name').textContent = file ? file.name : '아직 선택한 파일이 없습니다';
    $('process-button').disabled = !selectedFile;
    $('status').textContent = file ? '설정을 확인한 뒤 선택한 파일을 처리하세요.' : '파일을 선택하거나 합성 예제로 시작하세요.';
    const validationError = file ? validateFile(file, $('mode').value) : null;
    if (validationError) {
      selectedFile = null;
      $('process-button').disabled = true;
      showError(validationError);
    }
  }

  function validateFile(file, mode) {
    if (mode === 'compressed') {
      if (!compressedAvailable) return 'MP3·M4A 기능을 사용할 수 없는 실행 환경입니다. WAV 파일과 WAV 처리 방식을 선택해 주세요.';
      if (/\.(mp3|m4a)$/i.test(file.name) && file.size > 0 && file.size <= fileLimitMiB(mode) * 1024 * 1024) return null;
      return '0바이트보다 크고 32MiB 이하인 MP3 또는 AAC M4A 파일을 선택해 주세요. 디코딩 후 최대 10분·44.1/48kHz·모노/스테레오만 지원하며, ALAC M4A는 지원하지 않습니다.';
    }
    if (/\.wav$/i.test(file.name) && file.size > 0 && file.size <= fileLimitMiB(mode) * 1024 * 1024) return null;
    const format = mode === 'precision'
      ? '고해상도 WAV 모드는 최대 10분·44.1·48kHz·모노·스테레오·PCM16·PCM24·Float32를 지원합니다.'
      : mode === 'long'
      ? '긴 WAV 모드는 최대 10분·44.1·48kHz·모노·스테레오·PCM16만 지원합니다.'
      : (mode === 'compatible' ? '일반 WAV 모드는 44.1·48kHz 모노·스테레오 PCM16만 지원합니다.' : '이 모드는 48kHz·모노·PCM16만 지원합니다. 스테레오나 44.1kHz 파일은 일반 WAV 모드를 선택하세요.');
    return `0바이트보다 크고 ${fileLimitMiB(mode)}MiB 이하인 WAV 파일을 선택해 주세요. ${format}`;
  }

  function showError(message) {
    $('error').textContent = message;
    $('error').hidden = false;
    $('result-state').textContent = '확인 필요';
    $('result-state').className = 'state-label failed';
    $('status').textContent = '처리 결과를 만들지 않았습니다. 입력이나 설정을 확인해 주세요.';
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
    if (files.length !== 1) { setFile(null); showError('한 번에 오디오 파일 하나만 선택해 주세요.'); return; }
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
    $('attenuation-options').hidden = bandControls || mode === 'profile';
    $('profile-options').hidden = mode !== 'profile';
    $('compatible-notice').hidden = mode !== 'compatible';
    $('long-notice').hidden = mode !== 'long';
    $('precision-notice').hidden = mode !== 'precision';
    $('compressed-notice').hidden = mode !== 'compressed';
    $('audio-file').accept = mode === 'compressed' ? '.mp3,.m4a,audio/mpeg,audio/mp4,audio/x-m4a' : '.wav,audio/wav,audio/x-wav';
    $('file-type-label').textContent = mode === 'compressed' ? 'MP3·M4A 파일 선택' : 'WAV 파일 선택';
    $('input-contract').textContent = mode === 'compressed'
      ? 'MP3·AAC M4A · 최대 32MiB. 디코딩 후 44.1/48kHz · 모노/스테레오 · 최대 10분 또는 128MiB 중 먼저 닿는 한도. 출력은 Float32 WAV이며 MP3·M4A로 다시 압축하지 않습니다.'
      : mode === 'precision'
      ? '44.1·48kHz · 모노·스테레오 · PCM16·PCM24·Float32 WAV · 최대 10분 또는 128MiB. 원본 형식과 길이를 유지하며, 고해상도 스테레오는 크기 한도에 먼저 닿을 수 있습니다.'
      : mode === 'long'
      ? '44.1·48kHz · 모노·스테레오 · PCM16 WAV · 최대 10분 · 128MiB. 샘플 수·샘플레이트·채널 수를 유지합니다. 44.1kHz 변환 안내를 확인하세요.'
      : mode === 'compatible'
      ? '44.1·48kHz · 모노·스테레오 · PCM16 WAV · 최대 60초 · 32MiB. 아래 안내대로 샘플레이트를 변환하며 채널 수를 유지합니다.'
      : '48kHz · 모노 · PCM16 · 최대 60초 · 32MiB. 다른 형식은 자동 변환하지 않습니다.';
    for (let n = 1; n <= 3; n++) {
      const label = `${Number($('hum-hz').value) * n} Hz · ${n === 1 ? '기본' : n + '배'}`;
      $('harmonic-label-' + n).textContent = label;
      $('band-label-' + n).textContent = label;
      $('band-depth-value-' + n).textContent = `${$('band-depth-' + n).value} dB`;
    }
    $('attenuation-value').textContent = `${$('attenuation').value} dB`;
    $('profile-depth-value').textContent = `${$('profile-depth').value} dB`;
    const profileBase = Number($('hum-hz').value);
    $('profile-range').textContent = `${profileBase}Hz의 1~32배: ${profileBase}~${(profileBase * 32).toLocaleString('ko-KR')}Hz · 대역마다 ±1Hz`;
    $('band-depth-help').textContent = mode === 'compressed'
      ? '기본 설정은 6·3·0 dB입니다. 모두 0 dB이면 디코딩 원본 WAV의 오디오 데이터 값을 유지합니다. 압축 파일 보존·압축 전 원음 복원이 아니며, 설정값은 실제 잡음 감소량이 아닙니다.'
      : '기본 설정은 6·3·0 dB입니다. 모두 0 dB이면 원본 그대로이며, 설정값은 실제 잡음 감소량의 측정값이 아닙니다.';
    $('example-scope').textContent = mode === 'profile'
      ? '실험 모드 예제는 8초·48kHz·모노 PCM16 합성음입니다. 첫 0~2초에는 직접 만든 잡음만 있고, 2초부터 원하는 합성음이 더해집니다. 구간 0~2초를 입력하고 예제에 대한 확인 체크를 직접 해 주세요. 내 파일의 잡음 구간을 확인한 것은 아닙니다.'
      : mode === 'compressed'
      ? '6초·48kHz·스테레오 합성음을 로컬에서 MP3로 인코딩·디코딩한 예제입니다. PyAV 19.x가 없으면 실행되지 않으며 자동 설치하지 않습니다. 감쇠 설정과 무관하게 같은 합성 입력을 사용하며 주파수를 바꾸면 예제도 바뀝니다.'
      : mode === 'precision'
      ? '고해상도 예제는 직접 만든 6초·48kHz·스테레오 PCM24이며 Float32·44.1kHz 시험은 아닙니다. 감쇠 설정과 무관하게 같은 합성 입력을 사용하며, 주파수를 바꾸면 예제도 바뀝니다.'
      : mode === 'long'
      ? '긴 WAV 예제는 65초·48kHz·스테레오이며 샘플레이트를 변환하지 않습니다. 44.1kHz 변환을 확인하려면 해당 WAV를 직접 선택하세요. 감쇠 설정과 무관하게 같은 합성 입력을 사용하며, 주파수를 바꾸면 예제도 바뀝니다.'
      : bandControls
      ? `${mode === 'compatible' ? '일반 WAV 예제는 48kHz 스테레오입니다. ' : ''}감쇠 설정과 무관하게 같은 6초 합성 입력을 사용합니다. 주파수를 바꾸면 예제도 바뀝니다.`
      : '예제는 선택한 모드·대역에 맞춰 생성됩니다.';
  }

  for (const id of ['mode', 'hum-hz', 'attenuation', 'harmonic-1', 'harmonic-2', 'harmonic-3', 'band-depth-1', 'band-depth-2', 'band-depth-3', 'profile-start', 'profile-end', 'profile-orders', 'profile-depth', 'profile-confirm']) {
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
      $('status').textContent = '설정이 바뀌었습니다. 새 설정으로 다시 처리해 주세요.';
      if (id === 'mode' && selectedFile) {
        const validationError = validateFile(selectedFile, $('mode').value);
        if (validationError) {
          selectedFile = null;
          $('audio-file').value = '';
          $('process-button').disabled = true;
          $('file-name').textContent = '이 모드에 맞는 파일을 다시 선택해 주세요';
          showError(validationError);
        }
      }
    });
  }
  $('band-bypass').addEventListener('click', () => {
    if (interactionLocked()) return;
    clearResult();
    for (let n = 1; n <= 3; n++) $('band-depth-' + n).value = '0';
    updateSettingsControls();
    $('status').textContent = $('mode').value === 'compressed'
      ? '모든 대역을 0 dB로 설정했습니다. 새로 처리하면 디코딩 원본 WAV의 오디오 데이터를 그대로 비교합니다. 압축 전 원음 복원은 아닙니다.'
      : '모든 대역을 0 dB로 설정했습니다. 새로 처리하면 원본 그대로 비교할 수 있습니다.';
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
    $('status').textContent = '합성 예제용 0~2초만 입력했습니다. 내 파일의 잡음 구간은 확인하지 않았습니다. 예제를 실행하려면 확인 체크를 직접 해 주세요.';
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
      throw new Error('같은 입력의 잡음 구간 시작·끝을 초 단위로 입력하세요. 최소 2초, 최대 끝 60초이며 48kHz 샘플 경계에 맞아야 합니다. 실제 파일 길이는 처리기가 확인합니다.');
    }
    if (!$('profile-confirm').checked) throw new Error('지정한 구간에 원하는 소리가 없음을 직접 확인한 뒤 체크해 주세요. 예제와 내 파일은 별개의 확인입니다.');
    const orders = $('profile-orders').value.trim();
    if (orders !== 'all') {
      const parts = orders.split(',');
      if (!parts.length || parts.length > 32 || !parts.every(part => /^(?:[1-9]|[12]\d|3[0-2])$/.test(part))
        || !parts.every((part, index) => index === 0 || Number(part) > Number(parts[index - 1]))) {
        throw new Error('분석할 배수는 all 또는 1~32의 서로 다른 정수를 작은 수부터 쉼표로 입력하세요. 예: 1,6,21,24');
      }
    }
    const depth = Number($('profile-depth').value);
    if (!Number.isInteger(depth) || depth < 0 || depth > 12) throw new Error('실험 모드 감쇠량은 0~12 dB 정수로 설정하세요.');
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
      throw new Error('실험 모드의 대역 보호 근거 응답이 올바르지 않습니다.');
    }
    $('profile-result').hidden = false;
    $('profile-result-state').textContent = summary.processing_bypassed ? '그대로 · 처리 생략' : `${summary.approved_orders.length}개 대역 적용 가능`;
    $('profile-result-interval').textContent = `직접 지정한 잡음 구간 ${summary.profile_start_seconds}~${summary.profile_end_seconds}초 · 잡음만 있는지는 기계가 검증하지 않았습니다.`;
    const reasons = {
      profile_unsupported: '지정 구간에서 성분을 확인하지 못함',
      whole_file_amplitude_exceeds_profile: '진폭 증가로 파일 전체에서 제외',
      profile_supported_amplitude_consistent: '기준 구간·진폭 조건 충족 · 음질 보장 아님',
    };
    const list = $('profile-band-list');
    list.replaceChildren();
    if (!summary.bands.length) {
      const item = document.createElement('p');
      item.textContent = summary.bypass_reason === 'zero_depth' ? '0 dB: 분석과 필터를 건너뛰어 원본 그대로입니다.' : '적용할 대역이 없어 원본 그대로입니다.';
      list.append(item);
    }
    for (const band of summary.bands) {
      if (!Number.isInteger(band.order) || band.order < 1 || band.order > 32 || typeof band.reason !== 'string') throw new Error('실험 모드 대역 응답이 올바르지 않습니다.');
      const row = document.createElement('div');
      row.className = 'profile-band-row';
      const label = document.createElement('strong');
      label.textContent = `${band.order}배 · ${report.config.nominal_hum_hz * band.order}Hz`;
      const state = document.createElement('span');
      state.textContent = band.approved ? '후보 적용' : (band.amplitude_guard_rejected ? '보호 보류' : '성분 미확인');
      state.className = band.approved ? 'profile-band-approved' : 'profile-band-withheld';
      const reason = document.createElement('small');
      reason.textContent = reasons[band.reason] || `상세 보고서 확인: ${band.reason}`;
      row.append(label, state, reason);
      list.append(row);
    }
  }

  function resetListeningReview() {
    for (const id of ['review-hum', 'review-wanted', 'review-artifacts', 'review-notes']) $(id).value = '';
    $('review-confirm').checked = false;
    $('review-export').disabled = true;
    $('review-status').textContent = '세 항목을 기록한 후 직접 확인을 체크하세요. 서버에 평가를 저장하지 않습니다.';
  }

  function updateListeningReview() {
    const complete = ['review-hum', 'review-wanted', 'review-artifacts'].every(id => $(id).value !== '');
    $('review-export').disabled = !lastReport || !complete || !$('review-confirm').checked || interactionLocked();
  }
  for (const id of ['review-hum', 'review-wanted', 'review-artifacts', 'review-notes']) {
    $(id).addEventListener('input', () => {
      $('review-confirm').checked = false;
      updateListeningReview();
      $('review-status').textContent = '기록이 바뀌었습니다. 내용을 확인한 뒤 직접 확인을 다시 체크하세요.';
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
    $('review-status').textContent = '개인 청감 기록의 저장을 요청했습니다. 기계 검증·음질 합격·상용 승인은 아닙니다.';
  });

  function makeBlobURL(encoded, type) {
    if (typeof encoded !== 'string') throw new Error('처리 응답에 필요한 파일이 없습니다.');
    const binary = atob(encoded);
    const buffer = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) buffer[i] = binary.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([buffer], {type}));
    urls.push(url);
    return url;
  }

  function localAssetURL(value, filename) {
    if (typeof value !== 'string' || !value.startsWith('/api/long/assets/')) throw new Error('로컬 결과 파일 주소가 올바르지 않습니다.');
    const url = new URL(value, window.location.origin);
    const expectedPath = new RegExp('^/api/long/assets/[a-f0-9]{48}/' + filename.replace('.', '\\.') + '$');
    if (url.origin !== window.location.origin || url.search || url.hash || !expectedPath.test(url.pathname)) throw new Error('로컬 결과 파일 주소가 올바르지 않습니다.');
    return url.href;
  }

  function paintWaveform(canvas, bins, bound, color) {
    if (!Array.isArray(bins) || !bins.length) throw new Error('파형 응답 형식이 올바르지 않습니다.');
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
      if (!Array.isArray(range) || range.length !== 2 || !range.every(Number.isFinite)) throw new Error('파형 값이 올바르지 않습니다.');
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
    $('waveform-scale').textContent = `${lastChannels === 2 ? 'L/R 전체 공통축 포락 · 채널별 최솟값·최댓값, 합산 아님' : '두 파형의 공통 진폭 축'} ±${max.toFixed(4)} · 재생 음량 보정 없음`;
  }
  window.addEventListener('resize', drawWaveforms);

  async function run(isExample) {
    if (interactionLocked() || (!isExample && !selectedFile)) return;
    const mode = $('mode').value;
    if (mode === 'compressed' && !compressedAvailable) {
      clearResult(); showError('MP3·M4A 기능을 사용할 수 없는 실행 환경입니다. WAV 처리 방식을 선택해 주세요.'); return;
    }
    const isLong = mode === 'long';
    const isCompressed = mode === 'compressed';
    const isProfile = mode === 'profile';
    const isStoredFile = usesStoredFiles(mode);
    let profileConfig = null;
    if (isProfile) {
      const requestedSource = isExample ? 'example' : 'user_file';
      if ($('profile-confirm').checked && profileConfirmedSource !== requestedSource) {
        clearResult();
        $('profile-confirm').checked = false;
        profileConfirmedSource = null;
        showError(isExample
          ? '현재 확인은 내 파일에 대한 것입니다. 합성 예제를 실행하려면 예제용 구간 0~2초 입력 버튼을 누르고 예제에 대해 다시 확인해 주세요.'
          : '현재 확인은 합성 예제에 대한 것입니다. 내 파일에서 직접 확인한 잡음 구간 시작·끝을 입력한 뒤 다시 확인해 주세요.');
        return;
      }
      if (!isExample && exampleIntervalOnly) {
        clearResult(); $('profile-confirm').checked = false;
        profileConfirmedSource = null;
        showError('예제용 구간은 내 파일의 잡음 확인으로 사용할 수 없습니다. 내 파일에서 직접 확인한 구간 시작·끝을 다시 입력하세요.'); return;
      }
      try { profileConfig = profileParameters(); }
      catch (error) { clearResult(); showError(error.message); return; }
    }
    const orders = [1,2,3].filter(n => $('harmonic-' + n).checked);
    if ($('mode').value === 'harmonics' && !orders.length) {
      clearResult(); showError('처리할 기본·배수 대역을 하나 이상 선택해 주세요.'); return;
    }
    const bandDepths = [1,2,3].map(n => Number($('band-depth-' + n).value));
    if (usesBandControls($('mode').value) && !bandDepths.every(value => Number.isInteger(value) && value >= 0 && value <= 12)) {
      clearResult(); showError('각 대역의 감쇠량은 0~12 dB 사이 정수로 설정해 주세요. 모두 0 dB로 설정해도 됩니다.'); return;
    }
    clearResult();
    busy = true;
    dismissStopConfirmation();
    updateDesktopControls();
    $('settings-fields').disabled = true;
    $('empty-state').hidden = true;
    $('busy-state').hidden = false;
    $('result-state').textContent = '처리 중';
    $('result-state').className = 'state-label busy';
    $('status').textContent = isStoredFile
      ? `${modes[mode]}를 내 컴퓨터에서 처리 중입니다. 파일 길이·변환에 따라 수분이 걸릴 수 있으며 실시간 처리는 아닙니다.`
      : '내 컴퓨터에서 처리 중입니다. 파일은 외부로 전송하지 않습니다.';
    $('busy-copy').textContent = isStoredFile
      ? '44.1kHz의 왕복 FIR 변환은 처리 시간이 더 길 수 있습니다. 진행률은 제공하지 않으며, 완료 후에도 자동 재생하지 않습니다.'
      : '자동 재생하지 않습니다. 길이와 모드에 따라 잠시 걸릴 수 있습니다.';
    const parameters = new URLSearchParams(isStoredFile
      ? {hum_hz: $('hum-hz').value}
      : {mode, hum_hz: $('hum-hz').value});
    if (isProfile) {
      for (const [key, value] of Object.entries(profileConfig)) parameters.set(key, value);
    } else if (usesBandControls($('mode').value)) parameters.set('band_depths_db', bandDepths.join(','));
    else {
      parameters.set('attenuation_db', $('attenuation').value);
      if ($('mode').value === 'harmonics') parameters.set('orders', orders.join(','));
    }
    const sourceName = isExample ? `직접 만든 ${isLong ? '65' : (isProfile ? '8' : '6')}초 합성 예제` : selectedFile.name;
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
      if (!response.ok) throw new Error(typeof result.error === 'string' ? result.error : '처리하지 못했습니다.');
      if (!result.report || !result.waveforms || !modes[result.mode] || result.mode !== mode) throw new Error('처리 응답 형식이 올바르지 않습니다.');
      if (isStoredFile && (!Number.isFinite(result.duration_seconds) || result.duration_seconds <= 0 || result.duration_seconds > 600)) throw new Error('디코딩·WAV 길이 응답이 올바르지 않습니다.');
      const inputURL = isStoredFile ? localAssetURL(result.input_wav_url, 'input.wav') : makeBlobURL(result.input_wav_base64, 'audio/wav');
      const outputURL = isStoredFile ? localAssetURL(result.output_wav_url, 'processed.wav') : makeBlobURL(result.output_wav_base64, 'audio/wav');
      const zipURL = isStoredFile ? localAssetURL(result.download_zip_url, 'result.zip') : makeBlobURL(result.download_zip_base64, 'application/zip');
      audio[0].src = inputURL;
      audio[1].src = outputURL;
      $('download-wav').href = outputURL;
      $('download-zip').href = zipURL;
      const report = result.report;
      if (isProfile) renderProfileSummary(report);
      $('source-name').textContent = sourceName;
      $('result-settings').classList.toggle('band-settings', usesBandControls(result.mode));
      if (usesBandControls(result.mode)) {
        const depths = report.config.band_depths_db;
        if (!Array.isArray(depths) || depths.length !== 3 || !depths.every(value => Number.isFinite(value) && value >= 0 && value <= 12)) throw new Error('대역별 처리 설정 응답이 올바르지 않습니다.');
        $('result-settings').textContent = `${modes[result.mode]} · ` + depths.map((depth, index) => `${report.config.nominal_hum_hz * (index + 1)}Hz ${depth}dB${depth === 0 ? ' (끔)' : ''}`).join(' · ');
      } else {
        const orderLabel = result.mode === 'harmonics' ? ` · ${report.config.orders.join('·')}배 대역` : (isProfile ? ` · ${report.config.orders.length}개 후보 대역` : '');
        $('result-settings').textContent = `${modes[result.mode]} · ${report.config.nominal_hum_hz}Hz${orderLabel} · 설정 ${report.config.max_attenuation_db}dB`;
      }
      const inputRate = report.input.sample_rate_hz;
      const outputRate = report.output.sample_rate_hz;
      const inputChannels = report.input.channels ?? 1;
      const outputChannels = report.output.channels ?? 1;
      if (![inputRate, outputRate].every(value => Number.isInteger(value) && value > 0)
        || ![inputChannels, outputChannels].every(value => value === 1 || value === 2)) throw new Error('샘플레이트·채널 응답이 올바르지 않습니다.');
      if (isCompressed && (report.input.sample_format !== 'IEEE_FLOAT32_little_endian'
        || report.output.sample_format !== 'IEEE_FLOAT32_little_endian'
        || ![44100, 48000].includes(inputRate) || outputRate !== inputRate
        || outputChannels !== inputChannels || report.output.frames !== report.input.frames)) throw new Error('디코딩 원본·처리 WAV의 형식 또는 길이가 일치하지 않습니다.');
      $('input-track-title').textContent = isCompressed ? '디코딩 원본' : '원본';
      $('input-track-description').textContent = isCompressed ? '압축 파일에서 디코딩한 Float32 WAV' : '변경하지 않은 입력';
      $('output-track-title').textContent = isCompressed ? '처리 WAV' : '처리본';
      $('input-peak-label').textContent = isCompressed ? '디코딩 원본 피크' : '원본 피크';
      $('input-waveform').setAttribute('aria-label', isCompressed ? '디코딩 원본 WAV 오디오 파형' : '원본 오디오 파형');
      audio[0].setAttribute('aria-label', isCompressed ? '디코딩 원본 WAV 수동 재생' : '원본 수동 재생');
      $('listen-note').textContent = isCompressed
        ? '낮은 음량부터 직접 재생하세요. A는 압축 파일 자체가 아닌 디코딩 원본 Float32 WAV이며 B도 Float32 WAV입니다. 브라우저가 재생하지 못하면 저장해 오디오 편집기에서 확인하세요. A 또는 B 재생 시 다른 쪽은 멈춥니다.'
        : '낮은 음량부터 직접 재생하세요. A 또는 B 재생 시 다른 쪽은 멈춥니다. 브라우저 재생은 출력 장치의 영향을 받으며, 원본 파일의 수치 평가와는 다릅니다.';
      $('duration-stat').textContent = `${(report.input.frames / inputRate).toFixed(2)} 초`;
      $('format-stat').textContent = `${isCompressed ? '디코딩 원본' : '원본'} ${formatRate(inputRate)} · ${formatChannels(inputChannels)} · ${formatSamples(report.input)} → ${isCompressed ? '처리 WAV' : '처리본'} ${formatRate(outputRate)} · ${formatChannels(outputChannels)} · ${formatSamples(report.output)}`;
      $('input-peak').textContent = report.input.peak_abs.toFixed(4);
      $('output-peak').textContent = report.output.peak_abs.toFixed(4);
      $('elapsed-stat').textContent = `${report.timing.file_processing_seconds.toFixed(2)} 초`;
      $('report-json').textContent = JSON.stringify(report, null, 2);
      lastReport = report;
      lastReviewContext = {name: sourceName, example: isExample, mode, content_identity: 'see_embedded_processing_report_not_a_listening_attestation'};
      lastWaveforms = result.waveforms;
      lastChannels = inputChannels;
      $('results').hidden = false;
      drawWaveforms();
      $('result-state').textContent = '처리 완료';
      $('result-state').className = 'state-label';
      $('status').textContent = isStoredFile
        ? `${modes[mode]} 처리를 마쳤습니다. 새 긴 WAV·고해상도 WAV·MP3/M4A 작업이나 처리기 정상 종료 전에 필요한 파일을 저장하세요. 강제 종료 시 임시 사본이 남을 수 있습니다. 음질 판정은 아닙니다.`
        : (isProfile ? '실험 결과를 만들었습니다. 대역별 적용·보류와 직접 지정한 잡음 구간을 확인하고 직접 비교하세요. 자동 음질 판정은 아닙니다.' : '새 처리본을 만들었습니다. 직접 듣고 비교하거나 파일로 저장하세요. 음질 판정은 아닙니다.');
      if (isProfile && isExample) {
        // An assertion about a known synthetic fixture never carries over to a user's upload.
        resetProfileDeclaration();
        $('status').textContent += ' 예제 확인은 내 파일로 이어지지 않도록 구간과 체크를 비웠습니다.';
      }
    } catch (error) {
      clearResult();
      showError(error.name === 'AbortError'
        ? '응답 대기 시간이 끝났습니다. 서버 처리는 아직 진행 중일 수 있습니다. 잠시 후 더 짧은 파일로 다시 시도해 주세요.'
        : (error instanceof TypeError ? '로컬 처리기에 연결하지 못했습니다. 실행 창이 열려 있는지 확인해 주세요.' : (error.message || '처리 결과를 읽지 못했습니다. 다시 실행해 주세요.')));
    } finally {
      clearTimeout(timer);
      busy = false;
      updateDesktopControls();
      $('settings-fields').disabled = false;
      $('process-button').disabled = !selectedFile;
      $('busy-state').hidden = true;
      if (isProfile && isExample) resetProfileDeclaration();
      updateListeningReview();
    }
  }
  $('processor-form').addEventListener('submit', (event) => { event.preventDefault(); void run(false); });
  $('example-button').addEventListener('click', () => { void run(true); });
  window.addEventListener('beforeunload', () => { for (const url of urls) URL.revokeObjectURL(url); });
  updateSettingsControls();
  void loadCapabilities();
})();
