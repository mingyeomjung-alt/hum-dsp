"use strict";
// Local manual comparison only: no decoding DSP, gain normalization or autoplay.
(() => {
  function create({players, getElement: $, isLocked}) {
    const controls = $('ab-controls');
    const status = $('ab-status');
    let armed = false, ready = false, selected = 0, epoch = 0, pairGeneration = 0;
    let expected = 0, urls = [], loaded = [false, false];
    let wanted = false, pending = false, range = null, timer = null;
    let loadTimer = null, playTimer = null;
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const ownSource = i => players[i].src === urls[i]
      && (!players[i].currentSrc || players[i].currentSrc === urls[i]);
    const position = () => finite(players[selected].currentTime) ? players[selected].currentTime : 0;
    const clamp = value => Math.min(expected, Math.max(0, value));
    const clock = value => `${Math.floor(value / 60)}:${(value % 60).toFixed(2).padStart(5, '0')}`;

    function clearTimer() {
      if (timer !== null) clearInterval(timer);
      timer = null;
    }
    function pause() {
      epoch += 1; wanted = false; pending = false; clearTimer();
      if (playTimer !== null) clearTimeout(playTimer);
      playTimer = null;
      for (const player of players) player.pause();
    }
    function refresh() {
      controls.disabled = !ready || isLocked();
      if (ready) for (const player of players) player.controls = false;
      $('ab-a').setAttribute('aria-pressed', String(selected === 0));
      $('ab-b').setAttribute('aria-pressed', String(selected === 1));
      $('ab-play').disabled = !ready || isLocked() || wanted;
      $('ab-pause').disabled = !ready || isLocked() || !wanted;
      $('ab-position').value = String(clamp(position()));
      $('ab-position-label').textContent = `${selected === 0 ? 'A' : 'B'} · ${clock(clamp(position()))} / ${clock(expected)}`;
    }
    function fail(message) {
      pause(); ready = false; armed = false; range = null;
      if (loadTimer !== null) clearTimeout(loadTimer);
      loadTimer = null;
      for (const player of players) player.controls = !isLocked();
      status.textContent = message + ' You can still save the WAV files and compare them in an external editor.';
      refresh();
    }
    function reset() {
      pause(); armed = false; ready = false; range = null;
      pairGeneration += 1;
      if (loadTimer !== null) clearTimeout(loadTimer);
      loadTimer = null;
      selected = 0; expected = 0; urls = []; loaded = [false, false];
      for (const player of players) { player.controls = true; player.preload = 'none'; }
      $('ab-start').value = ''; $('ab-end').value = '';
      $('ab-position').max = '0';
      status.textContent = 'Comparison is available after processing. Playback never starts automatically.';
      refresh();
    }
    function suspend() {
      pause(); range = null;
      status.textContent = 'Playback stopped. Press Play yourself to start again.';
      refresh();
    }
    function setPair(metadata) {
      reset();
      const a = metadata?.input, b = metadata?.output;
      if (!a || !b || !Number.isSafeInteger(a.frames) || a.frames <= 0
        || !Number.isInteger(a.sample_rate_hz) || a.sample_rate_hz <= 0
        || ![1, 2].includes(a.channels ?? 1)
        || b.frames !== a.frames || b.sample_rate_hz !== a.sample_rate_hz
        || (b.channels ?? 1) !== (a.channels ?? 1)) {
        fail('Shared-position comparison is unavailable because frame count, sample rate or channels do not match.'); return;
      }
      expected = a.frames / a.sample_rate_hz;
      if (!finite(expected) || expected <= 0 || expected > 600 || players.some(p => !p.src)) {
        fail('The comparison audio metadata is invalid.'); return;
      }
      urls = players.map(p => p.src); armed = true;
      $('ab-position').max = String(expected);
      $('ab-start').value = '0'; $('ab-end').value = String(expected);
      status.textContent = 'Checking browser playback metadata for both WAV files. Playback will not start automatically.';
      const generation = pairGeneration;
      loadTimer = setTimeout(() => {
        if (armed && !ready && generation === pairGeneration) fail('Browser playback metadata loading timed out.');
      }, 10000);
      for (const player of players) { player.controls = false; player.preload = 'metadata'; player.load(); }
      refresh();
    }
    function metadataLoaded(index) {
      if (!armed || !ownSource(index)) return;
      // load()/src replacement can emit durationchange(NaN) while the new
      // resource still has HAVE_NOTHING. It is not a failed metadata load.
      if (players[index].readyState < 1) return;
      if (!finite(players[index].duration) || Math.abs(players[index].duration - expected) > 0.02) {
        fail('The duration read by the browser does not match the processing report.'); return;
      }
      loaded[index] = true;
      if (loaded.every(Boolean)) {
        if (loadTimer !== null) clearTimeout(loadTimer);
        loadTimer = null;
        ready = true;
        for (const player of players) { player.controls = false; player.playbackRate = 1; }
        status.textContent = 'A/B buttons preserve the playback position. Switching during playback continues on the selected track only.';
        refresh();
      }
    }
    function seek(time, both = false) {
      try {
        for (const i of both ? [0, 1] : [selected]) {
          if (!ownSource(i)) throw new Error('stale source');
          players[i].currentTime = clamp(time);
        }
        return true;
      } catch {
        fail('The browser could not seek to the requested position.'); return false;
      }
    }
    function tick() {
      if (!ready || isLocked()) { suspend(); return; }
      if (range && (position() >= range.end || position() < range.start - 0.02)) {
        const end = range.end;
        pause(); range = null;
        seek(end, true);
        status.textContent = 'Selected-interval playback stopped. Browser stopping is not guaranteed to be sample-exact.';
      }
      refresh();
    }
    function start(time) {
      if (!ready || isLocked()) return;
      pause();
      const operation = epoch, index = selected, player = players[index];
      if (!seek(time)) return;
      wanted = true; pending = true;
      playTimer = setTimeout(() => {
        if (operation === epoch && pending) {
          pause(); range = null;
          status.textContent = 'Playback startup timed out. Try again manually or save the WAV files.';
          refresh();
        }
      }, 10000);
      clearTimer(); timer = setInterval(tick, 30);
      refresh();
      // This function is called only from explicit click handlers, never from
      // metadata/seek/promise callbacks. Browsers may reject delayed playback.
      try {
        const attempt = player.play();
        Promise.resolve(attempt).then(() => {
          if (operation !== epoch || !ready || !wanted || selected !== index || isLocked()) {
            if (!wanted || selected !== index || isLocked() || !ready) player.pause();
            return;
          }
          if (playTimer !== null) clearTimeout(playTimer);
          playTimer = null;
          pending = false; tick();
        }, () => {
          if (operation !== epoch) return;
          pause(); range = null;
          status.textContent = 'Browser playback failed. Try again manually or save the WAV files.';
          refresh();
        });
      } catch {
        if (operation === epoch) {
          pause(); range = null;
          status.textContent = 'The browser could not start playback. Try again manually or save the WAV files.';
          refresh();
        }
      }
    }
    function choose(index) {
      if (!ready || isLocked()) return;
      const time = clamp(position()), resume = wanted;
      pause(); selected = index;
      if (!seek(time)) return;
      if (resume) start(time);
      else {
        status.textContent = 'Switched to the same position. Press Play yourself to start playback.';
        refresh();
      }
    }
    function readRange() {
      const values = [$('ab-start').value, $('ab-end').value];
      if (values.some(value => !/^(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(String(value).trim()))) return null;
      const [start, end] = values.map(Number);
      return finite(start) && finite(end) && start >= 0 && end > start && end <= expected
        ? {start, end} : null;
    }
    $('ab-a').addEventListener('click', () => choose(0));
    $('ab-b').addEventListener('click', () => choose(1));
    $('ab-play').addEventListener('click', () => {
      if (!ready || isLocked()) return;
      range = null; start(position() >= expected ? 0 : clamp(position()));
    });
    $('ab-pause').addEventListener('click', () => {
      pause(); status.textContent = 'Paused.'; refresh();
    });
    $('ab-position').addEventListener('input', () => {
      if (!ready || isLocked()) return;
      const time = Number($('ab-position').value);
      pause(); range = null;
      if (finite(time)) seek(clamp(time), true);
      refresh();
    });
    $('ab-range-play').addEventListener('click', () => {
      if (!ready || isLocked()) return;
      pause(); range = readRange();
      if (!range) {
        status.textContent = 'Enter an interval in seconds with 0 ≤ start < end ≤ file duration.';
        refresh(); return;
      }
      if (seek(range.start, true)) start(range.start);
    });
    function clearRange() {
      if (isLocked()) return;
      pause(); range = null;
      status.textContent = 'Interval playback cleared. Press Play yourself to start again.';
      refresh();
    }
    $('ab-range-clear').addEventListener('click', clearRange);
    for (const id of ['ab-start', 'ab-end']) $(id).addEventListener('input', clearRange);
    $('ab-volume').addEventListener('input', () => {
      if (isLocked()) return;
      const value = Number($('ab-volume').value);
      if (!finite(value) || value < 0 || value > 1) return;
      for (const player of players) player.volume = value;
    });
    $('ab-mute').addEventListener('input', () => {
      for (const player of players) player.muted = $('ab-mute').checked;
    });
    for (const [i, player] of players.entries()) {
      player.addEventListener('loadedmetadata', () => metadataLoaded(i));
      player.addEventListener('durationchange', () => metadataLoaded(i));
      player.addEventListener('error', () => {
        if (armed && ownSource(i) && player.error) fail('The browser could not load the comparison audio.');
      });
      player.addEventListener('play', () => {
        if (!ready) {
          if (armed || isLocked() || !urls.length || !ownSource(i)) player.pause();
          return;
        }
        if (!wanted || selected !== i || isLocked() || !ownSource(i)) { player.pause(); return; }
        for (const [j, other] of players.entries()) if (j !== i) other.pause();
      });
      player.addEventListener('pause', () => {
        if (ready && selected === i && wanted && player.paused) {
          pause(); refresh();
        }
      });
      player.addEventListener('ended', () => {
        if (ready && selected === i && ownSource(i) && position() >= expected - 0.02) {
          pause(); range = null; refresh();
        }
      });
      player.addEventListener('timeupdate', () => {
        if (ready && selected === i) tick();
      });
      player.addEventListener('ratechange', () => { if (ready && player.playbackRate !== 1) player.playbackRate = 1; });
      player.addEventListener('volumechange', () => {
        for (const other of players) {
          if (other.volume !== player.volume) other.volume = player.volume;
          if (other.muted !== player.muted) other.muted = player.muted;
        }
        $('ab-volume').value = String(player.volume);
        $('ab-mute').checked = player.muted;
      });
    }
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && wanted) suspend();
    });
    reset();
    return {setPair, reset, suspend, refresh};
  }
  window.HumABReview = Object.freeze({create});
})();
