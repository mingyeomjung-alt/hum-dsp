# Persistent hum processing

`Persistent hum · offline` is an explicit, opt-in processor for steady 50/60 Hz
hum families whose higher harmonics may remain after the simpler fixed filter.
It is implemented locally with NumPy DSP. It does not call an external denoising
API, and it does not replace the default processing mode.

## Use it

Start the local interface and select **Persistent hum · offline**, then choose
50 or 60 Hz and an attenuation from 0 to 18 dB. You can process a local WAV or
run the built-in synthetic example. Playback is always manual.

The same processor is available from the command line:

```powershell
python tools/process_persistent_hum.py input.wav --output-dir results/new-001 --hum-hz 60
```

Or from Python:

```python
from hum_dsp.persistent_processing import process_persistent_file

result = process_persistent_file(
    "input.wav", output_dir="results/new-002", hum_hz=60, attenuation_db=12
)
print(result.output_wav)
print(result.report_path)
```

Use a new output directory for every command-line run. A successful run writes
`processed.wav`, `report.json`, and `persistent_trace.json`. Errors use the
existing `HumDSPError` contract. Nothing records, uploads, or automatically
plays audio.

Supported input is mono or stereo PCM16, packed PCM24, or normalized Float32
WAV at native 44.1 or 48 kHz, up to 60 seconds and 128 MiB. MP3 and M4A are not
accepted by this mode. The entire signal and FFT workspace are held in memory,
so this is an offline processor rather than a real-time or bounded-stream path.

## Algorithm

1. Analyze per-channel power in two-second Hann windows at one-second hops.
2. Search up to 32 harmonics of the selected 50/60 Hz family below 2 kHz.
3. Require each candidate to be at least 10 dB above local spectral flanks,
   present in at least 75% of windows, and stable within 0.25 Hz median absolute
   deviation.
4. Require at least four consistent harmonic orders. An isolated steady note is
   therefore insufficient.
5. Freeze the accepted centers and apply one real, nonnegative spectral mask to
   every channel. The mask uses a 1 Hz flat half-width and a cosine transition
   out to 3 Hz.
6. Reflect-pad by at least two seconds, process offline, and crop back to the
   original frame count.

If there is insufficient evidence, fewer than three analysis windows, or zero
attenuation, the validated audio payload is returned unchanged. Active results
preserve sample format, sample rate, channel count, and frame count. Invalid or
unrepresentable output is rejected rather than clipped or normalized.

## Fixed engineering controls

The initial frozen configuration passed the declared synthetic controls for
four-line 50 Hz and 60 Hz hum, opposite-phase stereo hum, and a 44.1 kHz stereo
piano-plus-artificial-hum development mixture. Known injected hum reduction was
about 11.7 dB in those controls. The piano control had about 0.93% relative RMS
error in its isolated wanted component.

An intentional collision control is equally important: when wanted audio
occupied a selected harmonic, the wanted component suffered about 59.6%
relative RMS error. This is severe damage and demonstrates why detected spectral
lines cannot be assumed to be unwanted.

These are controlled engineering measurements, not independent perceptual,
field, or commercial-quality results. They must not be generalized to arbitrary
recordings.

## Limits

- A stable wanted harmonic family can look like hum and be attenuated.
- Drifting, transient, broadband, or non-harmonic noise may remain unchanged.
- Accepted orders are spectral observations, not sound-source identification.
- Whole-file lookahead can introduce boundary, pre-ringing, or post-ringing
  artifacts even though the spectral mask is zero phase.
- Container metadata outside the audio payload is not guaranteed to survive.
- This mode is not real-time ANC, source separation, speech enhancement, or an
  automatic quality decision.

Always keep the original, compare manually at a safe listening level, and treat
an unchanged result as an abstention rather than proof that the recording is
clean.
