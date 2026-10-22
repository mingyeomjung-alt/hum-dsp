# Evaluation scope

Hum DSP v0.0.14 is released for personal, noncommercial **evaluation**, not as
a general-purpose or quality-certified denoiser. Publication permission and
measured audio quality are separate decisions.

## Reproducible implementation checks

The public test suite contains self-contained core DSP, input-validation,
frozen replay, synthetic fixture, report, local HTTP, desktop lifecycle and UI
contracts. It does not copy private recordings, downloaded datasets, model
weights or historical output folders.

Five internal tool-specific test files are not included:

- `test_portable_build.py` and `test_portable_runtime_qa.py`: internal build/QA tools.
- `test_h2_audio_preparation.py`: retained-audio preparation tool.
- `test_extended_cli.py` and `test_profile_cli.py`: repository-only research launchers.

Five individual historical-preflight tests were also omitted from the copied
evaluation test modules. They validate internal retained source/output history
rather than a self-contained public input. The remaining evaluation contracts
use synthetic temporary fixtures or mocks; no historical output files are
fabricated to make those preflight checks pass.

Some historical study modules remain in the source for provenance and unit
contracts; running a complete historical study requires its separately supplied
catalog, source permissions and history. Their existence is not a promise that
private historical study directories are distributed here.

The publication checklist will record the actual public-suite result and fresh
synthetic review evidence below. The internal 2,874-test total must not be
presented as the number of tests shipped in this public copy.

## Release verification record

On 2026-10-22 the frozen development study was rerun without changing DSP
parameters or targets. The 9 cases, 18 renders and 36 input/reference/output WAV
hash comparisons exactly reproduced the previous run. Declared structural and
development checks passed. Replay discrepancy was zero; maximum component
additivity discrepancy was approximately `9.79e-13`.

The following diagnostics use a **12 dB setting**, 48 kHz mono PCM16 input,
selected orders 1–32 and a declared 0–2 s noise-only interval. Results below are
for the **2–8 s program window**, measured in floating-point component replay
**before PCM16 output rounding**. The runtime does not receive the known clean
and noise components used for these measurements. No alignment or output-gain
fitting is applied.

| Development case | Noise attenuation, baseline → profile guard | Wanted-level change, baseline → profile guard | Interpretation |
| --- | --- | --- | --- |
| Selective 50 Hz | 10.691 → 10.691 dB | −0.000181 → −0.000181 dB | No improvement over baseline in this fixture |
| Selective 60 Hz | 10.690 → 10.690 dB | −2.236 → −0.000236 dB | Avoided a wanted-tone band in this fixture |
| Strong wanted overlap | 10.118 → 5.031 dB | −8.179 → −0.000032 dB | Wanted preservation trades away noise removal |
| Late strong overlap | 10.118 → 5.031 dB | −7.294 → −0.000036 dB | Uses future lookahead, not causal protection |
| Empty profile | Not defined | 0 dB; exact input copy | Abstention, not denoising success |
| Growing noise | 10.227 → **0 dB** | Guard returns exact input | Noise remains untreated |
| Masked overlap limit | Guard: 10.558 dB | Guard: **−9.699 dB** | Severe wanted-content damage |
| Contaminated profile limit | Not defined | Guard: **−10.687 dB** | Severe wanted loss from an invalid profile assertion |
| Recorded hum + piano construction | 1.273 → 1.273 dB | Guard: −0.000777 dB | Modest total noise reduction; no improvement over baseline |

Eight cases are owned deterministic synthetic signals. The final case is a new
construction from previously seen CC0 hum/piano recordings, **not independent
field data**. It does not establish general natural-noise effectiveness. The
recording files and raw reports are not redistributed here. Reviewed metrics,
definitions, source hashes and explicit failure cases are available in
[evaluation-summary.json](evaluation-summary.json).

Human listening, clean-PC coverage and commercial quality were **not** approved
by this study. The final public-suite count and binary publication audit are
recorded separately; no automated check should be read as a listening score.

The public release is **source-only**. The internal unsigned Windows binary is
withheld pending bundled-runtime rights/notice review, rather than presented as
cleared by a successful technical test.

### Public-source regression check

The sanitized public copy was tested on 2026-10-22 using Python **3.12.14**,
NumPy **2.5.3**, SciPy **1.18.1**, and the developer host's optional PyAV 19.x.
Result: **2,728 passed, 1 skipped**. The skip is a symlink test because this
Windows host cannot create the required symlink. The command was
`python -m pytest -q tests`, with a fresh temporary directory and local-loopback
network permission. No audio was played or recorded. The same public suite is
configured for Windows and Ubuntu CI; passing those jobs must be checked on
GitHub rather than inferred from this Windows run. CI does not install PyAV,
so optional codec tests skip there.

A locally built, pure-Python wheel was checked for exact equality of all **73**
runtime Python/policy/web files, packaged H2c/H2d policies and UI assets,
isolated import, and native profile bypass behavior. No source kernel was
changed for publication. This packaging check does not clear native-runtime
binary redistribution rights.

## Hard limitations, including failed cases

- A notch cannot identify whether energy at its target frequency is unwanted.
  Wanted bass, tones, music or speech components at that frequency can be lost.
- The experimental profile guard uses a user assertion that an interval is
  noise-only. It cannot verify that assertion.
- Whole-file amplitude veto uses future samples and can leave growing noise
  completely untreated. This is deliberate abstention, not successful removal.
- Equal-amplitude overlapping wanted content and contaminated profiles can still
  suffer substantial attenuation; no automatic preservation guarantee is made.
- The profile guard's fixed-frequency probes do not guarantee protection
  throughout every tracked frequency trajectory.
- Development mixtures and synthetic test signals are not independent field
  recordings, human listening scores or proof of broad generalization.
- Browser audio-format support differs. Processing-file contracts are distinct
  from whether a browser can preview PCM24 or Float32.
- An unsigned binary can trigger operating-system warnings. A developer-host
  relocation test is not a clean Windows PC test or a code-signing certificate.

Human listening and real-use quality remain unverified unless separately
reported with a concrete protocol and provenance. Never relabel an automated
pass count as an audio-quality score.
