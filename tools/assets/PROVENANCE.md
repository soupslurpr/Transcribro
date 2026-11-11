# Vendored asset provenance

## silero-vad.ort

The inner Silero VAD model of the Moonshine Medium streaming recognizer, in
ONNX Runtime `.ort` flatbuffer form. The model converter
(`tools/src/convert.rs`) embeds its values into every converted
`medium.trm`; the bytes are verified on every model run against the SHA-256
pinned as `vad.asset_sha256` in `speech/native-inputs.lock.json` and against
the identical value compiled into the converter (`VAD_SHA256`).

- Bytes: 2,421,272
- SHA-256: `a620103ef77fbdae23c7798b9271506a48a883997e55f851d2884551e0947437`

### Origin

Extracted 2026-11-10 from `core/silero-vad-model-data.h` in the pinned
Moonshine tree:

- Repository: moonshine-ai/moonshine @
  `234f60faa0eb388b01cdf7e60aca232af37aefda`
- Tarball SHA-256:
  `6bd927a48234dced3cc6f208522fe176c248686e9283273c51eddc8d7994e001`
- Header SHA-256:
  `df9af551f0c4a4afffed0c64fae265122fa55fcd8247df0f64cc37a0c8551884`

The header is generated upstream by `scripts/generate-silero-vad-data.py`:
Silero VAD's `silero_vad.onnx` converted to the `.ort` format (optimization
level: disable) and emitted as the C byte array `silero_vad_onnx`. The
extraction applied here was the same mechanical scan the converter
previously ran at conversion time — every `0xNN` literal in the header
text, in order — with the raw bytes stored unchanged. Before this
vendoring, every fresh setup downloaded the 176,944,864-byte Moonshine
tarball for this one file; the tarball pin is gone from the lock.

### License

The file was distributed in the Moonshine repository under its MIT terms
(LICENSE Section 1; Copyright (c) 2025 Useful Sensors, Inc. (dba Moonshine
AI)). The full upstream license text is reproduced in the repo-root
`CREDITS`, section "Moonshine Medium model and streaming policy". The
underlying Silero VAD model is MIT-licensed by the Silero Team (Copyright
(c) 2020-present Silero Team); see the "Silero VAD model" section of
`CREDITS`.
