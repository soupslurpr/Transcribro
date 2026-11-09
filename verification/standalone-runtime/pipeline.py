#!/usr/bin/env python3
"""Research streaming controller using only the native fixed-model interface.

The evaluation policy follows the approved app's Moonshine streaming path:
VAD smoothing, retained onset, 80 ms frontend blocks, stable preview windows,
complete final encoding, and bounded speculative decoding. This is an oracle
port for accuracy evaluation, not the eventual Android controller.
"""
import ctypes as ct
import math
from pathlib import Path

import numpy as np
from verify_components import Model, FrontendState


class VadState(ct.Structure):
    _fields_ = [
        ("context", ct.c_float * 64),
        ("hidden", ct.c_float * 128),
        ("cell", ct.c_float * 128),
    ]


class Pipeline:
    def __init__(self, library, archive, tokenizer):
        self.model = Model(library, archive)
        self.model.library.tm_vad.argtypes = [ct.c_void_p] * 4
        self.vad_state = VadState()
        self.vocabulary = []
        raw, at = Path(tokenizer).read_bytes(), 0
        while at < len(raw):
            size = raw[at]
            at += 1
            if size >= 128:
                if at >= len(raw):
                    raise ValueError("truncated tokenizer length")
                size = size - 128 + raw[at] * 128
                at += 1
            if size > len(raw) - at:
                raise ValueError("truncated tokenizer entry")
            self.vocabulary.append(raw[at : at + size])
            at += size
        if len(self.vocabulary) != 32768:
            raise ValueError("wrong Medium vocabulary")

    def close(self):
        self.model.close()

    def text(self, tokens):
        chunks = []
        for token in tokens:
            data = self.vocabulary[token]
            if not data:
                raise ValueError("empty vocabulary entry")
            if len(data) > 2 and data[:1] == b"<" and data[-1:] == b">":
                continue
            chunks.append(data)
        # The pinned English model uses the SentencePiece space marker.
        return (
            b"".join(chunks)
            .replace("▁".encode(), b" ")
            .strip()
            .decode("utf8", errors="replace")
        )

    def probability(self, chunk):
        result = ct.c_float()
        status = self.model.library.tm_vad(
            self.model.handle,
            chunk.ctypes.data,
            ct.byref(self.vad_state),
            ct.byref(result),
        )
        if status:
            raise RuntimeError(self.model.library.tm_error().decode())
        return np.float32(result.value)

    def transcribe(self, audio, previews=True):
        """Mirror the fixed app-pattern benchmark, including its silent drain."""
        audio = np.pad(np.asarray(audio, "f4"), (0, -len(audio) % 512))
        audio = np.concatenate([audio, np.zeros(8192, "f4")])
        probabilities = np.zeros(32, "f4")
        lookbehind = np.zeros(8192, "f4")
        segments = []
        active, processed, last_end, probability_index = False, 0, 0, 0
        current = np.empty(0, "f4")
        interval = 16000 if previews else 8000
        pending, analyzed, sequence = 0, 0, None
        # The caller adds 8192-sample chunks. A short last input waits for the
        # drain. Updates consume the entire pending buffer after the interval.
        ends = list(range(8192, audio.size - 8192 + 1, 8192)) + [audio.size]
        for end in ends:
            pending = end - analyzed
            final = end == audio.size
            if pending < interval and not final:
                continue
            updated = set()
            for offset in range(analyzed, end - 511, 512):
                chunk = audio[offset : offset + 512]
                processed += 512
                lookbehind[:-512] = lookbehind[512:]
                lookbehind[-512:] = chunk
                probabilities[probability_index] = self.probability(chunk)
                probability_index = (probability_index + 1) % 32
                total = np.float32(0)
                for value in probabilities:
                    total = np.float32(total + value)
                smoothed = np.float32(total / np.float32(32))
                if current.size > 160000:
                    fade = np.float32(
                        np.float32(current.size - 160000) / np.float32(160000)
                    )
                    smoothed = np.float32(smoothed * fade)
                voice = smoothed > np.float32(0.3)
                if voice and not active:
                    size = min(8192, processed - last_end)
                    current = lookbehind[-size:].copy()
                    segments.append(
                        {
                            "audio": current,
                            "complete": False,
                            "text": "",
                            "tokens": [],
                            "end": processed,
                        }
                    )
                    updated.add(len(segments) - 1)
                elif active:
                    current = np.concatenate([current, chunk])
                    segments[-1].update(
                        audio=current, complete=not voice, end=processed
                    )
                    updated.add(len(segments) - 1)
                    if not voice:
                        last_end = processed
                        current = np.empty(0, "f4")
                active = voice
            analyzed = end
            if final and active:
                segments[-1]["complete"] = True
                updated.add(len(segments) - 1)
                active = False
            for index in sorted(updated):
                if sequence is None or sequence.identifier != index:
                    sequence = Sequence(self, index)
                segment = segments[index]
                segment["tokens"] = sequence.update(
                    segment["audio"], segment["complete"], previews
                )
                segment["text"] = (
                    self.text(segment["tokens"]) if segment["tokens"] else ""
                )
                if segment["complete"]:
                    segment["audio"] = np.empty(0, "f4")
        return " ".join(s["text"] for s in segments if s["complete"]).strip()


class Sequence:
    def __init__(self, pipeline, identifier):
        self.pipeline, self.identifier = pipeline, identifier
        self.frontend = FrontendState()
        self.processed = 0
        self.features = np.empty((1, 0, 768), "f4")
        self.memory = np.empty((1, 0, 640), "f4")
        self.emitted = 0
        self.tokens = []
        self.cross = None
        self.keys = self.values = None

    def reset_decoder(self):
        self.keys = np.empty((14, 1, 10, 0, 64), "f4")
        self.values = np.empty_like(self.keys)

    def decode(self, tokens):
        result = self.pipeline.model.run(
            "decoder_kv",
            {
                "token": np.array([tokens], "i8"),
                "k_self": self.keys,
                "v_self": self.values,
                "out_k_cross": self.cross[0],
                "out_v_cross": self.cross[1],
            },
        )
        self.keys, self.values = result[1:3]
        return result[0][0]

    def update(self, audio, final, previews):
        is_new = self.processed == 0
        if self.processed < audio.size:
            features = []
            while self.processed + 1280 <= audio.size:
                features.append(
                    self.pipeline.model.frontend(
                        audio[self.processed : self.processed + 1280], self.frontend
                    )
                )
                self.processed += 1280
            if features:
                self.features = np.concatenate([self.features, *features], axis=1)
            if final:
                self.emitted = 0
                self.memory = np.empty((1, 0, 640), "f4")
                self.cross = None
            total = self.features.shape[1]
            stable = total if final else max(0, total - 16)
            if stable > self.emitted:
                start = max(0, self.emitted - 224)
                encoded = self.pipeline.model.run(
                    "encoder",
                    {"features": np.ascontiguousarray(self.features[:, start:])},
                )[0]
                encoded = np.ascontiguousarray(
                    encoded[:, self.emitted - start : stable - start]
                )
                memory = self.pipeline.model.run(
                    "adapter",
                    {"encoded": encoded, "pos_offset": np.array([self.emitted], "i8")},
                )[0]
                self.memory = np.concatenate([self.memory, memory], axis=1)
                self.emitted, self.cross = stable, None
        if not self.memory.shape[1] or (not final and not previews):
            return self.tokens
        if self.cross is None:
            self.cross = self.pipeline.model.run("cross_kv", {"memory": self.memory})
        budget = min(
            256,
            math.ceil(
                float(
                    np.float32(np.float32(audio.size) / np.float32(16000))
                    * np.float32(20 if final else 6.5)
                )
            ),
        )
        self.reset_decoder()
        draft = [t for t in self.tokens if t not in {1, 2}][:budget]
        accepted = []
        if draft and not is_new:
            predictions = self.decode([1] + draft).argmax(-1).tolist()
            while (
                len(accepted) < len(draft)
                and predictions[len(accepted)] == draft[len(accepted)]
            ):
                accepted.append(draft[len(accepted)])
            if len(accepted) == len(draft):
                next_token = predictions[-1]
            else:
                self.reset_decoder()
                next_token = int(self.decode([1] + accepted)[-1].argmax())
            while next_token != 2 and len(accepted) < budget:
                accepted.append(next_token)
                next_token = int(self.decode([next_token])[-1].argmax())
            self.tokens = [1] + accepted
        else:
            self.tokens = [1]
            for _ in range(budget):
                next_token = int(self.decode([self.tokens[-1]])[-1].argmax())
                self.tokens.append(next_token)
                if next_token == 2:
                    break
        return self.tokens
