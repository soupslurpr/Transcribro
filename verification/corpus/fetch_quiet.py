#!/usr/bin/env python3
"""Focused quiet-recording evaluation; all downloaded data stays outside the repo.

EARS is CC BY-NC 4.0. Its recordings/transcripts are local evaluation inputs,
never app assets. Inputs are independent of recognition output.
"""

import argparse
import array
import concurrent.futures
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import urllib.request
import wave
import zipfile
import zlib



ROOT = Path(__file__).resolve().parents[2]
EARS_REVISION = "613935413a2eac115d191fffc5c74b34ca7cc884"
SPEAKERS = [f"p{n:03d}" for n in (1, 16, 31, 46, 61, 76, 91, 106)]
SENTENCES = (1, 4, 8)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return response.read()


def read_pcm(path):
    return array.array("h", Path(path).read_bytes())


def level(pcm):
    return dict(rms_dbfs=20 * math.log10(max(
        math.sqrt(sum(float(v) ** 2 for v in pcm) / max(1, len(pcm))) / 32768, 1e-12)),
        peak_dbfs=20 * math.log10(max(max((abs(v) for v in pcm), default=0) / 32768, 1e-12)))


class RemoteArchive:
    """Read selected public ZIP members without downloading a 600 MB speaker archive."""

    def __init__(self, asset):
        self.asset = asset
        self.url = asset["browser_download_url"]
        start = asset["size"] - 65557
        tail = self.range(start, asset["size"] - 1)
        end = tail.rfind(b"PK\x05\x06")
        if end < 0:
            raise RuntimeError("Missing ZIP end record")
        fields = struct.unpack_from("<4s4H2IH", tail, end)
        central_start = fields[6]
        if central_start < start:
            raise RuntimeError("ZIP directory exceeds requested tail")
        self.central_start = central_start
        self.index = zipfile.ZipFile(io.BytesIO(tail[central_start - start:]))

    def range(self, start, end):
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=45) as response:
            expected = f"bytes {start}-{end}/{self.asset['size']}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Server did not honor the bounded ZIP request")
            data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError("Incomplete ZIP range")
        return data

    def member(self, name):
        info = self.index.getinfo(name)
        offset = info.header_offset + self.central_start
        header = self.range(offset, offset + 29)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError("Invalid ZIP member header")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        offset += 30 + name_length + extra_length
        packed = self.range(offset, offset + info.compress_size - 1)
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(packed, -15)
        elif info.compress_type == zipfile.ZIP_STORED:
            data = packed
        else:
            raise RuntimeError("Unsupported ZIP compression")
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise RuntimeError("ZIP member integrity check failed")
        return data


def prepare(args):
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Keep downloaded recordings and transcripts outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    sources.mkdir(exist_ok=True)
    (sources / "EdAcc-LICENSE.txt").write_bytes(
        (ROOT / "verification/corpus/CC-BY-SA-4.0.txt").read_bytes())
    base = f"https://raw.githubusercontent.com/facebookresearch/ears_dataset/{EARS_REVISION}"
    for filename in ("LICENSE", "README.md", "transcripts.json"):
        (sources / f"EARS-{filename}").write_bytes(fetch(f"{base}/{filename}"))
    transcripts = json.loads((sources / "EARS-transcripts.json").read_text())
    release = json.loads(fetch("https://api.github.com/repos/facebookresearch/ears_dataset/releases/tags/dataset"))
    assets = {a["name"]: a for a in release["assets"]}

    def speaker_cases(speaker):
        asset = assets[f"{speaker}.zip"]
        archive = RemoteArchive(asset)
        cases = []
        names = [(f"rainbow_{n:02d}_{style}", style) for n in SENTENCES
                 for style in ("regular", "whisper")]
        names += [("vegetative_coughing", "nonverbal"), ("vegetative_throat", "nonverbal")]
        for stem, style in names:
            name = f"{speaker}/{stem}.wav"
            raw = archive.member(name)
            source = sources / f"{speaker}-{stem}.wav"
            source.write_bytes(raw)
            pcm = output / f"ears-{speaker}-{stem}.pcm"
            command = ["ffmpeg", "-v", "error", "-y", "-i", str(source)]
            if style == "nonverbal":
                command += ["-t", "5"]
            command += ["-ar", "16000", "-ac", "1", "-f", "s16le", str(pcm)]
            subprocess.run(command, check=True)
            samples = read_pcm(pcm)
            cases.append(dict(name=pcm.stem, path=str(pcm), group=f"ears-{style}",
                              speaker=speaker, reference=transcripts[stem] if style != "nonverbal" else "",
                              seconds=len(samples) / 16000, sha256=digest(pcm.read_bytes()),
                              source_sha256=digest(raw), source_member=name,
                              source_url=asset["browser_download_url"], source_asset_id=asset["id"],
                              source_archive_bytes=asset["size"], **level(samples)))
        print(f"Fetched {speaker}: {len(cases)} recordings", flush=True)
        return cases

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        cases = [case for group in pool.map(speaker_cases, SPEAKERS) for case in group]

    edacc = json.loads((args.edacc / "clips.json").read_text())
    ranked = []
    for clip in edacc["clips"]:
        wav = args.edacc / clip["wav"]
        if digest(wav.read_bytes()) != clip["sha256"]:
            raise RuntimeError(f"Existing EdAcc recording changed: {wav}")
        with wave.open(str(wav)) as audio:
            if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) != (1, 16000, 2):
                raise RuntimeError("Unexpected existing EdAcc format")
            pcm = audio.readframes(audio.getnframes())
        ranked.append((level(array.array("h", pcm))["rms_dbfs"], clip["index"], clip, pcm))
    counts = {}
    kept = 0
    for _, _, clip, pcm in sorted(ranked):
        speaker = clip["speaker"]
        if counts.get(speaker, 0) >= 2:
            continue
        counts[speaker] = counts.get(speaker, 0) + 1
        path = output / f"edacc-{clip['index']}.pcm"
        path.write_bytes(pcm)
        cases.append(dict(name=path.stem, path=str(path), group="edacc-low-level",
                          speaker=speaker, reference=clip["reference"], seconds=clip["seconds"],
                          sha256=digest(pcm), source_sha256=clip["sha256"],
                          source_dataset=edacc["dataset"], source_revision=edacc["revision"],
                          source_index=clip["index"], **level(array.array("h", pcm))))
        kept += 1
        if kept == 12:
            break
    manifest = dict(selection={
        "ears_revision": EARS_REVISION, "ears_release_id": release["id"],
        "ears_speakers": SPEAKERS, "ears_sentences": SENTENCES,
        "ears_nonverbal": "First five seconds of each selected speaker's coughing and throat recordings",
        "edacc": "Lowest full-clip RMS from the existing fixed 57 clips; maximum two per speaker; 12 total",
        "selection_uses_recognition_output": False,
        "processing": "16 kHz mono PCM16 resampling only; no gain or loudness normalization",
        "limitations": "EARS whispering is distinct from softly voiced speech. EdAcc level does not identify vocal effort or acoustic SPL."},
        cases=cases)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (sources / "PROVENANCE.md").write_text(
        "# Local evaluation inputs\n\n"
        "EARS: Richter, Wu, Krenn, Welker, Lay, Watanabe, Richard and Gerkmann, "
        "Interspeech 2024, https://arxiv.org/abs/2406.06185. CC BY-NC 4.0; "
        "verbatim license in EARS-LICENSE. Metadata revision: " + EARS_REVISION + ". "
        "Sources are copied unchanged and verified by member CRC and SHA-256. "
        "PCM derivatives are resampled; nonverbal controls are trimmed to five seconds. "
        "Per-file source URLs, asset IDs and hashes are in ../manifest.json. "
        "No recordings or transcripts are packaged in the app or repository.\n\n"
        "EdAcc: Edinburgh International Accents of English Corpus, "
        "https://huggingface.co/datasets/edinburghcstr/edacc. "
        "Ramon Sanabria, Nikolay Bogoychev, Nina Markl, Andrea Carmantini, "
        "Ondrej Klejch, and Peter Bell, ICASSP 2023. CC BY-SA 4.0; "
        "verbatim license in EdAcc-LICENSE.txt. "
        "Reused from the existing local evaluation subset at the revision in the manifest. "
        "PCM preserves the samples in those 16 kHz WAVs.\n")
    print(f"Prepared {len(cases)} fixed cases: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edacc", type=Path, required=True)
    prepare(parser.parse_args())
