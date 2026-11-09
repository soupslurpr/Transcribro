#!/usr/bin/env python3
"""Acquire the pinned EdAcc phrase subset for native correctness comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request
import wave

DATASET = "edinburghcstr/edacc"
REVISION = "d9ae7bd344f0562b766ec93ee5ce8f2f9568ce66"


def get_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    revision = get_json(f"https://huggingface.co/api/datasets/{DATASET}")["sha"]
    if revision != REVISION:
        raise RuntimeError("The dataset viewer has changed; review its revision before sampling.")
    clips = []
    # Fixed windows, independent of any model output. Short acknowledgements and
    # long multi-turn recordings are outside this voice-phrase comparison.
    for offset in range(0, 9289, 500):
        query = urllib.parse.urlencode(dict(dataset=DATASET, config="default", split="test",
                                            offset=offset, length=40))
        page = get_json("https://datasets-server.huggingface.co/rows?" + query)
        kept = 0
        for item in page["rows"]:
            row = item["row"]
            if not 8 <= len(row["text"].split()) <= 45:
                continue
            index = item["row_idx"]
            source = row["audio"][0]["src"]
            if f"/{REVISION}/" not in source:
                raise RuntimeError("Unexpected audio revision")
            raw, wav = root / f"{index}-source.wav", root / f"{index}.wav"
            if not wav.exists():
                urllib.request.urlretrieve(source, raw)
                try:
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-ar", "16000",
                                    "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
                finally:
                    raw.unlink(missing_ok=True)
            with wave.open(str(wav)) as audio:
                duration = audio.getnframes() / audio.getframerate()
            if not 2 <= duration <= 25:
                continue
            clips.append(dict(index=index, speaker=row["speaker"], accent=row["accent"],
                              reference=row["text"], seconds=duration, wav=wav.name,
                              sha256=hashlib.sha256(wav.read_bytes()).hexdigest()))
            kept += 1
            if kept == 3:
                break
        print(f"window {offset}: {kept} clips", flush=True)
    (root / "clips.json").write_text(json.dumps(dict(dataset=DATASET, revision=REVISION, clips=clips), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    fetch(parser.parse_args().audio)
