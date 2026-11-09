#!/usr/bin/env python3
"""Fetch a predetermined phrase sample, not the full Earnings-22 benchmark.

Audio and reference text are local correctness inputs, excluded from app assets.
The script records the fixed dataset revision and every supplied WAV digest.
"""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.parse
import urllib.request
import urllib.error
import wave

DATASET = "distil-whisper/earnings22"
REVISION = "0a034f9ed86d33a3859d9025d3e621cf243773ab"
ROWS = 57391


def download(url):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except Exception as error:
            if attempt == 7:
                raise
            delay = attempt + 1
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                delay = max(60, int(error.headers.get("Retry-After", "60")))
            time.sleep(delay)


def window(root, offset):
    cache = root / f"window-{offset}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    query = urllib.parse.urlencode(dict(dataset=DATASET, config="chunked", split="test",
                                        offset=offset, length=min(40, ROWS - offset)))
    page = json.loads(download("https://datasets-server.huggingface.co/rows?" + query))
    if page["num_rows_total"] != ROWS:
        raise RuntimeError("Dataset row count changed")
    clips = []
    for item in page["rows"]:
        row = item["row"]
        reference = re.sub(r"<[^>]+>", "", row["transcription"])
        if not 8 <= len(reference.split()) <= 45:
            continue
        if not 2 <= row["end_ts"] - row["start_ts"] <= 25:
            continue
        index = item["row_idx"]
        url = row["audio"][0]["src"]
        if f"/{REVISION}/" not in url:
            raise RuntimeError("Unexpected dataset revision")
        raw, wav = root / f"{index}-source.wav", root / f"{index}.wav"
        if not wav.exists():
            raw.write_bytes(download(url))
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-ar",
                                "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
            finally:
                raw.unlink(missing_ok=True)
        with wave.open(str(wav)) as audio:
            duration = audio.getnframes() / audio.getframerate()
        if not 2 <= duration <= 25:
            continue
        clips.append(dict(index=index, file_id=row["file_id"], segment_id=row["segment_id"],
                          speaker=None, accent=None, reference=reference,
                          original_reference=row["transcription"], seconds=duration,
                          wav=wav.name, sha256=hashlib.sha256(wav.read_bytes()).hexdigest()))
        if len(clips) == 4:
            break
    print(f"window {offset}: {len(clips)} clips", flush=True)
    cache.write_text(json.dumps(clips, indent=2) + "\n")
    return clips


def main(root):
    info = json.loads(download(f"https://huggingface.co/api/datasets/{DATASET}"))
    if info["sha"] != REVISION:
        raise RuntimeError("Dataset revision changed; review before sampling")
    root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        windows = list(pool.map(lambda offset: window(root, offset), range(0, ROWS, 1000)))
    clips = [clip for group in windows for clip in group]
    manifest = dict(dataset=DATASET, revision=REVISION, config="chunked", split="test",
                    sampling="First four qualifying clips in each 40-row window at offsets "
                             "0, 1000, ..., 57000; 8–45 whitespace-delimited reference words "
                             "and 2–25 seconds; no model output used for selection.",
                    reference_transform="Remove angle-bracket nonlexical annotations; "
                                        "retain original references alongside scored text.",
                    clips=clips)
    (root / "clips.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(clips=len(clips), calls=len({c['file_id'] for c in clips}),
                          seconds=sum(c["seconds"] for c in clips))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    main(parser.parse_args().audio)
