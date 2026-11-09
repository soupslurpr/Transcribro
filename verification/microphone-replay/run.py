#!/usr/bin/env python3
"""Replay synthetic speech through a virtual microphone or authorized physical speakers."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", required=True, type=Path)
    parser.add_argument("--protocol", type=Path,
                        help="Temporary directory of stubs generated from the installed emulator SDK")
    parser.add_argument("--fixture", required=True, type=Path, help="Synthetic mono 16 kHz PCM16 WAV")
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--speakers", action="store_true",
                        help="Play the synthetic WAV through this computer, with the phone nearby")
    parser.add_argument("--test", choices=("practice", "send"), default="practice")
    parser.add_argument("--host", choices=("compose", "native", "web-input", "web-textarea", "web-rich"),
                        default="compose")
    parser.add_argument("--automatic", action="store_true")
    parser.add_argument("--no-live-text", action="store_true")
    parser.add_argument("--trace-editor", action="store_true",
                        help="Keep an in-memory browser input trace, printed only on a counting failure")
    parser.add_argument("--between-phrases-ms", type=int, default=0,
                        help="Extra listening time before the second message (0–15000 ms)")
    args = parser.parse_args()
    if not 0 <= args.between_phrases_ms <= 15000:
        parser.error("--between-phrases-ms must be between 0 and 15000")
    adb = [str(args.adb), "-s", args.serial]
    with wave.open(str(args.fixture)) as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Expected mono 16 kHz PCM16 WAV")
        pcm = source.readframes(source.getnframes())
    channel = None
    if args.speakers:
        if args.serial.startswith("emulator-"):
            raise ValueError("Use direct injection for the emulator")
        def replay():
            subprocess.run(["pw-play", "--volume=0.5", str(args.fixture)], check=True, timeout=35)
            print(f"Speaker replay: {len(pcm)/32000:.3f}s synthetic speech", flush=True)
    else:
        if args.protocol is None:
            raise ValueError("--protocol is required for emulator injection")
        sys.path.insert(0, str(args.protocol))
        import grpc
        import emulator_controller_pb2 as pb
        import emulator_controller_pb2_grpc as rpc
        from google.protobuf.empty_pb2 import Empty

        discovery = subprocess.check_output(adb + ["emu", "avd", "discoverypath"], text=True).splitlines()[0]
        settings = dict(line.split("=", 1) for line in Path(discovery).read_text().splitlines() if "=" in line)
        if args.serial != "emulator-" + settings["port.serial"]:
            raise RuntimeError("Discovery does not match the selected emulator")
        channel = grpc.insecure_channel("127.0.0.1:" + settings["grpc.port"])
        client = rpc.EmulatorControllerStub(channel)
        metadata = [("authorization", "Bearer " + settings["grpc.token"])]
        if client.getMicrophoneState(Empty(), metadata=metadata, timeout=5).realAudioEnabled:
            raise RuntimeError("Disable the emulator host microphone before replaying synthetic audio")
        audio_format = pb.AudioFormat(
            samplingRate=16000, channels=pb.AudioFormat.Mono, format=pb.AudioFormat.AUD_FMT_S16,
            mode=pb.AudioFormat.MODE_UNSPECIFIED)

        def replay():
            started = time.monotonic()
            # Let the emulator pace consumption, including during model startup.
            packets = (pb.AudioPacket(format=audio_format, audio=pcm[offset:offset + 640])
                       for offset in range(0, len(pcm), 640))
            client.injectAudio(packets, metadata=metadata, timeout=35)
            # The RPC can finish while the virtual microphone still has ~1 second buffered.
            # Let the device consume it before ending replay or starting another injection.
            remaining = len(pcm) / 32000 - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            print(f"Microphone replay: {len(pcm)/32000:.3f}s PCM in {time.monotonic()-started:.3f}s", flush=True)

    method = ("microphoneReplayReachesTenInTheActualPracticeField" if args.test == "practice"
              else "microphoneReplaySendsSuccessivePhrasesThroughTheActualKeyboard")
    command = adb + ["shell", "am", "instrument", "-w", "-r", "-e", "class",
                     "dev.soupslurpr.transcribro.KeyboardDeviceTest#" + method,
                     "-e", "microphoneReplay", "true", "-e", "host", args.host,
                     "-e", "automatic", str(args.automatic).lower(),
                     "-e", "liveText", str(not args.no_live_text).lower(),
                     "-e", "editorTrace", str(args.trace_editor).lower(),
                     "-e", "betweenPhrasesMs", str(args.between_phrases_ms),
                     "dev.soupslurpr.transcribro.reimagined.test/androidx.test.runner.AndroidJUnitRunner"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    try:
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)
            marker = "INSTRUMENTATION_STATUS: transcribro_replay="
            if line.startswith(marker):
                token = line[len(marker):].strip()
                replay()
                subprocess.run(adb + ["shell", "am", "broadcast", "-a",
                                      "dev.soupslurpr.transcribro.MICROPHONE_REPLAY_COMPLETE",
                                      "-p", "dev.soupslurpr.transcribro.reimagined", "--es", "token", token],
                               check=True, stdout=subprocess.DEVNULL)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()
        if channel is not None:
            channel.close()
    if "OK (1 test)" not in "".join(output):
        raise SystemExit("Microphone integration test failed")


if __name__ == "__main__":
    main()
