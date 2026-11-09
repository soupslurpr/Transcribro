#!/usr/bin/env python3
"""Verify the current app with supplied PCM, restoring the prior device state."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from device_quiet import SpeechActivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["adb", "debug", "staging", "tests", "helper", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--serial", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    adb = [str(args.adb), "-s", args.serial]
    target = "dev.soupslurpr.transcribro.reimagined"
    tests = target + ".test"
    helper = "dev.soupslurpr.transcribro.verification"
    observer = SpeechActivity(adb)
    time.sleep(30)
    if observer.busy():
        raise RuntimeError("Another speech task owns this device")

    def run(*command, **kwargs):
        return subprocess.run(adb + list(command), check=True, **kwargs)

    def shell(command):
        return run("shell", command, capture_output=True, text=True).stdout.strip()

    def apk_path(package):
        result = subprocess.run(adb + ["shell", "pm path " + package], capture_output=True, text=True)
        value = result.stdout.strip()
        if not value:
            return None
        result.check_returncode()
        if value.count("package:") != 1:
            raise RuntimeError("Cannot restore a split APK with this runner")
        return value.removeprefix("package:")

    def top():
        return [line.strip() for line in shell("dumpsys activity activities").splitlines()
                if "topResumedActivity=" in line]

    def audio_granted():
        return "android.permission.RECORD_AUDIO: granted=true" in shell("dumpsys package " + target)

    keys = ["secure/voice_recognition_service", "secure/default_input_method",
            "secure/enabled_input_methods", "system/screen_brightness",
            "system/screen_brightness_mode", "system/screen_off_timeout"]
    prior_path = apk_path(target)
    state = dict(settings={key: shell("settings get " + key.replace("/", " ")) for key in keys},
                 foreground=top(), awake="mWakefulness=Awake" in shell("dumpsys power"),
                 record_audio_granted=audio_granted(), apk_sha256=None)
    if apk_path(helper) or apk_path(tests):
        raise RuntimeError("Existing verification packages must be preserved")
    prior = args.output / "prior.apk"
    if prior_path:
        run("pull", prior_path, str(prior), stdout=subprocess.DEVNULL)
        state["apk_sha256"] = hashlib.sha256(prior.read_bytes()).hexdigest()
    (args.output / "state-before.json").write_text(json.dumps(state, indent=2) + "\n")
    results = []

    def install(path):
        if observer.busy():
            raise RuntimeError("Another speech task acquired this device")
        run("install", "-r", "-t", str(path))

    def instrument(name, package, classes):
        log = args.output / (name + ".log")
        command = adb + ["shell", "am", "instrument", "-w", "-r", "-e", "class",
                         ",".join("dev.soupslurpr.transcribro." + c for c in classes),
                         package + "/androidx.test.runner.AndroidJUnitRunner"]
        with log.open("w") as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1800
            try:
                while process.poll() is None:
                    time.sleep(5)
                    if observer.busy():
                        raise RuntimeError("Another speech task acquired this device")
                    if time.monotonic() > deadline:
                        raise RuntimeError("Instrumentation deadline: " + str(log))
                if process.returncode:
                    raise RuntimeError("Instrumentation failed: " + str(log))
            finally:
                if process.poll() is None:
                    shell("am force-stop " + package)
                    shell("am force-stop " + target)
                    process.terminate()
                    process.wait(timeout=30)
        text = log.read_text()
        match = re.search(r"OK \((\d+) tests?\)", text)
        if not match or "FAILURES" in text or "AssertionError" in text:
            raise RuntimeError("Instrumentation failed: " + str(log))
        results.append(dict(name=name, tests=int(match[1]), passed=True))
        print(name, "passed", match[1], "tests", flush=True)

    try:
        install(args.debug)
        install(args.tests)
        shell("input keyevent 224")
        shell("wm dismiss-keyguard")
        shell("settings put system screen_off_timeout 1800000")
        shell("pm grant " + target + " android.permission.RECORD_AUDIO")
        instrument("native-finalization", tests, ["NativeFinalizationTest"])
        instrument("worker-regressions", tests, [
            "BufferedStartupTest", "SpeechDetectionTest", "ManualPhraseTest",
            "WorkerRealtimeTest#pacedAudioCompletesWithFinalWordsAndLivePreviews",
            "WorkerRealtimeTest#inputThatOutrunsLiveInferenceDrainsCapturedWordsWithoutSending",
            "WorkerRealtimeTest#successivePacedCountsKeepListeningAndRetainEveryNumber",
        ])
        shell("pm revoke " + target + " android.permission.RECORD_AUDIO")
        instrument("audio-files", tests, ["AudioFileIntegrationTest"])
        files = shell("run-as " + target + " sh -c 'find files cache code_cache no_backup -type f 2>/dev/null || true'")
        (args.output / "private-files-after-debug.txt").write_text(files + "\n")
        if any(re.search(r"(?:medium.*\.trm|\.ort$|tokenizer\.bin$)", line) for line in files.splitlines()):
            raise RuntimeError("Unexpected extracted speech model")
        run("uninstall", tests)
        install(args.staging)
        install(args.helper)
        shell("pm grant " + helper + " android.permission.RECORD_AUDIO")
        # Explicit methods exclude the real-microphone test and all playback.
        instrument("external-supplied-pcm", helper, ["RecognitionIntegrationTest#" + name for name in [
            "missingSuppliedDescriptorDoesNotFallBackToTheMicrophone",
            "serviceWithoutMicrophonePermissionAcceptsPcmButRejectsLiveCapture",
            "generatedSpeechPassesThroughTheRealIsolatedEngine",
            "silenceDoesNotHallucinateText",
            "pauseBoundaryFollowsCommittedSpeechThroughTheExternalRecognitionContract",
            "legacyContinuousOptionRetainsSuccessivePhrasesAndNewOptionTakesPrecedence",
            "wrongSampleRateIsRejectedInsteadOfMisinterpreted",
            "incompletePcmSampleIsRejected", "emptyInputTerminates",
        ]])
    finally:
        for package in [target, helper, tests]:
            shell("am force-stop " + package)
        if prior_path:
            run("install", "-r", "-t", str(prior))
            shell("pm " + ("grant " if state["record_audio_granted"] else "revoke ") +
                  target + " android.permission.RECORD_AUDIO")
        elif apk_path(target):
            run("uninstall", target)
        for package in [helper, tests]:
            if apk_path(package):
                run("uninstall", package)
        for key, value in state["settings"].items():
            scope, name = key.split("/")
            shell("settings delete " + scope + " " + name if value == "null" else
                  "settings put " + scope + " " + name + " " + shlex.quote(value))
        foreground = re.search(r"u\d+ ([^ ]+)", state["foreground"][0]) if state["foreground"] else None
        if foreground:
            shell("am start -W -n " + shlex.quote(foreground[1]))
        if not state["awake"]:
            shell("input keyevent 223")
        current_path = apk_path(target)
        restored = dict(apk_sha256=shell("sha256sum " + shlex.quote(current_path)).split()[0] if current_path else None,
                        record_audio_granted=audio_granted(), foreground=top(),
                        awake="mWakefulness=Awake" in shell("dumpsys power"),
                        helper_present=bool(apk_path(helper)), tests_present=bool(apk_path(tests)),
                        settings_match={key: shell("settings get " + key.replace("/", " ")) == value
                                        for key, value in state["settings"].items()})
        (args.output / "restored.json").write_text(json.dumps(restored, indent=2) + "\n")
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        assert restored["apk_sha256"] == state["apk_sha256"]
        assert restored["record_audio_granted"] == state["record_audio_granted"]
        assert restored["awake"] == state["awake"]
        assert not restored["helper_present"] and not restored["tests_present"]
        assert all(restored["settings_match"].values())
        if foreground and state["awake"]:
            assert any(foreground[1] in line for line in restored["foreground"])
        print("Prior APK, permissions, settings and foreground restored", flush=True)


if __name__ == "__main__":
    main()
