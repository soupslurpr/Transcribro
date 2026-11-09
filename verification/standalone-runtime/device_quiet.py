"""Observe other speech experiments without stopping their processes."""

import re
import subprocess
import time


class SpeechActivity:
    def __init__(self, adb):
        self.adb = adb
        self.ticks = int(self.shell("getconf CLK_TCK"))
        self.previous = self.snapshot()
        self.sampled = time.monotonic()

    def shell(self, command, check=True):
        return subprocess.run(
            self.adb + ["shell", command], check=check, capture_output=True, text=True
        ).stdout.strip()

    def snapshot(self):
        result = {}
        for line in self.shell("ps -A -o PID,NAME").splitlines():
            if not re.search(r"speechservices|runevox", line, re.I):
                continue
            pid, name = line.split(None, 1)
            stat = self.shell("cat /proc/" + pid + "/stat", check=False)
            if stat:
                fields = stat.split(") ", 1)[1].split()
                result[pid] = dict(
                    name=name,
                    ticks=int(fields[11]) + int(fields[12]),
                    started=fields[19],
                )
            elif self.shell("test -d /proc/" + pid + " && echo present", check=False):
                raise RuntimeError(
                    "Cannot verify that another speech process is idle: " + name
                )
        return result

    def busy(self):
        current = self.snapshot()
        now = time.monotonic()
        seconds = max(now - self.sampled, 1.0)
        active = [
            value["name"]
            for pid, value in current.items()
            if pid not in self.previous
            or value["started"] != self.previous[pid]["started"]
            or value["ticks"] - self.previous[pid]["ticks"]
            > self.ticks * seconds * 0.02
        ]
        self.previous, self.sampled = current, now
        return active
