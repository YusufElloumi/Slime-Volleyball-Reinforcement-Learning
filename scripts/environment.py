from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from scripts.agent import Discretizer
    from scripts.config import TrainingConfig
except ImportError:
    from agent import Discretizer
    from config import TrainingConfig


class NodeGameEnvironment:
    def __init__(self, project_root: str | None = None, binary: str = "node", script_name: str = "headless_env.js"):
        default_root = Path(__file__).resolve().parent.parent
        self.project_root = Path(project_root or default_root)
        self.binary = binary
        self.script_name = script_name
        self.proc = None
        self._start()

    def _start(self):
        script_path = self.project_root / "scripts" / self.script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Headless JS environment not found: {script_path}")
        self.proc = subprocess.Popen(
            [self.binary, str(script_path)],
            cwd=str(self.project_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _send(self, message):
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Environment is not running.")
        payload = json.dumps(message)
        self.proc.stdin.write(payload + "\n")
        self.proc.stdin.flush()
        response_line = self.proc.stdout.readline()
        if not response_line:
            error_output = self.proc.stderr.read()
            raise RuntimeError(f"Environment failed while waiting for response: {error_output}")
        return json.loads(response_line)

    def reset(self, server_side: bool = True, reset_scores: bool = True):
        result = self._send({"cmd": "reset", "server_side": server_side, "reset_scores": reset_scores})
        return result.get("state", {})

    def get_state(self, side: str = "left"):
        result = self._send({"cmd": "get_state", "side": side})
        return result.get("state", {})

    def step(self, action):
        result = self._send({"cmd": "step", "action": action})
        return result

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self._send({"cmd": "quit"})
            self.proc.stdin.close()
            if self.proc.stdout is not None:
                self.proc.stdout.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait(timeout=5)
        self.proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Discretizer(Discretizer):
    pass


def build_default_environment():
    return NodeGameEnvironment()
