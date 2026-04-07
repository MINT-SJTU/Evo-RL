from __future__ import annotations

import logging
import subprocess
import threading


def _speak_async(text: str) -> None:
    """Non-blocking Chinese TTS via spd-say (speech-dispatcher, same as lerobot's say())."""
    def _run():
        try:
            subprocess.run(["spd-say", "-l", "zh", text], capture_output=True, timeout=5)
        except FileNotFoundError:
            logging.debug("spd-say not found, TTS unavailable")
        except subprocess.TimeoutExpired:
            logging.debug("TTS timed out for: %s", text)
    threading.Thread(target=_run, daemon=True).start()


def say_start() -> None:
    _speak_async("开始")


def say_success() -> None:
    _speak_async("成功")


def say_failure() -> None:
    _speak_async("失败")
