from __future__ import annotations

import logging
import subprocess
import threading


def _speak_async(text: str) -> None:
    """Non-blocking Chinese TTS via espeak-ng or espeak fallback."""
    def _run():
        try:
            subprocess.run(
                ["espeak-ng", "-v", "cmn", text],
                capture_output=True, timeout=3,
            )
        except FileNotFoundError:
            try:
                subprocess.run(
                    ["espeak", "-v", "zh", text],
                    capture_output=True, timeout=3,
                )
            except FileNotFoundError:
                logging.debug("No TTS engine available (espeak-ng / espeak not found)")
        except subprocess.TimeoutExpired:
            logging.debug("TTS timed out for text: %s", text)
    threading.Thread(target=_run, daemon=True).start()


def say_start() -> None:
    _speak_async("开始")


def say_success() -> None:
    _speak_async("成功")


def say_failure() -> None:
    _speak_async("失败")
