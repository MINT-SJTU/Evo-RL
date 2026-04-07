from __future__ import annotations

import logging
import threading


def _speak_async(text: str) -> None:
    """Non-blocking Chinese TTS via pyttsx3 (espeak backend)."""
    def _run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 180)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            logging.debug("TTS unavailable for: %s", text)
    threading.Thread(target=_run, daemon=True).start()


def say_start() -> None:
    _speak_async("开始")


def say_success() -> None:
    _speak_async("成功")


def say_failure() -> None:
    _speak_async("失败")
