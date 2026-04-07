from __future__ import annotations

import logging


class CriticalPhaseTracker:
    """Tracks critical phase intervals during recording via spacebar toggles."""

    def __init__(self):
        self._intervals: list[tuple[int, int, int]] = []  # (ep_idx, start, end)
        self._current_start: int | None = None
        self._current_episode: int = 0

    def on_episode_start(self, episode_idx: int) -> None:
        """Call at the start of each episode. Auto-closes any unclosed interval."""
        if self._current_start is not None:
            logging.warning(
                "Auto-closing unclosed critical phase at episode boundary "
                "(episode %d, from frame %d)", self._current_episode, self._current_start,
            )
            self._current_start = None
        self._current_episode = episode_idx

    def toggle(self, frame_index: int) -> None:
        """Toggle critical phase marking. Odd call = start, even call = end."""
        if self._current_start is None:
            self._current_start = frame_index
            logging.info(f"[CP] START at episode {self._current_episode}, frame {frame_index}")
        else:
            self._intervals.append((self._current_episode, self._current_start, frame_index))
            logging.info(
                f"[CP] END at episode {self._current_episode}, frame {frame_index} "
                f"(segment: {self._current_start}-{frame_index}, "
                f"{frame_index - self._current_start} frames)"
            )
            self._current_start = None

    def on_episode_end(self, total_frames: int) -> None:
        """Call before save_episode. Auto-closes unclosed interval."""
        if self._current_start is not None:
            self._intervals.append((self._current_episode, self._current_start, total_frames))
            logging.info(
                f"[CP] Auto-closed at episode {self._current_episode}, frame {total_frames}"
            )
            self._current_start = None

    def discard_episode(self, episode_idx: int) -> None:
        """Discard all intervals for a given episode (on rerecord)."""
        before = len(self._intervals)
        self._intervals = [(e, s, end) for e, s, end in self._intervals if e != episode_idx]
        discarded = before - len(self._intervals)
        if discarded > 0:
            logging.info(f"[CP] Discarded {discarded} intervals for episode {episode_idx}")
        self._current_start = None

    def get_intervals(self) -> list[tuple[int, int, int]]:
        """Return all recorded intervals as (episode_idx, start_frame, end_frame)."""
        return list(self._intervals)

    def __len__(self) -> int:
        return len(self._intervals)

    @property
    def is_active(self) -> bool:
        """True if currently inside a critical phase (start pressed, end not yet)."""
        return self._current_start is not None
