"""Per-stage CUDA peak memory that keeps the process-wide peak.

``torch.cuda.reset_peak_memory_stats()`` starts a measurement window.  Before every reset the window's
peaks are folded into the process-wide maxima, so :meth:`StagePeakMemory.process_peak` returns what
``torch.cuda.max_memory_allocated()`` / ``max_memory_reserved()`` would return without any reset.
"""

from typing import Any, Optional


class StagePeakMemory:
    def __init__(self, cuda: Optional[Any] = None) -> None:
        if cuda is None:
            import torch

            cuda = torch.cuda
        self._cuda = cuda
        self._allocated = 0
        self._reserved = 0

    def window_peak(self) -> tuple[int, int]:
        """Peak allocated and reserved bytes since the current window started."""

        return int(self._cuda.max_memory_allocated()), int(self._cuda.max_memory_reserved())

    def start_window(self) -> tuple[int, int]:
        """Close the current window (returning its peaks), keep them in the process peak, and reset."""

        allocated, reserved = self.window_peak()
        self._allocated = max(self._allocated, allocated)
        self._reserved = max(self._reserved, reserved)
        self._cuda.reset_peak_memory_stats()
        return allocated, reserved

    def process_peak(self) -> tuple[int, int]:
        """Peak allocated and reserved bytes since the process started, across all windows."""

        allocated, reserved = self.window_peak()
        return max(self._allocated, allocated), max(self._reserved, reserved)
