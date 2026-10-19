"""Slide overlapping quarter windows across a bill image.

The window is half the page in each direction. The step is a quarter of the
page, so the next window shares a border with the last one. That shared border
is read in both windows. The unshared interior belongs to one window.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """One quarter-sized view, in pixels of the full page."""

    x: int
    y: int
    width: int
    height: int
    page_width: int
    page_height: int

    def interior(self) -> tuple[int, int, int, int]:
        """Return the unshared interior as x, y, width, height.

        A side that meets another window is inset. A side on the page edge
        stays, because nothing is shared past the edge of the page.
        """
        inset_x = max(0, self.page_width // 8)
        inset_y = max(0, self.page_height // 8)
        left = self.x + (inset_x if self.x > 0 else 0)
        top = self.y + (inset_y if self.y > 0 else 0)
        right = self.x + self.width - (
            inset_x if self.x + self.width < self.page_width else 0
        )
        bottom = self.y + self.height - (
            inset_y if self.y + self.height < self.page_height else 0
        )
        if right <= left:
            left, right = self.x, self.x + self.width
        if bottom <= top:
            top, bottom = self.y, self.y + self.height
        return left, top, right - left, bottom - top


def quarter_windows(page_width: int, page_height: int) -> list[Window]:
    """Return the overlapping windows that cover the page.

    The last window is pulled back so the far edge of the page is included
    when the step does not land on it.
    """
    if page_width < 2 or page_height < 2:
        return [Window(0, 0, max(page_width, 1), max(page_height, 1), page_width, page_height)]
    kernel_w = max(1, page_width // 2)
    kernel_h = max(1, page_height // 2)
    stride_w = max(1, page_width // 4)
    stride_h = max(1, page_height // 4)
    windows: list[Window] = []
    for top in _starts(page_height, kernel_h, stride_h):
        for left in _starts(page_width, kernel_w, stride_w):
            windows.append(
                Window(left, top, kernel_w, kernel_h, page_width, page_height)
            )
    return windows


def _starts(length: int, kernel: int, stride: int) -> list[int]:
    if kernel >= length:
        return [0]
    starts = list(range(0, length - kernel + 1, stride))
    last = length - kernel
    if starts[-1] != last:
        starts.append(last)
    return starts
