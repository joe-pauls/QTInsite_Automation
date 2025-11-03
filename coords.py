"""
coords.py - Simple mouse coordinate helper

- Prints current mouse coordinates every interval (default 0.5s)
- Or print only when you click (Windows) using --on-click
- Windows: uses Win32 API via ctypes (no extra installs)
- Other OS: falls back to Tkinter if available for position polling

Usage (PowerShell on Windows):
    python coords.py                # default 0.5s interval, single-line updating
    python coords.py -i 0.2         # faster updates
    python coords.py --newline      # print a new line each tick (useful for logging)
    python coords.py --on-click     # print only when clicking (Windows global)
    python coords.py --on-click --button right  # only on right-click

Output format on click:
    click saved to : (x, y)

Press Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import sys
import time

# Windows-specific imports are guarded
try:
    import ctypes
    import ctypes.wintypes as wintypes
except Exception:  # not on Windows or not needed
    ctypes = None  # type: ignore
    wintypes = None  # type: ignore

# Optional Tk fallback for non-Windows
try:
    import tkinter as _tk
except Exception:
    _tk = None  # type: ignore


def _get_mouse_pos_windows() -> tuple[int, int]:
    if ctypes is None or wintypes is None:
        raise RuntimeError("ctypes/wintypes unavailable for Windows API")

    class POINT(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    pt = POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
        raise OSError("GetCursorPos failed")
    return int(pt.x), int(pt.y)


class _TkMouse:
    def __init__(self) -> None:
        if _tk is None:
            raise RuntimeError("Tkinter not available")
        self._root = _tk.Tk()
        self._root.withdraw()

    def get_pos(self) -> tuple[int, int]:
        # Update the Tk event loop to refresh pointer info
        self._root.update()
        return int(self._root.winfo_pointerx()), int(self._root.winfo_pointery())


def get_mouse_pos() -> tuple[int, int]:
    # Prefer Windows API when on Windows
    if sys.platform.startswith("win"):
        return _get_mouse_pos_windows()

    # Fallback to Tk on other platforms if available
    if _tk is not None:
        global _tk_fallback
        try:
            _tk_fallback
        except NameError:
            _tk_fallback = _TkMouse()  # type: ignore[var-annotated]
        return _tk_fallback.get_pos()  # type: ignore[name-defined]

    raise RuntimeError(
        "No method available to read mouse position on this platform. "
        "Install Tkinter or run on Windows."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print mouse coordinates at a fixed interval")
    p.add_argument("-i", "--interval", type=float, default=0.5, help="Seconds between updates (default: 0.5)")
    p.add_argument("--newline", action="store_true", help="Print a new line each update instead of updating in-place")
    p.add_argument("--on-click", action="store_true", help="Print only when a mouse click occurs (Windows only)")
    p.add_argument(
        "--button",
        choices=["left", "right", "middle", "any"],
        default="any",
        help="Button to listen for in --on-click mode (default: any)",
    )
    return p.parse_args(argv)


def _monitor_clicks_windows(interval: float, button: str) -> None:
    """Poll mouse buttons and print coordinates on click-down events.

    Uses GetAsyncKeyState to detect transitions from up->down.
    """
    if ctypes is None or wintypes is None:
        raise RuntimeError("Windows APIs unavailable")

    # Virtual-Key codes
    VK = {
        "left": 0x01,   # VK_LBUTTON
        "right": 0x02,  # VK_RBUTTON
        "middle": 0x04, # VK_MBUTTON
    }

    watch_keys: list[int]
    if button == "any":
        watch_keys = [VK["left"], VK["right"], VK["middle"]]
    else:
        watch_keys = [VK[button]]

    # Track last pressed state per key to detect edges
    last_down: dict[int, bool] = {k: False for k in watch_keys}

    # GetAsyncKeyState from user32
    GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState

    def is_down(vk: int) -> bool:
        # High-order bit set if key is down
        return (GetAsyncKeyState(vk) & 0x8000) != 0

    print("Listening for clicks... Press Ctrl+C to stop.")
    while True:
        for vk in watch_keys:
            down = is_down(vk)
            if down and not last_down[vk]:
                x, y = get_mouse_pos()
                print(f"click saved to : ({x}, {y})", flush=True)
            last_down[vk] = down
        time.sleep(max(0.005, interval))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or [])
    interval = max(0.01, float(args.interval))  # clamp to a sane minimum

    # Fixed-width formatting to avoid leftover characters when updating in-place
    fmt = "X={:>5}  Y={:>5}"
    try:
        # On-click mode (Windows only)
        if args.on_click:
            if not sys.platform.startswith("win"):
                raise RuntimeError("--on-click is only supported on Windows")
            _monitor_clicks_windows(interval=interval, button=args.button)
            return 0

        if args.newline:
            while True:
                x, y = get_mouse_pos()
                print(fmt.format(x, y), flush=True)
                time.sleep(interval)
        else:
            last = ""
            while True:
                x, y = get_mouse_pos()
                line = fmt.format(x, y)
                # Pad with spaces to clear previous longer text
                pad = " " * max(0, len(last) - len(line))
                print("\r" + line + pad, end="", flush=True)
                last = line
                time.sleep(interval)
    except KeyboardInterrupt:
        # Ensure we end with a newline for clean prompt return
        print()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
