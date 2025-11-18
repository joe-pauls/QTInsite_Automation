"""
QT Insite Automation GUI

This app automates QT Insite: launch/login, edit a single known test sequence,
run tests (once or on a schedule), and save outputs into a timestamped folder.
It also generates a PDF report and can pull recent weather for context.

Design notes:
- GUI automation is timing-sensitive. We intentionally slow down motions and
    always activate/center the window before clicking to avoid missed targets. The mouse should not be touched while the automation is running.
- If QT Insite was already open when you start, we click Logout first to avoid
    odd session states before logging in again.

Author: Joseph Pauls

TODO 
✅ Add checks for whether QT Insite is already running before launching or user is already logged in (auto-click logout before loggin in) 
✅ Add options for user to input test voltage level, grounded/ungrounded checkbox, and integrate those values into the automation sequence 
✅  - The same sequence will get edited every time to avoid having to make a new sequence (placeholder is edit_test_sequence function)
✅ Add option to run a single test immediately to see results without scheduling (with its own configuration panel) 
✅ Add input box on gui for airport ICAO code or latitude and longitude coordinates for the weather data 
✅ When the test is run, a new folder with the datetime is created and all files are outputted into it. The summary folder is then created inside that folder.
✅ Check for rain in the past 3 days since testing and add the total rainfall amount for the past 3 days to the report summary table
✅ Get rid of weather previews
✅ make sure only one timestamped folder is being created when a test is run, not when the reporting is generated
✅ add check for making sure qt insite is open and logged in before running every test (in case it crashes)
✅ change output path at bottom right of gui to show the current output path being used
✅ Make sure the stop button completely stops the automation, it seems to continue moving the mouse even after stopping
"""

import sys
import time
import subprocess
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyautogui
import pygetwindow as gw
from PyQt5 import QtWidgets, QtGui, QtCore

# Import report generation module (must exist in your project)
import Reporting

# ========== CONFIGURATION ==========
USERNAME = "testuser"
PASSWORD = "AtlanticElectric"
JAVAW_PATH = r"C:\Program Files\Vitrek\QtInsite\jx\bin\javaw.exe"
START_IN_PATH = r"C:\Program Files\Vitrek\QtInsite\app"
QTINSITE_JAR = "QtInsite.jar"
WINDOW_TITLE_PART = "Vitrek QT Insite"

DEFAULT_COORDS = {
    "username_field": (875, 600),
    "run_field": (850, 405),
    "start_field": (1425, 490),
    "system_config": (510, 400),
    "verify_button": (800, 865),
    "logout_button": (955, 330),
    "test_sequence_menu": (700, 400),
    "test_sequence_select": (500, 565),
    "test_sequence_voltage": (1045, 585),
    "test_sequence_save":(660, 598),
    "test_sequence_reporting_tab": (905, 470),
    "test_sequence_csv_path_start": (1095, 541),
    "test_sequence_pdf_path_start": (1095, 627),
}

MOVE_DURATION = 0.5  # unified mouse move duration (slower for reliability)
ACTION_PAUSE = 0.2   # unified pause between actions (slower to stabilize UI)
pyautogui.PAUSE = ACTION_PAUSE  # apply same pause to all pyautogui calls

LOGO_PATH = "AtlanticElectricLogo.png"

DURATION_UNITS = [("Minutes", 60), ("Hours", 3600), ("Days", 86400)]
INTERVAL_UNITS = [("Seconds", 1), ("Minutes", 60), ("Hours", 3600)]

ACCENT_RED = "#b71c1c"
BG = "#f6f6f6"
TEXT = "#222"
MUTED = "#555"

# Thread-safe stop event used to abort automation quickly
STOP_EVENT = threading.Event()  

# ========== HELPERS ==========
def log_msg(tag: str, msg: str) -> str:
    s = f"[{time.strftime('%H:%M:%S')}] {tag}: {msg}"
    print(s)
    return s

def find_qt_window(title_part: str = WINDOW_TITLE_PART) -> Optional[object]:
    for w in gw.getAllWindows():
        try:
            if title_part in w.title:
                return w
        except Exception:
            continue
    return None

def activate_qt_insite_window() -> Optional[object]:
    win = find_qt_window()
    if not win:
        return None
    if win.isMinimized:
        win.restore()
        time.sleep(ACTION_PAUSE)
    win.activate()
    time.sleep(ACTION_PAUSE)
    return win

def _center_qt_window(win) -> bool:
    """Center the QT Insite window on the primary screen for predictable clicks.

    Why: hard-coded coordinates are more reliable when the app window is roughly
    centered on the primary monitor. This reduces drift when the user previously
    moved or resized the window.

    Returns True if the window was moved or already near center, False on error.
    """
    try:
        screen_w, screen_h = pyautogui.size()
        w, h = win.width, win.height
        target_x = max(0, (screen_w - w) // 2)
        target_y = max(0, (screen_h - h) // 2)

        # Consider it centered if within a small threshold
        thresh = 20
        if abs(win.left - target_x) <= thresh and abs(win.top - target_y) <= thresh:
            return True

        win.moveTo(target_x, target_y)
        time.sleep(ACTION_PAUSE)
        return True
    except Exception:
        return False

def ensure_qt_open_and_centered(*, login_if_launched: bool = True) -> bool:
    """Make sure QT Insite is open, focused, and centered before clicking.

    Behavior:
    - Not open: launch, focus, and optionally log in
    - Already open: focus and center the window
    - Always aborts early if STOP_EVENT is set

    Returns True when ready for GUI interactions, False on failure/stop.
    """
    if STOP_EVENT.is_set():
        return False

    win = activate_qt_insite_window()
    if not win:
        # Not open — try to launch
        if not launch_qtinsite_using_shortcut_style():
            return False
        if STOP_EVENT.is_set():
            return False
        win = activate_qt_insite_window()
        if not win:
            return False
        # If we just launched, optionally login
        if login_if_launched and not STOP_EVENT.is_set():
            login_coords()

    # Center window if needed
    _center_qt_window(win)
    return True

def launch_qtinsite_using_shortcut_style(wait: float = 6.0) -> bool:
    """
    Launch QT Insite jar and wait in short slices so STOP_EVENT can interrupt.
    """
    if find_qt_window():
        return True
    try:
        subprocess.Popen([JAVAW_PATH, "-jar", QTINSITE_JAR], cwd=START_IN_PATH)
        elapsed = 0.0
        step = 0.2
        while elapsed < wait:
            if STOP_EVENT.is_set():
                return False
            time.sleep(step)
            elapsed += step
        return find_qt_window() is not None
    except Exception as e:
        log_msg("LAUNCH", str(e))
        return False

def click_at(x: int, y: int, dur: float = MOVE_DURATION):
    """Move to (x, y) and click, honoring STOP_EVENT at each small step.

    We split waits into short slices so a stop request cancels immediately
    instead of waiting for long sleeps/movements to finish.
    """
    if STOP_EVENT.is_set():
        return
    safe_dur = dur
    try:
        if STOP_EVENT.is_set():
            return
        pyautogui.moveTo(x, y, duration=safe_dur)
        # cooperative pause (interruptible)
        elapsed = 0.0
        step = 0.05
        while elapsed < ACTION_PAUSE:
            if STOP_EVENT.is_set():
                return
            time.sleep(step)
            elapsed += step
        if STOP_EVENT.is_set():
            return
        pyautogui.click()
        elapsed = 0.0
        while elapsed < ACTION_PAUSE:
            if STOP_EVENT.is_set():
                return
            time.sleep(step)
            elapsed += step
    except Exception:
        # swallow to avoid raising from UI thread; logs are available elsewhere
        return

def login_coords() -> bool:
    try:
        ux, uy = DEFAULT_COORDS["username_field"]
        click_at(ux, uy)
        pyautogui.typewrite(USERNAME, interval=0.05)
        pyautogui.press("tab")
        pyautogui.typewrite(PASSWORD, interval=0.05)
        pyautogui.press("enter")
        return True
    except Exception:
        return False

def verify_connection_coords() -> bool:
    try:
        if not ensure_qt_open_and_centered(login_if_launched=True):
            return False
        sx, sy = DEFAULT_COORDS["system_config"]
        vx, vy = DEFAULT_COORDS["verify_button"]
        click_at(sx, sy)
        click_at(vx, vy)
        return True
    except Exception:
        return False

def run_test_coords() -> bool:
    try:
        if not ensure_qt_open_and_centered(login_if_launched=True):
            return False
        rx, ry = DEFAULT_COORDS["run_field"]
        sx, sy = DEFAULT_COORDS["start_field"]
        click_at(rx, ry)
        click_at(sx, sy)
        return True
    except Exception:
        return False

# ========== WORKER ==========
class AutomationWorker(QtCore.QThread):
    """
    Background worker thread for running automated test sequences.
    Uses STOP_EVENT and stop_flag cooperatively to allow immediate stopping.
    """
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(bool)
    report_signal = QtCore.pyqtSignal(str)

    def __init__(self, total_s: int, interval_s: int,
                 icao_code: str, test_voltage: int, is_grounded: bool,
                 generate_report: bool = True,
                 output_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.total_s = int(total_s)
        self.interval_s = max(1, int(interval_s))
        self.icao_code = icao_code
        self.test_voltage = test_voltage
        self.is_grounded = is_grounded
        self.generate_report = generate_report
        self.output_dir = Path(output_dir) if output_dir else None
        self.stop_flag = False

    def log(self, m: str):
        self.log_signal.emit(log_msg("WORKER", m))

    def stop(self):
        """Request immediate stop."""
        self.stop_flag = True
        STOP_EVENT.set()

    def edit_test_sequence(self):
        """Edit the test sequence in QT Insite according to provided parameters.

        Steps:
        - Ensure QT Insite window is active and centered
        - Open test sequence (menu -> select)
        - Set voltage: double-click field, delete, type value
        - Save
        - Go to Reporting tab
        - For CSV path: click, press Ctrl+A to select all, Backspace to clear, type '.../Test_$D_$T.csv'
        - For PDF path: click, press Ctrl+A to select all, Backspace to clear, type '.../PDFTest_$D_$T.pdf'
        """
        if self.stop_flag or STOP_EVENT.is_set():
            return

        self.log("Editing test sequence...")
        self.log(f"  - Voltage: {self.test_voltage}V")
        self.log(f"  - Circuit: {'Grounded' if self.is_grounded else 'Ungrounded'}")

        # Helper: cooperative short sleep
        def _pause():
            elapsed = 0.0
            step = 0.05
            while elapsed < ACTION_PAUSE:
                if self.stop_flag or STOP_EVENT.is_set():
                    return False
                time.sleep(step)
                elapsed += step
            return True

        # 1) Ensure window active and centered (launch+login if needed)
        if not ensure_qt_open_and_centered(login_if_launched=True):
            self.log("  - Warning: Could not prepare QT Insite window")
            return

        if self.stop_flag or STOP_EVENT.is_set():
            return

        c = DEFAULT_COORDS

        # 2) Open test sequence via menu/select
        try:
            mx, my = c["test_sequence_menu"]
            sx, sy = c["test_sequence_select"]
            click_at(mx, my)
            if not _pause():
                return
            click_at(sx, sy)
            if not _pause():
                return
        except Exception:
            self.log("  - Warning: Could not open test sequence via menu/select")

        if self.stop_flag or STOP_EVENT.is_set():
            return

        # 3) Set voltage
        try:
            vx, vy = c["test_sequence_voltage"]
            if self.stop_flag or STOP_EVENT.is_set():
                return
            pyautogui.moveTo(vx, vy, duration=MOVE_DURATION)
            if not _pause():
                return
            pyautogui.doubleClick()
            if not _pause():
                return
            pyautogui.press('backspace')
            if not _pause():
                return
            pyautogui.typewrite(str(self.test_voltage), interval=0.02)
            if not _pause():
                return
        except Exception:
            self.log("  - Warning: Could not set test sequence voltage")

        if self.stop_flag or STOP_EVENT.is_set():
            return

        # 4) Save the sequence
        try:
            svx, svy = c["test_sequence_save"]
            click_at(svx, svy)
            if not _pause():
                return
        except Exception:
            self.log("  - Warning: Could not save test sequence")

        if self.stop_flag or STOP_EVENT.is_set():
            return

        # 5) Go to Reporting tab
        try:
            rtx, rty = c["test_sequence_reporting_tab"]
            click_at(rtx, rty)
            if not _pause():
                return
        except Exception:
            self.log("  - Warning: Could not open Reporting tab")

        # Compose output paths
        try:
            base = self.output_dir if self.output_dir else Reporting.get_default_output_root()
        except Exception:
            base = None
        base_str = (base.as_posix() if base else str(Path.home().as_posix()))
        csv_path = f"{base_str}/Test_$D_$T.csv"
        pdf_path = f"{base_str}/PDFTest_$D_$T.pdf"

        if self.stop_flag or STOP_EVENT.is_set():
            return

        # 6) CSV path: click, Ctrl+A, Backspace, type
        try:
            cx, cy = c["test_sequence_csv_path_start"]
            click_at(cx, cy)
            if not _pause():
                return
            pyautogui.hotkey('ctrl', 'a')
            if not _pause():
                return
            pyautogui.press('backspace')
            if not _pause():
                return
            pyautogui.typewrite(csv_path, interval=0.01)
            if not _pause():
                return
        except Exception:
            self.log("  - Warning: Could not set CSV output path")

        if self.stop_flag or STOP_EVENT.is_set():
            return

        # 7) PDF path: click, Ctrl+A, Backspace, type
        try:
            px, py = c["test_sequence_pdf_path_start"]
            click_at(px, py)
            if not _pause():
                return
            pyautogui.hotkey('ctrl', 'a')
            if not _pause():
                return
            pyautogui.press('backspace')
            if not _pause():
                return
            pyautogui.typewrite(pdf_path, interval=0.01)
            if not _pause():
                return
        except Exception:
            self.log("  - Warning: Could not set PDF output path")

        if self.stop_flag or STOP_EVENT.is_set():
            return

        # 8) Final save
        try:
            svx, svy = c["test_sequence_save"]
            click_at(svx, svy)
            if not _pause():
                return
        except Exception:
            self.log("  - Warning: Could not perform final save")

        self.log("  - Test sequence updated")

    def run(self):
        """
        Main worker execution method. Clears STOP_EVENT at start and ensures
        cooperative checks during long waits and mouse operations.
        """
        # clear stop event at beginning of run
        STOP_EVENT.clear()
        try:
            self.log("Launching QT Insite...")
            if STOP_EVENT.is_set() or self.stop_flag:
                self.finished_signal.emit(False)
                return

            # Determine if QT Insite is already open before launching
            pre_existing = find_qt_window() is not None

            if not launch_qtinsite_using_shortcut_style():
                self.log("❌ Launch failed")
                self.finished_signal.emit(False)
                return

            if STOP_EVENT.is_set() or self.stop_flag:
                self.log("⚠️  Stopped before activation")
                self.finished_signal.emit(False)
                return

            win = activate_qt_insite_window()
            if not win:
                self.log("❌ Window not found")
                self.finished_signal.emit(False)
                return
            # Ensure centered before any login typing begins
            _center_qt_window(win)

            # If already open when we started, log out first
            if pre_existing and not (self.stop_flag or STOP_EVENT.is_set()):
                try:
                    self.log("Logging out existing session...")
                    lx, ly = DEFAULT_COORDS.get("logout_button", (None, None))
                    if lx is None or ly is None:
                        self.log("  - Warning: 'logout_button' coordinate not set; skipping logout")
                    else:
                        click_at(int(lx), int(ly))
                        # cooperative short wait using ACTION_PAUSE
                        elapsed = 0.0
                        step = 0.05
                        while elapsed < ACTION_PAUSE:
                            if self.stop_flag or STOP_EVENT.is_set():
                                break
                            time.sleep(step)
                            elapsed += step
                except Exception:
                    self.log("  - Warning: Could not click logout; proceeding to login")

            self.log("Logging in...")
            if STOP_EVENT.is_set() or self.stop_flag:
                self.log("⚠️  Stopped before login")
                self.finished_signal.emit(False)
                return
            login_coords()

            self.log("Verifying connection...")
            if STOP_EVENT.is_set() or self.stop_flag:
                self.log("⚠️  Stopped before verification")
                self.finished_signal.emit(False)
                return
            verify_connection_coords()

            # edit sequence cooperatively
            self.edit_test_sequence()

            runs = max(1, self.total_s // self.interval_s)
            self.log(f"Starting {runs} test runs (interval: {self.interval_s}s)")

            for i in range(1, runs + 1):
                if self.stop_flag or STOP_EVENT.is_set():
                    self.log("⚠️  Stopped by user request")
                    self.finished_signal.emit(False)
                    return

                self.log(f"▶️  Run {i}/{runs}")
                run_test_coords()

                # cooperative wait between runs
                elapsed = 0.0
                step = 0.2
                while elapsed < self.interval_s:
                    if self.stop_flag or STOP_EVENT.is_set():
                        self.log("⚠️  Stopped by user request")
                        self.finished_signal.emit(False)
                        return
                    time.sleep(step)
                    elapsed += step

            # Step: Generate report (if enabled)
            if self.generate_report:
                if not self.output_dir:
                    self.log("⚠️  No output directory configured; skipping report generation")
                else:
                    self.log(f"📊 Generating report in {self.output_dir}...")
                    self.report_signal.emit("Generating report, please wait...")
                    try:
                        result = Reporting.generate_report(self.output_dir, test_voltage=self.test_voltage)
                        pdf_path = result.get("pdf")
                        self.report_signal.emit("✅ Report generated successfully!")
                        if pdf_path:
                            self.log(f"✅ Report saved to {pdf_path}")
                        else:
                            self.log("✅ Report generation complete")
                    except Exception as e:
                        error_msg = f"Report generation failed: {str(e)}"
                        self.report_signal.emit(f"❌ {error_msg}")
                        self.log(f"❌ {error_msg}")

            self.log("✅ Automation completed successfully")
            self.finished_signal.emit(True)
        except Exception:
            error_trace = traceback.format_exc()
            self.log(f"❌ Error:\n{error_trace}")
            self.finished_signal.emit(False)
        finally:
            # ensure STOP_EVENT is cleared for subsequent runs
            STOP_EVENT.clear()

# ========== STYLE ==========
STYLE = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: 'Segoe UI';
    font-size: 16px;
}}
QGroupBox {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    margin-top: 0px;
    padding-top: 12px;
    padding-bottom: 8px;
}}
QLabel {{
    background: transparent;
}}
QPushButton {{
    background: {ACCENT_RED};
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
}}
QPushButton:disabled {{
    background: #e0a3a3;
}}
QSpinBox, QComboBox, QLineEdit {{
    background: white;
    border: 1px solid #ccc;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 16px;
    min-height: 32px;
}}
QTextEdit {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 15px;
    padding: 8px;
}}
QCheckBox {{
    spacing: 8px;
    font-size: 16px;
    background: transparent;
}}
QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid #ccc;
    background: white;
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_RED};
    border-color: {ACCENT_RED};
}}
QLabel#headerTitle {{
    color: {ACCENT_RED};
    font-size: 28px;
    font-weight: 800;
}}
QLabel#subtitle {{
    color: {MUTED};
    font-size: 18px;
}}
"""

# ========== MAIN WINDOW ==========
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QT Insite Automation — Atlantic Electric")
        self.setFixedSize(800, 800)
        self.worker: Optional[AutomationWorker] = None
        self.current_output_dir: Optional[Path] = None
        self._build_ui()
        self.setStyleSheet(STYLE)

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # HEADER
        header_h = QtWidgets.QHBoxLayout()
        header_h.setSpacing(8)
        text_v = QtWidgets.QVBoxLayout()
        text_v.setSpacing(4)

        title = QtWidgets.QLabel("QT Insite Automation")
        title.setObjectName("headerTitle")
        subtitle = QtWidgets.QLabel("Automated Insulation Resistance Testing Tool")
        subtitle.setObjectName("subtitle")

        text_v.addWidget(title)
        text_v.addWidget(subtitle)
        text_v.addStretch()
        header_h.addLayout(text_v, stretch=3)

        # Logo
        logo_lbl = QtWidgets.QLabel()
        logo_lbl.setFixedSize(200, 100)
        logo_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        logo_path = Path(LOGO_PATH)
        if logo_path.exists():
            pix = QtGui.QPixmap(str(logo_path)).scaled(
                200, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        else:
            # Fallback placeholder if the logo asset isn't present
            pix = QtGui.QPixmap(200, 100)
            pix.fill(QtGui.QColor(ACCENT_RED))
        logo_lbl.setPixmap(pix)
        logo_container = QtWidgets.QHBoxLayout()
        logo_container.addStretch()
        logo_container.addWidget(logo_lbl)
        header_h.addLayout(logo_container, stretch=1)

        outer.addLayout(header_h)

        # MAIN CONTENT GRID
        content = QtWidgets.QGridLayout()
        content.setHorizontalSpacing(8)
        content.setVerticalSpacing(8)
        outer.addLayout(content)

        # ======= Test Parameters bold label (top-left) =======
        test_params_label = QtWidgets.QLabel("<b>Test Parameters</b>")
        test_params_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        content.addWidget(test_params_label, 0, 0, 1, 1)

        # ======= Test Parameters GroupBox (no visible title) =======
        g_params = QtWidgets.QGroupBox()
        g_params.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        g_params.setFixedHeight(150)

        params_layout = QtWidgets.QHBoxLayout()
        params_layout.setContentsMargins(16, 6, 16, 6)
        params_layout.setSpacing(8)
        g_params.setLayout(params_layout)

        # Left column: voltage (above grounding)
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(6)

        # Voltage (top-left)
        voltage_box = QtWidgets.QHBoxLayout()
        voltage_label = QtWidgets.QLabel("Test Voltage (V):")
        voltage_label.setFixedWidth(150)
        voltage_label.setFixedHeight(36)
        self.voltage_spin = QtWidgets.QSpinBox()
        self.voltage_spin.setRange(1, 5000)
        self.voltage_spin.setValue(1500)
        self.voltage_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.voltage_spin.setFixedHeight(36)
        self.voltage_spin.setMinimumWidth(150)
        self.voltage_spin.setToolTip("Set the test voltage (1-5000V)")
        voltage_box.addWidget(voltage_label)
        voltage_box.addWidget(self.voltage_spin)
        voltage_box.addStretch()
        left_col.addLayout(voltage_box)

        # Grounding (below voltage)
        grounding_box = QtWidgets.QHBoxLayout()
        grounding_box.setAlignment(QtCore.Qt.AlignVCenter)
        grounding_label = QtWidgets.QLabel("Grounding:")
        grounding_label.setFixedWidth(150)
        grounding_label.setFixedHeight(36)
        self.grounded_cb = QtWidgets.QCheckBox("Circuit is grounded")
        self.grounded_cb.setChecked(True)
        self.grounded_cb.setFixedHeight(36)
        self.grounded_cb.setToolTip("Check if the circuit under test is grounded")
        grounding_box.addWidget(grounding_label)
        grounding_box.addWidget(self.grounded_cb)
        grounding_box.addStretch()
        left_col.addLayout(grounding_box)

        params_layout.addLayout(left_col, stretch=1)

        # Right column: duration (above interval)
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(6)

        # Total Duration (top-right)
        dur_box = QtWidgets.QHBoxLayout()
        dur_label = QtWidgets.QLabel("Total Duration:")
        dur_label.setFixedWidth(150)
        dur_label.setFixedHeight(36)
        self.duration_spin = QtWidgets.QSpinBox()
        self.duration_spin.setRange(0, 1000000)
        self.duration_spin.setValue(5)
        self.duration_spin.setFixedHeight(36)
        self.duration_spin.setFixedWidth(90)
        self.duration_unit = QtWidgets.QComboBox()
        for lbl, sec in DURATION_UNITS:
            self.duration_unit.addItem(lbl, sec)
        self.duration_unit.setFixedHeight(36)
        self.duration_unit.setFixedWidth(150)
        dur_box.addWidget(dur_label)
        dur_box.addWidget(self.duration_spin)
        dur_box.addWidget(self.duration_unit)
        dur_box.addStretch()
        right_col.addLayout(dur_box)

        # Interval (below duration)
        int_box = QtWidgets.QHBoxLayout()
        int_box.setAlignment(QtCore.Qt.AlignVCenter)
        int_label = QtWidgets.QLabel("Interval:")
        int_label.setMinimumWidth(150)
        int_label.setFixedHeight(36)
        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(1, 1000000)
        self.interval_spin.setValue(60)
        self.interval_spin.setFixedHeight(36)
        self.interval_spin.setFixedWidth(90)
        self.interval_unit = QtWidgets.QComboBox()
        for lbl, sec in INTERVAL_UNITS:
            self.interval_unit.addItem(lbl, sec)
        self.interval_unit.setFixedHeight(36)
        self.interval_unit.setFixedWidth(150)
        int_box.addWidget(int_label)
        int_box.addWidget(self.interval_spin)
        int_box.addWidget(self.interval_unit)
        int_box.addStretch()
        right_col.addLayout(int_box)

        params_layout.addLayout(right_col, stretch=1)

        # place the groupbox under the Test Parameters label
        content.addWidget(g_params, 1, 0, 1, 2)

        # ======= Reporting bold label (top-left for its section) =======
        report_label = QtWidgets.QLabel("<b>Reporting</b>")
        report_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        content.addWidget(report_label, 2, 0, 1, 1)

        # ======= Reporting GroupBox (no visible title) =======
        g_report = QtWidgets.QGroupBox()
        g_report.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        g_report.setFixedHeight(120)

        report_layout = QtWidgets.QVBoxLayout()
        report_layout.setContentsMargins(16, 8, 16, 8)
        report_layout.setSpacing(8)
        g_report.setLayout(report_layout)

        # Report generation checkbox
        self.generate_report_cb = QtWidgets.QCheckBox(
            "Generate Circuit Integrity Report after test completion"
        )
        self.generate_report_cb.setChecked(True)
        self.generate_report_cb.setFixedHeight(36)
        self.generate_report_cb.setToolTip(
            "Automatically generate PDF report with plots and weather data when tests complete"
        )
        report_layout.addWidget(self.generate_report_cb)

        # ICAO input moved into Reporting panel (aligned and sized like other controls)
        icao_row = QtWidgets.QHBoxLayout()
        icao_row.setAlignment(QtCore.Qt.AlignVCenter)
        icao_label = QtWidgets.QLabel("Airport ICAO Code:")
        icao_label.setFixedWidth(150)
        icao_label.setFixedHeight(36)
        # icao_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.icao_input = QtWidgets.QLineEdit("KCUB")
        self.icao_input.setPlaceholderText("e.g. KCUB")
        self.icao_input.setMaxLength(4)
        self.icao_input.setFixedHeight(36)
        self.icao_input.setFixedWidth(90)
        # self.icao_input.setAlignment(QtCore.Qt.AlignVCenter)
        self.icao_input.setToolTip("Enter the 4-letter ICAO code for the nearest airport (e.g., KCUB)")
        icao_row.addWidget(icao_label)
        icao_row.addWidget(self.icao_input)
        icao_row.addStretch()
        report_layout.addLayout(icao_row)

        # place the reporting groupbox below its label
        content.addWidget(g_report, 3, 0, 1, 2)

        # Balanced columns
        content.setColumnStretch(0, 1)
        content.setColumnStretch(1, 1)

        # ======= Buttons Row =======
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setContentsMargins(0, 8, 0, 0)

        self.start_btn = QtWidgets.QPushButton("Start Automation")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.setMinimumWidth(160)
        self.start_btn.setToolTip("Start automated test execution")

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setMinimumHeight(42)
        self.stop_btn.setMinimumWidth(100)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop automation at next opportunity")

        self.run_single_btn = QtWidgets.QPushButton("Run Single Test")
        self.run_single_btn.setMinimumHeight(42)
        self.run_single_btn.setMinimumWidth(160)
        self.run_single_btn.setToolTip("Run a single test run immediately using current test parameters")

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_single_btn)

        outer.addLayout(btn_row)

        # ======= Log area =======
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        self.log.setPlaceholderText("Status and log messages will appear here...")
        outer.addWidget(self.log, stretch=1)

        # Output path display
        path_row = QtWidgets.QHBoxLayout()
        path_row.addStretch()
        path_label = QtWidgets.QLabel("Output Path:")
        path_label.setStyleSheet("color: #555; font-size: 14px;")
        self.output_path_value = QtWidgets.QLabel("Not set")
        self.output_path_value.setStyleSheet("color: #555; font-size: 14px;")
        path_row.addWidget(path_label)
        path_row.addWidget(self.output_path_value)
        outer.addLayout(path_row)

        # Connect signals
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.run_single_btn.clicked.connect(self.run_single_test)

    # ======= Helpers for UI/worker control =======
    def append_log(self, t: str):
        self.log.append(t)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _update_output_path_label(self, path: Optional[Path]):
        if path:
            self.output_path_value.setText(str(path))
        else:
            self.output_path_value.setText("Not set")

    def _prepare_output_directory(self) -> Path:
        base_root = Reporting.get_default_output_root()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_root / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def start(self):
        total = int(self.duration_spin.value()) * int(self.duration_unit.currentData())
        interval = int(self.interval_spin.value()) * int(self.interval_unit.currentData())

        if interval <= 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Interval", "Interval must be greater than 0")
            return

        generate_report = self.generate_report_cb.isChecked()
        icao_code = self.icao_input.text()
        test_voltage = self.voltage_spin.value()
        is_grounded = self.grounded_cb.isChecked()

        if not icao_code or len(icao_code) != 4:
            QtWidgets.QMessageBox.warning(self, "Invalid ICAO Code", "Please enter a valid 4-character airport ICAO code.")
            return

        # prepare to run: clear any prior stop signals
        STOP_EVENT.clear()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.append_log(log_msg("GUI", f"Starting automation: total={total}s, interval={interval}s, report={'enabled' if generate_report else 'disabled'}"))

        output_dir: Optional[Path] = None
        if generate_report:
            try:
                output_dir = self._prepare_output_directory()
                self.current_output_dir = output_dir
                self._update_output_path_label(output_dir)
            except Exception as exc:
                self.append_log(log_msg("GUI", f"Failed to prepare output directory: {exc}"))
                QtWidgets.QMessageBox.critical(self, "Output Directory Error", "Could not prepare the output directory. Please check file permissions and try again.")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
        else:
            self.current_output_dir = None
            self._update_output_path_label(None)

        self.worker = AutomationWorker(total, interval, icao_code, test_voltage, is_grounded, generate_report, output_dir=output_dir)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.done)
        self.worker.report_signal.connect(self.append_log)
        self.worker.start()

    def run_single_test(self):
        interval = int(self.interval_spin.value()) * int(self.interval_unit.currentData())
        if interval <= 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Interval", "Interval must be greater than 0")
            return

        generate_report = self.generate_report_cb.isChecked()
        icao_code = self.icao_input.text()
        test_voltage = self.voltage_spin.value()
        is_grounded = self.grounded_cb.isChecked()

        if not icao_code or len(icao_code) != 4:
            QtWidgets.QMessageBox.warning(self, "Invalid ICAO Code", "Please enter a valid 4-character airport ICAO code.")
            return

        total = interval

        # clear any prior stop signals
        STOP_EVENT.clear()

        output_dir: Optional[Path] = None
        if generate_report:
            try:
                output_dir = self._prepare_output_directory()
                self.current_output_dir = output_dir
                self._update_output_path_label(output_dir)
            except Exception as exc:
                self.append_log(log_msg("GUI", f"Failed to prepare output directory: {exc}"))
                QtWidgets.QMessageBox.critical(self, "Output Directory Error", "Could not prepare the output directory. Please check file permissions and try again.")
                return
        else:
            self.current_output_dir = None
            self._update_output_path_label(None)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.append_log(log_msg("GUI", f"Running single test: interval={interval}s, report={'enabled' if generate_report else 'disabled'}"))

        self.worker = AutomationWorker(total, interval, icao_code, test_voltage, is_grounded, generate_report, output_dir=output_dir)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.done)
        self.worker.report_signal.connect(self.append_log)
        self.worker.start()

    def stop(self):
        """
        Request worker to stop immediately and set STOP_EVENT so low-level helpers abort.
        """
        if self.worker:
            self.worker.stop()  # sets stop_flag and STOP_EVENT
        else:
            STOP_EVENT.set()  # still set event so any ongoing low-level ops abort
        self.append_log(log_msg("GUI", "Stop requested..."))
        self.stop_btn.setEnabled(False)

    def done(self, ok: bool):
        status = "completed successfully" if ok else "stopped or encountered errors"
        self.append_log(log_msg("GUI", f"Automation {status}"))

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None

# ========== MAIN ==========
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    screen = app.primaryScreen().availableGeometry()
    win.resize(1400, 900)
    win.move((screen.width() - win.width()) // 2, (screen.height() - win.height()) // 2)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
