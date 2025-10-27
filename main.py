

"""
QT Insite Automation GUI

Provides a professional GUI for automating Vitrek QT Insite testing and generating
circuit integrity reports with resistance measurements and weather data.

Features:
- Automated QT Insite application control via PyAutoGUI
- Scheduled test execution with configurable duration and intervals
- Integrated report generation with plots and weather data
- Atlantic Electric branded interface
- Real-time logging and status updates

Dependencies:
- PyQt5: GUI framework
- pyautogui: UI automation
- pygetwindow: Window management
- Reporting.py: Report generation module

Author: Joseph Pauls

TODO
- Add checks for whether QT Insite is already running before launching or user is already logged in (auto-click logout before loggin in)
- Add options for user to input test voltage level, grounded/ungrounded checkbox, minimum resistance, and integrate those values into the automation sequence
        --- The same sequence will get edited every time to avoid having to make a new sequence
- Add option to run a single test immediately to see results without scheduling (with its own configuration panel)
- Add input box  on gui for airport ICAO code or latitude and longitude coordinates for the weather data
- Add option to control output folder name and where the files will go (browse for folder, option to create new folder, OR, purely automated based on timestamp of when test was run)

"""

import sys
import time
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyautogui
import pygetwindow as gw
from PyQt5 import QtWidgets, QtGui, QtCore

# Import report generation module
import Reporting

# ========== CONFIGURATION ==========
# QT Insite Application Settings
USERNAME = "testuser"
PASSWORD = "AtlanticElectric"
JAVAW_PATH = r"C:\Program Files\Vitrek\QtInsite\jx\bin\javaw.exe"
START_IN_PATH = r"C:\Program Files\Vitrek\QtInsite\app"
QTINSITE_JAR = "QtInsite.jar"
WINDOW_TITLE_PART = "Vitrek QT Insite"

# UI Automation Coordinates (screen positions for clicks)
DEFAULT_COORDS = {
    "username_field": (875, 600),
    "run_field": (850, 405),
    "start_field": (1425, 490),
    "system_config": (510, 400),
    "verify_button": (800, 865),
}

# Timing Configuration
MOVE_DURATION = 0.5      # Mouse movement duration in seconds
ACTION_PAUSE = 0.18      # Pause after each action
pyautogui.PAUSE = 0.05   # PyAutoGUI default pause

# Visual Assets
LOGO_PATH = "AtlanticElectricLogo.png"

# Time Unit Conversion
DURATION_UNITS = [("Minutes", 60), ("Hours", 3600), ("Days", 86400)]
INTERVAL_UNITS = [("Seconds", 1), ("Minutes", 60), ("Hours", 3600)]

# Atlantic Electric Brand Colors
ACCENT_RED = "#b71c1c"
BG = "#f6f6f6"
TEXT = "#222"
MUTED = "#555"
# ====================================


# ========== HELPER FUNCTIONS ==========

def log_msg(tag: str, msg: str) -> str:
    """
    Create timestamped log message.
    
    Args:
        tag: Category/source of the message (e.g., "GUI", "WORKER")
        msg: The log message content
        
    Returns:
        Formatted log string with timestamp
    """
    s = f"[{time.strftime('%H:%M:%S')}] {tag}: {msg}"
    print(s)
    return s


def find_qt_window(title_part: str = WINDOW_TITLE_PART) -> Optional[object]:
    """
    Find QT Insite application window by title substring.
    
    Args:
        title_part: Substring to search for in window titles
        
    Returns:
        Window object if found, None otherwise
    """
    for w in gw.getAllWindows():
        try:
            if title_part in w.title:
                return w
        except Exception:
            continue
    return None


def activate_qt_insite_window() -> Optional[object]:
    """
    Find, restore (if minimized), and activate the QT Insite window.
    
    Returns:
        Window object if successfully activated, None otherwise
    """
    win = find_qt_window()
    if not win:
        return None
    
    # Restore if minimized
    if win.isMinimized:
        win.restore()
        time.sleep(0.5)
    
    # Bring to foreground
    win.activate()
    time.sleep(0.6)
    return win


def launch_qtinsite_using_shortcut_style(wait: float = 6.0) -> bool:
    """
    Launch QT Insite application using Java command.
    
    Args:
        wait: Seconds to wait after launch before checking if window opened
        
    Returns:
        True if application launched successfully, False otherwise
    """
    # Check if already running
    if find_qt_window():
        return True
    
    try:
        subprocess.Popen(
            [JAVAW_PATH, "-jar", QTINSITE_JAR],
            cwd=START_IN_PATH
        )
        time.sleep(wait)
        return find_qt_window() is not None
    except Exception as e:
        log_msg("LAUNCH", str(e))
        return False


def click_at(x: int, y: int, dur: float = MOVE_DURATION):
    """
    Move mouse to coordinates and click.
    
    Args:
        x: Screen X coordinate
        y: Screen Y coordinate
        dur: Duration of mouse movement in seconds
    """
    pyautogui.moveTo(x, y, duration=dur)
    time.sleep(ACTION_PAUSE)
    pyautogui.click()
    time.sleep(ACTION_PAUSE)


def login_coords() -> bool:
    """
    Perform login using hardcoded screen coordinates.
    
    Types username, tabs to password field, types password, and presses enter.
    
    Returns:
        True if successful, False if exception occurred
    """
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
    """
    Click system config and verify button using hardcoded coordinates.
    
    Returns:
        True if successful, False if exception occurred
    """
    try:
        sx, sy = DEFAULT_COORDS["system_config"]
        vx, vy = DEFAULT_COORDS["verify_button"]
        click_at(sx, sy)
        click_at(vx, vy)
        return True
    except Exception:
        return False


def run_test_coords() -> bool:
    """
    Start a test run using hardcoded screen coordinates.
    
    Clicks the run field and then the start button.
    
    Returns:
        True if successful, False if exception occurred
    """
    try:
        rx, ry = DEFAULT_COORDS["run_field"]
        sx, sy = DEFAULT_COORDS["start_field"]
        click_at(rx, ry)
        click_at(sx, sy)
        return True
    except Exception:
        return False


# ========== WORKER THREAD ==========

class AutomationWorker(QtCore.QThread):
    """
    Background worker thread for running automated test sequences.
    
    Handles launching QT Insite, performing login, running scheduled tests,
    and generating reports when complete. Runs in a separate thread to avoid
    blocking the GUI.
    
    Signals:
        log_signal: Emits log messages (str) for display in GUI
        finished_signal: Emits completion status (bool) when automation finishes
        report_signal: Emits status message when report generation completes
    """
    
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(bool)
    report_signal = QtCore.pyqtSignal(str)

    def __init__(self, total_s: int, interval_s: int, generate_report: bool = True, 
                 output_dir: Optional[Path] = None, parent=None):
        """
        Initialize automation worker.
        
        Args:
            total_s: Total duration to run tests (seconds)
            interval_s: Time between test runs (seconds)
            generate_report: Whether to generate PDF report at completion
            output_dir: Directory where report artifacts should be written
            parent: Parent QObject
        """
        super().__init__(parent)
        self.total_s = int(total_s)
        self.interval_s = max(1, int(interval_s))
        self.generate_report = generate_report
        self.output_dir = Path(output_dir) if output_dir else None
        self.stop_flag = False

    def log(self, m: str):
        """Emit log message to GUI."""
        self.log_signal.emit(log_msg("WORKER", m))

    def stop(self):
        """Request worker to stop at next opportunity."""
        self.stop_flag = True

    def run(self):
        """
        Main worker execution method.
        
        Workflow:
        1. Launch QT Insite application
        2. Perform login and connection verification
        3. Run tests at scheduled intervals
        4. Generate report (if enabled)
        5. Emit completion signal
        """
        try:
            # Step 1: Launch application
            self.log("Launching QT Insite...")
            if not launch_qtinsite_using_shortcut_style():
                self.log("❌ Launch failed")
                self.finished_signal.emit(False)
                return
            
            # Step 2: Activate window
            win = activate_qt_insite_window()
            if not win:
                self.log("❌ Window not found")
                self.finished_signal.emit(False)
                return
            
            # Step 4: Login
            self.log("Logging in...")
            login_coords()
            
            # Step 5: Verify connection
            self.log("Verifying connection...")
            verify_connection_coords()
            
            # Step 6: Calculate number of test runs
            runs = max(1, self.total_s // self.interval_s)
            self.log(f"Starting {runs} test runs (interval: {self.interval_s}s)")
            
            # Step 7: Execute scheduled tests
            for i in range(1, runs + 1):
                if self.stop_flag:
                    self.log("⚠️  Stopped by user request")
                    self.finished_signal.emit(False)
                    return
                
                self.log(f"▶️  Run {i}/{runs}")
                run_test_coords()
                
                # Wait for interval (checking stop flag periodically)
                elapsed = 0.0
                while elapsed < self.interval_s:
                    if self.stop_flag:
                        self.log("⚠️  Stopped by user request")
                        self.finished_signal.emit(False)
                        return
                    time.sleep(0.5)
                    elapsed += 0.5
            
            # Step 8: Generate report if enabled
            if self.generate_report:
                if not self.output_dir:
                    self.log("⚠️  No output directory configured; skipping report generation")
                else:
                    self.log(f"📊 Generating report in {self.output_dir}...")
                    self.report_signal.emit("Generating report, please wait...")
                    try:
                        result = Reporting.generate_report(self.output_dir)
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
            
            # Step 9: Complete
            self.log("✅ Automation completed successfully")
            self.finished_signal.emit(True)
            
        except Exception:
            error_trace = traceback.format_exc()
            self.log(f"❌ Error:\n{error_trace}")
            self.finished_signal.emit(False)


# ========== GUI STYLING ==========

# Atlantic Electric GUI Style Sheet
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
    margin-top: 8px;
    padding-top: 12px;
}}
QGroupBox::title {{
    left: 10px;
    padding: 4px;
    font-weight: bold;
    color: {MUTED};
    font-size: 15px;
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
    padding: 8px;
    border-radius: 6px;
    font-size: 16px;
    min-height: 36px;
}}
QTextEdit {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 15px;
}}
QCheckBox {{
    spacing: 8px;
    font-size: 16px;
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
    """
    Main application window for QT Insite Automation.
    
    Provides GUI for:
    - Configuring test schedule (duration and interval)
    - Starting/stopping automated tests
    - Generating circuit integrity reports
    - Viewing real-time logs
    """
    
    def __init__(self):
        """Initialize main window and UI components."""
        super().__init__()
        self.setWindowTitle("QT Insite Automation — Atlantic Electric")
        self.setMinimumSize(920, 560)
        self.worker: Optional[AutomationWorker] = None
        self.current_output_dir: Optional[Path] = None
        self._build_ui()
        self.setStyleSheet(STYLE)

    def _build_ui(self):
        """
        Build the user interface layout.
        
        Layout structure:
        - Header: Title and logo
        - Schedule: Duration and interval configuration
        - Options: Report generation checkbox
        - Buttons: Start and Stop controls
        - Log: Real-time status and log display
        """
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(14)

        # ========== HEADER SECTION ==========
        header_h = QtWidgets.QHBoxLayout()
        header_h.setSpacing(8)

        # Left: Title and subtitle
        text_v = QtWidgets.QVBoxLayout()
        text_v.setSpacing(6)
        
        title = QtWidgets.QLabel("QT Insite Automation")
        title.setObjectName("headerTitle")
        
        subtitle = QtWidgets.QLabel("Atlantic Electric — Automated Test Runner")
        subtitle.setObjectName("subtitle")
        
        text_v.addWidget(title)
        text_v.addWidget(subtitle)
        text_v.addStretch()
        header_h.addLayout(text_v, stretch=3)

        # Right: Logo
        logo_lbl = QtWidgets.QLabel()
        logo_lbl.setFixedSize(200, 200)
        logo_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        logo_path = Path(LOGO_PATH)
        if logo_path.exists():
            pix = QtGui.QPixmap(str(logo_path)).scaled(
                200, 200, 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
        else:
            # Placeholder if logo not found
            pix = QtGui.QPixmap(200, 200)
            pix.fill(QtGui.QColor(ACCENT_RED))
        logo_lbl.setPixmap(pix)

        # Logo container (right-aligned)
        logo_container = QtWidgets.QHBoxLayout()
        logo_container.addStretch()
        logo_container.addWidget(logo_lbl)
        header_h.addLayout(logo_container, stretch=1)

        outer.addLayout(header_h)

        # ========== SCHEDULE SECTION ==========
        g_sched = QtWidgets.QGroupBox("Test Schedule")
        sched_v = QtWidgets.QVBoxLayout()
        sched_v.setContentsMargins(12, 16, 12, 12)
        sched_v.setSpacing(6)
        g_sched.setLayout(sched_v)

        # Schedule row: Duration and Interval side-by-side
        schedule_row = QtWidgets.QHBoxLayout()
        schedule_row.setSpacing(16)
        schedule_row.setContentsMargins(0, 0, 0, 0)

        # Duration controls
        duration_widget = QtWidgets.QWidget()
        dur_h = QtWidgets.QHBoxLayout(duration_widget)
        dur_h.setContentsMargins(0, 0, 0, 0)
        dur_h.setSpacing(6)
        
        dur_label = QtWidgets.QLabel("Total Duration:")
        dur_label.setMinimumWidth(120)
        
        self.duration_spin = QtWidgets.QSpinBox()
        self.duration_spin.setRange(0, 1000000)
        self.duration_spin.setValue(5)
        self.duration_spin.setMinimumHeight(36)
        
        self.duration_unit = QtWidgets.QComboBox()
        for lbl, sec in DURATION_UNITS:
            self.duration_unit.addItem(lbl, sec)
        self.duration_unit.setMinimumHeight(36)
        
        dur_h.addWidget(dur_label)
        dur_h.addWidget(self.duration_spin)
        dur_h.addWidget(self.duration_unit)
        dur_h.addStretch()

        # Interval controls
        interval_widget = QtWidgets.QWidget()
        int_h = QtWidgets.QHBoxLayout(interval_widget)
        int_h.setContentsMargins(0, 0, 0, 0)
        int_h.setSpacing(6)
        
        int_label = QtWidgets.QLabel("Interval:")
        int_label.setMinimumWidth(60)
        
        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(1, 1000000)
        self.interval_spin.setValue(60)
        self.interval_spin.setMinimumHeight(36)
        
        self.interval_unit = QtWidgets.QComboBox()
        for lbl, sec in INTERVAL_UNITS:
            self.interval_unit.addItem(lbl, sec)
        self.interval_unit.setMinimumHeight(36)
        
        int_h.addWidget(int_label)
        int_h.addWidget(self.interval_spin)
        int_h.addWidget(self.interval_unit)
        int_h.addStretch()

        # Add both to schedule row
        schedule_row.addWidget(duration_widget, stretch=0)
        schedule_row.addWidget(interval_widget, stretch=0)
        schedule_row.addStretch(1)

        sched_v.addLayout(schedule_row)
        outer.addWidget(g_sched)

        # ========== OPTIONS SECTION ==========
        g_options = QtWidgets.QGroupBox("Options")
        options_v = QtWidgets.QVBoxLayout()
        options_v.setContentsMargins(12, 16, 12, 12)
        options_v.setSpacing(6)
        g_options.setLayout(options_v)
        
        # Report generation checkbox
        self.generate_report_cb = QtWidgets.QCheckBox(
            "Generate Circuit Integrity Report after test completion"
        )
        self.generate_report_cb.setChecked(True)
        self.generate_report_cb.setToolTip(
            "Automatically generate PDF report with plots and weather data when tests complete"
        )
        options_v.addWidget(self.generate_report_cb)
        
        outer.addWidget(g_options)

        # ========== BUTTONS SECTION ==========
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)
        
        self.start_btn = QtWidgets.QPushButton("Start Automation")
        self.start_btn.setToolTip("Start automated test execution")
        
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop automation at next opportunity")
        
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        
        outer.addLayout(btn_row)

        # ========== LOG SECTION ==========
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(240)
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

        # Connect button signals
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def append_log(self, t: str):
        """
        Append message to log display and auto-scroll to bottom.
        
        Args:
            t: Message text to append
        """
        self.log.append(t)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum()
        )

    def _update_output_path_label(self, path: Optional[Path]):
        """Update the output path label to reflect the active directory."""
        if path:
            self.output_path_value.setText(str(path))
        else:
            self.output_path_value.setText("Not set")

    def _prepare_output_directory(self) -> Path:
        """Create and return a timestamped output directory for this run."""
        base_root = Reporting.get_default_output_root()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_root / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def start(self):
        """
        Start automation workflow.
        
        Validates inputs, disables start button, enables stop button,
        creates worker thread, and begins execution.
        """
        # Calculate total and interval in seconds
        total = int(self.duration_spin.value()) * int(self.duration_unit.currentData())
        interval = int(self.interval_spin.value()) * int(self.interval_unit.currentData())
        
        # Validate interval
        if interval <= 0:
            QtWidgets.QMessageBox.warning(
                self, 
                "Invalid Interval", 
                "Interval must be greater than 0"
            )
            return
        
        # Get report generation option
        generate_report = self.generate_report_cb.isChecked()
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Log start
        self.append_log(log_msg(
            "GUI", 
            f"Starting automation: total={total}s, interval={interval}s, "
            f"report={'enabled' if generate_report else 'disabled'}"
        ))

        # Configure output directory if report generation is enabled
        output_dir: Optional[Path] = None
        if generate_report:
            try:
                output_dir = self._prepare_output_directory()
                self.current_output_dir = output_dir
                self._update_output_path_label(output_dir)
            except Exception as exc:
                self.append_log(log_msg("GUI", f"Failed to prepare output directory: {exc}"))
                QtWidgets.QMessageBox.critical(
                    self,
                    "Output Directory Error",
                    "Could not prepare the output directory. Please check file permissions and try again."
                )
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
        else:
            self.current_output_dir = None
            self._update_output_path_label(None)
        
        # Create and start worker thread
        self.worker = AutomationWorker(total, interval, generate_report, output_dir=output_dir)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.done)
        self.worker.report_signal.connect(self.append_log)
        self.worker.start()

    def stop(self):
        """
        Request worker to stop automation.
        
        Disables stop button and sends stop signal to worker thread.
        """
        if self.worker:
            self.worker.stop()
            self.append_log(log_msg("GUI", "Stop requested..."))
        self.stop_btn.setEnabled(False)

    def done(self, ok: bool):
        """
        Handle automation completion.
        
        Args:
            ok: True if automation completed successfully, False if stopped/error
        """
        status = "completed successfully" if ok else "stopped or encountered errors"
        self.append_log(log_msg("GUI", f"Automation {status}"))
        
        # Reset UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None


# ========== MAIN ENTRY POINT ==========

def main():
    """
    Main application entry point.
    
    Creates QApplication, initializes main window, centers it on screen,
    and starts the event loop.
    """
    app = QtWidgets.QApplication(sys.argv)
    
    # Create main window
    win = MainWindow()
    
    # Center window on primary screen
    screen = app.primaryScreen().availableGeometry()
    win.resize(1000, 640)
    win.move(
        (screen.width() - win.width()) // 2,
        (screen.height() - win.height()) // 2
    )
    
    # Show window and start event loop
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
