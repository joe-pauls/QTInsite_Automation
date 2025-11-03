
import sys, time, subprocess, traceback
from pathlib import Path
import pyautogui, pygetwindow as gw
from PyQt5 import QtWidgets, QtGui, QtCore

# ---------- CONFIG ----------
USERNAME = "testuser"
PASSWORD = "AtlanticElectric"
JAVAW_PATH = r"C:\Program Files\Vitrek\QtInsite\jx\bin\javaw.exe"
START_IN_PATH = r"C:\Program Files\Vitrek\QtInsite\app"
QTINSITE_JAR = "QtInsite.jar"
LOGO_PATH = "AtlanticElectricLogo.png"

DEFAULT_COORDS = {
    "username_field": (875, 600),
    "run_field": (850, 405),
    "start_field": (1425, 490),
    "system_config": (510, 400),
    "verify_button": (800, 865),
}
MOVE_DURATION = 0.5
ACTION_PAUSE = 0.18
pyautogui.PAUSE = 0.05
WINDOW_TITLE_PART = "Vitrek QT Insite"

# ---------- HELPERS ----------
def log_msg(tag, msg):
    s = f"[{time.strftime('%H:%M:%S')}] {tag}: {msg}"
    print(s)
    return s

def find_qt_window(title_part=WINDOW_TITLE_PART):
    for w in gw.getAllWindows():
        try:
            if title_part in w.title:
                return w
        except Exception:
            continue
    return None

def activate_qt_insite_window():
    win = find_qt_window()
    if not win: return None
    if win.isMinimized: 
        win.restore()
        time.sleep(0.5)
    win.activate()
    time.sleep(0.6)
    return win

def launch_qtinsite_using_shortcut_style(wait=6.0):
    if find_qt_window(): return True
    try:
        subprocess.Popen([JAVAW_PATH, "-jar", QTINSITE_JAR], cwd=START_IN_PATH)
        time.sleep(wait)
        return find_qt_window() is not None
    except Exception as e:
        log_msg("LAUNCH", str(e))
        return False

def click_at(x, y, dur=MOVE_DURATION):
    pyautogui.moveTo(x, y, duration=dur)
    time.sleep(ACTION_PAUSE)
    pyautogui.click()
    time.sleep(ACTION_PAUSE)

def login_coords():
    try:
        ux, uy = DEFAULT_COORDS["username_field"]
        click_at(ux, uy)
        pyautogui.typewrite(USERNAME, interval=.05)
        pyautogui.press("tab")
        pyautogui.typewrite(PASSWORD, interval=.05)
        pyautogui.press("enter")
        return True
    except Exception:
        return False

def verify_connection_coords():
    try:
        sx, sy = DEFAULT_COORDS["system_config"]
        vx, vy = DEFAULT_COORDS["verify_button"]
        click_at(sx, sy)
        click_at(vx, vy)
        return True
    except Exception:
        return False

def run_test_coords():
    try:
        rx, ry = DEFAULT_COORDS["run_field"]
        sx, sy = DEFAULT_COORDS["start_field"]
        click_at(rx, ry)
        click_at(sx, sy)
        return True
    except Exception:
        return False

# ---------- WORKER ----------
class AutomationWorker(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(bool)

    def __init__(self, total_s, interval_s, parent=None):
        super().__init__(parent)
        self.total_s = int(total_s)
        self.interval_s = max(1, int(interval_s))
        self.stop_flag = False

    def log(self, m):
        self.log_signal.emit(log_msg("WORKER", m))

    def stop(self):
        self.stop_flag = True

    def run(self):
        try:
            if not launch_qtinsite_using_shortcut_style():
                self.log("Launch failed")
                self.finished_signal.emit(False)
                return
            win = activate_qt_insite_window()
            if not win:
                self.log("No window")
                self.finished_signal.emit(False)
                return
            login_coords()
            verify_connection_coords()
            runs = max(1, self.total_s // self.interval_s)
            for i in range(1, runs + 1):
                if self.stop_flag:
                    self.finished_signal.emit(False)
                    return
                self.log(f"Run {i}/{runs}")
                run_test_coords()
                t = 0.0
                while t < self.interval_s:
                    if self.stop_flag:
                        self.finished_signal.emit(False)
                        return
                    time.sleep(0.5)
                    t += 0.5
            self.finished_signal.emit(True)
        except Exception:
            self.log(traceback.format_exc())
            self.finished_signal.emit(False)

# ---------- GUI THEME ----------
ACCENT_RED = "#b71c1c"
BG = "#f6f6f6"
TEXT = "#222"
MUTED = "#555"

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
    /* ensure internal spacing so title doesn't overlap content */
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

DURATION_UNITS = [("Minutes", 60), ("Hours", 3600), ("Days", 86400)]
INTERVAL_UNITS = [("Seconds", 1), ("Minutes", 60), ("Hours", 3600)]

# ---------- MAIN WINDOW ----------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QT Insite Automation — Atlantic Electric")
        self.setMinimumSize(920, 560)
        self.worker = None
        self._build_ui()
        self.setStyleSheet(STYLE)

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(14)

        # --- Header: text left, logo right ---
        header_h = QtWidgets.QHBoxLayout()
        header_h.setSpacing(8)

        # Left text column
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

        # Right logo (bigger)
        logo_lbl = QtWidgets.QLabel()
        logo_lbl.setFixedSize(200, 200)
        logo_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        logo_path = Path(LOGO_PATH)
        if logo_path.exists():
            pix = QtGui.QPixmap(str(logo_path)).scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        else:
            pix = QtGui.QPixmap(200, 200)
            pix.fill(QtGui.QColor(ACCENT_RED))
        logo_lbl.setPixmap(pix)

        # put little padding around logo using a horizontal layout so it hugs the right edge
        logo_container = QtWidgets.QHBoxLayout()
        logo_container.addStretch()
        logo_container.addWidget(logo_lbl)
        header_h.addLayout(logo_container, stretch=1)

        outer.addLayout(header_h)

        # --- Launcher info group (white background control) ---
        g_launch = QtWidgets.QGroupBox("Launcher (hardcoded)")
        g_launch_layout = QtWidgets.QVBoxLayout()
        # ensure internal spacing so title doesn't overlap and content has breathing room
        g_launch_layout.setContentsMargins(12, 16, 12, 12)
        g_launch_layout.setSpacing(6)
        g_launch.setLayout(g_launch_layout)

        # Read-only line edit styled like the white card
        launcher_path_edit = QtWidgets.QLineEdit(f"{JAVAW_PATH} -jar {QTINSITE_JAR}")
        launcher_path_edit.setReadOnly(True)
        launcher_path_edit.setMinimumHeight(40)
        launcher_path_edit.setCursorPosition(0)  # show beginning of path
        g_launch_layout.addWidget(launcher_path_edit)
        outer.addWidget(g_launch)

        # --- Schedule group: keep labels close to their spinboxes on one line ---
        g_sched = QtWidgets.QGroupBox("Schedule")
        sched_v = QtWidgets.QVBoxLayout()
        sched_v.setContentsMargins(12, 16, 12, 12)
        sched_v.setSpacing(6)
        g_sched.setLayout(sched_v)

        # single horizontal row containing both duration and interval groups
        schedule_row = QtWidgets.QHBoxLayout()
        schedule_row.setSpacing(16)
        schedule_row.setContentsMargins(0, 0, 0, 0)

        # Duration group (compact)
        duration_widget = QtWidgets.QWidget()
        dur_h = QtWidgets.QHBoxLayout(duration_widget)
        dur_h.setContentsMargins(0, 0, 0, 0)
        dur_h.setSpacing(6)
        dur_label = QtWidgets.QLabel("Total Duration:")
        dur_label.setMinimumWidth(120)  # keep label width steady
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
        dur_h.addStretch()  # keep controls left-aligned within their area

        # Interval group (compact)
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

        # add both compact groups to the same row; use stretch to keep them left-aligned and not overly spread
        schedule_row.addWidget(duration_widget, stretch=0)
        schedule_row.addWidget(interval_widget, stretch=0)
        schedule_row.addStretch(1)

        # add the row to the schedule group
        sched_v.addLayout(schedule_row)
        outer.addWidget(g_sched)

        # --- Buttons row ---
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)
        self.start_btn = QtWidgets.QPushButton("Start Automation")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        # --- Log view ---
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(240)
        outer.addWidget(self.log, stretch=1)

        # Connect buttons (these signals assume other methods exist on the class)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def append_log(self, t):
        self.log.append(t)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def start(self):
        total = int(self.duration_spin.value()) * int(self.duration_unit.currentData())
        interval = int(self.interval_spin.value()) * int(self.interval_unit.currentData())
        if interval <= 0:
            QtWidgets.QMessageBox.warning(self, "Error", "Interval must be > 0")
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.append_log(log_msg("GUI", f"Starting: total={total}s, interval={interval}s"))
        self.worker = AutomationWorker(total, interval)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.done)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.append_log(log_msg("GUI", "Stop requested"))
        self.stop_btn.setEnabled(False)

    def done(self, ok):
        self.append_log(log_msg("GUI", "Automation finished" if ok else "Automation stopped/errors"))
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None

# ---------- MAIN ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    # center window on screen
    screen = app.primaryScreen().availableGeometry()
    win.resize(1000, 640)
    win.move((screen.width() - win.width()) // 2, (screen.height() - win.height()) // 2)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()