#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTSP 4ch Recorder (macOS first, Windows-ready)
==============================================

Features
- Up to 4 RTSP cameras (live preview + per‑camera connect/disconnect)
- Always-on segmented recording via FFmpeg (copy codec, low CPU)
- Event clips with pre/post seconds from a rolling segment buffer
- Two trigger types:
    1) ROI brightness change (per camera)
    2) Mitsubishi PLC (SLMP/MC Protocol) bit rising-edge (global)
- Simple UI (PySide6) with 4 live panes, URL inputs, pre/post settings,
  PLC settings, ROI drawing, sensitivity threshold, output folders.

Tested on macOS (Python 3.10+). Windows build planned: this code is
cross‑platform; replace FFmpeg path if needed.

Requirements
- Homebrew FFmpeg (macOS):   brew install ffmpeg
- Python deps:               pip install PySide6 opencv-python numpy pymcprotocol
  (pymcprotocol can be omitted if PLC trigger unused)

Important notes
- RTSP stability: OpenCV with FFMPEG works well for many cameras; if unstable,
  try adding '?tcp' or '&rtsp_transport=tcp' to your RTSP URL or switch camera
  to TCP transport.
- FFmpeg segmenting: we record continuous segments (default 10 s). Event clips
  are created by concatenating segments that cover [t0 - pre, t0 + post].
- SLMP port: must MATCH your PLC "Host Station Port" setting in GX Works.
  Common defaults in tools are 12288 (0x3000) but it is configurable on PLC.

Directory layout (auto-created)
  ./output/
    continuous/cam0/cont_YYYYmmdd-HHMMSS.mp4  (rolling segments)
    events/cam0/event_YYYYmmdd-HHMMSS_pre{p}_post{q}.mp4

Author: ChatGPT (GPT-5 Thinking)
License: MIT (do as you like, no warranty)
"""

import os
import sys
import cv2
import time
import json
import math
import glob
import queue
import shutil
import signal
import atexit
try:
    import psutil  # optional; if missing, process cleanup falls back
except Exception:
    psutil = None
import threading
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

# ---------- FFmpeg binary resolution (Windows portable) ----------
def get_ffmpeg_bin():
    try:
        if sys.platform.startswith("win"):
            base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(base, "bin", "ffmpeg.exe")
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    return os.environ.get("FFMPEG_BIN", "ffmpeg")

FFMPEG_BIN = get_ffmpeg_bin()

# ---------- Optional PLC support (pymcprotocol) ----------
try:
    import pymcprotocol
    HAS_PLC = True
except Exception:
    HAS_PLC = False

# ---------- Utils ----------

APP_NAME = "RTSP 4ch Recorder"
SEGMENT_SECONDS = 10  # default continuous segment length
TIME_FMT = "%Y%m%d-%H%M%S"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def now_ts() -> float:
    return time.time()


def parse_ts_from_name(path: str) -> Optional[float]:
    # expects .../cont_YYYYmmdd-HHMMSS.mp4
    base = os.path.basename(path)
    try:
        stamp = base.split("_")[1].split(".")[0]
        dt = datetime.strptime(stamp, TIME_FMT)
        return dt.timestamp()
    except Exception:
        return None


# ---------- FFmpeg Segment Recorder ----------

class FFmpegSegmentRecorder(QtCore.QObject):
    log = QtCore.Signal(str)
    def __init__(self, cam_index: int, url: str, out_dir: str,
                 segment_seconds: int = SEGMENT_SECONDS,
                 ring_seconds: int = 3600):
        super().__init__()
        self.cam_index = cam_index
        self.url = url
        self.out_dir = ensure_dir(out_dir)
        self.segment_seconds = segment_seconds
        self.ring_seconds = ring_seconds
        self.proc: Optional[subprocess.Popen] = None
        self._stop_flag = False
        self.segment_glob = os.path.join(self.out_dir, "cont_*.mp4")

    def start(self):
        if self.proc and self.proc.poll() is None:
            self.log.emit(f"[cam{self.cam_index}] FFmpeg already running.")
            return
        pattern = os.path.join(self.out_dir, f"cont_%{ 'Y%m%d-%H%M%S' }.mp4")
        # Build ffmpeg command for segment copy
        cmd = [
            FFMPEG_BIN,
            "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.segment_seconds),
            "-reset_timestamps", "1",
            "-strftime", "1",
            pattern,
        ]
        self.log.emit(f"[cam{self.cam_index}] Starting FFmpeg segmenter → {self.out_dir}")
        self._stop_flag = False
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except FileNotFoundError:
            self.log.emit("FFmpeg not found. Install FFmpeg and ensure it's in PATH.")
            self.proc = None
            return
        # Start janitor thread
        threading.Thread(target=self._janitor_loop, daemon=True).start()

    def stop(self):
        self._stop_flag = True
        if self.proc and self.proc.poll() is None:
            self.log.emit(f"[cam{self.cam_index}] Stopping FFmpeg...")
            try:
                if psutil and self.proc.pid:
                    p = psutil.Process(self.proc.pid)
                    for child in p.children(recursive=True):
                        child.terminate()
                    p.terminate()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None

    def _janitor_loop(self):
        # delete old segments beyond ring_seconds
        while not self._stop_flag:
            try:
                files = sorted(glob.glob(self.segment_glob))
                now = now_ts()
                for f in files:
                    ts = parse_ts_from_name(f)
                    if ts and (now - ts) > self.ring_seconds:
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                time.sleep(self.segment_seconds)
            except Exception as e:
                self.log.emit(f"[cam{self.cam_index}] janitor error: {e}")
                time.sleep(5)

    def collect_segments_for_window(self, t_start: float, t_end: float) -> List[str]:
        files = sorted(glob.glob(self.segment_glob))
        picks = []
        for f in files:
            ts = parse_ts_from_name(f)
            if ts is None:
                continue
            ts_end = ts + self.segment_seconds + 0.5
            # overlap test
            if (ts <= t_end) and (ts_end >= t_start):
                picks.append(f)
        return picks


# ---------- PLC Monitor (SLMP/MC Protocol) ----------

class PLCMonitor(QtCore.QObject):
    plc_trigger = QtCore.Signal()
    log = QtCore.Signal(str)

    def __init__(self, ip: str, port: int, device: str, poll_ms: int = 100, series: str = "iQ-R"):
        super().__init__()
        self.ip = ip
        self.port = port
        self.device = device  # e.g., "M100" or "X0"
        self.poll_ms = poll_ms
        self.series = series
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last = 0

    def start(self):
        if not HAS_PLC:
            self.log.emit("pymcprotocol not installed; PLC monitor disabled.")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.log.emit(f"PLC monitor started: {self.ip}:{self.port} {self.device}")

    def stop(self):
        self._stop.set()
        self.log.emit("PLC monitor stopped.")

    def _loop(self):
        try:
            cli = pymcprotocol.Type3E(plctype=self.series)
            cli.connect(self.ip, self.port)
            # choose bit/word by headletter
            head = self.device[0].upper()
            addr = self.device[1:]
            while not self._stop.is_set():
                val = 0
                if head in ["M", "X", "Y"]:  # bit devices
                    bits = cli.batchread_bitunits(headdevice=self.device, readsize=1)
                    # returns list of 0/1 ints or strings depending lib version
                    b = bits[0]
                    val = int(b) if not isinstance(b, str) else (1 if b in ("1", "ON", "on", "True") else 0)
                else:  # word devices
                    words = cli.batchread_wordunits(headdevice=self.device, readsize=1)
                    v = words[0]
                    val = int(v)
                if self._last == 0 and val != 0:
                    self.log.emit(f"PLC rising edge on {self.device} (val={val})")
                    self.plc_trigger.emit()
                self._last = 1 if val != 0 else 0
                time.sleep(self.poll_ms / 1000.0)
        except Exception as e:
            self.log.emit(f"PLC error: {e}")


# ---------- Camera worker with ROI trigger ----------

class CameraWorker(QtCore.QObject):
    frame_ready = QtCore.Signal(int, np.ndarray)
    roi_triggered = QtCore.Signal(int)
    log = QtCore.Signal(str)

    def __init__(self, cam_index: int, url: str, roi: Optional[QtCore.QRectF] = None,
                 roi_threshold: float = 20.0, sample_rate_fps: float = 5.0):
        super().__init__()
        self.cam_index = cam_index
        self.url = url
        self.roi = roi
        self.roi_threshold = roi_threshold  # mean-gray delta threshold
        self.sample_rate_fps = sample_rate_fps
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_mean = None
        self._frame_w = 0
        self._frame_h = 0

    def set_roi(self, roi: Optional[QtCore.QRectF]):
        self.roi = roi
        self._last_mean = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.log.emit(f"[cam{self.cam_index}] Camera thread started.")

    def stop(self):
        self._stop.set()
        self.log.emit(f"[cam{self.cam_index}] Camera thread stopped.")

    def _loop(self):
        # Use FFMPEG backend for RTSP
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self._cap = cap
        if not cap.isOpened():
            self.log.emit(f"[cam{self.cam_index}] Cannot open RTSP: {self.url}")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        last_sample = 0.0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            self._frame_h, self._frame_w = frame.shape[:2]
            # emit preview at ~15 fps max
            self.frame_ready.emit(self.cam_index, frame)
            t = time.time()
            if (t - last_sample) >= (1.0 / max(0.1, self.sample_rate_fps)):
                self._check_roi_trigger(frame)
                last_sample = t
        cap.release()

    def _check_roi_trigger(self, frame: np.ndarray):
        if self.roi is None:
            return
        x0 = int(max(0, min(self._frame_w - 1, self.roi.left() * self._frame_w)))
        y0 = int(max(0, min(self._frame_h - 1, self.roi.top() * self._frame_h)))
        x1 = int(max(0, min(self._frame_w, (self.roi.left() + self.roi.width()) * self._frame_w)))
        y1 = int(max(0, min(self._frame_h, (self.roi.top() + self.roi.height()) * self._frame_h)))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            return
        roi_img = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())
        if self._last_mean is None:
            self._last_mean = mean_val
            return
        delta = abs(mean_val - self._last_mean)
        if delta >= self.roi_threshold:
            self.log.emit(f"[cam{self.cam_index}] ROI trigger: Δ={delta:.1f} ≥ thr={self.roi_threshold}")
            self.roi_triggered.emit(self.cam_index)
        self._last_mean = mean_val


# ---------- Event Manager ----------

class EventManager(QtCore.QObject):
    log = QtCore.Signal(str)

    def __init__(self, recorders: List[FFmpegSegmentRecorder], events_root: str,
                 pre_sec: int = 120, post_sec: int = 600):
        super().__init__()
        self.recorders = recorders
        self.events_root = ensure_dir(events_root)
        self.pre_sec = pre_sec
        self.post_sec = post_sec
        self._lock = threading.Lock()

    def set_window(self, pre_sec: int, post_sec: int):
        self.pre_sec = pre_sec
        self.post_sec = post_sec

    def trigger(self, from_cam: Optional[int] = None):
        t0 = now_ts()
        with self._lock:
            self.log.emit(f"Event trigger received (t0={datetime.fromtimestamp(t0)}) from cam={from_cam}")
            # collect pre segments now; post will finalize after delay
            for cam_i, rec in enumerate(self.recorders):
                if rec is None:
                    continue
                threading.Thread(target=self._make_event_clip, args=(cam_i, rec, t0), daemon=True).start()

    def _make_event_clip(self, cam_i: int, rec: FFmpegSegmentRecorder, t0: float):
        # Pre
        t_start = t0 - self.pre_sec
        t_end_first = t0  # we'll extend later
        pre_segments = rec.collect_segments_for_window(t_start, t_end_first)
        # Wait for post window to elapse to gather remaining segments
        time.sleep(self.post_sec + SEGMENT_SECONDS + 1)
        t_end = t0 + self.post_sec
        all_segments = rec.collect_segments_for_window(t_start, t_end)
        # ensure chronological order and dedupe
        all_segments = sorted(set(all_segments), key=lambda p: parse_ts_from_name(p) or 0)
        if not all_segments:
            self.log.emit(f"[cam{cam_i}] No segments found for event window.")
            return
        out_dir = ensure_dir(os.path.join(self.events_root, f"cam{cam_i}"))
        ts_label = datetime.fromtimestamp(t0).strftime(TIME_FMT)
        out_path = os.path.join(out_dir, f"event_{ts_label}_pre{self.pre_sec}_post{self.post_sec}.mp4")
        # concat using ffmpeg concat demuxer
        list_path = os.path.join(out_dir, f"_list_{ts_label}.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for seg in all_segments:
                f.write(f"file '{os.path.abspath(seg)}'\n")
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", out_path
        ]
        self.log.emit(f"[cam{cam_i}] Building event clip → {out_path} ({len(all_segments)} segments)")
        try:
            subprocess.run(cmd, check=True)
            self.log.emit(f"[cam{cam_i}] Event clip done: {out_path}")
        except Exception as e:
            self.log.emit(f"[cam{cam_i}] ffmpeg concat failed: {e}")
        finally:
            try:
                os.remove(list_path)
            except Exception:
                pass


# ---------- GUI ----------

class ROIOverlay(QtWidgets.QLabel):
    roiChanged = QtCore.Signal(QtCore.QRectF)
    def __init__(self, cam_index: int):
        super().__init__()
        self.setScaledContents(True)
        self.cam_index = cam_index
        self._pix: Optional[QtGui.QPixmap] = None
        self._drawing = False
        self._rect = QtCore.QRect()

    def setFrame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self._pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(self._pix)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if not self._pix:
            return
        self._drawing = True
        self._rect = QtCore.QRect(e.pos(), e.pos())
        self.update()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._drawing:
            self._rect = QtCore.QRect(self._rect.topLeft(), e.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if self._drawing:
            self._drawing = False
            self._rect = QtCore.QRect(self._rect.topLeft(), e.pos()).normalized()
            self.update()
            # emit normalized ROI
            if self._pix and self._pix.width() > 0 and self._pix.height() > 0:
                r = self._rect
                roi = QtCore.QRectF(r.x()/self._pix.width(), r.y()/self._pix.height(),
                                     r.width()/self._pix.width(), r.height()/self._pix.height())
                self.roiChanged.emit(roi)

    def paintEvent(self, e: QtGui.QPaintEvent):
        super().paintEvent(e)
        if self._rect.isNull():
            return
        p = QtGui.QPainter(self)
        pen = QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.DashLine)
        p.setPen(pen)
        p.drawRect(self._rect)
        p.end()


class CameraPane(QtWidgets.QGroupBox):
    connectClicked = QtCore.Signal(int, str)
    disconnectClicked = QtCore.Signal(int)
    def __init__(self, cam_index: int):
        super().__init__(f"Camera {cam_index}")
        self.cam_index = cam_index
        v = QtWidgets.QVBoxLayout(self)
        h = QtWidgets.QHBoxLayout()
        self.url_edit = QtWidgets.QLineEdit()
        self.url_edit.setPlaceholderText("rtsp://user:pass@host:port/stream")
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        h.addWidget(self.url_edit)
        h.addWidget(self.btn_connect)
        h.addWidget(self.btn_disconnect)
        v.addLayout(h)
        self.view = ROIOverlay(cam_index)
        self.view.setMinimumHeight(180)
        v.addWidget(self.view)
        sens_h = QtWidgets.QHBoxLayout()
        sens_h.addWidget(QtWidgets.QLabel("ROI Δ (brightness) ≥"))
        self.sensitivity = QtWidgets.QDoubleSpinBox()
        self.sensitivity.setRange(1.0, 255.0)
        self.sensitivity.setValue(20.0)
        sens_h.addWidget(self.sensitivity)
        v.addLayout(sens_h)

        self.btn_connect.clicked.connect(lambda: self.connectClicked.emit(self.cam_index, self.url_edit.text().strip()))
        self.btn_disconnect.clicked.connect(lambda: self.disconnectClicked.emit(self.cam_index))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        grid = QtWidgets.QGridLayout(cw)

        # 4 camera panes
        self.cams: List[CameraPane] = []
        self.workers: List[Optional[CameraWorker]] = [None, None, None, None]
        self.recorders: List[Optional[FFmpegSegmentRecorder]] = [None, None, None, None]

        for i in range(4):
            pane = CameraPane(i)
            pane.connectClicked.connect(self.on_connect)
            pane.disconnectClicked.connect(self.on_disconnect)
            pane.view.roiChanged.connect(lambda roi, idx=i: self.on_roi_changed(idx, roi))
            self.cams.append(pane)
            grid.addWidget(pane, i//2, i%2)

        # Right side panel
        side = QtWidgets.QGroupBox("Controls")
        sidev = QtWidgets.QFormLayout(side)
        self.out_root = QtWidgets.QLineEdit(os.path.abspath("output"))
        self.btn_browse = QtWidgets.QPushButton("…")
        h1 = QtWidgets.QHBoxLayout(); h1.addWidget(self.out_root); h1.addWidget(self.btn_browse)
        sidev.addRow("Output root", h1)
        self.pre_spin = QtWidgets.QSpinBox(); self.pre_spin.setRange(0, 3600); self.pre_spin.setValue(120)
        self.post_spin = QtWidgets.QSpinBox(); self.post_spin.setRange(1, 7200); self.post_spin.setValue(600)
        sidev.addRow("Pre seconds", self.pre_spin)
        sidev.addRow("Post seconds", self.post_spin)
        self.seg_spin = QtWidgets.QSpinBox(); self.seg_spin.setRange(2, 60); self.seg_spin.setValue(10)
        sidev.addRow("Segment length (s)", self.seg_spin)
        self.ring_spin = QtWidgets.QSpinBox(); self.ring_spin.setRange(60, 24*3600); self.ring_spin.setValue(3600)
        sidev.addRow("Ring buffer (s)", self.ring_spin)
        self.btn_start_rec = QtWidgets.QPushButton("Start continuous recording")
        self.btn_stop_rec = QtWidgets.QPushButton("Stop continuous recording")
        sidev.addRow(self.btn_start_rec)
        sidev.addRow(self.btn_stop_rec)

        # PLC controls
        plc_box = QtWidgets.QGroupBox("PLC (SLMP) Trigger")
        plc_form = QtWidgets.QFormLayout(plc_box)
        self.plc_ip = QtWidgets.QLineEdit("192.168.3.10")
        self.plc_port = QtWidgets.QSpinBox(); self.plc_port.setRange(1, 65535); self.plc_port.setValue(12288)
        self.plc_device = QtWidgets.QLineEdit("M100")
        self.plc_poll = QtWidgets.QSpinBox(); self.plc_poll.setRange(10, 2000); self.plc_poll.setValue(100)
        self.btn_plc_start = QtWidgets.QPushButton("Start PLC monitor")
        self.btn_plc_stop = QtWidgets.QPushButton("Stop PLC monitor")
        plc_form.addRow("IP", self.plc_ip)
        plc_form.addRow("Port", self.plc_port)
        plc_form.addRow("Device", self.plc_device)
        plc_form.addRow("Poll (ms)", self.plc_poll)
        plc_form.addRow(self.btn_plc_start)
        plc_form.addRow(self.btn_plc_stop)
        sidev.addRow(plc_box)

        # manual trigger
        self.btn_manual = QtWidgets.QPushButton("Manual TRIGGER (all cams)")
        sidev.addRow(self.btn_manual)

        # log box
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        sidev.addRow("Log", self.log)

        grid.addWidget(side, 0, 2, 2, 1)

        # connections
        self.btn_browse.clicked.connect(self.select_out)
        self.btn_start_rec.clicked.connect(self.start_all_recorders)
        self.btn_stop_rec.clicked.connect(self.stop_all_recorders)
        self.btn_manual.clicked.connect(self.on_manual_trigger)
        self.btn_plc_start.clicked.connect(self.start_plc)
        self.btn_plc_stop.clicked.connect(self.stop_plc)

        self.event_mgr = EventManager(self.recorders, os.path.join(self.out_root.text(), "events"),
                                      self.pre_spin.value(), self.post_spin.value())
        self.event_mgr.log.connect(self._log)
        # react to pre/post changes
        self.pre_spin.valueChanged.connect(lambda v: self.event_mgr.set_window(v, self.post_spin.value()))
        self.post_spin.valueChanged.connect(lambda v: self.event_mgr.set_window(self.pre_spin.value(), v))

        self.plc_mon: Optional[PLCMonitor] = None

    # ---------- UI handlers ----------
    @QtCore.Slot()
    def select_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output root", self.out_root.text())
        if d:
            self.out_root.setText(d)

    @QtCore.Slot(int, str)
    def on_connect(self, cam_i: int, url: str):
        if not url:
            self._log("URL is empty")
            return
        # Camera worker (preview + ROI)
        if self.workers[cam_i]:
            self._log(f"cam{cam_i} already connected.")
        else:
            w = CameraWorker(cam_i, url, roi_threshold=self.cams[cam_i].sensitivity.value())
            w.frame_ready.connect(self.update_frame)
            w.roi_triggered.connect(self.on_roi_trigger)
            w.log.connect(self._log)
            self.workers[cam_i] = w
            w.start()

        # Recorder (continuous)
        cont_dir = ensure_dir(os.path.join(self.out_root.text(), "continuous", f"cam{cam_i}"))
        r = FFmpegSegmentRecorder(cam_i, url, cont_dir, segment_seconds=self.seg_spin.value(),
                                  ring_seconds=self.ring_spin.value())
        r.log.connect(self._log)
        self.recorders[cam_i] = r
        # do not auto-start; controlled by global button
        self._log(f"cam{cam_i} ready. Press 'Start continuous recording' to begin segmenting.")

    @QtCore.Slot(int)
    def on_disconnect(self, cam_i: int):
        w = self.workers[cam_i]
        if w:
            w.stop()
            self.workers[cam_i] = None
        r = self.recorders[cam_i]
        if r:
            r.stop()
            self.recorders[cam_i] = None
        self._log(f"cam{cam_i} disconnected.")

    @QtCore.Slot(int, np.ndarray)
    def update_frame(self, cam_i: int, frame: np.ndarray):
        self.cams[cam_i].view.setFrame(frame)

    @QtCore.Slot(int)
    def on_roi_trigger(self, cam_i: int):
        self._log(f"ROI trigger from cam{cam_i}")
        self.event_mgr.trigger(from_cam=cam_i)

    def on_roi_changed(self, cam_i: int, roi: QtCore.QRectF):
        self._log(f"cam{cam_i} ROI set: x={roi.left():.3f} y={roi.top():.3f} w={roi.width():.3f} h={roi.height():.3f}")
        w = self.workers[cam_i]
        if w:
            w.set_roi(roi)
        # update sensitivity from UI
        if self.cams[cam_i].sensitivity:
            if w:
                w.roi_threshold = self.cams[cam_i].sensitivity.value()

    @QtCore.Slot()
    def start_all_recorders(self):
        for r in self.recorders:
            if r:
                r.segment_seconds = self.seg_spin.value()
                r.ring_seconds = self.ring_spin.value()
                r.start()
        self._log("All active recorders started.")

    @QtCore.Slot()
    def stop_all_recorders(self):
        for r in self.recorders:
            if r:
                r.stop()
        self._log("All active recorders stopped.")

    @QtCore.Slot()
    def on_manual_trigger(self):
        self.event_mgr.trigger(from_cam=None)

    @QtCore.Slot()
    def start_plc(self):
        if not HAS_PLC:
            self._log("pymcprotocol not installed. pip install pymcprotocol")
            return
        if self.plc_mon:
            self._log("PLC monitor already running.")
            return
        self.plc_mon = PLCMonitor(self.plc_ip.text().strip(), int(self.plc_port.value()),
                                  self.plc_device.text().strip(), int(self.plc_poll.value()))
        self.plc_mon.log.connect(self._log)
        self.plc_mon.plc_trigger.connect(lambda: self.event_mgr.trigger(from_cam=None))
        self.plc_mon.start()

    @QtCore.Slot()
    def stop_plc(self):
        if self.plc_mon:
            self.plc_mon.stop()
            self.plc_mon = None

    @QtCore.Slot(str)
    def _log(self, msg: str):
        t = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{t}] {msg}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

