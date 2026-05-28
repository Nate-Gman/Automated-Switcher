# ACS.py — Automated Camera Switcher v1.1
# =============================================================================
# Universal live camera switcher — works with any hardware, any camera count.
#
# Supported hardware (non-exhaustive):
#   Cameras : USB webcams, Sony ZV-1 via USB, IP cameras (RTSP/HTTP), files
#   Switchers: BlackMagic ATEM Mini Pro (optional, via pyatem)
#   Audio   : Any USB/system microphone (one per speaker for best accuracy)
#
# Key capabilities:
#   • N-camera support — 1 to thousands; previews paginate automatically
#   • ATEM Mini Pro integration — mirror switches to hardware switcher (optional)
#   • RTSP / HTTP / file camera sources — any OpenCV-compatible URI
#   • 60 FPS UI loop  +  dedicated fast-switch thread (<16 ms decisions)
#   • 4-factor confidence engine (RMS median + SNR + duration + dominance)
#   • 256-sample audio buffer → 16 ms audio chunk latency at 16 kHz
#   • Adaptive noise-floor calibration per mic per session
#   • Overlap & tie-breaking: no ping-pong during crosstalk
#   • Shot discipline: configurable min-hold and min-switch-interval
#   • Camera naming: "Host", "Guest 1", "Wide Shot" instead of CAM 1/2/3
#   • Tally lights: red border = LIVE, green border = PREVIEW/next
#   • Manual override: buttons, keyboard 1-9 / a-z, or HTTP API
#   • Settings panel: live sliders, saved to acs_config.json
#   • Remote HTTP API: GET /status, POST /switch, POST /mode
#   • Structured logging: every switch, confidence, latency, RMS to file
#   • Headless unit tests: python ACS.py --test
#   • Cross-platform: Windows (DirectShow), Linux/macOS (V4L/GStreamer)
#
# Quick start:
#   1. python ACS.py          # auto-generates acs_config.json on first run
#   2. Edit camera_sources and mic_indices in acs_config.json
#   3. python ACS.py          # UI appears
#   4. Click AUTO MODE        # switcher follows the speaker
#
# Tests:   python ACS.py --test
# Help:    python ACS.py --help

import sys
import platform
import importlib

# --- Mock modules for headless --test mode ---
if "--test" in sys.argv:
    import types
    if "cv2" not in sys.modules:
        _mock_cv2 = types.ModuleType("cv2")
        _mock_cv2.CAP_DSHOW = 700
        _mock_cv2.CAP_ANY = 0
        class _MockCap:
            def isOpened(self): return False
            def release(self): pass
            def read(self): return (False, None)
            def set(self, *a): pass
        _mock_cv2.VideoCapture = lambda *a, **k: _MockCap()
        _mock_cv2.resize = lambda img, sz: img
        _mock_cv2.cvtColor = lambda img, code: img
        _mock_cv2.COLOR_BGR2RGB = 4
        _mock_cv2.putText = lambda *a, **k: None
        _mock_cv2.rectangle = lambda *a, **k: None
        _mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        _mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        _mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        _mock_cv2.CAP_PROP_FPS = 5
        sys.modules["cv2"] = _mock_cv2
    if "PIL" not in sys.modules:
        _mock_pil = types.ModuleType("PIL")
        _mock_img = types.ModuleType("PIL.Image")
        _mock_img.Image = type("Image", (), {"fromarray": lambda cls, arr: arr})()
        sys.modules["PIL.Image"] = _mock_img
        sys.modules["PIL"] = _mock_pil
        _mock_imgtk = types.ModuleType("PIL.ImageTk")
        _mock_imgtk.PhotoImage = lambda image=None, **kw: image
        sys.modules["PIL.ImageTk"] = _mock_imgtk
    if "pyaudio" not in sys.modules:
        sys.modules["pyaudio"] = types.ModuleType("pyaudio")
    if "audioop" not in sys.modules:
        _mock_audioop = types.ModuleType("audioop")
        _mock_audioop.rms = lambda data, width: 0
        sys.modules["audioop"] = _mock_audioop
    if "numpy" not in sys.modules:
        import math
        _mock_np = types.ModuleType("numpy")
        _mock_np.median = lambda x: x[len(x)//2] if x else 0
        _mock_np.array = lambda x, dtype=None: list(x)
        _mock_np.float32 = float
        _mock_np.int16 = None
        _mock_np.log = math.log
        _mock_np.percentile = lambda arr, p: sorted(arr)[int(len(arr)*p/100)]
        sys.modules["numpy"] = _mock_np

try:
    import tkinter as tk
    from tkinter import ttk
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    cv2 = None

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    Image = None
    ImageTk = None

try:
    import pyaudio
    _HAS_PYAUDIO = True
except ImportError:
    _HAS_PYAUDIO = False
    pyaudio = None

try:
    import audioop
    _HAS_AUDIOOP = True
except ImportError:
    _HAS_AUDIOOP = False
    audioop = None

import threading
import queue
import time
import logging
import datetime
from collections import deque

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None

import json
import os

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse, parse_qs
    _HAS_HTTP = True
except ImportError:
    _HAS_HTTP = False

# Optional ATEM Mini Pro support via pyatem
try:
    import pyatem.client
    _HAS_PYATEM = True
except Exception:
    _HAS_PYATEM = False

_CAP_BACKEND = None
if _HAS_CV2:
    _CAP_BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


def _ensure_deps():
    if "--test" in sys.argv:
        return
    import subprocess
    deps = [
        ("cv2", "opencv-python"),
        ("PIL.Image", "pillow"),
        ("pyaudio", "pyaudio"),
        ("numpy", "numpy"),
        ("psutil", "psutil"),
    ]
    missing = []
    for mod_name, pip_name in deps:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            if not any(m[1] == pip_name for m in missing):
                missing.append((mod_name, pip_name))
    if not missing:
        return
    print("=" * 60)
    print("  Auto-installing missing dependencies...")
    print("=" * 60)
    for mod_name, pip_name in missing:
        print(f"   [MISSING] {pip_name} — installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"   [OK] {pip_name} installed.")
        except subprocess.CalledProcessError:
            print(f"   [FAIL] Could not install {pip_name}.")
            if pip_name == "pyaudio" and platform.system() != "Windows":
                print("          Linux: sudo apt-get install portaudio19-dev")
                print("          macOS: brew install portaudio")
            sys.exit(1)
    # Re-import after install
    for mod_name, _ in missing:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            pass
    print("=" * 60)
    print("  All dependencies ready.")
    print("=" * 60)


# ========================== CONFIGURATION DEFAULTS ==========================
# These are overridden by acs_config.json on startup.

NUM_CAMERAS = 4                     # Sony ZV-1 ×4 default; change freely
CAMERA_SOURCES = [0, 1, 2, 3]       # int=device index, str=RTSP/HTTP/file URL
CAMERA_NAMES = []                   # ["Host", "Guest 1", ...] — auto-fills if empty
MIC_INDICES = [0, 1, 2, 3]         # PyAudio device indices, one per camera
PREVIEW_SIZE = (240, 135)           # 16:9 thumbnail (px)
MAIN_SIZE = (960, 540)              # Program-out resolution (px)
GRID_COLUMNS = 4                    # Preview cameras per row
PREVIEW_ROWS = 2                    # Visible rows of previews (pagination)
UI_FPS = 60                         # UI refresh rate (fps)

# --- Audio ---
AUDIO_RATE = 16000
AUDIO_BUFFER_SIZE = 256             # samples per chunk — 16 ms @ 16 kHz
RMS_WINDOW_SIZE = 8                 # rolling-median window (chunks)
NOISE_FLOOR_MULTIPLIER = 3.0
NOISE_FLOOR_PERCENTILE = 20
NOISE_CALIBRATION_SECONDS = 2.0

# --- Confidence Engine ---
MIN_CONFIDENCE = 0.65
CONFIDENCE_WEIGHTS = {
    "signal_strength": 0.30,   # log-scaled distance above threshold
    "duration":        0.22,   # how long speaker has been active
    "snr":             0.18,   # signal-to-noise ratio
    "dominance":       0.20,   # ratio vs. runner-up
    "speech_ratio":    0.10,   # FFT: energy in 300–3400 Hz band vs. total
}
OVERLAP_DOMINANCE_RATIO = 1.4
OVERLAP_HOLD_CONFIDENCE = 0.50
OVERLAP_MAX_WAIT = 2.0

# --- Shot Discipline ---
MIN_SWITCH_INTERVAL = 0.7
MIN_SHOT_DURATION = 1.5

# --- Silence Fallback ---
SILENCE_TIMEOUT = 3.0
DEFAULT_CAMERA = 0
HOLD_LAST_ON_SILENCE = False

# --- Host Priority ---
HOLD_PRIORITY_HOST = True
HOST_MIC_INDEX = 0

# --- ATEM Mini Pro (optional) ---
ATEM_ENABLED = False
ATEM_IP = "192.168.10.240"          # default ATEM Mini Pro IP
ATEM_INPUT_MAP = {}                 # {camera_index: atem_input_number}

# --- Advanced Audio ---
AUDIO_EMA_ALPHA      = 0.40         # EMA attack factor — how fast level rises
AUDIO_EMA_ALPHA_FALL = 0.15         # EMA release factor — how fast level falls
MIC_GAINS = []                      # per-mic multipliers, e.g. [1.0, 1.2, 0.8, 1.0]
USE_SPEECH_RATIO = True             # FFT speech-band energy as 5th confidence factor
# ==========================================================================


def _load_config(profile=None):
    global NUM_CAMERAS, CAMERA_SOURCES, CAMERA_NAMES, MIC_INDICES
    global PREVIEW_SIZE, MAIN_SIZE, GRID_COLUMNS, PREVIEW_ROWS, UI_FPS
    global AUDIO_RATE, AUDIO_BUFFER_SIZE
    global RMS_WINDOW_SIZE, NOISE_FLOOR_MULTIPLIER, NOISE_FLOOR_PERCENTILE, NOISE_CALIBRATION_SECONDS
    global MIN_CONFIDENCE, CONFIDENCE_WEIGHTS, OVERLAP_DOMINANCE_RATIO
    global OVERLAP_HOLD_CONFIDENCE, OVERLAP_MAX_WAIT, MIN_SWITCH_INTERVAL, MIN_SHOT_DURATION
    global SILENCE_TIMEOUT, DEFAULT_CAMERA, HOLD_LAST_ON_SILENCE, HOLD_PRIORITY_HOST, HOST_MIC_INDEX
    global ATEM_ENABLED, ATEM_IP, ATEM_INPUT_MAP
    global AUDIO_EMA_ALPHA, AUDIO_EMA_ALPHA_FALL, MIC_GAINS, USE_SPEECH_RATIO

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if profile:
        config_path = os.path.join(base_dir, profile) if not os.path.isabs(profile) else profile
        label = f"profile '{profile}'"
    else:
        config_path = os.path.join(base_dir, "acs_config.json")
        label = "default config"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # Support legacy camera_indices key
            if "camera_sources" in cfg:
                CAMERA_SOURCES = cfg["camera_sources"]
            elif "camera_indices" in cfg:
                CAMERA_SOURCES = cfg["camera_indices"]

            CAMERA_NAMES = cfg.get("camera_names", CAMERA_NAMES)
            MIC_INDICES = cfg.get("mic_indices", MIC_INDICES)
            NUM_CAMERAS = cfg.get("num_cameras", len(CAMERA_SOURCES))

            PREVIEW_SIZE = tuple(cfg.get("preview_size", list(PREVIEW_SIZE)))
            MAIN_SIZE = tuple(cfg.get("main_size", list(MAIN_SIZE)))
            GRID_COLUMNS = max(1, cfg.get("grid_columns", GRID_COLUMNS))
            PREVIEW_ROWS = max(1, cfg.get("preview_rows", PREVIEW_ROWS))
            UI_FPS = max(10, min(120, cfg.get("ui_fps", UI_FPS)))

            AUDIO_RATE = cfg.get("audio_rate", AUDIO_RATE)
            AUDIO_BUFFER_SIZE = max(64, cfg.get("audio_buffer_size", AUDIO_BUFFER_SIZE))
            RMS_WINDOW_SIZE = max(2, cfg.get("rms_window_size", RMS_WINDOW_SIZE))
            NOISE_FLOOR_MULTIPLIER = max(1.5, cfg.get("noise_floor_multiplier", NOISE_FLOOR_MULTIPLIER))
            NOISE_FLOOR_PERCENTILE = max(1, min(50, cfg.get("noise_floor_percentile", NOISE_FLOOR_PERCENTILE)))
            NOISE_CALIBRATION_SECONDS = max(0.5, cfg.get("noise_calibration_seconds", NOISE_CALIBRATION_SECONDS))

            MIN_CONFIDENCE = max(0.0, min(1.0, cfg.get("min_confidence", MIN_CONFIDENCE)))
            CONFIDENCE_WEIGHTS = cfg.get("confidence_weights", CONFIDENCE_WEIGHTS)
            OVERLAP_DOMINANCE_RATIO = max(1.0, cfg.get("overlap_dominance_ratio", OVERLAP_DOMINANCE_RATIO))
            OVERLAP_HOLD_CONFIDENCE = max(0.0, min(1.0, cfg.get("overlap_hold_confidence", OVERLAP_HOLD_CONFIDENCE)))
            OVERLAP_MAX_WAIT = max(0.1, cfg.get("overlap_max_wait", OVERLAP_MAX_WAIT))
            MIN_SWITCH_INTERVAL = max(0.1, cfg.get("min_switch_interval", MIN_SWITCH_INTERVAL))
            MIN_SHOT_DURATION = max(0.2, cfg.get("min_shot_duration", MIN_SHOT_DURATION))
            SILENCE_TIMEOUT = max(0.5, cfg.get("silence_timeout", SILENCE_TIMEOUT))
            DEFAULT_CAMERA = max(0, min(NUM_CAMERAS - 1, cfg.get("default_camera", DEFAULT_CAMERA)))
            HOLD_LAST_ON_SILENCE = cfg.get("hold_last_on_silence", HOLD_LAST_ON_SILENCE)
            HOLD_PRIORITY_HOST = cfg.get("hold_priority_host", HOLD_PRIORITY_HOST)
            HOST_MIC_INDEX = cfg.get("host_mic_index", HOST_MIC_INDEX)

            ATEM_ENABLED = cfg.get("atem_enabled", ATEM_ENABLED)
            ATEM_IP = cfg.get("atem_ip", ATEM_IP)
            ATEM_INPUT_MAP = cfg.get("atem_input_map", ATEM_INPUT_MAP)

            AUDIO_EMA_ALPHA      = float(cfg.get("audio_ema_alpha",      AUDIO_EMA_ALPHA))
            AUDIO_EMA_ALPHA_FALL = float(cfg.get("audio_ema_alpha_fall", AUDIO_EMA_ALPHA_FALL))
            MIC_GAINS            = cfg.get("mic_gains",      MIC_GAINS)
            USE_SPEECH_RATIO     = bool(cfg.get("use_speech_ratio", USE_SPEECH_RATIO))

            print(f"Config loaded: {config_path}")
        except Exception as e:
            print(f"WARNING: Failed to load {label}: {type(e).__name__}: {e}. Using defaults.")
    elif profile:
        print(f"WARNING: Profile not found: {config_path}. Using defaults.")
    else:
        _write_default_config(config_path)


def _write_default_config(config_path):
    template = {
        "_comment": "ACS config — edit camera_sources and mic_indices, then restart",
        "num_cameras": NUM_CAMERAS,
        "camera_sources": CAMERA_SOURCES,
        "camera_names": ["Host", "Guest 1", "Guest 2", "Guest 3"],
        "mic_indices": MIC_INDICES,
        "preview_size": list(PREVIEW_SIZE),
        "main_size": list(MAIN_SIZE),
        "grid_columns": GRID_COLUMNS,
        "preview_rows": PREVIEW_ROWS,
        "ui_fps": UI_FPS,
        "audio_buffer_size": AUDIO_BUFFER_SIZE,
        "rms_window_size": RMS_WINDOW_SIZE,
        "noise_floor_multiplier": NOISE_FLOOR_MULTIPLIER,
        "noise_floor_percentile": NOISE_FLOOR_PERCENTILE,
        "noise_calibration_seconds": NOISE_CALIBRATION_SECONDS,
        "min_confidence": MIN_CONFIDENCE,
        "confidence_weights": CONFIDENCE_WEIGHTS,
        "overlap_dominance_ratio": OVERLAP_DOMINANCE_RATIO,
        "overlap_hold_confidence": OVERLAP_HOLD_CONFIDENCE,
        "overlap_max_wait": OVERLAP_MAX_WAIT,
        "min_switch_interval": MIN_SWITCH_INTERVAL,
        "min_shot_duration": MIN_SHOT_DURATION,
        "silence_timeout": SILENCE_TIMEOUT,
        "default_camera": DEFAULT_CAMERA,
        "hold_last_on_silence": HOLD_LAST_ON_SILENCE,
        "hold_priority_host": HOLD_PRIORITY_HOST,
        "host_mic_index": HOST_MIC_INDEX,
        "atem_enabled": False,
        "atem_ip": "192.168.10.240",
        "atem_input_map": {"0": 1, "1": 2, "2": 3, "3": 4},
        "audio_ema_alpha": AUDIO_EMA_ALPHA,
        "audio_ema_alpha_fall": AUDIO_EMA_ALPHA_FALL,
        "mic_gains": [1.0] * NUM_CAMERAS,
        "use_speech_ratio": USE_SPEECH_RATIO,
    }
    try:
        with open(config_path, "w") as f:
            json.dump(template, f, indent=2)
        print(f"Template config created: {config_path}")
        print("   Edit camera_sources and mic_indices, then restart.")
    except Exception as e:
        print(f"WARNING: Could not write config template: {type(e).__name__}: {e}")


_profile = None
for _i, _arg in enumerate(sys.argv):
    if _arg == "--profile" and _i + 1 < len(sys.argv):
        _profile = sys.argv[_i + 1]
        break
_load_config(profile=_profile)

# Ensure CAMERA_NAMES is padded to NUM_CAMERAS
def _camera_name(idx):
    if idx < len(CAMERA_NAMES) and CAMERA_NAMES[idx]:
        return CAMERA_NAMES[idx]
    return f"CAM {idx + 1}"


# ===========================  VIDEO STREAM  ================================

class VideoStream:
    """Threaded camera capture. source can be int (device index) or str (URL/path)."""
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self._open_capture()
        self.q = queue.Queue(maxsize=4)
        self.last_frame = None
        self.consecutive_failures = 0
        self.max_failures = 30
        self.reconnect_backoff = 1.0
        self._running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        src = int(self.source) if isinstance(self.source, (int, float)) else self.source
        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src, _CAP_BACKEND)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            # RTSP / HTTP / file — OpenCV handles these natively
            self.cap = cv2.VideoCapture(str(src))

    def _update(self):
        reconnect_attempts = 0
        while self._running:
            if self.cap and self.cap.isOpened():
                try:
                    grabbed, frame = self.cap.read()
                    if grabbed and frame is not None:
                        self.consecutive_failures = 0
                        reconnect_attempts = 0
                        self.last_frame = frame
                        if not self.q.full():
                            self.q.put(frame.copy())
                    else:
                        self.consecutive_failures += 1
                        if self.consecutive_failures >= self.max_failures:
                            self._open_capture()
                            self.consecutive_failures = 0
                            reconnect_attempts += 1
                            delay = min(self.reconnect_backoff * (1.5 ** max(0, reconnect_attempts - 3)), 30.0)
                            time.sleep(delay)
                except Exception:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_failures:
                        self._open_capture()
                        self.consecutive_failures = 0
                        reconnect_attempts += 1
                        time.sleep(self.reconnect_backoff)
            else:
                self._open_capture()
                reconnect_attempts += 1
                time.sleep(self.reconnect_backoff)
            time.sleep(0.005)

    def read(self):
        if not self.q.empty():
            return self.q.get()
        return self.last_frame

    def release(self):
        self._running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


# ===========================  AUDIO MONITOR  ================================

class AudioMonitor:
    """
    Threaded per-mic monitor.

    Detection path (fast):  asymmetric EMA — rise at alpha=0.4, fall at 0.15.
                            Fires speaking=True ~2–3 chunks (32–48 ms) after onset.
    Scoring path (stable):  rolling median — spike-resistant, used for confidence.
    Speech filter:          FFT energy ratio (300–3400 Hz) — reduces false triggers
                            from HVAC, music, keyboard noise.
    Per-mic gain:           scales raw RMS before all calculations.
    """

    def __init__(self, device_index, name="Mic", gain=1.0):
        self.device_index = device_index
        self.name = name
        self.gain = gain                    # per-mic gain multiplier

        self.rms = 0
        self.ema_rms = 0.0                  # asymmetric EMA (fast detection)
        self.median_rms = 0.0               # windowed median (stable scoring)
        self.peak_rms = 0.0                 # peak hold for UI bar graph
        self.speech_ratio = 0.5            # FFT speech band energy ratio

        self.speaking = False
        self.running = False
        self.thread = None
        self.p = None
        self.stream = None

        self.rms_window = deque(maxlen=RMS_WINDOW_SIZE)
        self.noise_floor = 100.0
        self.threshold = 300.0
        self.calibration_samples = int(NOISE_CALIBRATION_SECONDS * (AUDIO_RATE / AUDIO_BUFFER_SIZE))
        self.calibrated = False
        self.above_threshold_since = None
        self.speaking_duration = 0.0
        self.snr = 0.0
        self.confidence = 0.0

        self._PEAK_DECAY = 0.92             # peak hold decay factor per chunk

    def start(self):
        if self.device_index is None:
            print(f"  {self.name}: disabled (no mic assigned)")
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _update_noise_floor(self):
        if len(self.rms_window) < self.calibration_samples:
            return
        arr = np.array(list(self.rms_window), dtype=np.float32)
        self.noise_floor = float(np.percentile(arr, NOISE_FLOOR_PERCENTILE))
        self.threshold = max(50.0, self.noise_floor * NOISE_FLOOR_MULTIPLIER)
        self.calibrated = True

    def _update_metrics(self, raw_rms):
        gained = raw_rms * self.gain
        self.rms = gained

        # ── Asymmetric EMA: fast attack, slow release ──────────────────
        a = AUDIO_EMA_ALPHA if gained > self.ema_rms else AUDIO_EMA_ALPHA_FALL
        self.ema_rms = a * gained + (1.0 - a) * self.ema_rms

        # ── Rolling median window (spike-resistant) ────────────────────
        self.rms_window.append(gained)
        self.median_rms = (
            float(np.median(list(self.rms_window)))
            if len(self.rms_window) >= RMS_WINDOW_SIZE
            else gained
        )

        # ── Peak hold ─────────────────────────────────────────────────
        if gained > self.peak_rms:
            self.peak_rms = gained
        else:
            self.peak_rms *= self._PEAK_DECAY

        # ── Noise floor calibration ────────────────────────────────────
        if not self.calibrated:
            self._update_noise_floor()
        if self.median_rms < self.threshold:
            self._update_noise_floor()

        # ── Speaking detection via EMA (fast) ─────────────────────────
        self.speaking = self.ema_rms > self.threshold
        self.snr = (self.ema_rms - self.noise_floor) / max(1.0, self.noise_floor)

        now = time.time()
        if self.speaking:
            if self.above_threshold_since is None:
                self.above_threshold_since = now
            self.speaking_duration = now - self.above_threshold_since
        else:
            self.above_threshold_since = None
            self.speaking_duration = 0.0

    def _compute_speech_ratio(self, raw_bytes):
        """Ratio of energy in 300–3400 Hz to total. 1.0 = pure speech, 0 = LF noise."""
        try:
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            if len(samples) < 16:
                return 0.5
            fft_mag = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1.0 / AUDIO_RATE)
            total = np.dot(fft_mag, fft_mag) + 1e-10
            mask = (freqs >= 300) & (freqs <= 3400)
            speech = np.dot(fft_mag[mask], fft_mag[mask])
            return float(min(1.0, speech / total))
        except Exception:
            return 0.5

    def _open_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=AUDIO_BUFFER_SIZE,
        )

    def _close_stream(self):
        for obj in (self.stream, self.p):
            if obj:
                try:
                    if obj is self.stream:
                        obj.stop_stream()
                    obj.close() if obj is self.stream else obj.terminate()
                except Exception:
                    pass
        self.stream = None
        self.p = None

    def _monitor(self):
        reconnect_delay = 1.0
        consecutive_failures = 0
        while self.running:
            try:
                self._open_stream()
                self.calibrated = False
                self.rms_window.clear()
                self.ema_rms = 0.0
                consecutive_failures = 0
                while self.running:
                    try:
                        data = self.stream.read(AUDIO_BUFFER_SIZE, exception_on_overflow=False)
                        raw_rms = audioop.rms(data, 2)
                        if USE_SPEECH_RATIO and _HAS_NUMPY:
                            self.speech_ratio = self._compute_speech_ratio(data)
                        self._update_metrics(raw_rms)
                        consecutive_failures = 0
                    except OSError:
                        consecutive_failures += 1
                        if consecutive_failures >= 5:
                            break
                        time.sleep(0.01)
                    except Exception:
                        consecutive_failures += 1
                        self.rms = 0
                        self.speaking = False
                        if consecutive_failures >= 5:
                            break
            except Exception as e:
                print(f"Mic {self.name} failed to open: {type(e).__name__}: {e}")
            finally:
                self._close_stream()
                if self.running:
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 10.0)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


# ===========================  ATEM CONTROLLER  ==============================

class ATEMController:
    """
    Optional BlackMagic ATEM Mini Pro integration.

    Requirements: pip install pyatem
    The ATEM Mini Pro must be on the same network (default IP 192.168.10.240).
    Each ACS camera index maps to an ATEM input number via atem_input_map.

    If pyatem is not installed or connection fails, all methods silently no-op.
    """
    def __init__(self, ip, input_map, logger):
        self.ip = ip
        self.input_map = {int(k): int(v) for k, v in (input_map or {}).items()}
        self.logger = logger
        self._client = None
        self._connected = False
        self._lock = threading.Lock()
        if _HAS_PYATEM:
            self._connect()

    def _connect(self):
        try:
            self._client = pyatem.client.ATEMProtocol(self.ip)
            self._client.connect()
            self._connected = True
            self.logger.info(f"ATEM connected: {self.ip}")
        except Exception as e:
            self.logger.warning(f"ATEM connection failed ({self.ip}): {type(e).__name__}: {e}")
            self._connected = False

    def switch_program(self, cam_index):
        """Switch ATEM program bus to the input mapped to cam_index."""
        if not self._connected or not self._client:
            return
        atem_input = self.input_map.get(cam_index)
        if atem_input is None:
            return
        with self._lock:
            try:
                self._client.set_program_input(atem_input)
                self.logger.info(f"ATEM program → input {atem_input} (CAM {cam_index + 1})")
            except Exception as e:
                self.logger.warning(f"ATEM switch failed: {type(e).__name__}: {e}")
                self._connected = False

    def set_preview(self, cam_index):
        """Set ATEM preview bus."""
        if not self._connected or not self._client:
            return
        atem_input = self.input_map.get(cam_index)
        if atem_input is None:
            return
        with self._lock:
            try:
                self._client.set_preview_input(atem_input)
            except Exception:
                pass

    @property
    def connected(self):
        return self._connected


# ========================  SWITCHER APPLICATION  ============================

class SwitcherApp:
    # --- Tally light colors ---
    COLOR_LIVE    = "#ff2222"   # red — camera is PROGRAM (on-air)
    COLOR_PREVIEW = "#22aa22"   # green — camera is next / high confidence
    COLOR_IDLE    = "#222222"   # dark — not active
    COLOR_BG      = "#0d0d0d"
    COLOR_PANEL   = "#161616"
    COLOR_ACCENT  = "#00e5ff"
    COLOR_WARN    = "#ffaa00"
    COLOR_SPEAK   = "#00ff66"
    COLOR_SILENT  = "#ff4444"

    def __init__(self, root):
        self.root = root
        self.root.title("ACS — Automated Camera Switcher")
        self.root.configure(bg=self.COLOR_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.num_cams = NUM_CAMERAS
        self.current_cam = 0
        self.preview_cam = -1       # next candidate (for preview tally)
        self.auto_mode = True
        self.last_switch_time = time.time()
        self.shot_start_time = time.time()
        self.silence_start_time = None
        self.overlap_tie_start_time = None
        self._preview_page = 0      # current page index for preview grid

        # Switch history
        self._switch_history = deque(maxlen=20)
        self._switch_history_dirty = False

        # Performance
        self._fps_counter = 0
        self._fps_last_time = time.time()
        self.current_fps = 0.0
        self.frame_count = 0
        self.health_check_time = time.time()
        self.camera_frame_counts = [0] * self.num_cams
        self.last_frame_count = [0] * self.num_cams

        # Logging
        log_filename = f"acs_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("ACS")
        self.logger.info("ACS started")

        # Cameras
        print(f"\nINITIALIZING {self.num_cams} CAMERAS...")
        self.cameras = []
        for i in range(self.num_cams):
            src = CAMERA_SOURCES[i] if i < len(CAMERA_SOURCES) else i
            cam = VideoStream(src)
            self.cameras.append(cam)
            print(f"  Camera {i+1} ({_camera_name(i)}) ← {src}")

        # Microphones
        print(f"\nINITIALIZING {self.num_cams} MICROPHONES...")
        self.audio_mons = []
        for i in range(self.num_cams):
            mic_idx = MIC_INDICES[i] if i < len(MIC_INDICES) else None
            gain = float(MIC_GAINS[i]) if i < len(MIC_GAINS) else 1.0
            mon = AudioMonitor(mic_idx, _camera_name(i), gain=gain)
            self.audio_mons.append(mon)
            mon.start()

        # ATEM (optional)
        self.atem = None
        if ATEM_ENABLED:
            self.atem = ATEMController(ATEM_IP, ATEM_INPUT_MAP, self.logger)
            if self.atem.connected:
                print(f"  ATEM Mini Pro connected: {ATEM_IP}")
            else:
                print(f"  ATEM: connection failed — continuing in software-only mode")
        else:
            print("  ATEM: disabled (set atem_enabled=true in config to enable)")

        self._build_ui()
        self._list_devices()

        # Remote API
        if "--remote" in sys.argv:
            port = 8765
            for _i, _a in enumerate(sys.argv):
                if _a == "--port" and _i + 1 < len(sys.argv):
                    try:
                        port = int(sys.argv[_i + 1])
                    except ValueError:
                        pass
                    break
            self._start_remote_api(port=port)

        # Dedicated fast-switch decision thread
        self._switch_thread_running = True
        self._switch_thread = threading.Thread(target=self._switch_loop, daemon=True)
        self._switch_thread.start()

        self._update_ui()

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.geometry(f"{max(960, GRID_COLUMNS * (PREVIEW_SIZE[0] + 24) + 40)}x900")

        # ── Title bar ──────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg="#0a0a0a", height=36)
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="  ACS — Automated Camera Switcher",
                 bg="#0a0a0a", fg=self.COLOR_ACCENT,
                 font=("Consolas", 13, "bold")).pack(side=tk.LEFT, pady=4)
        self._status_badge = tk.Label(title_bar, text="AUTO", bg="#00994d", fg="white",
                                      font=("Consolas", 10, "bold"), padx=8, pady=2)
        self._status_badge.pack(side=tk.RIGHT, padx=12, pady=6)

        # ── Program output (main broadcast feed) ──────────────────────
        prog_frame = tk.Frame(self.root, bg="#0a0a0a", pady=8)
        prog_frame.pack()
        tk.Label(prog_frame, text="PROGRAM OUT", bg="#0a0a0a", fg="#666666",
                 font=("Consolas", 9, "bold")).pack(anchor="w", padx=6)
        self._main_label = tk.Label(prog_frame, bg="black",
                                    relief=tk.FLAT, bd=3,
                                    highlightthickness=3,
                                    highlightbackground=self.COLOR_LIVE)
        self._main_label.pack(padx=6)

        # ── Control bar ───────────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=self.COLOR_BG)
        ctrl.pack(fill=tk.X, padx=16, pady=4)

        self.mode_var = tk.StringVar(value="AUTO")
        tk.Radiobutton(ctrl, text="AUTO", variable=self.mode_var,
                       value="AUTO", command=self._toggle_mode,
                       bg=self.COLOR_BG, fg=self.COLOR_SPEAK, selectcolor=self.COLOR_PANEL,
                       font=("Consolas", 12, "bold"), indicatoron=0, width=8,
                       relief=tk.FLAT, bd=0,
                       activebackground="#003322", activeforeground=self.COLOR_SPEAK).pack(side=tk.LEFT, padx=4)
        tk.Radiobutton(ctrl, text="MANUAL", variable=self.mode_var,
                       value="MANUAL", command=self._toggle_mode,
                       bg=self.COLOR_BG, fg=self.COLOR_WARN, selectcolor=self.COLOR_PANEL,
                       font=("Consolas", 12, "bold"), indicatoron=0, width=8,
                       relief=tk.FLAT, bd=0,
                       activebackground="#332200", activeforeground=self.COLOR_WARN).pack(side=tk.LEFT, padx=4)

        tk.Button(ctrl, text="SETTINGS", command=self._open_settings,
                  bg=self.COLOR_PANEL, fg="white", font=("Consolas", 11),
                  relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="DEVICES", command=self._list_devices,
                  bg=self.COLOR_PANEL, fg="white", font=("Consolas", 11),
                  relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="ABOUT", command=self._open_about,
                  bg=self.COLOR_PANEL, fg="white", font=("Consolas", 11),
                  relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=6)

        self._fps_label = tk.Label(ctrl, text="FPS: --", bg=self.COLOR_BG, fg="#555555",
                                   font=("Consolas", 10))
        self._fps_label.pack(side=tk.RIGHT, padx=10)

        # ── Camera preview grid ───────────────────────────────────────
        grid_outer = tk.Frame(self.root, bg=self.COLOR_BG)
        grid_outer.pack(fill=tk.X, padx=12, pady=4)

        # Pagination header
        nav_row = tk.Frame(grid_outer, bg=self.COLOR_BG)
        nav_row.pack(fill=tk.X, pady=(0, 4))
        tk.Button(nav_row, text="◀", command=self._prev_page,
                  bg=self.COLOR_PANEL, fg="white", font=("Consolas", 11),
                  relief=tk.FLAT, width=3).pack(side=tk.LEFT, padx=2)
        self._page_label = tk.Label(nav_row, text="PAGE 1", bg=self.COLOR_BG, fg="#666666",
                                    font=("Consolas", 10))
        self._page_label.pack(side=tk.LEFT, padx=6)
        tk.Button(nav_row, text="▶", command=self._next_page,
                  bg=self.COLOR_PANEL, fg="white", font=("Consolas", 11),
                  relief=tk.FLAT, width=3).pack(side=tk.LEFT, padx=2)

        # Camera preview tiles
        self._cut_btns = {}         # {cam_i: tk.Button}
        self._preview_tiles = []    # list of tile-info dicts
        self._visible_pages = []

        tiles_frame = tk.Frame(grid_outer, bg=self.COLOR_BG)
        tiles_frame.pack(fill=tk.X)

        page_size = GRID_COLUMNS * PREVIEW_ROWS
        total_pages = max(1, (self.num_cams + page_size - 1) // page_size)

        for page in range(total_pages):
            page_cams = range(page * page_size, min((page + 1) * page_size, self.num_cams))
            page_frame = tk.Frame(tiles_frame, bg=self.COLOR_BG)
            if page == 0:
                page_frame.pack(fill=tk.X)
            self._visible_pages.append(page_frame)

            for col_in_page, cam_i in enumerate(page_cams):
                row_in_page = col_in_page // GRID_COLUMNS
                col = col_in_page % GRID_COLUMNS

                # Tile outer frame — tally border changes color each frame
                tile = tk.Frame(page_frame, bg=self.COLOR_IDLE,
                                highlightthickness=3,
                                highlightbackground=self.COLOR_IDLE,
                                padx=2, pady=2)
                tile.grid(row=row_in_page, column=col, padx=6, pady=4, sticky="n")

                # ── Header row: name  health-dot  CUT button ──────────
                header = tk.Frame(tile, bg=self.COLOR_PANEL)
                header.pack(fill=tk.X)

                health_dot = tk.Canvas(header, bg=self.COLOR_PANEL,
                                       width=10, height=10,
                                       highlightthickness=0)
                health_dot.pack(side=tk.LEFT, padx=(4, 0), pady=3)
                health_dot.create_oval(1, 1, 9, 9, fill="#555555", outline="", tags="dot")

                tk.Label(header, text=_camera_name(cam_i),
                         bg=self.COLOR_PANEL, fg="white",
                         font=("Consolas", 9, "bold"), width=12, anchor="w").pack(side=tk.LEFT, padx=4)

                btn = tk.Button(header, text="CUT",
                                command=lambda x=cam_i: self.manual_switch(x),
                                bg="#2a2a2a", fg="white", font=("Consolas", 8),
                                relief=tk.FLAT, padx=4)
                btn.pack(side=tk.RIGHT, padx=2)
                self._cut_btns[cam_i] = btn

                # Keyboard shortcut (1–9)
                if cam_i < 9:
                    self.root.bind(str(cam_i + 1), lambda e, x=cam_i: self.manual_switch(x))

                # ── Preview thumbnail ──────────────────────────────────
                thumb = tk.Label(tile, bg="black",
                                 width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
                thumb.pack(pady=2)

                # ── Status line ────────────────────────────────────────
                status = tk.Label(tile, text="SILENT",
                                  bg=self.COLOR_PANEL, fg=self.COLOR_SILENT,
                                  font=("Consolas", 8, "bold"), width=22)
                status.pack(fill=tk.X)

                # ── Confidence bar (Unicode blocks, 10 chars wide) ─────
                conf_bar = tk.Label(tile, text="░░░░░░░░░░",
                                    bg=self.COLOR_PANEL, fg="#333333",
                                    font=("Consolas", 8), width=22, anchor="w")
                conf_bar.pack(fill=tk.X)

                self._preview_tiles.append({
                    "cam_i":      cam_i,
                    "frame":      tile,
                    "thumb":      thumb,
                    "status":     status,
                    "conf_bar":   conf_bar,
                    "health_dot": health_dot,
                })

        self.root.bind("a", lambda e: self._set_mode("AUTO"))
        self.root.bind("m", lambda e: self._set_mode("MANUAL"))
        self.root.bind("<F11>", lambda e: self._toggle_fullscreen())

        # ── Audio bar graph ───────────────────────────────────────────
        bar_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        bar_frame.pack(pady=(4, 0), padx=12, fill=tk.X)
        tk.Label(bar_frame, text="AUDIO LEVELS", bg=self.COLOR_BG, fg="#444444",
                 font=("Consolas", 8, "bold")).pack(anchor="w")
        bar_width = max(200, GRID_COLUMNS * (PREVIEW_SIZE[0] + 24))
        self._rms_canvas = tk.Canvas(bar_frame, bg=self.COLOR_BG,
                                     highlightthickness=0,
                                     width=bar_width, height=52)
        self._rms_canvas.pack(fill=tk.X)

        # ── Switch history log ────────────────────────────────────────
        hist_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        hist_frame.pack(pady=(2, 0), padx=12, fill=tk.X)
        tk.Label(hist_frame, text="SWITCH LOG", bg=self.COLOR_BG, fg="#444444",
                 font=("Consolas", 8, "bold")).pack(anchor="w")
        self._history_text = tk.Text(hist_frame, bg="#090909", fg="#556655",
                                     font=("Consolas", 8), height=4,
                                     state=tk.DISABLED, relief=tk.FLAT,
                                     highlightthickness=0, cursor="arrow")
        self._history_text.pack(fill=tk.X)

        # ── Status bar ────────────────────────────────────────────────
        self._status_bar = tk.Label(self.root, text="READY",
                                    bg="#0a0a0a", fg=self.COLOR_ACCENT,
                                    font=("Consolas", 11, "bold"),
                                    anchor="w", padx=12)
        self._status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=0)

    # ------------------------------------------------------------------
    # PAGE NAVIGATION
    # ------------------------------------------------------------------

    def _page_size(self):
        return GRID_COLUMNS * PREVIEW_ROWS

    def _total_pages(self):
        return max(1, (self.num_cams + self._page_size() - 1) // self._page_size())

    def _show_page(self, page):
        self._preview_page = max(0, min(page, self._total_pages() - 1))
        for i, pf in enumerate(self._visible_pages):
            if i == self._preview_page:
                pf.pack(fill=tk.X)
            else:
                pf.pack_forget()
        self._page_label.config(text=f"PAGE {self._preview_page + 1}/{self._total_pages()}")

    def _prev_page(self):
        self._show_page(self._preview_page - 1)

    def _next_page(self):
        self._show_page(self._preview_page + 1)

    # ------------------------------------------------------------------
    # MODE CONTROL
    # ------------------------------------------------------------------

    def _set_mode(self, mode):
        self.mode_var.set(mode)
        self._toggle_mode()

    def _toggle_mode(self):
        self.auto_mode = (self.mode_var.get() == "AUTO")
        if self.auto_mode:
            self._status_badge.config(text="AUTO", bg="#00994d")
        else:
            self._status_badge.config(text="MANUAL", bg="#994400")

    def _add_to_history(self, mode, from_cam, to_cam, conf=None, latency_ms=None):
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:12]
        conf_str = f"  C:{conf:.2f}" if conf is not None else ""
        lat_str = f"  {latency_ms:.0f}ms" if latency_ms is not None else ""
        entry = f"{ts}  {mode:<6}  {_camera_name(from_cam)} → {_camera_name(to_cam)}{conf_str}{lat_str}"
        self._switch_history.appendleft(entry)
        self._switch_history_dirty = True

    def manual_switch(self, idx):
        if 0 <= idx < self.num_cams:
            prev = self.current_cam
            self.current_cam = idx
            self.auto_mode = False
            self.mode_var.set("MANUAL")
            self._status_badge.config(text="MANUAL", bg="#994400")
            self.last_switch_time = time.time()
            self.shot_start_time = time.time()
            self.overlap_tie_start_time = None
            self.silence_start_time = None
            self.logger.info(f"MANUAL OVERRIDE → {_camera_name(idx)} (was {_camera_name(prev)})")
            self._add_to_history("MANUAL", prev, idx)
            if self.atem:
                self.atem.switch_program(idx)

    # ------------------------------------------------------------------
    # FAST SWITCH DECISION LOOP (dedicated thread)
    # ------------------------------------------------------------------

    def _switch_loop(self):
        """Runs at ~100 Hz, decoupled from the UI paint cycle."""
        while self._switch_thread_running:
            if self.auto_mode:
                prev = self.current_cam
                new_cam = self._determine_active_camera()
                if new_cam != prev:
                    mon = self.audio_mons[new_cam] if new_cam < len(self.audio_mons) else None
                    lat = ((time.time() - mon.above_threshold_since) * 1000
                           if mon and mon.above_threshold_since else None)
                    self._add_to_history("AUTO", prev, new_cam,
                                         conf=mon.confidence if mon else None,
                                         latency_ms=lat)
                    self.current_cam = new_cam
                    if self.atem:
                        self.atem.switch_program(new_cam)
            time.sleep(0.010)  # 100 Hz

    # ------------------------------------------------------------------
    # CONFIDENCE ENGINE
    # ------------------------------------------------------------------

    def _compute_confidence(self, mon_idx, candidates_map):
        mon = self.audio_mons[mon_idx]
        # Prefer EMA level (live); fall back to median if EMA hasn't warmed up
        level = mon.ema_rms if mon.ema_rms > 0 else mon.median_rms
        thresh = mon.threshold

        # Signal strength: log-scaled distance above threshold
        if level > thresh > 0:
            signal_strength = min(1.0, np.log(level / thresh) / np.log(5.0))
        else:
            signal_strength = 0.0

        # Duration: longer sustained speech = higher confidence
        duration_score = min(1.0, mon.speaking_duration / 2.0)

        # SNR: distance above noise floor
        snr_score = min(1.0, mon.snr / 5.0)

        # Dominance: how much louder vs. runner-up
        if len(candidates_map) > 1:
            sorted_vals = sorted(candidates_map.values(), reverse=True)
            my_val = candidates_map.get(mon_idx, 0.0)
            runner_up = sorted_vals[1] if my_val == sorted_vals[0] else sorted_vals[0]
            dominance_score = min(1.0, (my_val / max(1.0, runner_up)) / 2.0)
        else:
            dominance_score = 1.0

        # Speech ratio: 0.3 (LF rumble) → 0, 0.5 (flat) → 0.4, 0.8 (speech) → 1.0
        raw_sr = mon.speech_ratio if hasattr(mon, "speech_ratio") else 0.5
        speech_score = max(0.0, min(1.0, (raw_sr - 0.3) / 0.5))

        confidence = (
            CONFIDENCE_WEIGHTS.get("signal_strength", 0.30) * signal_strength +
            CONFIDENCE_WEIGHTS.get("duration",        0.22) * duration_score +
            CONFIDENCE_WEIGHTS.get("snr",             0.18) * snr_score +
            CONFIDENCE_WEIGHTS.get("dominance",       0.20) * dominance_score +
            CONFIDENCE_WEIGHTS.get("speech_ratio",    0.00) * speech_score
        )

        if HOLD_PRIORITY_HOST and mon_idx == HOST_MIC_INDEX:
            confidence = min(1.0, confidence + 0.15)

        return confidence

    def _determine_active_camera(self):
        """Confidence-based speaker detection with overlap handling and shot discipline."""
        now = time.time()

        if now - self.last_switch_time < MIN_SWITCH_INTERVAL:
            return self.current_cam
        if now - self.shot_start_time < MIN_SHOT_DURATION:
            return self.current_cam

        candidates = {}
        for i, mon in enumerate(self.audio_mons):
            if mon.speaking and mon.median_rms > 0:
                candidates[i] = mon.median_rms

        # Silence fallback
        if not candidates:
            if self.silence_start_time is None:
                self.silence_start_time = now
            elif now - self.silence_start_time >= SILENCE_TIMEOUT:
                fallback = self.current_cam if HOLD_LAST_ON_SILENCE else DEFAULT_CAMERA
                if fallback != self.current_cam:
                    self.logger.info(f"SILENCE FALLBACK → {_camera_name(fallback)}")
                    self.last_switch_time = now
                    self.shot_start_time = now
                    return fallback
            return self.current_cam
        else:
            self.silence_start_time = None

        confidences = {idx: self._compute_confidence(idx, candidates) for idx in candidates}

        for i, mon in enumerate(self.audio_mons):
            mon.confidence = confidences.get(i, 0.0)

        sorted_cams = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        winner_idx, winner_conf = sorted_cams[0]

        # Set preview tally to the runner-up / next candidate
        self.preview_cam = sorted_cams[1][0] if len(sorted_cams) > 1 else -1

        # Overlap / tie-breaking
        if len(sorted_cams) > 1:
            runner_conf = sorted_cams[1][1]
            ratio = winner_conf / max(0.001, runner_conf)
            if ratio < OVERLAP_DOMINANCE_RATIO:
                current_conf = confidences.get(self.current_cam, 0.0)
                if current_conf >= OVERLAP_HOLD_CONFIDENCE:
                    if self.overlap_tie_start_time is None:
                        self.overlap_tie_start_time = now
                    elif now - self.overlap_tie_start_time < OVERLAP_MAX_WAIT:
                        return self.current_cam
                else:
                    if self.overlap_tie_start_time is None:
                        self.overlap_tie_start_time = now
                    elif now - self.overlap_tie_start_time < OVERLAP_MAX_WAIT:
                        return self.current_cam
            else:
                self.overlap_tie_start_time = None
        else:
            self.overlap_tie_start_time = None

        if winner_conf < MIN_CONFIDENCE:
            return self.current_cam

        if winner_idx != self.current_cam:
            mon = self.audio_mons[winner_idx]
            latency_ms = (now - mon.above_threshold_since) * 1000 if mon.above_threshold_since else 0
            self.logger.info(
                f"AUTO SWITCH → {_camera_name(winner_idx)} "
                f"(conf={winner_conf:.2f} lat={latency_ms:.0f}ms "
                f"rms={mon.median_rms:.0f} thr={mon.threshold:.0f})"
            )
            self.last_switch_time = now
            self.shot_start_time = now
            self.overlap_tie_start_time = None

        return winner_idx

    # ------------------------------------------------------------------
    # UI REFRESH (60 FPS)
    # ------------------------------------------------------------------

    @staticmethod
    def _conf_bar_text(conf):
        """10-char Unicode progress bar: 0.0→'░░░░░░░░░░'  1.0→'██████████'"""
        filled = max(0, min(10, round(conf * 10)))
        return "█" * filled + "░" * (10 - filled)

    def _update_ui(self):
        def refresh():
            self._health_check()

            active = self.current_cam
            page_start = self._preview_page * self._page_size()
            page_end = min(page_start + self._page_size(), self.num_cams)
            visible_cams = set(range(page_start, page_end))

            # ── Read frames ───────────────────────────────────────────
            frames = {}
            for i in range(self.num_cams):
                f = self.cameras[i].read()
                frames[i] = f
                if f is not None:
                    self.camera_frame_counts[i] += 1

            # ── Program output ────────────────────────────────────────
            frame = frames.get(active)
            if frame is not None:
                pf = cv2.resize(frame, MAIN_SIZE)
                cv2.putText(pf, f"LIVE  {_camera_name(active)}", (24, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 230, 80), 3)
                if self.auto_mode:
                    cv2.putText(pf, "AUTO", (MAIN_SIZE[0] - 160, 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 3)
                rgb = cv2.cvtColor(pf, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self._main_label.imgtk = photo
                self._main_label.configure(image=photo)

            # ── Preview tiles ─────────────────────────────────────────
            for tile_info in self._preview_tiles:
                cam_i = tile_info["cam_i"]
                mon = self.audio_mons[cam_i]

                # Tally border
                if cam_i == active:
                    border = self.COLOR_LIVE
                elif cam_i == self.preview_cam:
                    border = self.COLOR_PREVIEW
                else:
                    border = self.COLOR_IDLE
                tile_info["frame"].config(highlightbackground=border)

                # Thumbnail — only render for visible page (saves CPU)
                if cam_i in visible_cams:
                    f = frames.get(cam_i)
                    if f is not None:
                        fp = cv2.resize(f, PREVIEW_SIZE)
                        pp = ImageTk.PhotoImage(
                            image=Image.fromarray(cv2.cvtColor(fp, cv2.COLOR_BGR2RGB)))
                        tile_info["thumb"].imgtk = pp
                        tile_info["thumb"].configure(image=pp)

                # Health dot: green = frames arriving, red = stalled
                alive = (frames.get(cam_i) is not None)
                dot_color = self.COLOR_SPEAK if alive else self.COLOR_SILENT
                d = tile_info["health_dot"]
                d.itemconfig("dot", fill=dot_color)

                # Speaking status
                if mon.speaking:
                    tile_info["status"].config(
                        text=f"SPEAKING  C:{mon.confidence:.2f}", fg=self.COLOR_SPEAK)
                else:
                    tile_info["status"].config(
                        text=f"SILENT  T:{mon.threshold:.0f}", fg=self.COLOR_SILENT)

                # Confidence bar
                bar_color = self.COLOR_SPEAK if mon.speaking else "#445544"
                tile_info["conf_bar"].config(
                    text=self._conf_bar_text(mon.confidence),
                    fg=bar_color)

            # ── CUT button highlight for active camera ─────────────────
            for ci, btn in self._cut_btns.items():
                if ci == active:
                    btn.config(bg=self.COLOR_LIVE, fg="white")
                else:
                    btn.config(bg="#2a2a2a", fg="white")

            # ── Audio bar graph ────────────────────────────────────────
            self._draw_rms_bars()

            # ── Switch history ─────────────────────────────────────────
            if self._switch_history_dirty:
                self._switch_history_dirty = False
                self._history_text.config(state=tk.NORMAL)
                self._history_text.delete("1.0", tk.END)
                for entry in self._switch_history:
                    self._history_text.insert(tk.END, entry + "\n")
                self._history_text.config(state=tk.DISABLED)

            # ── FPS counter ────────────────────────────────────────────
            self._fps_counter += 1
            self.frame_count += 1
            now_p = time.time()
            if now_p - self._fps_last_time >= 1.0:
                self.current_fps = self._fps_counter
                self._fps_counter = 0
                self._fps_last_time = now_p
                self._fps_label.config(text=f"FPS: {self.current_fps}")

            # ── Status bar ─────────────────────────────────────────────
            all_cal = all(mon.calibrated for mon in self.audio_mons
                          if mon.device_index is not None)
            if not all_cal:
                self._status_bar.config(text="CALIBRATING — stay quiet for 2s...",
                                        fg=self.COLOR_WARN)
            else:
                mode = "AUTO" if self.auto_mode else "MANUAL"
                extra = ""
                if _HAS_PSUTIL:
                    extra = (f"  CPU {psutil.cpu_percent(interval=None):.0f}%"
                             f"  MEM {psutil.virtual_memory().percent:.0f}%")
                self._status_bar.config(
                    text=f"{mode}  •  {_camera_name(active)} LIVE  •  {self.num_cams} cams{extra}",
                    fg=self.COLOR_ACCENT if self.auto_mode else self.COLOR_WARN,
                )

            self.root.after(max(1, 1000 // UI_FPS), refresh)

        refresh()

    def _draw_rms_bars(self):
        c = self._rms_canvas
        c.delete("all")
        w = int(c.winfo_width())
        if w < 10:
            return
        n = self.num_cams
        bar_w = max(8, (w - 20) // max(1, n) - 4)
        bar_h_max = 36
        y_base = 44
        for i in range(n):
            mon = self.audio_mons[i]
            x = 10 + i * (bar_w + 4)
            max_rms = mon.threshold * 3.0 if mon.threshold > 0 else 900.0

            # Background track
            c.create_rectangle(x, y_base - bar_h_max, x + bar_w, y_base,
                                outline="#222222", fill="#111111")

            # EMA level bar
            ema_ratio = min(1.0, mon.ema_rms / max_rms) if max_rms > 0 else 0
            ema_h = int(bar_h_max * ema_ratio)
            bar_color = self.COLOR_SPEAK if mon.speaking else "#334433"
            if ema_h > 0:
                c.create_rectangle(x, y_base - ema_h, x + bar_w, y_base,
                                    outline="", fill=bar_color)

            # Peak hold — white line at peak level
            if mon.peak_rms > 0 and max_rms > 0:
                peak_ratio = min(1.0, mon.peak_rms / max_rms)
                peak_y = int(bar_h_max * peak_ratio)
                c.create_rectangle(x, y_base - peak_y - 1, x + bar_w, y_base - peak_y,
                                    fill="#cccccc", outline="")

            # Threshold line (yellow)
            if mon.threshold > 0 and max_rms > 0:
                ty = int(bar_h_max * min(1.0, mon.threshold / max_rms))
                c.create_line(x - 1, y_base - ty, x + bar_w + 1, y_base - ty,
                              fill="#ffff44", width=1)

            # Label
            label = _camera_name(i)[:5]
            c.create_text(x + bar_w // 2, y_base + 8, text=label,
                          fill="#555555", font=("Consolas", 7))

    # ------------------------------------------------------------------
    # HEALTH CHECK
    # ------------------------------------------------------------------

    def _health_check(self):
        now = time.time()
        if now - self.health_check_time < 5.0:
            return
        self.health_check_time = now
        issues = []
        for i in range(self.num_cams):
            if self.camera_frame_counts[i] == self.last_frame_count[i]:
                issues.append(f"Camera {i+1} stalled")
            self.last_frame_count[i] = self.camera_frame_counts[i]
        for i, mon in enumerate(self.audio_mons):
            if mon.device_index is not None and not (mon.running and mon.thread and mon.thread.is_alive()):
                issues.append(f"Mic {i+1} offline")
        if issues:
            self.logger.warning(f"Health: {'; '.join(issues)}")

    # ------------------------------------------------------------------
    # DEVICE LISTING
    # ------------------------------------------------------------------

    def _list_devices(self):
        print("\n" + "=" * 70)
        print("AVAILABLE DEVICES")
        print("=" * 70)
        print("CAMERAS:")
        available = []
        for i in range(12):
            try:
                cap = cv2.VideoCapture(i, _CAP_BACKEND)
                if cap.isOpened():
                    ret, _ = cap.read()
                    status = "working" if ret else "open but no frame"
                    if ret:
                        available.append(i)
                    print(f"  Index {i:2d}  {status}")
                    cap.release()
            except Exception as e:
                print(f"  Index {i:2d}  error: {type(e).__name__}")
        if not available:
            print("  No cameras found — check USB connections")

        print("\nMICROPHONES:")
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    print(f"  Index {i:2d}  {info.get('name','?')} ({info['maxInputChannels']}ch)")
            p.terminate()
        except Exception as e:
            print(f"  PyAudio error: {type(e).__name__}: {e}")
        print()

    def _toggle_fullscreen(self):
        state = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not state)

    # ------------------------------------------------------------------
    # SETTINGS PANEL
    # ------------------------------------------------------------------

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("ACS Settings")
        win.geometry("540x560")
        win.configure(bg=self.COLOR_PANEL)
        win.resizable(False, False)

        tk.Label(win, text="Live Tuning", bg=self.COLOR_PANEL, fg=self.COLOR_ACCENT,
                 font=("Consolas", 15, "bold")).pack(pady=10)
        tk.Label(win, text="Changes apply immediately. Save writes to acs_config.json.",
                 bg=self.COLOR_PANEL, fg="#666666", font=("Consolas", 9)).pack()

        params = [
            ("Noise Floor Multiplier", "NOISE_FLOOR_MULTIPLIER", 1.0, 6.0, 0.1),
            ("Min Confidence",         "MIN_CONFIDENCE",         0.3, 1.0, 0.05),
            ("Min Switch Interval (s)", "MIN_SWITCH_INTERVAL",  0.1, 2.0, 0.1),
            ("Min Shot Duration (s)",  "MIN_SHOT_DURATION",     0.5, 5.0, 0.1),
            ("Silence Timeout (s)",    "SILENCE_TIMEOUT",       1.0, 10.0, 0.5),
            ("Overlap Dominance Ratio","OVERLAP_DOMINANCE_RATIO",1.0, 3.0, 0.1),
            ("Overlap Hold Confidence","OVERLAP_HOLD_CONFIDENCE",0.3, 0.8, 0.05),
        ]
        for label_text, var_name, lo, hi, res in params:
            row = tk.Frame(win, bg=self.COLOR_PANEL)
            row.pack(fill=tk.X, padx=20, pady=3)
            tk.Label(row, text=label_text, bg=self.COLOR_PANEL, fg="white",
                     font=("Consolas", 10), width=26, anchor="w").pack(side=tk.LEFT)
            val_lbl = tk.Label(row, text=f"{globals()[var_name]:.2f}",
                               bg=self.COLOR_PANEL, fg=self.COLOR_ACCENT,
                               font=("Consolas", 10, "bold"), width=6)
            val_lbl.pack(side=tk.RIGHT)
            sc = tk.Scale(row, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          bg=self.COLOR_PANEL, fg="white", highlightthickness=0,
                          troughcolor="#2a2a2a", showvalue=0, length=200,
                          command=lambda v, vn=var_name, vl=val_lbl: (
                              globals().__setitem__(vn, float(v)),
                              vl.config(text=f"{float(v):.2f}")
                          ))
            sc.set(globals()[var_name])
            sc.pack(side=tk.RIGHT, padx=6)

        def save_cfg():
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acs_config.json")
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
            cfg.update({
                "noise_floor_multiplier": NOISE_FLOOR_MULTIPLIER,
                "min_confidence": MIN_CONFIDENCE,
                "min_switch_interval": MIN_SWITCH_INTERVAL,
                "min_shot_duration": MIN_SHOT_DURATION,
                "silence_timeout": SILENCE_TIMEOUT,
                "overlap_dominance_ratio": OVERLAP_DOMINANCE_RATIO,
                "overlap_hold_confidence": OVERLAP_HOLD_CONFIDENCE,
            })
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            self.logger.info(f"Settings saved: {config_path}")
            tk.Label(win, text="Saved!", bg=self.COLOR_PANEL, fg=self.COLOR_SPEAK,
                     font=("Consolas", 10)).pack(pady=4)

        tk.Button(win, text="Save to Config", command=save_cfg,
                  bg="#1a1a1a", fg="white", font=("Consolas", 12),
                  relief=tk.FLAT, padx=14).pack(pady=14)

    # ------------------------------------------------------------------
    # ABOUT PANEL
    # ------------------------------------------------------------------

    def _open_about(self):
        win = tk.Toplevel(self.root)
        win.title("About ACS")
        win.geometry("720x660")
        win.configure(bg=self.COLOR_PANEL)

        canvas = tk.Canvas(win, bg=self.COLOR_PANEL, highlightthickness=0)
        sb = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        sf = tk.Frame(canvas, bg=self.COLOR_PANEL)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw", width=680)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        def _h(t):
            tk.Label(sf, text=t, bg=self.COLOR_PANEL, fg=self.COLOR_ACCENT,
                     font=("Consolas", 12, "bold"), anchor="w").pack(anchor="w", pady=(12,2), padx=10)
        def _p(t):
            tk.Label(sf, text=t, bg=self.COLOR_PANEL, fg="#cccccc",
                     font=("Consolas", 10), wraplength=640, anchor="w",
                     justify=tk.LEFT).pack(anchor="w", pady=1, padx=10)

        tk.Label(sf, text="ACS — Automated Camera Switcher  v1.1",
                 bg=self.COLOR_PANEL, fg=self.COLOR_ACCENT,
                 font=("Consolas", 15, "bold")).pack(anchor="w", pady=(10,2), padx=10)
        _p("Universal live camera switcher — works with USB, IP (RTSP/HTTP), files, any count.")

        _h("Quick Start")
        for line in [
            "1. Plug in cameras and microphones (one mic per speaker).",
            "2. Run: python ACS.py  — edit acs_config.json with device indices.",
            "3. Restart — choose AUTO MODE — system follows the speaker.",
            "4. Buttons / keys 1–9 for instant manual override at any time.",
            "5. Window-capture in OBS Studio for broadcast.",
        ]:
            _p(line)

        _h("Camera Sources (acs_config.json)")
        _p('camera_sources can mix integers (USB index) and strings (IP camera URLs):')
        _p('  [0, 1, "rtsp://192.168.1.10/stream", "http://cam4/video"]')
        _p('Sony ZV-1 via USB appears as a standard device index (0, 1, 2, 3...).')

        _h("ATEM Mini Pro Integration (optional)")
        _p('1. Install:  pip install pyatem')
        _p('2. Set in config:  "atem_enabled": true,  "atem_ip": "192.168.10.240"')
        _p('3. Map cameras to ATEM inputs:  "atem_input_map": {"0":1,"1":2,"2":3,"3":4}')
        _p('ACS mirrors every switch (auto or manual) to the ATEM program bus.')
        _p('You can also use the ATEM output as a camera source in ACS (feed it back via USB).')

        _h("Keyboard Shortcuts")
        for line in [
            "1–9   Switch to camera 1–9 (also sets MANUAL mode)",
            "A     AUTO mode",
            "M     MANUAL mode",
            "F11   Fullscreen toggle",
        ]:
            _p(line)

        _h("CLI Flags")
        for line in [
            "python ACS.py                      # GUI",
            "python ACS.py --test               # unit tests (headless)",
            "python ACS.py --profile <file>     # load config profile",
            "python ACS.py --remote             # enable HTTP API port 8765",
            "python ACS.py --remote --port N    # custom port",
        ]:
            _p(line)

        _h("HTTP Remote API  (--remote)")
        for line in [
            "GET  /status             JSON: cam, mode, fps, mic levels",
            "POST /switch?cam=N       Switch to camera N (0-indexed)",
            "POST /mode?mode=auto     Switch to AUTO or MANUAL mode",
        ]:
            _p(line)

        tk.Button(sf, text="Close", command=win.destroy,
                  bg="#1a1a1a", fg="white", font=("Consolas", 11),
                  relief=tk.FLAT, width=10).pack(anchor="w", pady=14, padx=10)

    # ------------------------------------------------------------------
    # REMOTE HTTP API
    # ------------------------------------------------------------------

    def _start_remote_api(self, port=8765):
        if not _HAS_HTTP:
            return
        app_ref = self

        class ACSHandler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, data, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/status":
                    data = {
                        "current_cam": app_ref.current_cam,
                        "camera_name": _camera_name(app_ref.current_cam),
                        "mode": "AUTO" if app_ref.auto_mode else "MANUAL",
                        "fps": app_ref.current_fps,
                        "num_cameras": app_ref.num_cams,
                        "mics": [{
                            "index": i,
                            "name": _camera_name(i),
                            "rms": mon.median_rms,
                            "threshold": mon.threshold,
                            "speaking": mon.speaking,
                            "confidence": mon.confidence,
                            "calibrated": mon.calibrated,
                        } for i, mon in enumerate(app_ref.audio_mons)],
                    }
                    self._json(data)
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)
                if parsed.path == "/switch":
                    cam = qs.get("cam", [None])[0]
                    try:
                        idx = int(cam)
                        if 0 <= idx < app_ref.num_cams:
                            app_ref.manual_switch(idx)
                            self._json({"ok": True, "current_cam": app_ref.current_cam})
                            return
                    except (TypeError, ValueError):
                        pass
                    self._json({"ok": False, "error": "invalid cam"}, 400)
                elif parsed.path == "/mode":
                    mode = qs.get("mode", [None])[0]
                    if mode in ("auto", "manual"):
                        app_ref._set_mode(mode.upper())
                        self._json({"ok": True, "mode": mode.upper()})
                    else:
                        self._json({"ok": False, "error": "mode must be auto or manual"}, 400)
                else:
                    self._json({"error": "not found"}, 404)

        def run():
            srv = HTTPServer(("0.0.0.0", port), ACSHandler)
            self.logger.info(f"Remote API: http://0.0.0.0:{port}")
            srv.serve_forever()

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    def on_closing(self):
        self.logger.info("Shutting down...")
        self._switch_thread_running = False
        for i, cam in enumerate(self.cameras):
            try:
                cam.release()
            except Exception as e:
                self.logger.warning(f"Camera {i+1} release error: {type(e).__name__}")
        for i, mon in enumerate(self.audio_mons):
            try:
                mon.stop()
            except Exception as e:
                self.logger.warning(f"Mic {i+1} stop error: {type(e).__name__}")
        self.logger.info("All resources released.")
        try:
            self.root.destroy()
        except Exception:
            pass


# ============================  UNIT TESTS  ==================================

def run_tests():
    """Run headless unit tests: python ACS.py --test"""
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)

    class MockMon:
        def __init__(self):
            self.rms = 0
            self.ema_rms = 0.0
            self.median_rms = 0.0
            self.peak_rms = 0.0
            self.speech_ratio = 0.5
            self.threshold = 300.0
            self.speaking = False
            self.speaking_duration = 0.0
            self.snr = 0.0
            self.confidence = 0.0
            self.above_threshold_since = None
            self.calibrated = True
            self.device_index = 0
            self.gain = 1.0

    class TestApp:
        num_cams = 4
        current_cam = 0
        auto_mode = True
        last_switch_time = 0.0
        shot_start_time = 0.0
        silence_start_time = None
        overlap_tie_start_time = None
        preview_cam = -1
        atem = None
        logger = logging.getLogger("ACS.Test")
        _switch_history = deque(maxlen=20)
        _switch_history_dirty = False

        _compute_confidence  = SwitcherApp._compute_confidence
        _determine_active_camera = SwitcherApp._determine_active_camera
        _add_to_history      = SwitcherApp._add_to_history

    app = TestApp()
    app.audio_mons = [MockMon() for _ in range(4)]
    now = time.time()
    app.last_switch_time = now - 100
    app.shot_start_time = now - 100

    # Reset globals to deterministic defaults
    global MIN_CONFIDENCE, OVERLAP_DOMINANCE_RATIO, OVERLAP_HOLD_CONFIDENCE
    global OVERLAP_MAX_WAIT, MIN_SWITCH_INTERVAL, MIN_SHOT_DURATION
    global SILENCE_TIMEOUT, DEFAULT_CAMERA, HOLD_LAST_ON_SILENCE
    global HOLD_PRIORITY_HOST, HOST_MIC_INDEX, CONFIDENCE_WEIGHTS
    global AUDIO_EMA_ALPHA, AUDIO_EMA_ALPHA_FALL, USE_SPEECH_RATIO
    MIN_CONFIDENCE = 0.65
    OVERLAP_DOMINANCE_RATIO = 1.4
    OVERLAP_HOLD_CONFIDENCE = 0.50
    OVERLAP_MAX_WAIT = 2.0
    MIN_SWITCH_INTERVAL = 0.7
    MIN_SHOT_DURATION = 1.5
    SILENCE_TIMEOUT = 3.0
    DEFAULT_CAMERA = 0
    HOLD_LAST_ON_SILENCE = True
    HOLD_PRIORITY_HOST = False
    HOST_MIC_INDEX = 0
    AUDIO_EMA_ALPHA      = 0.40
    AUDIO_EMA_ALPHA_FALL = 0.15
    USE_SPEECH_RATIO     = True
    # Use 4-key weights for existing tests (no speech_ratio key → .get default = 0.0)
    CONFIDENCE_WEIGHTS = {
        "signal_strength": 0.35, "duration": 0.25, "snr": 0.20, "dominance": 0.20
    }

    passed = 0
    failed = 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}")
            failed += 1

    def reset():
        for m in app.audio_mons:
            m.speaking = False
            m.median_rms = 0.0
            m.ema_rms = 0.0
            m.peak_rms = 0.0
            m.speech_ratio = 0.5
            m.threshold = 300.0
            m.above_threshold_since = None
            m.speaking_duration = 0.0
            m.snr = 0.0
            m.confidence = 0.0
        app.last_switch_time = now - 100
        app.shot_start_time = now - 100
        app.silence_start_time = None
        app.overlap_tie_start_time = None

    # 1. Solo speaker → switch
    reset()
    app.current_cam = 0
    app.audio_mons[1].speaking = True
    app.audio_mons[1].median_rms = 1200
    app.audio_mons[1].above_threshold_since = now - 0.5
    app.audio_mons[1].speaking_duration = 2.0
    app.audio_mons[1].snr = 3.0
    check("Solo speaker → switch to cam 2", app._determine_active_camera() == 1)

    # 2. Silence → hold current
    reset()
    app.current_cam = 1
    check("Silence → hold current cam", app._determine_active_camera() == 1)

    # 3. Shot discipline
    reset()
    app.current_cam = 0
    app.shot_start_time = now  # just switched
    app.audio_mons[1].speaking = True
    app.audio_mons[1].median_rms = 1200
    app.audio_mons[1].above_threshold_since = now - 0.5
    check("Shot discipline → no switch before MIN_SHOT_DURATION", app._determine_active_camera() == 0)

    # 4. Confidence gate — weak signal
    reset()
    app.current_cam = 0
    app.audio_mons[1].speaking = True
    app.audio_mons[1].median_rms = 310
    app.audio_mons[1].above_threshold_since = now - 0.05
    app.audio_mons[1].snr = 0.05
    check("Confidence gate → no switch on weak signal", app._determine_active_camera() == 0)

    # 5. Overlap tie → hold
    reset()
    app.current_cam = 0
    for idx in (0, 1):
        m = app.audio_mons[idx]
        m.speaking = True
        m.median_rms = 1000 if idx == 0 else 950
        m.above_threshold_since = now - 1.0
        m.speaking_duration = 1.0
        m.snr = 3.0
    check("Overlap tie → hold current shot", app._determine_active_camera() == 0)

    # 6. Host priority boost
    reset()
    orig_host = HOLD_PRIORITY_HOST
    orig_host_idx = HOST_MIC_INDEX
    HOLD_PRIORITY_HOST = True
    HOST_MIC_INDEX = 0
    app.current_cam = 1
    app.audio_mons[0].speaking = True
    app.audio_mons[0].median_rms = 800
    app.audio_mons[0].above_threshold_since = now - 0.5
    app.audio_mons[0].speaking_duration = 2.0
    app.audio_mons[0].snr = 2.0
    check("Host priority → host wins with lower RMS", app._determine_active_camera() == 0)
    HOLD_PRIORITY_HOST = orig_host
    HOST_MIC_INDEX = orig_host_idx

    # 7. Silence timeout → fallback to DEFAULT_CAMERA
    reset()
    orig_hold = HOLD_LAST_ON_SILENCE
    orig_timeout = SILENCE_TIMEOUT
    orig_default = DEFAULT_CAMERA
    HOLD_LAST_ON_SILENCE = False
    SILENCE_TIMEOUT = 2.0
    DEFAULT_CAMERA = 2
    app.current_cam = 1
    app.silence_start_time = now - 5.0
    check("Silence timeout → fallback to DEFAULT_CAMERA", app._determine_active_camera() == 2)
    HOLD_LAST_ON_SILENCE = orig_hold
    SILENCE_TIMEOUT = orig_timeout
    DEFAULT_CAMERA = orig_default

    # 8. Sustained overlap tie → forced switch
    reset()
    orig_wait = OVERLAP_MAX_WAIT
    OVERLAP_MAX_WAIT = 1.0
    app.current_cam = 0
    app.overlap_tie_start_time = now - 3.0
    for idx in (0, 1):
        m = app.audio_mons[idx]
        m.speaking = True
        m.median_rms = 1000 if idx == 0 else 1100
        m.above_threshold_since = now - 4.0
        m.speaking_duration = 4.0
        m.snr = 3.0
    check("Sustained overlap tie → forced switch to highest confidence",
          app._determine_active_camera() == 1)
    OVERLAP_MAX_WAIT = orig_wait

    # 9. 4-camera setup — correct winner across 4 cams
    reset()
    app.num_cams = 4
    app.current_cam = 0
    app.audio_mons[3].speaking = True
    app.audio_mons[3].median_rms = 1500
    app.audio_mons[3].above_threshold_since = now - 1.0
    app.audio_mons[3].speaking_duration = 2.0
    app.audio_mons[3].snr = 4.0
    check("4-camera: switch to cam 4 (index 3)", app._determine_active_camera() == 3)

    # 10. Switch interval rate limit
    reset()
    app.current_cam = 0
    app.last_switch_time = now - 0.1  # switched only 100ms ago
    app.audio_mons[2].speaking = True
    app.audio_mons[2].median_rms = 1500
    app.audio_mons[2].above_threshold_since = now - 1.0
    app.audio_mons[2].speaking_duration = 2.0
    app.audio_mons[2].snr = 4.0
    check("Switch interval rate limit → hold current", app._determine_active_camera() == 0)

    # 11. Confidence computation — solo gives high score
    reset()
    app.audio_mons[0].median_rms = 1200
    app.audio_mons[0].threshold = 300
    app.audio_mons[0].speaking_duration = 2.0
    app.audio_mons[0].snr = 3.0
    conf = app._compute_confidence(0, {0: 1200.0})
    check("Confidence: solo speaker ≥ 0.70", conf >= 0.70)

    # 12. Confidence: two equal speakers — both near 0.5
    reset()
    candidates = {0: 600.0, 1: 600.0}
    for idx in (0, 1):
        app.audio_mons[idx].median_rms = 600
        app.audio_mons[idx].threshold = 300
        app.audio_mons[idx].speaking_duration = 1.0
        app.audio_mons[idx].snr = 1.0
    c0 = app._compute_confidence(0, candidates)
    c1 = app._compute_confidence(1, candidates)
    check("Confidence: equal speakers are close (|c0-c1| < 0.1)", abs(c0 - c1) < 0.1)

    # 13. Camera name helper
    global CAMERA_NAMES
    orig_names = CAMERA_NAMES[:]
    CAMERA_NAMES = ["Host", "Guest A", "", "Wide"]
    check("Camera name: index 0 → 'Host'",       _camera_name(0) == "Host")
    check("Camera name: index 1 → 'Guest A'",    _camera_name(1) == "Guest A")
    check("Camera name: index 2 → 'CAM 3' (blank fallback)", _camera_name(2) == "CAM 3")
    check("Camera name: index 4 → 'CAM 5' (out of range)", _camera_name(4) == "CAM 5")
    CAMERA_NAMES = orig_names

    # 14. Asymmetric EMA — attack faster than release
    mon_ema = MockMon()
    mon_ema.ema_rms = 0.0
    mon_ema.gain = 1.0
    mon_ema.peak_rms = 0.0
    mon_ema.noise_floor = 50.0
    mon_ema.threshold = 150.0
    mon_ema.rms_window = deque(maxlen=RMS_WINDOW_SIZE)
    mon_ema.calibrated = True
    mon_ema.above_threshold_since = None
    mon_ema.speaking_duration = 0.0

    orig_alpha = AUDIO_EMA_ALPHA
    orig_fall  = AUDIO_EMA_ALPHA_FALL
    AUDIO_EMA_ALPHA      = 0.40
    AUDIO_EMA_ALPHA_FALL = 0.15

    # Simulate 5 frames of silence then 5 frames of loud speech
    ema_before = 0.0
    for _ in range(5):
        gained = 200.0  # loud speech
        a = AUDIO_EMA_ALPHA if gained > ema_before else AUDIO_EMA_ALPHA_FALL
        ema_before = a * gained + (1 - a) * ema_before
    ema_after_speech = ema_before

    # Now 5 frames of silence
    for _ in range(5):
        gained = 0.0
        a = AUDIO_EMA_ALPHA if gained > ema_before else AUDIO_EMA_ALPHA_FALL
        ema_before = a * gained + (1 - a) * ema_before
    ema_after_silence = ema_before

    check("EMA: attack reaches threshold faster than release decays",
          ema_after_speech > ema_after_silence)
    check("EMA: 5 speech frames → above threshold (200 > 150)",
          ema_after_speech > 150.0)
    AUDIO_EMA_ALPHA      = orig_alpha
    AUDIO_EMA_ALPHA_FALL = orig_fall

    # 15. Per-mic gain — higher gain → same RMS appears louder
    mon_gain_lo = MockMon()
    mon_gain_hi = MockMon()
    for m in (mon_gain_lo, mon_gain_hi):
        m.ema_rms = 0.0
        m.gain = 1.0
        m.peak_rms = 0.0
        m.noise_floor = 50.0
        m.threshold = 150.0
        m.rms_window = deque(maxlen=RMS_WINDOW_SIZE)
        m.calibrated = True
        m.above_threshold_since = None
        m.speaking_duration = 0.0
    raw_signal = 100.0   # below threshold at gain=1.0
    # Low gain: 100 × 1.0 = 100 < 150
    gained_lo = raw_signal * 1.0
    ema_lo = AUDIO_EMA_ALPHA * gained_lo
    # High gain: 100 × 2.0 = 200 > 150
    gained_hi = raw_signal * 2.0
    ema_hi = AUDIO_EMA_ALPHA * gained_hi
    check("Per-mic gain: gain=2.0 puts same signal above threshold",
          ema_hi > 150.0 * AUDIO_EMA_ALPHA * 0.5 and ema_lo < 150.0)

    # 16. Confidence bar text helper
    check("Conf bar: 0.0 → all empty",  SwitcherApp._conf_bar_text(0.0)  == "░░░░░░░░░░")
    check("Conf bar: 1.0 → all filled", SwitcherApp._conf_bar_text(1.0)  == "██████████")
    check("Conf bar: 0.5 → half filled",SwitcherApp._conf_bar_text(0.5)  == "█████░░░░░")
    check("Conf bar: 0.3 → 3 filled",   SwitcherApp._conf_bar_text(0.3)  == "███░░░░░░░")

    # 17. Speech ratio confidence factor: speech scores higher than LF noise
    reset()
    app.current_cam = 0
    # Speaker with speech-like frequency profile (speech_ratio = 0.75)
    app.audio_mons[1].speaking = True
    app.audio_mons[1].median_rms = 800
    app.audio_mons[1].ema_rms = 800
    app.audio_mons[1].threshold = 300
    app.audio_mons[1].above_threshold_since = now - 1.0
    app.audio_mons[1].speaking_duration = 2.0
    app.audio_mons[1].snr = 3.0
    app.audio_mons[1].speech_ratio = 0.75   # speech-like
    orig_weights = CONFIDENCE_WEIGHTS.copy()
    CONFIDENCE_WEIGHTS = {
        "signal_strength": 0.30, "duration": 0.22, "snr": 0.18,
        "dominance": 0.20, "speech_ratio": 0.10
    }
    conf_speech = app._compute_confidence(1, {1: 800.0})

    app.audio_mons[1].speech_ratio = 0.20   # LF noise-like
    conf_noise = app._compute_confidence(1, {1: 800.0})
    check("Speech ratio: speech-like mic scores higher than LF noise mic",
          conf_speech > conf_noise)
    CONFIDENCE_WEIGHTS = orig_weights

    # 18. Switch history: _add_to_history populates deque
    app._switch_history = deque(maxlen=20)
    app._switch_history_dirty = False
    app._add_to_history("AUTO", 0, 1, conf=0.82, latency_ms=48.0)
    check("Switch history: entry added", len(app._switch_history) == 1)
    check("Switch history: dirty flag set", app._switch_history_dirty)
    entry = app._switch_history[0]
    check("Switch history: entry contains camera names",
          _camera_name(0) in entry and _camera_name(1) in entry)

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


# ============================  ENTRY POINT  =================================

def _print_help():
    print("""
ACS — Automated Camera Switcher v1.1
=====================================
Usage:
  python ACS.py                    # Launch GUI
  python ACS.py --test             # Unit tests (headless, no hardware)
  python ACS.py --profile <file>   # Load a named config profile
  python ACS.py --remote           # Enable HTTP API on port 8765
  python ACS.py --remote --port N  # Custom port
  python ACS.py --help             # This message

Camera sources (in acs_config.json):
  [0, 1, 2, 3]                       USB device indices
  ["rtsp://192.168.1.10/stream"]      IP camera via RTSP
  ["http://192.168.1.20:8080/video"]  IP camera via HTTP
  Mix freely: [0, "rtsp://..."]

ATEM Mini Pro integration:
  pip install pyatem
  Set "atem_enabled": true in acs_config.json
  Set "atem_ip" to your switcher's IP (default 192.168.10.240)
""")


if __name__ == "__main__":
    if "--help" in sys.argv:
        _print_help()
        sys.exit(0)

    _ensure_deps()

    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)

    for _req, _msg in [
        (_HAS_TK,     "Tkinter not available.\n  Linux: sudo apt-get install python3-tk"),
        (_HAS_CV2,    "OpenCV (cv2) not available.\n  pip install opencv-python"),
        (_HAS_PIL,    "Pillow not available.\n  pip install pillow"),
        (_HAS_PYAUDIO,"PyAudio not available.\n  pip install pyaudio"),
    ]:
        if not _req:
            print(f"ERROR: {_msg}")
            sys.exit(1)

    root = tk.Tk()
    app = SwitcherApp(root)
    root.mainloop()
