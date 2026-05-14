import atexit
import threading
import time
from collections import deque
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template

from app.config import AppConfig
from app.pipeline import VideoPipeline


class WebStreamRuntime:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = VideoPipeline(config)

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_metrics = {
            "left": 0,
            "right": 0,
            "today_total": 0,
            "session_total": 0,
            "active_tracks": 0,
            "fps": 0.0,
            "stairs_total": 0,
            "timestamp": datetime.now().isoformat(),
            "distance_vertical_m": 0.0,
            "session_steps": 0,
            "session_distance_m": 0.0,
            "session_distance_vertical_m": 0.0,
        }
        self._history = deque(maxlen=3600)
        self._last_history_ts = 0.0

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._running = True
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            self._running = False

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message

    def _run_loop(self) -> None:
        cap = self.pipeline.open_capture()
        if cap is None:
            self._set_error("Cannot open video source")
            with self._lock:
                self._running = False
            return

        frame_idx = 0
        fps_counter = 0
        fps_value = 0.0
        fps_started = time.time()

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if not self.config.use_camera:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                fps_counter += 1
                now = time.time()
                if now - fps_started >= 1.0:
                    fps_value = fps_counter / (now - fps_started)
                    fps_counter = 0
                    fps_started = now

                processed, metrics = self.pipeline.process_frame(
                    frame,
                    frame_idx=frame_idx,
                    fps_value=fps_value,
                    draw_hud_overlay=False,
                )

                ok, encoded = cv2.imencode(
                    ".jpg",
                    processed,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )
                if not ok:
                    continue

                metrics["timestamp"] = datetime.now().isoformat()
                with self._lock:
                    self._latest_jpeg = encoded.tobytes()
                    self._latest_metrics = metrics
                    if now - self._last_history_ts >= 1.0:
                        self._history.append(metrics.copy())
                        self._last_history_ts = now
        except Exception as exc:
            self._set_error(str(exc))
        finally:
            cap.release()
            with self._lock:
                self._running = False

    def snapshot(self) -> dict:
        with self._lock:
            payload = self._latest_metrics.copy()
            payload["running"] = self._running
            payload["error"] = self._last_error
            payload["history"] = list(self._history)
            payload["source"] = (
                f"camera:{self.config.camera_index}"
                if self.config.use_camera
                else self.config.video_path
            )
            return payload

    def iter_mjpeg(self):
        boundary = b"--frame\r\n"
        while True:
            with self._lock:
                frame = self._latest_jpeg
                running = self._running
            if frame is None:
                if not running:
                    time.sleep(0.2)
                else:
                    time.sleep(0.03)
                continue

            yield (
                boundary
                + b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
            time.sleep(0.03)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    runtime = WebStreamRuntime(AppConfig())
    runtime.start()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video-feed")
    def video_feed():
        return Response(
            runtime.iter_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/metrics")
    def metrics():
        return jsonify(runtime.snapshot())

    @app.route("/api/restart")
    def restart():
        runtime.stop()
        runtime.start()
        return jsonify({"ok": True})

    @app.route("/api/reset-today")
    def reset_today():
        runtime.pipeline.stats.reset_today()
        return jsonify({"ok": True})

    @atexit.register
    def _cleanup_runtime():
        runtime.stop()

    return app


app = create_app()
