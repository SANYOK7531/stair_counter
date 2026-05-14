import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from app.daily_stats import DailyStats
from app.detector import PersonTracker
from app.directional_zone_counter import DirectionalZoneCounter
from app.track_handoff import TrackHandoff
from app.visualizer import (
    COLOR_ENTRY, COLOR_EXIT, COLOR_TRANSIT, draw_hud, draw_track, draw_zone,
)


class VideoPipeline:
    def __init__(self, config):
        self.config = config

        self.tracker = PersonTracker(
            model_path=config.model_path,
            tracker_config=config.tracker_config,
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            person_class_id=config.person_class_id,
            imgsz=config.imgsz,
            augment=config.augment,
        )

        zones = self._load_zones(config.zones_path)
        transit_polygon = zones.get("transit")
        if transit_polygon is None:
            print(
                "[!] zones.json: 'transit' zone is missing — pass-through "
                "cancellation is disabled. Re-run the calibrator to enable."
            )
        self.transit_polygon = transit_polygon

        # Resolution of the frame used during calibration. If the live frame
        # comes at a different size, polygons are rescaled on first frame.
        cal_res = zones.get("_resolution")
        self._calibration_resolution: tuple[int, int] | None = (
            (int(cal_res[0]), int(cal_res[1]))
            if isinstance(cal_res, (list, tuple)) and len(cal_res) == 2
            else None
        )
        self._zones_scaled = False

        self.left_counter = DirectionalZoneCounter(
            name="left",
            entry_polygon=zones["left_entry"],
            exit_polygon=zones["left_exit"],
            transit_polygon=transit_polygon,
            min_frames_in_zone=config.min_frames_in_zone,
            min_exit_coverage=config.min_exit_coverage,
        )
        self.right_counter = DirectionalZoneCounter(
            name="right",
            entry_polygon=zones["right_entry"],
            exit_polygon=zones["right_exit"],
            transit_polygon=transit_polygon,
            min_frames_in_zone=config.min_frames_in_zone,
            min_exit_coverage=config.min_exit_coverage,
        )
        self.counters = {"left": self.left_counter, "right": self.right_counter}

        self.handoff = TrackHandoff(
            max_age_frames=config.handoff_max_age_frames,
            max_distance_px=config.handoff_max_distance_px,
            size_ratio_tolerance=config.handoff_size_ratio_tolerance,
            score_threshold=config.handoff_score_threshold,
            max_predicted_speed_px=config.handoff_max_predicted_speed_px,
        )

        self.stats = DailyStats(config.stats_path)

        self._track_last_seen: dict[int, int] = {}

    @staticmethod
    def _load_zones(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"zones file not found: '{path}'. Run the calibrator: "
                f"python calibrate.py <video_or_camera>"
            )
        with open(p, "r", encoding="utf-8") as f:
            zones = json.load(f)
        required = {"left_entry", "left_exit", "right_entry", "right_exit"}
        missing = required - set(zones.keys())
        if missing:
            raise ValueError(f"missing zones in {path}: {missing}")
        return zones

    def _open_capture(self) -> cv2.VideoCapture | None:
        if self.config.use_camera:
            cap = cv2.VideoCapture(self.config.camera_index)
            source = f"camera {self.config.camera_index}"
        else:
            cap = cv2.VideoCapture(self.config.video_path)
            source = self.config.video_path
        if not cap.isOpened():
            print(f"[ERROR] cannot open source: {source}")
            return None
        return cap

    def open_capture(self) -> cv2.VideoCapture | None:
        return self._open_capture()

    def _flush_new_counts(self) -> None:
        # Each event carries its own datetime so a decrement targets the
        # same hourly bucket as its original increment.
        for name, counter in self.counters.items():
            for action, when in counter.pop_events():
                if action == "increment":
                    self.stats.increment(name, when=when)
                elif action == "decrement":
                    self.stats.decrement(name, when=when)

    def _scale_zones_to_frame(self, frame) -> None:
        """
        Scale calibration polygons to the current frame resolution. Runs once
        on the first frame; later calls are no-ops. Required so a calibration
        produced from a 1920x1080 video stays valid for a 1280x720 live feed
        without re-running the calibrator.
        """
        if self._zones_scaled:
            return
        self._zones_scaled = True

        if self._calibration_resolution is None:
            return

        cw, ch = self._calibration_resolution
        h, w = frame.shape[:2]
        if (w, h) == (cw, ch) or cw <= 0 or ch <= 0:
            return

        sx = w / cw
        sy = h / ch

        def scale(arr: np.ndarray) -> np.ndarray:
            scaled = arr.astype(np.float32).copy()
            scaled[:, 0] *= sx
            scaled[:, 1] *= sy
            return scaled.astype(np.int32)

        for counter in self.counters.values():
            counter.entry_poly = scale(counter.entry_poly)
            counter.exit_poly = scale(counter.exit_poly)
            if counter.transit_poly is not None:
                counter.transit_poly = scale(counter.transit_poly)
            counter._exit_area = float(cv2.contourArea(counter.exit_poly))

        if self.transit_polygon is not None:
            transit_arr = np.array(self.transit_polygon, dtype=np.int32)
            self.transit_polygon = scale(transit_arr).tolist()

        print(
            f"[zones] rescaled from calibration {cw}x{ch} to frame {w}x{h} "
            f"(sx={sx:.3f}, sy={sy:.3f})"
        )

    def _draw_zones(self, frame, visibility: float = 1.0) -> None:
        if visibility <= 0.0:
            return
        draw_zone(frame, self.left_counter.entry_poly.tolist(), COLOR_ENTRY,
                  label="L-entry", visibility=visibility)
        draw_zone(frame, self.left_counter.exit_poly.tolist(), COLOR_EXIT,
                  label="L-exit", visibility=visibility)
        draw_zone(frame, self.right_counter.entry_poly.tolist(), COLOR_ENTRY,
                  label="R-entry", visibility=visibility)
        draw_zone(frame, self.right_counter.exit_poly.tolist(), COLOR_EXIT,
                  label="R-exit", visibility=visibility)
        if self.transit_polygon is not None:
            draw_zone(frame, self.transit_polygon, COLOR_TRANSIT,
                      label="transit", visibility=visibility)

    def _zones_visibility(self) -> float:
        """Returns zone-overlay visibility in [0..1] for the current moment."""
        period = self.config.zones_show_period_s
        duration = self.config.zones_show_duration_s
        fade = self.config.zones_fade_s

        if period <= 0.0:
            return 1.0
        if duration <= 0.0:
            return 0.0

        fade = min(fade, duration / 2.0)

        t = time.time() % period
        if t >= duration:
            return 0.0
        if fade > 0.0:
            if t < fade:
                return t / fade
            if t > duration - fade:
                return max(0.0, (duration - t) / fade)
        return 1.0

    @staticmethod
    def _extract_active_tracks(result):
        active_tracks: dict[int, tuple[int, int, int, int]] = {}
        track_confs: dict[int, float] = {}
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            for box, tid, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                active_tracks[tid] = (x1, y1, x2, y2)
                track_confs[tid] = conf
        return active_tracks, track_confs

    def _update_tracks_and_counters(
        self,
        frame,
        active_tracks: dict[int, tuple[int, int, int, int]],
        track_confs: dict[int, float],
        frame_idx: int,
    ) -> set[int]:
        id_mapping = self.handoff.resolve(active_tracks, frame_idx)

        active_track_ids: set[int] = set()
        for tid, bbox in active_tracks.items():
            eff_id = id_mapping[tid]
            active_track_ids.add(tid)
            active_track_ids.add(eff_id)
            self._track_last_seen[tid] = frame_idx
            self._track_last_seen[eff_id] = frame_idx

            x1, y1, x2, y2 = bbox
            foot_point = ((x1 + x2) // 2, y2)

            for counter in self.counters.values():
                counter.update(eff_id, bbox)

            in_zone = any(
                c._point_in_zone(foot_point) is not None
                for c in self.counters.values()
            )
            label_id = tid if tid == eff_id else f"{tid}<-{eff_id}"
            draw_track(
                frame,
                bbox,
                label_id,
                in_any_zone=in_zone,
                conf=track_confs[tid],
                show_conf=self.config.show_detection_conf,
            )

        return active_track_ids

    def _cleanup_stale_tracks(self, frame_idx: int) -> None:
        ttl = self.config.track_ttl_frames
        stale = {
            tid for tid, last in self._track_last_seen.items()
            if frame_idx - last > ttl
        }
        if not stale:
            return

        for counter in self.counters.values():
            counter.cleanup(stale)
        self.handoff.forget(stale)
        for tid in stale:
            self._track_last_seen.pop(tid, None)

    def process_frame(
        self,
        frame,
        frame_idx: int,
        fps_value: float,
        draw_hud_overlay: bool = True,
    ):
        """Process one frame and return (annotated_frame, metrics)."""
        self._scale_zones_to_frame(frame)
        self._draw_zones(frame, visibility=self._zones_visibility())

        result = self.tracker.track_frame(frame)
        active_tracks, track_confs = self._extract_active_tracks(result)

        active_track_ids = self._update_tracks_and_counters(
            frame,
            active_tracks,
            track_confs,
            frame_idx,
        )

        self._cleanup_stale_tracks(frame_idx)
        self._flush_new_counts()

        left_total = self.left_counter.total_count
        right_total = self.right_counter.total_count
        today_total = self.stats.today_total()
        session_total = left_total + right_total

        total_steps = today_total * self.config.stairs_per_person
        session_steps = session_total * self.config.stairs_per_person
        step_path_m = math.hypot(self.config.step_run_m, self.config.step_rise_m)
        distance_m = total_steps * step_path_m
        distance_vertical_m = total_steps * self.config.step_rise_m
        session_distance_m = session_steps * step_path_m
        session_distance_vertical_m = session_steps * self.config.step_rise_m

        if draw_hud_overlay:
            draw_hud(
                frame,
                counters={"Left": left_total, "Right": right_total},
                today_total=today_total,
                stairs_total=total_steps,
                active_tracks=len(active_track_ids),
                fps=fps_value,
            )

        metrics = {
            "left": left_total,
            "right": right_total,
            "today_total": today_total,
            "session_total": session_total,
            "active_tracks": len(active_track_ids),
            "fps": round(fps_value, 2),
            "stairs_total": total_steps,
            "distance_m": round(distance_m, 2),
            "distance_vertical_m": round(distance_vertical_m, 2),
            "session_steps": session_steps,
            "session_distance_m": round(session_distance_m, 2),
            "session_distance_vertical_m": round(session_distance_vertical_m, 2),
        }
        return frame, metrics

    def run(self) -> None:
        cap = self._open_capture()
        if cap is None:
            return

        fps_start = time.time()
        fps_counter = 0
        fps_value = 0.0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                fps_counter += 1
                now = time.time()
                if now - fps_start >= 1.0:
                    fps_value = fps_counter / (now - fps_start)
                    fps_counter = 0
                    fps_start = now

                frame, _ = self.process_frame(
                    frame,
                    frame_idx=frame_idx,
                    fps_value=fps_value,
                    draw_hud_overlay=True,
                )

                cv2.imshow(self.config.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
