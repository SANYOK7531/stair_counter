import json
import sys
from pathlib import Path

import cv2
import numpy as np


# Camera faces the stairs. Side staircases going down are hidden behind
# walls — only the dark openings are visible. A climbing person first
# appears inside the opening (head + shoulders), then steps onto the
# landing in front of it. The transit zone is the central staircase /
# corridor area beyond the landing — passing through it cancels the count.
ZONES_TO_CAPTURE = [
    ("left_entry", "LEFT OPENING: inside area (where head and shoulders show up)"),
    ("left_exit", "LEFT OPENING: floor in front of the opening"),
    ("right_entry", "RIGHT OPENING: inside area (where head and shoulders show up)"),
    ("right_exit", "RIGHT OPENING: floor in front of the opening"),
    ("transit", "TRANSIT zone (shared): central stairs going up / corridor away "
                "from landing. Passing through here cancels the count."),
]

INSTRUCTIONS = [
    "LMB - add point",
    "RMB or ENTER - close polygon, go to next zone",
    "U - undo last point",
    "R - restart current zone",
    "S - save and exit (after all zones)",
    "Q or ESC - exit without saving",
]


class ZoneCalibrator:
    def __init__(self, frame, output_path: str):
        self.base_frame = frame
        self.output_path = Path(output_path)

        self.current_zone_idx = 0
        self.current_points: list[tuple[int, int]] = []
        self.zones: dict[str, list[list[int]]] = {}

    @property
    def current_zone_name(self) -> str | None:
        if self.current_zone_idx >= len(ZONES_TO_CAPTURE):
            return None
        return ZONES_TO_CAPTURE[self.current_zone_idx][0]

    @property
    def current_zone_prompt(self) -> str:
        if self.current_zone_idx >= len(ZONES_TO_CAPTURE):
            return "All zones done. Press S to save."
        return ZONES_TO_CAPTURE[self.current_zone_idx][1]

    def _on_mouse(self, event, x, y, flags, param):
        if self.current_zone_name is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._finish_current_zone()

    def _finish_current_zone(self) -> None:
        if len(self.current_points) < 3:
            print("[!] need at least 3 points for a polygon")
            return
        name = self.current_zone_name
        self.zones[name] = [list(p) for p in self.current_points]
        print(f"[+] zone '{name}' saved ({len(self.current_points)} points)")
        self.current_points = []
        self.current_zone_idx += 1

    def _render(self) -> np.ndarray:
        canvas = self.base_frame.copy()

        for name, pts in self.zones.items():
            arr = np.array(pts, dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [arr], (120, 120, 120))
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, dst=canvas)
            cv2.polylines(canvas, [arr], True, (180, 180, 180), 2)
            top_idx = int(np.argmin(arr[:, 1]))
            cv2.putText(
                canvas, name, (arr[top_idx][0], max(20, arr[top_idx][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2,
            )

        if self.current_points:
            for i, p in enumerate(self.current_points):
                cv2.circle(canvas, p, 5, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(canvas, self.current_points[i - 1], p, (0, 255, 255), 2)
            if len(self.current_points) >= 2:
                cv2.line(
                    canvas, self.current_points[-1], self.current_points[0],
                    (0, 255, 255), 1, lineType=cv2.LINE_AA,
                )

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            canvas, f"[{self.current_zone_idx + 1}/{len(ZONES_TO_CAPTURE)}] "
                    f"{self.current_zone_prompt}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        h = canvas.shape[0]
        cv2.rectangle(
            canvas,
            (0, h - 8 * len(INSTRUCTIONS) - 2),
            (120, h),
            (0, 0, 0),
            -1
        )
        for i, line in enumerate(INSTRUCTIONS):
            cv2.putText(
                canvas,
                line,
                (2, h - 8 * (len(INSTRUCTIONS) - i - 1) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.15,
                (255, 255, 255),
                1,
            )

        return canvas

    def save(self) -> bool:
        if len(self.zones) != len(ZONES_TO_CAPTURE):
            missing = [z[0] for z in ZONES_TO_CAPTURE if z[0] not in self.zones]
            print(f"[!] not all zones defined. Missing: {missing}")
            return False
        # Resolution of the calibration frame is needed at runtime so the
        # pipeline can rescale polygons when the live source has a different
        # resolution than the one used for calibration.
        h, w = self.base_frame.shape[:2]
        payload = {"_resolution": [int(w), int(h)], **self.zones}
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[OK] saved to {self.output_path} (calibration resolution {w}x{h})")
        return True

    def run(self, window_name: str = "Zone calibrator") -> bool:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)

        while True:
            cv2.imshow(window_name, self._render())
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                print("[!] exit without saving")
                cv2.destroyWindow(window_name)
                return False
            elif key in (ord("\r"), 13, 10):
                self._finish_current_zone()
            elif key == ord("u"):
                if self.current_points:
                    self.current_points.pop()
            elif key == ord("r"):
                self.current_points = []
            elif key == ord("s"):
                if self.save():
                    cv2.destroyWindow(window_name)
                    return True


def grab_frame(source) -> np.ndarray | None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] cannot open source: {source}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] failed to read frame")
        return None
    return frame


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.zone_calibrator <video_path|camera_index> "
              "[output.json]")
        print("Examples:")
        print("  python -m app.zone_calibrator videos/stairs.mp4")
        print("  python -m app.zone_calibrator 0 zones.json")
        sys.exit(1)

    src = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "zones.json"

    try:
        src_arg = int(src)
    except ValueError:
        src_arg = src

    frame = grab_frame(src_arg)
    if frame is None:
        sys.exit(1)

    calibrator = ZoneCalibrator(frame, output)
    calibrator.run()


if __name__ == "__main__":
    main()
