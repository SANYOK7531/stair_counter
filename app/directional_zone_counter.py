from datetime import datetime

import cv2
import numpy as np


class DirectionalZoneCounter:
    """
    Counts people moving ENTRY -> EXIT for one staircase.

    Increment is emitted immediately on ENTRY -> EXIT. If the same track later
    enters the shared transit zone, the increment is rolled back through a
    decrement event timestamped with the original increment moment, so that
    daily stats stay consistent across hour boundaries.
    """

    ENTRY = "entry"
    EXIT = "exit"

    def __init__(
        self,
        name: str,
        entry_polygon: list[tuple[int, int]],
        exit_polygon: list[tuple[int, int]],
        transit_polygon: list[tuple[int, int]] | None = None,
        min_frames_in_zone: int = 3,
        min_exit_coverage: float = 0.35,
    ):
        self.name = name
        self.entry_poly = np.array(entry_polygon, dtype=np.int32)
        self.exit_poly = np.array(exit_polygon, dtype=np.int32)
        self.transit_poly: np.ndarray | None = (
            np.array(transit_polygon, dtype=np.int32)
            if transit_polygon is not None
            else None
        )
        self.min_frames_in_zone = min_frames_in_zone
        self.min_exit_coverage = min_exit_coverage

        self._exit_area = float(cv2.contourArea(self.exit_poly))
        if self._exit_area <= 0:
            raise ValueError(f"zone '{name}' exit polygon has zero area")

        self._track_zone_history: dict[int, list[str]] = {}
        self._track_consecutive_frames: dict[int, dict[str, int]] = {}
        self._counted_ids: set[int] = set()
        # Counted, then walked into transit: never counted again, never
        # decremented again.
        self._cancelled_ids: set[int] = set()
        # Original increment timestamp per track, used so a later decrement
        # targets the same hourly bucket as its increment.
        self._counted_at: dict[int, datetime] = {}
        self._events: list[tuple[str, datetime]] = []

        self.total_count = 0

    def pop_events(self) -> list[tuple[str, datetime]]:
        events = self._events
        self._events = []
        return events

    def _point_in_zone(self, point: tuple[int, int]) -> str | None:
        x, y = int(point[0]), int(point[1])
        if cv2.pointPolygonTest(self.entry_poly, (x, y), False) >= 0:
            return self.ENTRY
        if cv2.pointPolygonTest(self.exit_poly, (x, y), False) >= 0:
            return self.EXIT
        return None

    def _point_in_transit(self, point: tuple[int, int]) -> bool:
        if self.transit_poly is None:
            return False
        x, y = int(point[0]), int(point[1])
        return cv2.pointPolygonTest(self.transit_poly, (x, y), False) >= 0

    def _bbox_exit_coverage(self, bbox: tuple[int, int, int, int]) -> float:
        # Render the polygon mask and the bbox mask into a small buffer the
        # size of the exit polygon's bounding rect, not the whole frame.
        x1, y1, x2, y2 = map(int, bbox)
        zx, zy, zw, zh = cv2.boundingRect(self.exit_poly)

        ix1 = max(x1, zx)
        iy1 = max(y1, zy)
        ix2 = min(x2, zx + zw)
        iy2 = min(y2, zy + zh)
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        zone_mask = np.zeros((zh, zw), dtype=np.uint8)
        shifted_poly = self.exit_poly - np.array([[zx, zy]], dtype=np.int32)
        cv2.fillPoly(zone_mask, [shifted_poly], 255)

        bbox_mask = np.zeros((zh, zw), dtype=np.uint8)
        bbox_mask[iy1 - zy:iy2 - zy, ix1 - zx:ix2 - zx] = 255

        intersection = cv2.countNonZero(cv2.bitwise_and(zone_mask, bbox_mask))
        return intersection / self._exit_area

    def update(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
    ) -> None:
        x1, y1, x2, y2 = map(int, bbox)
        foot_point = ((x1 + x2) // 2, y2)

        # Already counted, but now walked into transit: roll back.
        if (
            track_id in self._counted_ids
            and self._point_in_transit(foot_point)
        ):
            self._counted_ids.discard(track_id)
            self._cancelled_ids.add(track_id)
            self.total_count -= 1
            when = self._counted_at.pop(track_id, datetime.now())
            self._events.append(("decrement", when))
            return

        current_zone = self._point_in_zone(foot_point)

        # Tiny bbox (only a head visible in the opening) can foot-touch EXIT
        # when ENTRY and EXIT overlap; require a coverage fraction.
        if current_zone == self.EXIT:
            if self._bbox_exit_coverage(bbox) < self.min_exit_coverage:
                current_zone = None

        if track_id not in self._track_consecutive_frames:
            self._track_consecutive_frames[track_id] = {}
            self._track_zone_history[track_id] = []

        consecutive = self._track_consecutive_frames[track_id]

        if current_zone is None:
            consecutive.clear()
            return

        for z in (self.ENTRY, self.EXIT):
            if z == current_zone:
                consecutive[z] = consecutive.get(z, 0) + 1
            else:
                consecutive[z] = 0

        if consecutive[current_zone] == self.min_frames_in_zone:
            history = self._track_zone_history[track_id]
            if not history or history[-1] != current_zone:
                history.append(current_zone)
                self._check_and_count(track_id)

    def _check_and_count(self, track_id: int) -> None:
        if (
            track_id in self._counted_ids
            or track_id in self._cancelled_ids
        ):
            return

        history = self._track_zone_history[track_id]
        if len(history) < 2:
            return

        for i in range(len(history) - 1):
            if history[i] == self.ENTRY and history[i + 1] == self.EXIT:
                self.total_count += 1
                self._counted_ids.add(track_id)
                now = datetime.now()
                self._counted_at[track_id] = now
                self._events.append(("increment", now))
                return

    def cleanup(self, track_ids_to_forget: set[int]) -> None:
        for tid in track_ids_to_forget:
            self._track_zone_history.pop(tid, None)
            self._track_consecutive_frames.pop(tid, None)
            self._counted_ids.discard(tid)
            self._cancelled_ids.discard(tid)
            self._counted_at.pop(tid, None)
