"""
Track-ID handoff.

BoT-SORT can lose a track during occlusions and reassign a fresh track_id
to the same person. Without compensation, the new id has no zone history
and the climb is never counted.

Strategy:
  - Per active track, keep the last two bboxes to estimate per-frame
    velocity.
  - When a track disappears, snapshot its last bbox + velocity + the frame
    it was lost on.
  - For each fresh track_id, pick the best lost track using a score that
    combines distance to a velocity-extrapolated position, age, and
    bbox-size similarity.
  - Two collision guards prevent two simultaneously visible tracks from
    sharing one effective id.
"""
from dataclasses import dataclass


@dataclass
class TrackState:
    bbox: tuple[int, int, int, int]
    prev_bbox: tuple[int, int, int, int] | None = None
    last_update_frame: int = 0


@dataclass
class LostTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    velocity: tuple[float, float]
    lost_at_frame: int
    claimed: bool = False


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _size(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, (x2 - x1) * (y2 - y1))


class TrackHandoff:
    def __init__(
        self,
        max_age_frames: int = 60,
        max_distance_px: int = 150,
        size_ratio_tolerance: float = 0.5,
        score_threshold: float = 0.3,
        max_predicted_speed_px: float = 100.0,
    ):
        self.max_age_frames = max_age_frames
        self.max_distance_px = max_distance_px
        self.size_ratio_tolerance = size_ratio_tolerance
        self.score_threshold = score_threshold
        self.max_predicted_speed_px = max_predicted_speed_px

        self._active: dict[int, TrackState] = {}
        self._lost: dict[int, LostTrack] = {}
        self._id_mapping: dict[int, int] = {}

    def _clamp_velocity(self, vx: float, vy: float) -> tuple[float, float]:
        # A single noisy detection should not project the lost track
        # arbitrarily far during extrapolation.
        speed = (vx * vx + vy * vy) ** 0.5
        if speed > self.max_predicted_speed_px:
            k = self.max_predicted_speed_px / speed
            return vx * k, vy * k
        return vx, vy

    def _predict_center(
        self,
        lost: LostTrack,
        frame_idx: int,
    ) -> tuple[float, float]:
        cx, cy = _center(lost.bbox)
        age = frame_idx - lost.lost_at_frame
        vx, vy = self._clamp_velocity(*lost.velocity)
        return cx + vx * age, cy + vy * age

    def _match_score(
        self,
        new_bbox: tuple[int, int, int, int],
        lost: LostTrack,
        frame_idx: int,
    ) -> float:
        age = frame_idx - lost.lost_at_frame
        if age > self.max_age_frames or age < 0:
            return 0.0

        px, py = self._predict_center(lost, frame_idx)
        ncx, ncy = _center(new_bbox)
        distance = ((ncx - px) ** 2 + (ncy - py) ** 2) ** 0.5
        if distance > self.max_distance_px:
            return 0.0

        new_size = _size(new_bbox)
        lost_size = _size(lost.bbox)
        size_ratio = min(new_size, lost_size) / max(new_size, lost_size)
        if size_ratio < self.size_ratio_tolerance:
            return 0.0

        distance_score = 1.0 - distance / self.max_distance_px
        age_score = 1.0 - age / self.max_age_frames
        return distance_score * age_score * size_ratio

    def _compute_velocity(
        self,
        state: TrackState,
        current_frame: int,
    ) -> tuple[float, float]:
        if state.prev_bbox is None:
            return (0.0, 0.0)
        dt = max(1, current_frame - state.last_update_frame)
        cx0, cy0 = _center(state.prev_bbox)
        cx1, cy1 = _center(state.bbox)
        return ((cx1 - cx0) / dt, (cy1 - cy0) / dt)

    def resolve(
        self,
        active_tracks: dict[int, tuple[int, int, int, int]],
        frame_idx: int,
    ) -> dict[int, int]:
        """Returns {real_track_id: effective_track_id} for this frame."""
        # 1. Move tracks that disappeared this frame into _lost with velocity.
        newly_lost_ids = set(self._active.keys()) - set(active_tracks.keys())
        for tid in newly_lost_ids:
            state = self._active.pop(tid)
            velocity = self._compute_velocity(state, frame_idx)
            self._lost[tid] = LostTrack(
                track_id=tid,
                bbox=state.bbox,
                velocity=velocity,
                lost_at_frame=frame_idx,
            )

        # 1.5. Drop stale mappings whose root collides with a currently
        # visible raw id — BoT-SORT may recycle an old id for a new person
        # while another track still maps to that root.
        active_raw = set(active_tracks.keys())
        stale_mappings = [
            mapped
            for mapped, root in self._id_mapping.items()
            if root in active_raw and mapped != root
        ]
        for mapped in stale_mappings:
            self._id_mapping.pop(mapped, None)

        # 2. For each fresh track_id pick a parent. busy_roots are roots
        # already in use by a visible track; assigning them again would
        # produce two bboxes with the same effective id on one frame.
        busy_roots = {
            self._id_mapping.get(other_tid, other_tid)
            for other_tid in active_tracks
        }
        for tid, bbox in active_tracks.items():
            if tid in self._active or tid in self._id_mapping:
                continue

            forbidden = busy_roots - {tid}

            best_parent_id = None
            best_score = self.score_threshold
            for lost_id, lost in self._lost.items():
                if lost.claimed:
                    continue
                lost_root = self._id_mapping.get(lost_id, lost_id)
                if lost_root in forbidden:
                    continue
                score = self._match_score(bbox, lost, frame_idx)
                if score > best_score:
                    best_score = score
                    best_parent_id = lost_id

            if best_parent_id is not None:
                self._lost[best_parent_id].claimed = True
                root = self._id_mapping.get(best_parent_id, best_parent_id)
                self._id_mapping[tid] = root
                busy_roots.add(root)

        # 3. Refresh active state for next-frame velocity computation.
        for tid, bbox in active_tracks.items():
            if tid in self._active:
                prev = self._active[tid]
                self._active[tid] = TrackState(
                    bbox=bbox,
                    prev_bbox=prev.bbox,
                    last_update_frame=frame_idx,
                )
            else:
                self._active[tid] = TrackState(
                    bbox=bbox,
                    prev_bbox=None,
                    last_update_frame=frame_idx,
                )

        # 4. Drop expired or claimed lost entries.
        self._lost = {
            tid: lost
            for tid, lost in self._lost.items()
            if (frame_idx - lost.lost_at_frame) <= self.max_age_frames
            and not lost.claimed
        }

        return {
            tid: self._id_mapping.get(tid, tid)
            for tid in active_tracks
        }

    def effective_id(self, track_id: int) -> int:
        return self._id_mapping.get(track_id, track_id)

    def forget(self, track_ids: set[int]) -> None:
        for tid in track_ids:
            self._active.pop(tid, None)
            self._lost.pop(tid, None)
            self._id_mapping.pop(tid, None)
