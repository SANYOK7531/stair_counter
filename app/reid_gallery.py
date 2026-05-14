"""
Appearance gallery: stores reference embeddings of seen people and finds
the closest match by cosine similarity. Knows nothing about track_id —
the track_id <-> person_id link lives in TrackHandoff.

Notes:
  - Matching is vectorised (single matmul over the gallery).
  - Embeddings are updated via EMA so a single bad frame cannot wreck a
    reference.
  - TTL drops embeddings that have not been refreshed for a while
    (clothes change between days, so old references are useless).
"""
import numpy as np


class ReIDGallery:
    def __init__(
        self,
        similarity_threshold: float = 0.6,
        ema_alpha: float = 0.8,
        ttl_frames: int = 600,
    ):
        # Embeddings are L2-normalised, so cosine similarity == dot product.
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha
        self.ttl_frames = ttl_frames

        self._next_person_id = 1
        self._embeddings: dict[int, np.ndarray] = {}
        self._last_update_frame: dict[int, int] = {}

    def match(
        self,
        embedding: np.ndarray,
    ) -> tuple[int | None, float]:
        """Returns (person_id, score) if score >= threshold, else (None, score)."""
        if embedding is None or not self._embeddings:
            return None, -1.0

        ids = list(self._embeddings.keys())
        matrix = np.stack([self._embeddings[pid] for pid in ids])
        scores = matrix @ embedding

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.similarity_threshold:
            return ids[best_idx], best_score
        return None, best_score

    def register(self, embedding: np.ndarray, frame_idx: int) -> int:
        pid = self._next_person_id
        self._next_person_id += 1
        self._embeddings[pid] = embedding.astype(np.float32)
        self._last_update_frame[pid] = frame_idx
        return pid

    def update(
        self,
        person_id: int,
        embedding: np.ndarray,
        frame_idx: int,
    ) -> None:
        if person_id not in self._embeddings:
            return
        old = self._embeddings[person_id]
        new = self.ema_alpha * old + (1.0 - self.ema_alpha) * embedding
        # Re-normalise after the convex combination.
        norm = float(np.linalg.norm(new))
        if norm > 0:
            new = new / norm
        self._embeddings[person_id] = new.astype(np.float32)
        self._last_update_frame[person_id] = frame_idx

    def prune(self, current_frame: int) -> None:
        stale = [
            pid for pid, last in self._last_update_frame.items()
            if current_frame - last > self.ttl_frames
        ]
        for pid in stale:
            self._embeddings.pop(pid, None)
            self._last_update_frame.pop(pid, None)

    @property
    def size(self) -> int:
        return len(self._embeddings)
