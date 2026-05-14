import cv2
import numpy as np
import torch
from torchreid.utils import FeatureExtractor


class OSNetReID:
    """Extract L2-normalised appearance embeddings from bbox crops via OSNet."""

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path="",
            device=self.device,
        )

    def extract_embedding(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame_bgr.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        try:
            feats = self.extractor([crop_rgb])
            emb = feats[0].cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[reid] embedding extraction failed: {e}")
            return None

        # Normalise so cosine similarity reduces to a dot product.
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb
