import cv2
import numpy as np

# BGR
COLOR_ENTRY = (0, 255, 0)
COLOR_EXIT = (255, 100, 0)
COLOR_TRANSIT = (180, 0, 220)
COLOR_TRACK = (255, 255, 255)
COLOR_TRACK_HOT = (0, 200, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_FPS = (0, 255, 0)


def draw_zone(
    frame,
    polygon,
    color,
    label: str | None = None,
    alpha: float = 0.18,
    visibility: float = 1.0,
):
    """
    Draw a zone polygon. visibility (0..1) fades the whole zone — fill,
    outline and label together — by compositing a fully drawn overlay
    over the frame.
    """
    if visibility <= 0.0:
        return

    pts = np.array(polygon, dtype=np.int32)

    overlay = frame.copy()
    fill_layer = overlay.copy()
    cv2.fillPoly(fill_layer, [pts], color)
    cv2.addWeighted(fill_layer, alpha, overlay, 1.0 - alpha, 0, dst=overlay)
    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

    if label:
        top_idx = int(np.argmin(pts[:, 1]))
        anchor = tuple(pts[top_idx])
        cv2.putText(
            overlay, label, (anchor[0], max(20, anchor[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    cv2.addWeighted(overlay, visibility, frame, 1.0 - visibility, 0, dst=frame)


def draw_track(
    frame,
    box,
    track_id: int | str,
    in_any_zone: bool,
    conf: float | None = None,
    show_conf: bool = True,
):
    x1, y1, x2, y2 = map(int, box)
    color = COLOR_TRACK_HOT if in_any_zone else COLOR_TRACK

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{track_id}"
    if show_conf and conf is not None:
        label += f" {conf:.2f}"

    cv2.putText(
        frame, label, (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
    )

    # Foot point — bottom-middle of the bbox; used as the zone-test point.
    px = (x1 + x2) // 2
    py = y2
    cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
    return px, py


def draw_hud(
    frame,
    counters: dict[str, int],
    today_total: int,
    active_tracks: int,
    fps: float,
    stairs_total: int | None = None,
):
    y = 35
    cv2.putText(
        frame, f"Today total: {today_total}",
        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
    )
    y += 32

    if stairs_total is not None:
        cv2.putText(
            frame, f"Stairs: {stairs_total}",
            (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2,
        )
        y += 30

    for name, value in counters.items():
        cv2.putText(
            frame, f"{name}: {value}",
            (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2,
        )
        y += 28

    cv2.putText(
        frame, f"Active tracks: {active_tracks}",
        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1,
    )
    y += 24

    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FPS, 1,
    )
