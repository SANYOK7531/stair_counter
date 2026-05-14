from dataclasses import dataclass


@dataclass
class AppConfig:
    use_camera: bool = True
    camera_index: int = 0
    video_path: str = "videos/stairs.mp4"

    model_path: str = "yolov10m.pt"
    tracker_config: str = "trackers/botsort_person.yaml"
    # Low YOLO confidence on purpose: BoT-SORT filters weak detections via
    # its own track_high_thresh / new_track_thresh, so we let more candidates
    # through to improve recall on partially visible people in the openings.
    conf_threshold: float = 0.2
    iou_threshold: float = 0.5
    person_class_id: int = 0
    # 640 (YOLO default) loses heads inside dark openings on FullHD frames.
    imgsz: int = 800
    # Test-time augmentation roughly doubles inference cost.
    augment: bool = False

    zones_path: str = "zones.json"
    stats_path: str = "stats.json"

    min_frames_in_zone: int = 2
    min_exit_coverage: float = 0.0
    track_ttl_frames: int = 150

    # Handoff knobs: re-attach split track IDs after occlusions.
    # max_age must stay below track_ttl_frames.
    handoff_max_age_frames: int = 180
    handoff_max_distance_px: int = 300
    handoff_size_ratio_tolerance: float = 0.5
    handoff_score_threshold: float = 0.1
    handoff_max_predicted_speed_px: float = 200.0

    show_detection_conf: bool = True
    window_name: str = "Stair counter"

    # Periodic flash of zone overlays. period_s=0 → always visible,
    # show_duration_s=0 → never visible.
    zones_show_period_s: float = 10.0
    zones_show_duration_s: float = 2.0
    zones_fade_s: float = 0.4

    # Stair-distance estimation.
    stairs_per_person: int = 20
    step_rise_m: float = 0.17
    step_run_m: float = 0.28
