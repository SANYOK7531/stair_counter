from ultralytics import YOLO


class PersonTracker:
    def __init__(
        self,
        model_path: str,
        tracker_config: str,
        conf: float,
        iou: float,
        person_class_id: int,
        imgsz: int = 640,
        augment: bool = False,
    ):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.classes = [person_class_id]
        self.imgsz = imgsz
        self.augment = augment

    def track_frame(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            augment=self.augment,
            verbose=False,
        )
        return results[0]
