import cv2
import numpy as np
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path="yolov8x.pt", confidence=0.5):
        # load yolo model
        # using the yolov8x.pt model for better accuracy in player detection
        # at first launch will automatically download the model from ultralytics repository
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.color = np.random.randint(0,255, size=(100,3), dtype="int")  # different random colors for each player

        # forsce use GPU if available for faster processing:
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando device: {self.device}")



    def process_frame(self, frame):
        # recieve a video frame and return the frame with detected players
        results = self.model.track(
            frame,
            persist = True, # keep tracking the same players across frames
            classes =  [0], # only detect person class (class 0 in COCO dataset)
            conf=self.confidence, 
            verbose = False,
            device = self.device
        )

        if results[0].boxes is None:
            return frame # no players detected, return original frame

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # get bounding box coordinates
            track_id = int(box.id[0]) if box.id is not None else -1 # get tracking ID, if available or None if not available
            conf = float(box.conf[0]) # get confidence score
            color = self.color[track_id%100].tolist() if track_id >= 0 else (0,255,0) # assign a color based on the tracking ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # draw bounding box

            if track_id >= 0:
                label = f"ID:{track_id} {conf:.2f}" # create label with tracking ID and confidence
            else:
                label = f"Conf:{conf:.2f}" # if no tracking ID, just show confidence
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # put label above the bounding box
            

        return frame
    

    def get_detections(self, frame):
        # similar to process_frame but returns also raw data, useful for further analysis and statistics
        # returns: the frame, list of dict with id, bbox, confidence)

        results = self.model.track(
            frame, 
            persist = True,
            classes =  [0],
            conf=self.confidence,
            verbose = False,
            device = self.device
        )

        detections = []

        if results[0].boxes is None: #if no players detected, return original frame and empty list
            return frame, detections
        
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # get bounding box coordinates
            track_id = int(box.id[0]) if box.id is not None else -1 # get tracking ID, if available or None if not available
            conf = float(box.conf[0])  # get confidence score

            detections.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "conf": conf
            })

            color = self.color[track_id%100].tolist() if track_id >= 0 else (0,255,0) # assign a color based on the tracking ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # draw bounding box
            if track_id >= 0:
                label = f"ID:{track_id} {conf:.2f}" # create label with tracking ID and confidence
            else:
                label = f"Conf:{conf:.2f}" # if no tracking ID, just show confidence
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # put label above the bounding box

        return frame, detections