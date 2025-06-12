import cv2
import numpy as np
from core.camera import CameraManager
from core.detector_manager import DetectorManager
from core.event_manager import EventManager
from core.pipeline import DetectionPipeline
from detectors.face.face_detector import FaceDetector

def draw_detections(frame, detections):
    for detection in detections:
        for bbox in detection['results']:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detection['detector'], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def on_frame_processed(data):
    frame = data['frame']
    detections = data['detections']

    draw_detections(frame, detections)
    cv2.imshow("VISIT Core Test - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        pipeline.stop()

if __name__ == "__main__":
    camera = CameraManager()
    detector_manager = DetectorManager()
    face_detector = FaceDetector()
    detector_manager.add_detector(face_detector)
    event_manager = EventManager()
    event_manager.subscribe('frame_processed', on_frame_processed)
    event_manager.subscribe('error', lambda data: print(f"Error: {data['error']}"))

    pipeline = DetectionPipeline(camera, detector_manager, event_manager)
    pipeline.start()
