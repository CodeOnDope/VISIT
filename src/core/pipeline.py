 
import threading
import time

class DetectionPipeline:
    def __init__(self, camera_manager, detector_manager, event_manager, max_fps=30):
        self.camera_manager = camera_manager
        self.detector_manager = detector_manager
        self.event_manager = event_manager
        self.running = False
        self.thread = None
        self.max_fps = max_fps

    def start(self):
        self.camera_manager.start()
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        frame_time = 1.0 / self.max_fps
        while self.running:
            start_time = time.time()
            try:
                frame = self.camera_manager.read_frame()
                detections = self.detector_manager.process_frame(frame)
                self.event_manager.publish('frame_processed', {
                    'frame': frame,
                    'detections': detections
                })
            except Exception as e:
                self.event_manager.publish('error', {'error': str(e)})
            elapsed = time.time() - start_time
            time_to_sleep = frame_time - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.camera_manager.stop()
