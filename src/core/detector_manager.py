 
class DetectorManager:
    def __init__(self):
        self.detectors = []

    def add_detector(self, detector):
        self.detectors.append(detector)

    def remove_detector(self, detector):
        if detector in self.detectors:
            self.detectors.remove(detector)

    def process_frame(self, frame):
        all_results = []
        for detector in self.detectors:
            if detector.active:
                results = detector.process(frame)
                all_results.append({
                    'detector': detector.__class__.__name__,
                    'results': results
                })
        return all_results
