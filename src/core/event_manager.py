import threading
from collections import defaultdict

class EventManager:
    def __init__(self):
        self.listeners = defaultdict(list)
        self.lock = threading.Lock()

    def subscribe(self, event_type, listener):
        with self.lock:
            self.listeners[event_type].append(listener)

    def unsubscribe(self, event_type, listener):
        with self.lock:
            if listener in self.listeners[event_type]:
                self.listeners[event_type].remove(listener)

    def publish(self, event_type, data=None):
        with self.lock:
            listeners = list(self.listeners[event_type])  # snapshot to avoid concurrency issues

        for listener in listeners:
            try:
                listener(data)
            except Exception as e:
                print(f"Error in event listener for {event_type}: {e}")

