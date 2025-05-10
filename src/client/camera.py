import threading
import time

import cv2


class Camera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()

    def start(self):
        if self.is_running:
            return

        # Open the camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera {self.camera_id}")

        self.is_running = True
        self.thread = threading.Thread(target=self._capture)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

    def _capture(self):
        while self.is_running:
            # Read a frame from the camera
            ret, frame = self.cap.read()
            if ret:
                with self.lock:  # Mutex lock to prevent race condition
                    self.current_frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:  # Mutex lock to prevent race condition
            if self.current_frame is None:
                return None
            return self.current_frame.copy()
