import threading
import time
import cv2

class Camera:
    def __init__(self, camera_index=0, video_path=None, resize=None):
        self.thread = None
        self.current_frame = None
        self.last_access = None
        self.is_running = False
        self.camera = None

        if video_path is not None:
            self.camera = cv2.VideoCapture(video_path)
        else:
            self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise Exception("Could not open video source")

        if resize is not None:
            width, height = resize
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __del__(self):
        self.camera.release()

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._capture)
            self.thread.start()

    def get_frame(self):
        self.last_access = time.time()
        return self.current_frame

    def stop(self):
        self.is_running = False
        self.thread.join()
        self.thread = None

    def _capture(self):
        self.is_running = True
        self.last_access = time.time()
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                ret, encoded = cv2.imencode(".jpg", frame)
                if ret:
                    self.current_frame = encoded
                else:
                    print("Failed to encode frame")
            else:
                print("Failed to capture frame")
        print("Reading thread stopped")
        self.thread = None
        self.is_running = False