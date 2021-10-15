import collections
import time

import cv2
import numpy as np


class WebcamSource:
    """
    Helper class for OpenCV VideoCapture. Can be used as an iterator.
    """

    def __init__(self, camera_id=0, width=1280, height=720, fps=30, buffer_size=1):
        self.__name = "WebcamSource"
        self.__capture = cv2.VideoCapture(camera_id)
        self.__capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.__capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.__capture.set(cv2.CAP_PROP_FPS, fps)
        self.__capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.buffer_size = buffer_size

        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.fps_deque = collections.deque(maxlen=fps)

    def __iter__(self):
        if not self.__capture.isOpened():
            raise StopIteration
        return self

    def __next__(self):
        """
        Get next frame from webcam or stop iteration when no frame can be grabbed from webcam

        :return: None
        """
        ret, frame = self.__capture.read()

        if not ret:
            raise StopIteration

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise StopIteration

        return frame

    def clear_frame_buffer(self):
        for _ in range(self.buffer_size):
            self.__capture.read()

    def __del__(self):
        self.__capture.release()
        cv2.destroyAllWindows()

    def show(self, frame, only_print=False):
        self.new_frame_time = time.time()
        self.fps_deque.append(1 / (self.new_frame_time - self.prev_frame_time))
        self.prev_frame_time = self.new_frame_time

        if only_print:
            print(f'{self.__name} - FPS: {np.mean(self.fps_deque):5.2f}')
        else:
            cv2.imshow('show_frame', frame)
            cv2.setWindowTitle("show_frame", f'{self.__name} - FPS: {np.mean(self.fps_deque):5.2f}')
