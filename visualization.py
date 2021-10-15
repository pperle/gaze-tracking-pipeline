import numpy as np
from matplotlib import pyplot as plt


class Plot3DScene:
    def __init__(self, face_model: np.ndarray, screen_width_mm: int, screen_height_mm: int, screen_height_mm_offset: int):
        self.face_model = face_model.T
        self.screen_width_mm = screen_width_mm
        self.screen_height_mm = screen_height_mm
        self.screen_height_mm_offset = screen_height_mm_offset

        self.face_scatter = None
        self.center_point = None
        self.point_on_screen = None

        self.__setup_figure()
        self.__plot_screen()

        self.fig.show()
        self.fig.canvas.draw()

        self.plot_legend = True

    def __setup_figure(self) -> None:
        self.fig = plt.figure(figsize=(20, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-100, 700)
        self.ax.set_zlim(-10, 800 - 10)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

    def __plot_screen(self) -> None:
        self.ax.plot(0, 0, 0, linestyle="", marker="o", color='b')  # webcam

        screen_x = [-self.screen_width_mm / 2, self.screen_width_mm / 2]
        screen_y = [self.screen_height_mm_offset, self.screen_height_mm + self.screen_height_mm_offset]
        self.ax.plot(
            [screen_x[0], screen_x[1], screen_x[1], screen_x[0], screen_x[0]],
            [screen_y[0], screen_y[0], screen_y[1], screen_y[1], screen_y[0]],
            [0, 0, 0, 0, 0],
            color='r'
        )

    def plot_face_landmarks(self, face_landmarks: np.ndarray) -> None:
        if self.face_scatter is None:
            self.face_scatter = self.ax.plot(face_landmarks[0, :], face_landmarks[1, :], face_landmarks[2, :], linestyle="", marker="o", color='#7f7f7f', markersize=1, label='face landmarks')[0]
        else:
            self.face_scatter.set_data_3d(face_landmarks[0, :], face_landmarks[1, :], face_landmarks[2, :])

    def plot_center_point(self, center_point: np.ndarray, gaze_vector_3d_normalized) -> None:
        point = center_point.reshape(3) + 1000 * gaze_vector_3d_normalized.reshape(3)

        if self.center_point is None:
            self.center_point = self.ax.plot([center_point[0], point[0]], [center_point[1], point[1]], [center_point[2], point[2]], color='#2ca02c', label='gaze vector')[0]
        else:
            self.center_point.set_data_3d([center_point[0], point[0]], [center_point[1], point[1]], [center_point[2], point[2]])

    def plot_point_on_screen(self, point_on_screen):
        if self.point_on_screen is None:
            self.point_on_screen = self.ax.plot(point_on_screen[0], point_on_screen[1], point_on_screen[2], linestyle="", marker="X", color='#9467bd', markersize=5, label='target on screen')[0]
        else:
            self.point_on_screen.set_data_3d(point_on_screen[0], point_on_screen[1], point_on_screen[2])

    def update_canvas(self) -> None:
        """
        Update face, gaze and point of screen in matplotlib plot

        :param rotation_vector: rotation vector from object coordinate system to the camera coordinate system
        :param translation_vector: translation vector from object coordinate system to the camera coordinate system
        :param pitchyaw: predicted pitch and yaw in radians
        """
        if self.plot_legend:
            self.plot_legend = False
            self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
