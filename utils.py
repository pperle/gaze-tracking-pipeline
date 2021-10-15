from enum import Enum
from typing import Tuple, Union

import cv2
import numpy as np
import yaml


def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """
    Get monitor dimensions from Gdk.
    from on https://github.com/NVlabs/few_shot_gaze/blob/master/demo/monitor.py
    :return: tuple of monitor width and height in mm and pixels or None
    """
    try:
        import pgi

        pgi.install_as_gi()
        import gi.repository

        gi.require_version('Gdk', '3.0')
        from gi.repository import Gdk

        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        num = default_screen.get_number()

        h_mm = default_screen.get_monitor_height_mm(num)
        w_mm = default_screen.get_monitor_width_mm(num)

        h_pixels = default_screen.get_height()
        w_pixels = default_screen.get_width()

        return (w_mm, h_mm), (w_pixels, h_pixels)

    except ModuleNotFoundError:
        return None, None


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2


class TargetOrientation(Enum):
    UP = 82
    DOWN = 84
    LEFT = 81
    RIGHT = 83


def get_camera_matrix(calibration_matrix_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera_matrix and dist_coefficients from `calibration_matrix_path`.

    :param base_path: base path of data
    :return: camera intrinsic matrix and dist_coefficients
    """
    with open(calibration_matrix_path, 'r') as file:
        calibration_matrix = yaml.safe_load(file)
    camera_matrix = np.asarray(calibration_matrix['camera_matrix']).reshape(3, 3)
    dist_coefficients = np.asarray(calibration_matrix['dist_coeff'])
    return camera_matrix, dist_coefficients


def get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, shape, results, face_model, face_model_all, landmarks_ids):
    """
    Fit `face_model` onto `face_landmarks` using `solvePnP`.

    :param camera_matrix: camera intrinsic matrix
    :param dist_coefficients: distortion coefficients
    :param shape: image shape
    :param results: output of MediaPipe FaceMesh
    :return: full face model in the camera coordinate system
    """
    height, width, _ = shape
    face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
    face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

    rvec, tvec = None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Initial fit
    for _ in range(10):
        success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy

    head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))
    return np.dot(head_rotation_matrix, face_model.T) + tvec.reshape((3, 1)), np.dot(head_rotation_matrix, face_model_all.T) + tvec.reshape((3, 1))  # 3D positions of facial landmarks


def gaze_2d_to_3d(gaze: np.ndarray) -> np.ndarray:
    """
    pitch and gaze to 3d vector

    :param gaze: pitch and gaze vector
    :return: 3d vector
    """
    x = -np.cos(gaze[0]) * np.sin(gaze[1])
    y = -np.sin(gaze[0])
    z = -np.cos(gaze[0]) * np.cos(gaze[1])
    return np.array([x, y, z])


def ray_plane_intersection(support_vector: np.ndarray, direction_vector: np.ndarray, plane_normal: np.ndarray, plane_d: np.ndarray) -> np.ndarray:
    """
    Calulate the intersection between the gaze ray and the plane that represents the monitor.

    :param support_vector: support vector of the gaze
    :param direction_vector: direction vector of the gaze
    :param plane_normal: normal of the plane
    :param plane_d: d of the plane
    :return: point in 3D where the the person is looking at on the screen
    """
    a11 = direction_vector[1]
    a12 = -direction_vector[0]
    b1 = direction_vector[1] * support_vector[0] - direction_vector[0] * support_vector[1]

    a22 = direction_vector[2]
    a23 = -direction_vector[1]
    b2 = direction_vector[2] * support_vector[1] - direction_vector[1] * support_vector[2]

    line_w = np.array([[a11, a12, 0], [0, a22, a23]])
    line_b = np.array([[b1], [b2]])

    matrix = np.insert(line_w, 2, plane_normal, axis=0)
    bias = np.insert(line_b, 2, plane_d, axis=0)

    return np.linalg.solve(matrix, bias).reshape(3)


def plane_equation(rmat: np.ndarray, tmat: np.ndarray) -> np.ndarray:
    """
    Computes the equation of x-y plane.
    The normal vector of the plane is z-axis in rotation matrix. And tmat provide on point in the plane.

    :param rmat: rotation matrix
    :param tmat: translation matrix
    :return: (a, b, c, d), where the equation of plane is ax + by + cz = d
    """

    assert type(rmat) == type(np.zeros(0)) and rmat.shape == (3, 3), "There is an error about rmat."
    assert type(tmat) == type(np.zeros(0)) and tmat.size == 3, "There is an error about tmat."

    n = rmat[:, 2]
    origin = np.reshape(tmat, (3))

    a = n[0]
    b = n[1]
    c = n[2]

    d = origin[0] * n[0] + origin[1] * n[1] + origin[2] * n[2]
    return np.array([a, b, c, d])


def get_point_on_screen(monitor_mm: Tuple[float, float], monitor_pixels: Tuple[float, float], result: np.ndarray) -> Tuple[int, int]:
    """
    Calculate point in screen in pixels.

    :param monitor_mm: dimensions of the monitor in mm
    :param monitor_pixels: dimensions of the monitor in pixels
    :param result: predicted point on the screen in mm
    :return: point in screen in pixels
    """
    result_x = result[0]
    result_x = -result_x + monitor_mm[0] / 2
    result_x = result_x * (monitor_pixels[0] / monitor_mm[0])

    result_y = result[1]
    result_y = result_y - 20  # 20 mm offset
    result_y = min(result_y, monitor_mm[1])
    result_y = result_y * (monitor_pixels[1] / monitor_mm[1])

    return tuple(np.asarray([result_x, result_y]).round().astype(int))
