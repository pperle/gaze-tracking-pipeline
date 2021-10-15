from typing import Tuple

import cv2
import numpy as np


def get_matrices(camera_matrix: np.ndarray, distance_norm: int, center_point: np.ndarray, focal_norm: int, head_rotation_matrix: np.ndarray, image_output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate rotation, scaling and transformation matrix.

    :param camera_matrix: intrinsic camera matrix
    :param distance_norm: normalized distance of the camera
    :param center_point: position of the center in the image
    :param focal_norm: normalized focal length
    :param head_rotation_matrix: rotation of the head
    :param image_output_size: output size of the output image
    :return: rotation, scaling and transformation matrix
    """
    # normalize image
    distance = np.linalg.norm(center_point)  # actual distance between center point and original camera
    z_scale = distance_norm / distance

    cam_norm = np.array([
        [focal_norm, 0, image_output_size[0] / 2],
        [0, focal_norm, image_output_size[1] / 2],
        [0, 0, 1.0],
    ])

    scaling_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    forward = (center_point / distance).reshape(3)
    down = np.cross(forward, head_rotation_matrix[:, 0])
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)

    rotation_matrix = np.asarray([right, down, forward])
    transformation_matrix = np.dot(np.dot(cam_norm, scaling_matrix), np.dot(rotation_matrix, np.linalg.inv(camera_matrix)))

    return rotation_matrix, scaling_matrix, transformation_matrix


def equalize_hist_rgb(rgb_img: np.ndarray) -> np.ndarray:
    """
    Equalize the histogram of a RGB image.

    :param rgb_img: RGB image
    :return: equalized RGB image
    """
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)  # convert from RGB color-space to YCrCb
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])  # equalize the histogram of the Y channel
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)  # convert back to RGB color-space from YCrCb
    return equalized_img


def normalize_single_image(image: np.ndarray, head_rotation, gaze_target: np.ndarray, center_point: np.ndarray, camera_matrix: np.ndarray, is_eye: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The normalization process of a single image, creates a normalized eye image or a face image, depending on `is_eye`.

    :param image: original image
    :param head_rotation: rotation of the head
    :param gaze_target: 3D target of the gaze
    :param center_point: 3D point on the face to focus on
    :param camera_matrix: intrinsic camera matrix
    :param is_eye: if true the `distance_norm` and `image_output_size` values for the eye are used
    :return: normalized image, normalized gaze and rotation matrix
    """
    # normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 500 if is_eye else 1600  # normalized distance between eye and camera
    image_output_size = (96, 64) if is_eye else (96, 96)  # size of cropped eye image

    # compute estimated 3D positions of the landmarks
    if gaze_target is not None:
        gaze_target = gaze_target.reshape((3, 1))

    head_rotation_matrix, _ = cv2.Rodrigues(head_rotation)
    rotation_matrix, scaling_matrix, transformation_matrix = get_matrices(camera_matrix, distance_norm, center_point, focal_norm, head_rotation_matrix, image_output_size)

    img_warped = cv2.warpPerspective(image, transformation_matrix, image_output_size)  # image normalization
    img_warped = equalize_hist_rgb(img_warped)  # equalizes the histogram (normalization)

    if gaze_target is not None:
        # normalize gaze vector
        gaze_normalized = gaze_target - center_point  # gaze vector
        # For modified data normalization, scaling is not applied to gaze direction, so here is only R applied.
        gaze_normalized = np.dot(rotation_matrix, gaze_normalized)
        gaze_normalized = gaze_normalized / np.linalg.norm(gaze_normalized)
    else:
        gaze_normalized = np.zeros(3)

    return img_warped, gaze_normalized.reshape(3), rotation_matrix
