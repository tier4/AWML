import cv2
import numpy as np
import transforms3d


def intensity_to_rainbow_color(value):
    h = value * 5.0 + 1.0
    i = h // 1  # floor(h)
    f = h - i
    if i % 2 == 0:
        f = 1.0 - f

    n = 1 - f

    if i <= 1:
        r, g, b = n, 0, 1.0
    elif i == 2:
        r, g, b = 0.0, n, 1.0
    elif i == 3:
        r, g, b = 0.0, 1.0, n
    elif i == 4:
        r, g, b = n, 1.0, 0.0
    elif i >= 5:
        r, g, b = 1.0, n, 0

    return int(255 * r), int(255 * g), int(255 * b)


def bounded_gaussian(center, min_value, max_value, scale):
    """Generates a value from a Gaussian distribution centered at `center`,
    bounded by `min_value` and `max_value`.

    Args:
        center (float): Mean of the Gaussian distribution.
        min_value (float): Minimum value of the output.
        max_value (float): Maximum value of the output.
        scale (float): Standard deviation of the Gaussian distribution.

    Returns:
        float: Bounded random value.
    """
    while True:
        value = np.random.normal(loc=center, scale=scale)
        if min_value <= value <= max_value:
            return value


def random_unit_sphere():
    """This method generates a single random point in the unit sphere"""

    u1 = np.random.rand()
    u2 = np.random.rand()

    z = 2 * u1 - 1.0
    x = np.sqrt(1 - z * z) * np.cos(2 * np.pi * u2)
    y = np.sqrt(1 - z * z) * np.sin(2 * np.pi * u2)

    return np.array([x, y, z])


def random_rotation_matrix(angle):
    """This method generates a rotion matrix to a coordinate system that is the result of rotating the original system by angle radians.
    The rotation axis is also random an non contrained

    angle: the angle in radians to rotate the coordinate system
    """
    x, y, z = random_unit_sphere()
    w = np.sqrt(0.5 * (1.0 + np.cos(angle)))
    factor = np.sqrt(0.5 * (1.0 - np.cos(angle)))
    return transforms3d.quaternions.quat2mat((w, factor * x, factor * y, factor * z))


def random_translation(radius):
    return radius * random_unit_sphere()


def alter_calibration(
    camera_to_lidar_pose,
    min_augmentation_angle,
    max_augmentation_angle,
    min_augmentation_radius,
    max_augmentation_radius,
):
    # Generate random rotation and translation using bounded Gaussian distribution
    noise_angle = bounded_gaussian(
        center=min_augmentation_angle,
        min_value=min_augmentation_angle,
        max_value=max_augmentation_angle,
        scale=(max_augmentation_angle - min_augmentation_angle) / 1.5,
    )
    noise_radius = bounded_gaussian(
        center=min_augmentation_radius,
        min_value=min_augmentation_radius,
        max_value=max_augmentation_radius,
        scale=(max_augmentation_radius - min_augmentation_radius) / 1.5,
    )
    # random sign
    if np.random.rand() > 0.5:
        noise_angle = -noise_angle
    if np.random.rand() > 0.5:
        noise_radius = -noise_radius

    noise_transform = np.eye(4)
    noise_transform[0:3, 0:3] = random_rotation_matrix(np.pi * noise_angle / 180.0)
    noise_transform[0:3, 3] = random_translation(noise_radius)
    non_calibrated_transform = camera_to_lidar_pose @ noise_transform
    return non_calibrated_transform
