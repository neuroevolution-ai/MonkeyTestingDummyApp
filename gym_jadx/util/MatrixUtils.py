import numpy as np
from PIL import Image
from numpy import ndarray
from typing import Tuple

from gym_jadx.util.PathUtils import PathUtils


class MatrixUtils:
    """
    This class contains static utility functions for matrix computations
    """

    @staticmethod
    def __blit_single_channel_inplace(dest: ndarray, src: ndarray, x: int, y: int):
        dest[x:x + src.shape[0], y:y + src.shape[1]] = src

    @staticmethod
    def blit_image_inplace(dest: ndarray, src: ndarray, x: int, y: int):
        """
        Blits the source array on the destination array at a specified location inplace.
        Destination array will be changed as a result.

        :param dest: Destination array, changed with the resulting matrix after the operation
        :param src: Source array, remains unchanged
        :param x: X coordinate of the destination array on which the blit will take place
        :param y: Y coordinate of the destination array on which the blit will take place
        """
        MatrixUtils.__blit_single_channel_inplace(dest[:, :, 0], src[:, :, 0], x, y)
        MatrixUtils.__blit_single_channel_inplace(dest[:, :, 1], src[:, :, 1], x, y)
        MatrixUtils.__blit_single_channel_inplace(dest[:, :, 2], src[:, :, 2], x, y)

    @staticmethod
    def includes_point(click_coordinates: ndarray, absolute_coordinates: ndarray, width: int, height: int) -> bool:
        """
        Checks if a click has landed within the boundaries of a rectangle. A click counts as within boundaries
        if it lies on a boundary as well

        :param click_coordinates: Coordinates of the click position
        :param absolute_coordinates: Absolute coordinates of the upper left corner of the rectangle
        :param width: Width of the rectangle
        :param height: Height of the rectangle
        :return: True if the click is in the rectangle, False if not
        """
        if absolute_coordinates[0] <= click_coordinates[0] and absolute_coordinates[1] <= click_coordinates[1]:
            if (absolute_coordinates[0] + width) >= click_coordinates[0] and (absolute_coordinates[1] + height) >= \
                    click_coordinates[1]:
                return True
        return False

    @staticmethod
    def get_numpy_array_of_image(filename: str, resized: bool = True) -> ndarray:
        """
        Loads an image and converts it to a numpy array.

        :param filename: A filename (string)
        :param resized: True if the image should be resized, False else
        :return: A numpy array of the loaded image
        """
        path = PathUtils.get_image_path(filename, resized)
        pic = Image.open(path).convert('RGB')
        return np.transpose(np.array(pic), (1, 0, 2))

    @staticmethod
    def get_blank_image_as_numpy_array(color: Tuple[int, int, int], width: int, height: int) -> ndarray:
        """
        Creates a numpy array of a single color RGB image

        :param color: Desired color of the image as an RGB tuple
        :param width: Width of the image
        :param height: Height of the image
        :return: A numpy array of the desired single color image
        """
        red = np.full((width, height), color[0])
        green = np.full((width, height), color[1])
        blue = np.full((width, height), color[2])
        return np.dstack((red, green, blue))
