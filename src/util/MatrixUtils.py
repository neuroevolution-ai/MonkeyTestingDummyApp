import numpy as np
from PIL import Image
from numpy import ndarray


class MatrixUtils:
    """
    This class contains static utility functions for matrix computations
    """
    @staticmethod
    def __blit_single_channel(dest: ndarray, src: ndarray, loc: ndarray) -> ndarray:
        d = np.array(dest, copy=True)
        target = d[tuple(slice(i, None) for i in loc)]
        s = src[tuple(slice(i) for i in target.shape)]
        target[tuple(slice(None, i) for i in src.shape)] = s
        return d

    @staticmethod
    def __merge_channels(red: ndarray, green: ndarray, blue: ndarray) -> ndarray:
        return np.dstack((red, green, blue))

    @staticmethod
    def blit_image(dest: ndarray, src: ndarray, loc: ndarray) -> ndarray:
        blitted_red = MatrixUtils.__blit_single_channel(dest[:, :, 0], src[:, :, 0], loc)
        blitted_green = MatrixUtils.__blit_single_channel(dest[:, :, 1], src[:, :, 1], loc)
        blitted_blue = MatrixUtils.__blit_single_channel(dest[:, :, 2], src[:, :, 2], loc)
        return MatrixUtils.__merge_channels(blitted_red, blitted_green, blitted_blue)

    @staticmethod
    def includes_point(click_coordinates: ndarray, absolute_coordinates: ndarray, width: int, height: int) -> bool:
        if absolute_coordinates[0] > click_coordinates[0] or absolute_coordinates[1] > click_coordinates[1]:
            return False
        if (absolute_coordinates[0] + width) < click_coordinates[0] or (absolute_coordinates[1] + height) < click_coordinates[1]:
            return False
        return True

    @staticmethod
    def get_numpy_array_of_image(path: str) -> ndarray:
        pic = Image.open(path).convert('RGB')
        return np.transpose(np.array(pic), (1, 0, 2))

    @staticmethod
    def get_blank_image_as_numpy_array(color: tuple[int, int, int], width: int, height: int) -> ndarray:
        red = np.full((width, height), color[0])
        green = np.full((width, height), color[1])
        blue = np.full((width, height), color[2])
        return MatrixUtils.__merge_channels(red, green, blue)
