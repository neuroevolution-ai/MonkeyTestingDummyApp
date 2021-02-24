import unittest
from unittest import TestCase
import numpy as np

from gym_jadx.util.MatrixUtils import MatrixUtils


class TestMatrixUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        destination_single_channel = np.zeros((4, 4))
        cls.destination_image = np.dstack((destination_single_channel, destination_single_channel,
                                           destination_single_channel))
        source_single_channel = np.ones((2, 2))
        cls.source_image = np.dstack((source_single_channel, source_single_channel, source_single_channel))
        cls.rectangle_origin = np.array([2, 2])
        cls.rectangle_width = 4
        cls.rectangle_height = 4

    def test_blit_image_inplace(self):
        MatrixUtils.blit_image_inplace(self.destination_image, self.source_image, 1, 1)
        result_single_channel = np.array([[0, 0, 0, 0],
                                          [0, 1, 1, 0],
                                          [0, 1, 1, 0],
                                          [0, 0, 0, 0]])
        result = np.dstack((result_single_channel, result_single_channel, result_single_channel))
        self.assertTrue(np.array_equal(result, self.destination_image))

    def test_includes_point_click_inside(self):
        click = np.array([4, 5])
        self.assertTrue(MatrixUtils.includes_point(click,
                                                   self.rectangle_origin,
                                                   self.rectangle_width,
                                                   self.rectangle_height))

    def test_includes_point_click_border(self):
        click = np.array([3, 6])
        self.assertTrue(MatrixUtils.includes_point(click,
                                                   self.rectangle_origin,
                                                   self.rectangle_width,
                                                   self.rectangle_height))

    def test_includes_point_click_outside(self):
        click = np.array([-2, 8])
        self.assertFalse(MatrixUtils.includes_point(click,
                                                    self.rectangle_origin,
                                                    self.rectangle_width,
                                                    self.rectangle_height))

    def test_get_blank_image_as_numpy_array_image_shape(self):
        image = MatrixUtils.get_blank_image_as_numpy_array((25, 85, 172), width=30, height=80)
        self.assertEqual(image.shape, (30, 80, 3))

    def test_get_blank_image_as_numpy_array_image_color(self):
        image = MatrixUtils.get_blank_image_as_numpy_array((25, 85, 172), width=2, height=2)
        result_red = np.array([[25, 25],
                               [25, 25]])
        result_green = np.array([[85, 85],
                                 [85, 85]])
        result_blue = np.array([[172, 172],
                                [172, 172]])
        result = np.dstack((result_red, result_green, result_blue))
        self.assertTrue(np.array_equal(image, result))

    def test_get_numpy_array_of_image_type(self):
        image = MatrixUtils.get_numpy_array_of_image('window_über.png')
        self.assertIsInstance(image, np.ndarray)

    def test_get_numpy_array_of_image_shape(self):
        image = MatrixUtils.get_numpy_array_of_image('window_über.png')
        self.assertEqual(image.shape, (95, 95, 3))


if __name__ == '__main__':
    unittest.main()
