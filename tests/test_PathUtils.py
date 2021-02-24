from unittest import TestCase

from gym_jadx.util.PathUtils import PathUtils


class TestPathUtils(TestCase):
    def test_get_image_resized(self):
        self.assertTrue(PathUtils.get_image_path('chk_clicked.png').is_file())

    def test_get_image_original(self):
        self.assertTrue(PathUtils.get_image_path('chk_clicked.png', resized=False).is_file())
