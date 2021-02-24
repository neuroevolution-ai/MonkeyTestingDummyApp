from pathlib import Path


class PathUtils:

    @staticmethod
    def get_image_path(filename: str, resized=True):
        """
        Returns a pathlib.Path object of the specified image file

        :param filename: Name of the image file
        :param resized: True if the image should be resized, False else
        :return: pathlib.Path object of the image
        """
        if resized:
            return PathUtils.__get_resized_drawables_path() / filename
        else:
            return PathUtils.__get_original_size_drawables_path() / filename

    @staticmethod
    def __get_project_root() -> Path:
        return Path(__file__).parent.parent.parent

    @staticmethod
    def __get_resized_drawables_path() -> Path:
        return PathUtils.__get_project_root() / 'resources' / 'drawables'

    @staticmethod
    def __get_original_size_drawables_path() -> Path:
        return PathUtils.__get_resized_drawables_path() / 'size_original'
