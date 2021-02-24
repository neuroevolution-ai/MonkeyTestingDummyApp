from abc import ABC, abstractmethod

from numpy import ndarray


class Drawable(ABC):
    """
    The parent class of all components which can be drawn on a screen
    """
    @abstractmethod
    def __init__(self, relative_coordinates: ndarray, width: int, height: int):
        self.__relative_coordinates = relative_coordinates
        self.__width = width
        self.__height = height
        self.__visible = True

    @property
    def relative_coordinates(self):
        return self.__relative_coordinates

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def visible(self):
        return self.__visible

    @visible.setter
    def visible(self, value):
        self.__visible = value

    @abstractmethod
    def click(self, click_coordinates: ndarray, parent_coordinates: ndarray) -> (int, bool, ndarray, ndarray):
        """
        Makes a click action on this drawable and returns the results of the click action

        :param click_coordinates: Coordinates of the click
        :param parent_coordinates: Coordinates of the parent of this drawable

        :return:
            - reward - Reward gained by clicking on this drawable
            - includes_point - True if the click landed somewhere on this drawable, False else
            - child_matrix - Matrix of the clicked Non-Composite child (eg Button), returns None if the click did not land on a clickable child
            - child_coordinates - Coordinates of the clicked Non-Composite child, returns None if the click did not land on a clickable child
        """
        pass

    @abstractmethod
    def draw_self(self, parent_matrix: ndarray):
        pass

    @abstractmethod
    def reset(self) -> bool:
        pass
