from numpy import ndarray
from typing import List

from gym_jadx.ui.CompositeDrawable import CompositeDrawable
from gym_jadx.ui.Drawable import Drawable
from gym_jadx.util.MatrixUtils import MatrixUtils


class Window(CompositeDrawable):
    """
    A Window is a :class:`CompositeDrawable` which can only be at the top of the component hierarchy
    """

    def __init__(self, matrix_self: ndarray, children: List[Drawable], relative_coordinates: ndarray,
                 modal: bool = True, auto_close: bool = False):
        super().__init__(matrix_self, children, relative_coordinates, matrix_self.shape[0], matrix_self.shape[1])
        self.__current_matrix = None
        self.draw_self()
        self.__modal = modal
        self.__auto_close = auto_close

    @property
    def current_matrix(self):
        return self.__current_matrix

    @property
    def modal(self):
        return self.__modal

    @property
    def auto_close(self):
        return self.__auto_close

    def click(self, click_coordinates: ndarray, parent_coordinates: ndarray = None) -> (int, bool, ndarray, ndarray):
        click_on_window = False
        if self.visible:
            if MatrixUtils.includes_point(click_coordinates, self.relative_coordinates, self.width, self.height):
                click_on_window = True
                for child in self.children:
                    reward, click_on_child, matrix, coords = child.click(click_coordinates, self.relative_coordinates)
                    if click_on_child:
                        MatrixUtils.blit_image_inplace(self.__current_matrix, matrix, coords[0], coords[1])
                        return reward, click_on_window, matrix, coords + self.relative_coordinates

        return 0, click_on_window, None, None

    def draw_self(self, parent_matrix: ndarray = None):
        temp = self.matrix_self.copy()
        for child in self.children:
            child.draw_self(temp)
        self.__current_matrix = temp

    def reset(self) -> bool:
        was_reset = False
        for child in self.children:
            res = child.reset()
            if res:
                was_reset = True
                child.draw_self(self.__current_matrix)

        return was_reset
