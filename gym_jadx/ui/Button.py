from __future__ import annotations
from typing import Callable

from numpy import ndarray

from gym_jadx.ui.Drawable import Drawable
from gym_jadx.util.MatrixUtils import MatrixUtils


class Button(Drawable):

    def __init__(self,
                 matrix_unclicked: ndarray,
                 relative_coordinates: ndarray,
                 matrix_clicked: ndarray = None,
                 matrix_disabled: ndarray = None,
                 reward: int = 0,
                 on_click_listener: Callable[[Button], None] = lambda b: None,
                 resettable: bool = True):
        """
        A simple button with an on click listener. Each button generates a fixed amount of reward when clicked for the
        first time.
        A button which was already clicked at least once will still call its listener when clicked, however it does not
        generate any more rewards.

        :param matrix_unclicked: Image matrix of the button in unclicked state
        :param relative_coordinates: Coordinates of the button relative to its parent Drawable
        :param matrix_clicked: Image matrix of the button in clicked state
        :param matrix_disabled: Image matrix of the button in disabled state
        :param reward: The amount of reward this button generates when clicked for the first time
        :param on_click_listener: Function which is called when the button is clicked
        :param resettable: Dictates if the status of this button should be reset, when its parent Window is closed
        """

        super().__init__(relative_coordinates, matrix_unclicked.shape[0], matrix_unclicked.shape[1])
        if matrix_clicked is None:
            matrix_clicked = matrix_unclicked.copy()
        if matrix_disabled is None:
            matrix_disabled = matrix_unclicked.copy()
        self.__matrix_clicked = matrix_clicked
        self.__matrix_unclicked = matrix_unclicked
        self.__matrix_disabled = matrix_disabled
        self.__reward = reward
        self.__on_click_listener = on_click_listener
        self.__clicked = False
        self.__disabled = False
        self.__reward_given = False
        self.__resettable = resettable

    @property
    def matrix_clicked(self):
        return self.__matrix_clicked

    @property
    def matrix_unclicked(self):
        return self.__matrix_unclicked

    @property
    def matrix_disabled(self):
        return self.__matrix_disabled

    @property
    def reward(self):
        return self.__reward

    @property
    def clicked(self):
        return self.__clicked

    @clicked.setter
    def clicked(self, value):
        self.__clicked = value

    @property
    def disabled(self):
        return self.__disabled

    @property
    def reward_given(self):
        return self.__reward_given

    def _select_matrix(self) -> ndarray:
        if self.__disabled:
            return self.__matrix_disabled
        if self.__clicked:
            return self.__matrix_clicked
        return self.__matrix_unclicked

    def click(self, click_coordinates: ndarray, parent_coordinates: ndarray) -> (int, bool, ndarray, ndarray):
        if self.visible:
            if MatrixUtils.includes_point(click_coordinates, self.relative_coordinates + parent_coordinates, self.width,
                                          self.height):
                self.__clicked = not self.__clicked
                if self.__reward_given:
                    reward = 0
                else:
                    self.__reward_given = True
                    reward = self.__reward
                self.__on_click_listener(self)
                return reward, True, self._select_matrix(), self.relative_coordinates

        return 0, False, None, None

    def draw_self(self, parent_matrix: ndarray):
        if self.visible:
            return MatrixUtils.blit_image_inplace(parent_matrix,
                                                  self._select_matrix(),
                                                  self.relative_coordinates[0],
                                                  self.relative_coordinates[1])

    def reset(self):
        if self.__resettable:
            if self.__clicked:
                self.__clicked = False
                return True
        return False
