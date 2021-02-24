from typing import Callable

from numpy import ndarray
import numpy as np

from gym_jadx.ui.Button import Button


class MenuButton(Button):

    def __init__(self,
                 matrix_unclicked: ndarray,
                 matrix_clicked: ndarray = None,
                 matrix_disabled: ndarray = None,
                 reward: int = 0,
                 on_click_listener: Callable[[Button], None] = lambda b: None,
                 resettable: bool = True):
        """
        A :class:`Button` which can be displayed on a dropdown menu when it is used as a parameter
        for :class:`DropdownButton`.

        :param matrix_unclicked: Image matrix of the button in unclicked state
        :param matrix_clicked: Image matrix of the button in clicked state
        :param matrix_disabled: Image matrix of the button in disabled state
        :param reward: The amount of reward this button generates when clicked for the first time
        :param on_click_listener: Function which is called when the button is clicked
        :param resettable: Dictates if the status of this button should be reset, when its parent Window is closed
        """
        super().__init__(matrix_unclicked, np.array([0, 0]), matrix_clicked, matrix_disabled, reward, on_click_listener,
                         resettable)

    @property
    def relative_coordinates(self):
        return self.__relative_coordinates

    @relative_coordinates.setter
    def relative_coordinates(self, value):
        self.__relative_coordinates = value
