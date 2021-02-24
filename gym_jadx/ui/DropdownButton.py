from numpy import ndarray
import numpy as np
from typing import List, Callable

from gym_jadx.ui.Button import Button
from gym_jadx.ui.MenuButton import MenuButton
from gym_jadx.util.MatrixUtils import MatrixUtils


class DropdownButton(Button):
    DROPDOWN_MENU_BORDER_WIDTH = 2
    DROPDOWN_MENU_BORDER_COLOR = (160, 160, 160)

    def __init__(self,
                 matrix_unclicked: ndarray,
                 relative_coordinates: ndarray,
                 menu_buttons: List[MenuButton],
                 matrix_clicked: ndarray = None,
                 matrix_disabled: ndarray = None,
                 reward: int = 0,
                 on_click_listener: Callable[[Button], None] = lambda b: None):
        """
        A :class:`Button` which opens a dropdown menu when clicked. The dropdown menu consists of :class:`MenuButton`'s

        :param matrix_unclicked: Image matrix of the button in unclicked state
        :param relative_coordinates: Coordinates of the button relative to its parent Drawable
        :param menu_buttons: A List of MenuButton's to be shown on the dropdown menu
        :param matrix_clicked: Image matrix of the button in clicked state
        :param matrix_disabled: Image matrix of the button in disabled state
        :param reward: The amount of reward this button generates when clicked for the first time
        """

        super().__init__(matrix_unclicked, relative_coordinates, matrix_clicked, matrix_disabled, reward,
                         on_click_listener)

        # Lazy init in click
        self.__parent_coords = None
        # Lazy init in click listener
        self.__menu = None

        dropdown_menu_width, dropdown_menu_height = DropdownButton.__calculate_menu_dimensions(menu_buttons)
        dropdown_menu_width += DropdownButton.DROPDOWN_MENU_BORDER_WIDTH
        dropdown_menu_height += DropdownButton.DROPDOWN_MENU_BORDER_WIDTH
        self.__background_matrix = MatrixUtils.get_blank_image_as_numpy_array(DropdownButton.DROPDOWN_MENU_BORDER_COLOR,
                                                                              dropdown_menu_width,
                                                                              dropdown_menu_height)
        self.__menu_buttons = menu_buttons
        # Set relative positions of menu buttons
        menu_button_pos = np.array([1, 1])
        for button in menu_buttons:
            button.relative_coordinates = menu_button_pos
            menu_button_pos = menu_button_pos + np.array([0, button.height])

    @property
    def menu(self):
        return self.__menu

    @menu.setter
    def menu(self, value):
        self.__menu = value

    @property
    def background_matrix(self):
        return self.__background_matrix

    @property
    def menu_buttons(self):
        return self.__menu_buttons

    @property
    def parent_coords(self):
        return self.__parent_coords

    def click(self, click_coordinates: ndarray, parent_coordinates: ndarray) -> (int, bool, ndarray, ndarray):
        # Capture the parent coordinates here, since the children don't know their parents
        self.__parent_coords = parent_coordinates
        return super().click(click_coordinates, parent_coordinates)

    @staticmethod
    def __calculate_menu_dimensions(menu_buttons: List[Button]) -> (int, int):
        width, height = 0, 0
        for button in menu_buttons:
            height = height + button.height
            if button.width > width:
                width = button.width
        return width, height
