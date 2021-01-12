from numpy import ndarray
import numpy as np

from src.ui.Button import Button
from src.ui.MenuButton import MenuButton
from src.ui.Window import Window
from src.util.MatrixUtils import MatrixUtils


class DropdownButton(Button):

    def __init__(self,
                 matrix_unclicked: ndarray,
                 relative_coordinates: ndarray,
                 menu_buttons: list[MenuButton],
                 app,
                 matrix_clicked: ndarray = None,
                 matrix_disabled: ndarray = None,
                 reward: int = 0):
        self.__parent_coords = None
        self.__just_reset = False
        width, height = self.__get_menu_dimensions(menu_buttons)
        background_matrix = MatrixUtils.get_blank_image_as_numpy_array((242, 242, 242), width, height)
        border_matrix = MatrixUtils.get_blank_image_as_numpy_array((160, 160, 160), width + 2, height + 2)
        background_matrix = MatrixUtils.blit_image(border_matrix, background_matrix, np.array([1, 1]))
        menu_button_pos = np.array([1, 1])
        for button in menu_buttons:
            button.relative_coordinates = menu_button_pos
            menu_button_pos = menu_button_pos + np.array([0, button.height])

        def on_click_listener(btn: Button):
            if not self.clicked:
                menu = Window(background_matrix, menu_buttons,
                              self.__parent_coords + self.relative_coordinates + np.array([0, self.height]), False,
                              True)
                app.add_window(menu)

        super().__init__(matrix_unclicked, relative_coordinates, matrix_clicked, matrix_disabled, reward,
                         on_click_listener)

    def click(self, click_coordinates: ndarray, parent_coordinates: ndarray) -> (int, bool, ndarray, ndarray):
        self.__parent_coords = parent_coordinates
        return super().click(click_coordinates, parent_coordinates)

    def __get_menu_dimensions(self, menu_buttons: list[Button]):
        width, height = 0, 0
        for button in menu_buttons:
            height = height + button.height
            if button.width > width:
                width = button.width
        return width, height
