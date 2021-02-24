from typing import Callable

from numpy import ndarray

from gym_jadx.ui.Button import Button
from gym_jadx.util.MatrixUtils import MatrixUtils


class Checkbox(Button):
    ARRAY_CLICKED = MatrixUtils.get_numpy_array_of_image('chk_clicked.png')
    ARRAY_UNCLICKED = MatrixUtils.get_numpy_array_of_image('chk_unclicked.png')
    ARRAY_CLICKED_DISABLED = MatrixUtils.get_numpy_array_of_image('chk_clicked_dis.png')
    ARRAY_UNCLICKED_DISABLED = MatrixUtils.get_numpy_array_of_image('chk_unclicked_dis.png')

    def __init__(self,
                 relative_coordinates: ndarray,
                 reward: int,
                 on_click_listener: Callable[[Button], None] = lambda b: None):
        """
        A :class:`Button` with checkbox appearance

        :param relative_coordinates: Coordinates of the button relative to its parent Drawable
        :param reward: The amount of reward this button generates when clicked for the first time
        :param on_click_listener: Function which is called when the button is clicked
        """
        super().__init__(Checkbox.ARRAY_UNCLICKED,
                         relative_coordinates,
                         Checkbox.ARRAY_CLICKED,
                         Checkbox.ARRAY_CLICKED_DISABLED,
                         reward,
                         on_click_listener,
                         False)
