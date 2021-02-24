from gym_jadx.ui.Button import Button
from gym_jadx.ui.Checkbox import Checkbox
from gym_jadx.ui.Drawable import Drawable
from gym_jadx.ui.DropdownButton import DropdownButton
from gym_jadx.ui.MenuButton import MenuButton
from gym_jadx.ui.Window import Window
import numpy as np
from numpy import ndarray
import cv2
import gym

from gym_jadx.util.MatrixUtils import MatrixUtils


class JadxEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__all_buttons = []
        main_window = self.__init_components()
        self.__windows = [main_window]
        self.__frame_buffer = main_window.current_matrix.copy()
        self.__width = main_window.width
        self.__height = main_window.height
        self.__should_re_stack = False
        self.__done = False
        self.__windows_to_be_removed = []

    @property
    def frame_buffer(self):
        return self.__frame_buffer

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    def reset(self) -> ndarray:
        self.__all_buttons = []
        main_window = self.__init_components()
        self.__windows = [main_window]
        self.__frame_buffer = main_window.current_matrix.copy()
        self.__should_re_stack = False
        self.__done = False
        return self.__frame_buffer

    def close(self):
        pass

    def render(self, mode='human'):
        cv2.imshow('App', np.transpose(self.__frame_buffer, (1, 0, 2)))
        cv2.waitKey(0)

    def get_progress(self) -> ndarray:
        """
        Returns a vector containing the value 1 for each clicked Button and the
        value 0 for each unclicked Button.

        :return: Numpy Array of Button states
        """
        progress_vector = []
        for button in self.__all_buttons:
            if button.reward_given:
                progress_vector.append(1)
            else:
                progress_vector.append(0)
        return np.array(progress_vector)

    def step(self, action: ndarray) -> (ndarray, int, bool):
        reward = 0
        number_of_windows_to_be_removed = 0

        # Click on windows starting with the topmost window down to the window at the bottom
        # Continue clicking only while the current window was not clicked AND
        # the current window is not modal
        for i in range(-1, -len(self.__windows) - 1, -1):
            index = i + number_of_windows_to_be_removed
            reward, window_includes_point, clicked_child_component_matrix, clicked_child_component_coords = \
                self.__windows[index].click(action)
            if window_includes_point:
                if clicked_child_component_matrix is not None:
                    # Blit the changed component directly on the frame buffer, only if the clicked component is
                    # in the topmost window
                    if not self.__should_re_stack:
                        MatrixUtils.blit_image_inplace(self.__frame_buffer,
                                                       clicked_child_component_matrix,
                                                       clicked_child_component_coords[0],
                                                       clicked_child_component_coords[1])
                break
            else:
                if self.__windows[index].modal:
                    break
                elif self.__windows[index].auto_close:
                    # Save the removed window for future reference and continue with the loop
                    removed_window = self.__remove_window()
                    self.__windows_to_be_removed.append(removed_window)
                    number_of_windows_to_be_removed += 1

        # Redraw all windows by stacking them up from the bottom to the top, if needed
        if self.__should_re_stack:
            self.__frame_buffer = self.__stack_windows()

        # Reset internal state
        self.__should_re_stack = False
        self.__windows_to_be_removed.clear()
        return self.__frame_buffer, reward, self.__done, None

    def __add_window(self, window: Window):
        self.__windows.append(window)
        MatrixUtils.blit_image_inplace(self.__frame_buffer,
                                       window.current_matrix,
                                       window.relative_coordinates[0], window.relative_coordinates[1])

    def __remove_window(self) -> Window:
        removed = self.__windows.pop()
        removed.reset()
        self.__windows[-1].reset()
        self.__should_re_stack = True
        return removed

    def __change_visibility(self, drawable: Drawable, value: bool):
        drawable.visible = value
        self.__windows[-1].draw_self()
        self.__should_re_stack = True

    def __is_window_going_to_be_removed(self, window: Window) -> bool:
        for win in self.__windows_to_be_removed:
            if win == window:
                return True
        return False

    def __stack_windows(self) -> ndarray:
        final = self.__windows[0].current_matrix.copy()
        for k in range(1, len(self.__windows)):
            window_coords = self.__windows[k].relative_coordinates
            MatrixUtils.blit_image_inplace(final, self.__windows[k].current_matrix,
                                           window_coords[0], window_coords[1])
        return final

    def __init_components(self) -> Window:
        # Load drawable images
        # --------------------------------------------------------------------------------------------------
        main_window_array = MatrixUtils.get_numpy_array_of_image('main_window.png')

        close_button_large_array = MatrixUtils.get_numpy_array_of_image('close_window_button_large_unclicked.png')

        uber_window_array = MatrixUtils.get_numpy_array_of_image('window_über.png')

        close_uber_window_button_array = MatrixUtils.get_numpy_array_of_image('close_über_window_button.png')
        close_button_array = MatrixUtils.get_numpy_array_of_image('close_window_button_unclicked.png')

        preferences_window_array = MatrixUtils.get_numpy_array_of_image('window_preferences.png')
        preferences_window_abbrechen_button_array = MatrixUtils.get_numpy_array_of_image('button_abbrechen_unclicked.png')
        preferences_window_speichern_button_array = MatrixUtils.get_numpy_array_of_image('button_speichern_unclicked.png')
        preferences_window_zuruecksetzen_unclicked_array = MatrixUtils.get_numpy_array_of_image('button_zuruecksetzen_unclicked.png')
        preferences_window_zuruecksetzen_clicked_array = MatrixUtils.get_numpy_array_of_image('button_zuruecksetzen_clicked.png')
        preferences_window_clipboard_unclicked_array = MatrixUtils.get_numpy_array_of_image('button_clipboard_unclicked.png')
        preferences_window_clipboard_clicked_array = MatrixUtils.get_numpy_array_of_image('button_clipboard_clicked.png')
        preferences_window_aendern_unclicked_array = MatrixUtils.get_numpy_array_of_image('button_aendern_unclicked.png')
        preferences_window_aendern_clicked_array = MatrixUtils.get_numpy_array_of_image('button_aendern_clicked.png')
        preferences_window_bearbeiten_unclicked_array = MatrixUtils.get_numpy_array_of_image('button_bearbeiten_unclicked.png')
        preferences_window_bearbeiten_clicked_array = MatrixUtils.get_numpy_array_of_image('button_bearbeiten_clicked.png')
        preferences_window_close_array = MatrixUtils.get_numpy_array_of_image('close_pref_button.png')

        dropdown_datei_unclicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_datei_unclicked.png')
        dropdown_datei_clicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_datei_clicked.png')
        dropdown_anzeigen_unclicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_anzeigen_unclicked.png')
        dropdown_anzeigen_clicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_anzeigen.png')
        dropdown_hilfe_unclicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_hilfe_unclicked.png')
        dropdown_hilfe_clicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_hilfe_clicked.png')
        dropdown_navigation_unclicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_navigation_unclicked.png')
        dropdown_navigation_clicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_navigation_clicked.png')
        dropdown_tools_unclicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_tools_unclicked.png')
        dropdown_tools_clicked_array = MatrixUtils.get_numpy_array_of_image('drpdwn_tools_clicked.png')

        small_button_arrays = []
        for i in range(1, 14):
            small_button_arrays.append(MatrixUtils.get_numpy_array_of_image('small_button_' + str(i) + '_unclicked.png'))
            small_button_arrays.append(MatrixUtils.get_numpy_array_of_image('small_button_' + str(i) + '_clicked.png'))

        menu_button_uber_unclicked_array = MatrixUtils.get_numpy_array_of_image('menu_über_unclicked.png')

        menu_button_other_unclicked_array = MatrixUtils.get_numpy_array_of_image('menu_button_1.png')
        menu_button_other_clicked_array = MatrixUtils.get_numpy_array_of_image('menu_button_1_clicked.png')

        menu_button_preferences_unclicked_array = MatrixUtils.get_numpy_array_of_image('menu_button_preferences_unclicked.png')

        # --------------------------------------------------------------------------------------------------

        # Define click listeners
        # --------------------------------------------------------------------------------------------------
        def open_uber(_):
            # Close dropdown menu first
            self.__remove_window()
            self.__add_window(uber_window)

        def open_preferences(_):
            # Close dropdown menu first
            self.__remove_window()
            self.__add_window(preferences_window)

        def open_dropdown_menu(btn: DropdownButton):
            # Button is clicked while its dropdown menu was active -> Make button unclicked instead of opening the menu
            if self.__is_window_going_to_be_removed(btn.menu):
                btn.clicked = False
            else:
                btn.menu = Window(btn.background_matrix, btn.menu_buttons,
                                  btn.parent_coords + btn.relative_coordinates + np.array([0, btn.height]),
                                  False,
                                  True)
                self.__add_window(btn.menu)

        """
        def hide_button(btn: Button):
            self.change_visibility(btn, not btn.visible)
        """

        def close_window(_):
            self.__remove_window()

        def close_application(_):
            self.__done = True

        # --------------------------------------------------------------------------------------------------

        # Initialize components
        # --------------------------------------------------------------------------------------------------

        # Init dropdown buttons in main window
        # --------------------------------------------------------------------------------------------------
        next_pos = 0
        main_window_children = []

        app_close_button = Button(close_button_large_array, np.array([380, 0]), reward=2,
                                  on_click_listener=close_application)
        self.__all_buttons.append(app_close_button)
        main_window_children.append(app_close_button)

        dropdown_button_datei_children = []
        for i in range(0, 10):
            button = MenuButton(menu_button_other_unclicked_array, menu_button_other_clicked_array, reward=2)
            dropdown_button_datei_children.append(button)
            self.__all_buttons.append(button)

        preferences_button = MenuButton(menu_button_preferences_unclicked_array,
                                        reward=2,
                                        on_click_listener=open_preferences)
        dropdown_button_datei_children.append(preferences_button)
        self.__all_buttons.append(preferences_button)

        dropdown_button_datei = DropdownButton(dropdown_datei_unclicked_array, np.array([next_pos, 12]),
                                               dropdown_button_datei_children,
                                               dropdown_datei_clicked_array, reward=2,
                                               on_click_listener=open_dropdown_menu)
        self.__all_buttons.append(dropdown_button_datei)

        next_pos += dropdown_button_datei.width
        main_window_children.append(dropdown_button_datei)

        dropdown_button_anzeigen_children = []
        for i in range(0, 3):
            button = MenuButton(menu_button_other_unclicked_array, menu_button_other_clicked_array, reward=2)
            dropdown_button_anzeigen_children.append(button)
            self.__all_buttons.append(button)

        dropdown_button_anzeigen = DropdownButton(dropdown_anzeigen_unclicked_array,
                                                  np.array([next_pos, 12]),
                                                  dropdown_button_anzeigen_children,
                                                  dropdown_anzeigen_clicked_array, reward=2,
                                                  on_click_listener=open_dropdown_menu)
        self.__all_buttons.append(dropdown_button_anzeigen)

        next_pos += dropdown_button_anzeigen.width
        main_window_children.append(dropdown_button_anzeigen)

        dropdown_button_navigation_children = []
        for i in range(0, 4):
            button = MenuButton(menu_button_other_unclicked_array, menu_button_other_clicked_array, reward=2)
            dropdown_button_navigation_children.append(button)
            self.__all_buttons.append(button)

        dropdown_button_navigation = DropdownButton(dropdown_navigation_unclicked_array,
                                                    np.array([next_pos, 12]),
                                                    dropdown_button_navigation_children,
                                                    dropdown_navigation_clicked_array, reward=2,
                                                    on_click_listener=open_dropdown_menu)
        self.__all_buttons.append(dropdown_button_navigation)

        next_pos += dropdown_button_navigation.width
        main_window_children.append(dropdown_button_navigation)

        dropdown_button_tools_children = []
        for i in range(0, 2):
            button = MenuButton(menu_button_other_unclicked_array, menu_button_other_clicked_array, reward=2)
            dropdown_button_tools_children.append(button)
            self.__all_buttons.append(button)

        dropdown_button_tools = DropdownButton(dropdown_tools_unclicked_array,
                                               np.array([next_pos, 12]),
                                               dropdown_button_tools_children,
                                               dropdown_tools_clicked_array, reward=2,
                                               on_click_listener=open_dropdown_menu)
        self.__all_buttons.append(dropdown_button_tools)

        next_pos += dropdown_button_tools.width
        main_window_children.append(dropdown_button_tools)

        menu_button_uber = MenuButton(menu_button_uber_unclicked_array, reward=2,
                                      on_click_listener=open_uber)
        self.__all_buttons.append(menu_button_uber)

        dropdown_button_hilfe = DropdownButton(dropdown_hilfe_unclicked_array,
                                               np.array([next_pos, 12]),
                                               [menu_button_uber],
                                               dropdown_hilfe_clicked_array, reward=2,
                                               on_click_listener=open_dropdown_menu)
        self.__all_buttons.append(dropdown_button_hilfe)

        main_window_children.append(dropdown_button_hilfe)
        # --------------------------------------------------------------------------------------------------

        # Init small buttons in main window
        # --------------------------------------------------------------------------------------------------
        k = 0
        for i in range(1, 86, 7):
            small_button = Button(small_button_arrays[k],
                                  np.array([i, 22]),
                                  small_button_arrays[k + 1],
                                  reward=2,
                                  resettable=False)
            self.__all_buttons.append(small_button)
            main_window_children.append(small_button)
            k += 2

        # --------------------------------------------------------------------------------------------------

        # Init preferences window
        # --------------------------------------------------------------------------------------------------
        preferences_window_children = []
        preferences_window_checkbox_coords = [[136, 25], [169, 102], [169, 113], [169, 125], [155, 145], [148, 206],
                                              [148, 218], [148, 230], [312, 64], [312, 82], [312, 100], [312, 118],
                                              [312, 136], [312, 154], [312, 172], [312, 190], [312, 208], [312, 226]]

        for coord in preferences_window_checkbox_coords:
            preferences_window_checkbox = Checkbox(np.array(coord), reward=2)
            preferences_window_children.append(preferences_window_checkbox)
            self.__all_buttons.append(preferences_window_checkbox)

        preferences_window_bearbeiten_button = Button(preferences_window_bearbeiten_unclicked_array,
                                                      np.array([310, 45]),
                                                      preferences_window_bearbeiten_clicked_array,
                                                      reward=2)
        self.__all_buttons.append(preferences_window_bearbeiten_button)
        preferences_window_children.append(preferences_window_bearbeiten_button)

        preferences_window_aendern_button = Button(preferences_window_aendern_unclicked_array,
                                                   np.array([146, 164]),
                                                   preferences_window_aendern_clicked_array,
                                                   reward=2)
        self.__all_buttons.append(preferences_window_aendern_button)
        preferences_window_children.append(preferences_window_aendern_button)

        preferences_window_abbrechen_button = Button(preferences_window_abbrechen_button_array, np.array([315, 248]),
                                                     reward=2,
                                                     on_click_listener=close_window)
        self.__all_buttons.append(preferences_window_abbrechen_button)
        preferences_window_children.append(preferences_window_abbrechen_button)

        preferences_window_speichern_button = Button(preferences_window_speichern_button_array, np.array([282, 248]),
                                                     reward=2,
                                                     on_click_listener=close_window)
        self.__all_buttons.append(preferences_window_speichern_button)
        preferences_window_children.append(preferences_window_speichern_button)

        preferences_window_zuruecksetzen_button = Button(preferences_window_zuruecksetzen_unclicked_array,
                                                         np.array([4, 248]),
                                                         preferences_window_zuruecksetzen_clicked_array,
                                                         reward=2)
        self.__all_buttons.append(preferences_window_zuruecksetzen_button)
        preferences_window_children.append(preferences_window_zuruecksetzen_button)

        preferences_window_clipboard_button = Button(preferences_window_clipboard_unclicked_array,
                                                     np.array([38, 248]),
                                                     preferences_window_clipboard_clicked_array,
                                                     reward=2)
        self.__all_buttons.append(preferences_window_clipboard_button)
        preferences_window_children.append(preferences_window_clipboard_button)

        close_preferences_button = Button(preferences_window_close_array, np.array([335, 1]), reward=2,
                                          on_click_listener=close_window)
        self.__all_buttons.append(close_preferences_button)
        preferences_window_children.append(close_preferences_button)

        preferences_window = Window(preferences_window_array, preferences_window_children, np.array([2, 2]))
        # --------------------------------------------------------------------------------------------------

        # Init über window
        # --------------------------------------------------------------------------------------------------
        close_uber_button_1 = Button(close_button_array, np.array([80, 1]), reward=2, on_click_listener=close_window)
        self.__all_buttons.append(close_uber_button_1)
        close_uber_button_2 = Button(close_uber_window_button_array, np.array([1, 87]), reward=2,
                                     on_click_listener=close_window)
        self.__all_buttons.append(close_uber_button_2)
        uber_window = Window(uber_window_array, [close_uber_button_1, close_uber_button_2], np.array([273, 123]))

        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------
        # Return main window
        return Window(main_window_array, main_window_children, np.array([0, 0]))
