from unittest import TestCase

import numpy as np

from gym_jadx.envs.jadx_env import JadxEnv


class TestJadxEnv(TestCase):
    def setUp(self):
        self.env = JadxEnv()

    def test_env_size(self):
        self.assertEqual(self.env.width, 400)
        self.assertEqual(self.env.height, 268)

    def test_step_click_button(self):
        # Click on a button
        point = np.array([54, 14])
        _, reward, done, _ = self.env.step(point)
        self.assertEqual(reward, 2)
        self.assertFalse(done)

    def test_step_clicked_button_reward(self):
        # Click on a button twice
        _, reward, _, _ = self.env.step(np.array([54, 14]))
        self.assertEqual(reward, 2)
        _, reward, _, _ = self.env.step(np.array([54, 14]))
        self.assertEqual(reward, 0)

    def test_get_progress(self):
        result = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0])
        # Click on a small button
        self.env.step(np.array([25, 25]))
        self.assertTrue(np.array_equal(self.env.get_progress(), result))

    def test_reset(self):
        initial_frame = self.env.frame_buffer.copy()
        # Open preferences window
        self.env.step(np.array([7, 14]))
        self.env.step(np.array([19, 94]))
        result_before_reset = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.array_equal(self.env.get_progress(), result_before_reset))
        observation = self.env.reset()
        self.assertTrue(np.array_equal(self.env.get_progress(), np.zeros(67)))
        self.assertTrue(np.array_equal(initial_frame, observation))

    def test_done(self):
        # Click on close button
        _, _, done, _ = self.env.step(np.array([390, 4]))
        self.assertTrue(done)

    def test_modal_window(self):
        initial_frame = self.env.frame_buffer.copy()
        # Open dropdown menu
        self.env.step(np.array([88, 15]))
        # Open über window
        observation_uber_open, _, _, _ = self.env.step(np.array([91, 23]))
        # Click on a button from the main window while the über window is open
        observation_click_elsewhere, reward, _, _ = self.env.step(np.array([7, 14]))
        self.assertTrue(np.array_equal(observation_uber_open, observation_click_elsewhere))
        self.assertEqual(reward, 0)
        # Close the über window
        observation_closed, reward, _, _ = self.env.step(np.array([358, 128]))
        self.assertTrue(np.array_equal(observation_closed, initial_frame))
        self.assertEqual(reward, 2)
        # Click on the same button from the main window
        observation_click_elsewhere, reward, _, _ = self.env.step(np.array([7, 14]))
        self.assertEqual(reward, 2)
