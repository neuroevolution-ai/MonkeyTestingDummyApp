from statistics import mean

import numpy as np

from gym_jadx.envs.jadx_env import JadxEnv
import random
import time

env = JadxEnv()
width = env.width - 1
height = env.height - 1


def test_with_stats(num_of_iterations: int):

    start_total = time.process_time()
    frame_times_in_ms = []
    for i in range(0, num_of_iterations):
        # Click at random
        point = np.array([random.randint(0, width), random.randint(0, height)])
        # Click on button
        # point = np.array([54, 14])
        # Click on an empty space
        # point = np.array([220, 115])
        start = time.process_time()
        _, _, _, _ = env.step(point)
        time_in_sec = time.process_time() - start
        frame_times_in_ms.append(time_in_sec * 1000)

    print('Total Time: ' + str(time.process_time() - start_total) + ' sec')
    print('Longest Frame Time: ' + str(max(frame_times_in_ms)) + ' ms')
    print('Mean Frame Time: ' + str(mean(frame_times_in_ms)) + ' ms')
    print(env.get_progress())


def test_without_stats(num_of_iterations: int):
    for i in range(0, num_of_iterations):
        point = np.array([random.randint(0, width), random.randint(0, height)])
        _, _, _, _ = env.step(point)


test_with_stats(100000)
