import gym

from gym_jadx.envs.jadx_env import JadxEnv
import pygame
import numpy as np
import cv2
from numpy import ndarray

env = gym.make('jadx-v0')
DISPLAY_SIZE = (1280, 720)
ENV_SIZE = (env.width, env.height)
WIDTH_RATIO = DISPLAY_SIZE[0] / ENV_SIZE[0]
HEIGHT_RATIO = DISPLAY_SIZE[1] / ENV_SIZE[1]

pygame.init()
display = pygame.display.set_mode(DISPLAY_SIZE)


def update_pygame_display(new_frame: ndarray):
    resized = cv2.resize(new_frame, dsize=(DISPLAY_SIZE[1], DISPLAY_SIZE[0]))
    pygame.surfarray.blit_array(display, resized)
    pygame.display.update()


update_pygame_display(env.frame_buffer)
i = 1
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Left click
            if event.button == 1:
                point = np.array([event.pos[0] / WIDTH_RATIO, event.pos[1] / HEIGHT_RATIO])
                observation, reward, done, _ = env.step(point)
                print('Click: ' + str(i) + ' Reward: ' + str(reward))
                i += 1
                if done:
                    print('Done: True')
                    print(env.get_progress())
                    observation = env.reset()
                update_pygame_display(observation)
            # Right click
            elif event.button == 3:
                observation = env.reset()
                update_pygame_display(observation)
