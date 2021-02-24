from gym_jadx.envs.jadx_env import JadxEnv
import pygame
import numpy as np
import random
import cv2

env = JadxEnv()
DISPLAY_SIZE = (1280, 720)
pygame.init()

display = pygame.display.set_mode(DISPLAY_SIZE)
resized = cv2.resize(env.frame_buffer, dsize=(DISPLAY_SIZE[1], DISPLAY_SIZE[0]))
pygame.surfarray.blit_array(display, resized)
pygame.display.update()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    point = np.array([random.randint(0, env.width), random.randint(0, env.height)])
    observation, _, _, _ = env.step(point)
    resized = cv2.resize(observation, dsize=(DISPLAY_SIZE[1], DISPLAY_SIZE[0]))
    pygame.surfarray.blit_array(display, resized)
    pygame.display.update()

pygame.quit()
