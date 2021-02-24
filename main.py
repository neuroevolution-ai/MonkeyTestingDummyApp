from gym_jadx.envs.jadx_env import JadxEnv
import pygame
import numpy as np
import time
import cv2

env = JadxEnv()
DISPLAY_SIZE = (1280, 720)
ENV_SIZE = (env.width, env.height)
WIDTH_RATIO = DISPLAY_SIZE[0] / ENV_SIZE[0]
HEIGHT_RATIO = DISPLAY_SIZE[1] / ENV_SIZE[1]

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
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Left click
            if event.button == 1:
                point = np.array([event.pos[0] / WIDTH_RATIO, event.pos[1] / HEIGHT_RATIO])
                start = time.process_time()
                observation, reward, done, _ = env.step(point)
                print('Frame Time: ' + str((time.process_time() - start) * 1000) + ' ms')
                print('Reward: ' + str(reward))
                print('Done: ' + str(done))
                if done:
                    print(env.get_progress())
                resized = cv2.resize(observation, dsize=(DISPLAY_SIZE[1], DISPLAY_SIZE[0]))
                pygame.surfarray.blit_array(display, resized)
                pygame.display.update()
            # Right click
            elif event.button == 3:
                observation = env.reset()
                resized = cv2.resize(observation, dsize=(DISPLAY_SIZE[1], DISPLAY_SIZE[0]))
                pygame.surfarray.blit_array(display, resized)
                pygame.display.update()


pygame.quit()
print(env.get_progress())
env.render()
