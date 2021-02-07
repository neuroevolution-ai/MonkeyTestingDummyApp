from src.Application2 import Application2
import pygame
import numpy as np
import time

env = Application2()

pygame.init()
display = pygame.display.set_mode((env.width, env.height))
pygame.surfarray.blit_array(display, env.current_matrix)
pygame.display.update()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                point = np.array([event.pos[0], event.pos[1]])
                start = time.process_time()
                result_matrix, reward, done = env.step(point)
                print('Frame Time: ' + str((time.process_time() - start) * 1000) + ' ms')
                print('Reward: ' + str(reward))
                print('Done: ' + str(done))
                if done:
                    print(env.get_progress())
                pygame.surfarray.blit_array(display, result_matrix)
                pygame.display.update()
            elif event.button == 3:
                result_matrix = env.reset()
                pygame.surfarray.blit_array(display, result_matrix)
                pygame.display.update()


pygame.quit()
print(env.get_progress())
env.render()
