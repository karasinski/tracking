from __future__ import division, print_function
import pygame
from pygame.locals import *
import time
import math
from array import array


def draw_drawing_region(width):
    pygame.draw.line(surface, BLACK,
                     [canvas_width / 2 - width, 0],
                     [canvas_width / 2 - width, canvas_height], 2)
    pygame.draw.line(surface, BLACK,
                     [canvas_width / 2 + width, 0],
                     [canvas_width / 2 + width, canvas_height], 2)


def draw_cross():
    _, y = pygame.mouse.get_pos()

    pygame.draw.line(surface, BLACK,
                     [canvas_width / 2, y - 5],
                     [canvas_width / 2, y + 5], 1)
    pygame.draw.line(surface, BLACK,
                     [canvas_width / 2 - 5, y],
                     [canvas_width / 2 + 5, y], 1)


def draw_sin(a, f, x, o=0):
    t = speed * time.time()
    return a * math.sin(f * ((x / canvas_width) * (2 * math.pi) + t + o))


def draw_path(path):
    # Update sine wave
    for x in range(0, canvas_width):
        y = canvas_height / 2  # center the line
        y += draw_sin(amplitudeA, frequencyA, x, offsetA)
        y += draw_sin(amplitudeB, frequencyB, x, offsetB)
        surface.set_at((x, int(y)), RED)

        if x == int(canvas_width / 2):
            path.append(int(y))


def update_trail(trail):
    _, y = pygame.mouse.get_pos()
    trail.append(y)
    return trail


def draw_trail():
    num = int(round(canvas_width / (2 * frequencyA * speed)))
    recent_trail = trail[-num:-1]
    for i in range(1, num):
        try:
            x = canvas_width / 2 - frequencyA * i * speed
            # If we've gone off the screen stop drawing
            if x < 0:
                print(i, test_num)
                break
            y = recent_trail[-i]
            surface.set_at((int(x), y), BLUE)
        except IndexError:
            pass
        # print(y)


# Some config width height settings
canvas_height = 500
canvas_width = int(canvas_height * 1.61803398875)

# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE  = (  0,   0, 255)
GREEN = (  0, 255,   0)
RED   = (255,   0,   0)

background_color = WHITE

pygame.init()
# Set the window title
pygame.display.set_caption("Tracking Experiment")

# Make a screen to see
screen = pygame.display.set_mode((canvas_width, canvas_height))
screen.fill(background_color)

# Make a surface to draw on
surface = pygame.Surface((canvas_width, canvas_height))
surface.fill(background_color)

# Disable mouse visbility
pygame.mouse.set_visible(False)

# Wave properties
frequencyA, frequencyB = 2, 7
offsetA, offsetB = 0, 17
amplitudeA, amplitudeB = canvas_height / 3, canvas_height / 6
speed = 0.2

# Other simulation properties
drawing_width = 50

# Create a trail array
trail, path = array('i'), array('i')

# Create Pygame clock object and set framerate
clock = pygame.time.Clock()
FPS = 60

# Simple main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == KEYDOWN and event.key == K_SPACE:
            running = False

    clock.tick(FPS)  # do not go faster than this framerate
    # Redraw the background
    surface.fill(background_color)

    # Draw the path
    draw_path(path)

    # Draw features
    draw_drawing_region(drawing_width)
    draw_cross()
    trail = update_trail(trail)
    draw_trail()

    # Put the surface we draw on, onto the screen
    screen.blit(surface, (0, 0))

    # Show it
    pygame.display.flip()
