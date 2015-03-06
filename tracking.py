from __future__ import division, print_function
import pygame
from pygame.locals import *
import time
from array import array
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np


# Font size definitions
rcParams['font.size'] = 11
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']

# Label settings
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 11
rcParams['legend.numpoints'] = 1


class KalmanFilter(object):
    def __init__(self, process_variance, est_measurement_variance):
        self.process_variance = process_variance
        self.est_measurement_variance = est_measurement_variance
        self.posteri_est = 0.0
        self.posteri_error_est = 1.0

    def input_latest_noisy_measurement(self, measurement):
        priori_est = self.posteri_est
        priori_error_est = self.posteri_error_est + self.process_variance

        gain = priori_error_est / (priori_error_est + self.est_measurement_variance)
        self.posteri_est = priori_est + gain * (measurement - priori_est)
        self.posteri_error_est = (1 - gain) * priori_error_est

    def get_latest_est_measurement(self):
        return self.posteri_est


class Canvas(object):
    # Define the colors we will use in RGB format
    BLACK = (0,   0,   0)
    WHITE = (255, 255, 255)
    BLUE = (0,   0, 255)
    GREEN = (0, 255,   0)
    RED = (255,   0,   0)

    # Some config width height settings
    height = 500
    width = int(height * 1.61803398875)
    background_color = WHITE
    FPS = 60

    def __init__(self, frequencyA, frequencyB,
                 offsetA, offsetB,
                 amplitudeA, amplitudeB,
                 speed,
                 drawing_width):

        # Path properties
        self.frequencyA = frequencyA
        self.frequencyB = frequencyB
        self.offsetA = offsetA
        self.offsetB = offsetB
        self.amplitudeA = amplitudeA
        self.amplitudeB = amplitudeB
        self.speed = speed

        # Other simulation properties
        self.drawing_width = drawing_width

        # Create a trail array
        self.trail, self.path = array('i'), array('i')

        # Create Pygame clock object
        self.clock = pygame.time.Clock()

        # Init pygame
        pygame.init()

        # Set the window title
        pygame.display.set_caption("Tracking Experiment")

        # Make a screen to see
        self.screen = pygame.display.set_mode((Canvas.width, Canvas.height))
        self.screen.fill(Canvas.background_color)

        # Make a surface to draw on
        self.surface = pygame.Surface((Canvas.width, Canvas.height))
        self.surface.fill(Canvas.background_color)

        # Disable mouse visbility
        pygame.mouse.set_visible(False)

    def draw_drawing_region(self):
        width = self.drawing_width
        pygame.draw.line(self.surface, Canvas.BLACK,
                         [Canvas.width / 2 - width, 0],
                         [Canvas.width / 2 - width, Canvas.height], 2)
        pygame.draw.line(self.surface, Canvas.BLACK,
                         [Canvas.width / 2 + width, 0],
                         [Canvas.width / 2 + width, Canvas.height], 2)

    def draw_cross(self):
        _, y = pygame.mouse.get_pos()

        pygame.draw.line(self.surface, Canvas.BLACK,
                         [Canvas.width / 2, y - 5],
                         [Canvas.width / 2, y + 5], 1)
        pygame.draw.line(self.surface, Canvas.BLACK,
                         [Canvas.width / 2 - 5, y],
                         [Canvas.width / 2 + 5, y], 1)

    def draw_sin(self, a, f, x, o=0):
        t = self.speed * time.time()
        return a * np.sin(f * ((x / Canvas.width) * (2 * np.pi) + t + o))

    def draw_path(self):
        # Update sine wave
        for x in range(0, Canvas.width):
            y = Canvas.height / 2  # center the line
            y += self.draw_sin(self.amplitudeA, self.frequencyA, x, self.offsetA)
            y += self.draw_sin(self.amplitudeB, self.frequencyB, x, self.offsetB)
            self.surface.set_at((x, int(y)), Canvas.RED)

            if x == int(Canvas.width / 2):
                self.path.append(int(y))

    def update_trail(self):
        _, y = pygame.mouse.get_pos()
        self.trail.append(y)

    def draw_trail(self):
        num = int(round(Canvas.width / (2 * speed)))
        recent_trail = self.trail[-num:-1]
        for i in range(1, num):
            try:
                x = int(round(Canvas.width / 2 - 2.155 * i * speed))

                # If we've gone off the screen stop drawing
                if x < 0:
                    break
                y = recent_trail[-i]
                self.surface.set_at((x, y), Canvas.BLUE)
            except IndexError:
                pass

    def simulation(self):
        # Simple main loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == KEYDOWN and event.key == K_SPACE:
                    running = False

            self.clock.tick(Canvas.FPS)  # do not go faster than this framerate
            # Redraw the background
            self.surface.fill(Canvas.background_color)

            # Draw the path
            self.draw_path()

            # Draw features
            self.draw_drawing_region()
            self.draw_cross()
            self.update_trail()
            self.draw_trail()

            # Put the surface we draw on, onto the screen
            self.screen.blit(self.surface, (0, 0))

            # Show it
            pygame.display.flip()


def error_plot(trail, path):
    t = np.array(trail)
    p = np.array(path)

    t = Canvas.height - t
    p = Canvas.height - p
    error = t - p

    time = np.linspace(0, len(t) / Canvas.FPS, len(t))

    plt.subplot(2, 1, 1)
    plt.plot(time, error, 'b')
    plt.xlim(0, time[-1])
    plt.ylabel('Error [px]')

    plt.subplot(2, 1, 2)
    plt.plot(time, p, 'g', label='Guidance')
    plt.plot(time, t, 'r', label='Subject')
    plt.xlim(0, time[-1])
    plt.ylabel('Position [px]')
    plt.xlabel('Time [s]')
    plt.legend(loc='best')
    plt.show()


def kalman_plot():
    iteration_count = len(c.trail)

    process_variance = 1e2
    est_measurement_variance = np.var(c.trail)

    kalman_filter = KalmanFilter(process_variance, est_measurement_variance)
    posteri_est_graph = []
    for iteration in xrange(1, iteration_count):
        kalman_filter.input_latest_noisy_measurement(c.trail[iteration])
        posteri_est_graph.append(kalman_filter.get_latest_est_measurement())

    time = np.linspace(0, len(c.trail) / Canvas.FPS, len(c.trail))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, c.trail, color='r', label='Subject')
    plt.plot(time[1:], posteri_est_graph, 'b-', label='Kalman Filter')
    plt.plot(time, c.path, color='g', label='Guidance')
    plt.legend(loc='best')
    plt.ylabel('Position [px]')
    plt.xlim(time[1], time[-1])


    plt.subplot(2, 1, 2)
    errs = np.array(c.path[1:]) - np.array(posteri_est_graph)
    plt.plot(time[1:], errs, 'b-', label='error between truth and kalman')
    plt.legend(loc='best')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [px]')
    plt.xlim(time[1], time[-1])
    plt.show()

# Wave properties
frequencyA, frequencyB = 1.2, 3.7
offsetA, offsetB = 3, 17
amplitudeA, amplitudeB = Canvas.height / 3 - 20, Canvas.height / 5 - 20
speed = 0.2

# Other simulation properties
drawing_width = 50

# Start the simulation
c = Canvas(frequencyA, frequencyB,
           offsetA, offsetB,
           amplitudeA, amplitudeB,
           speed,
           drawing_width)
c.simulation()
error_plot(c.trail, c.path)
kalman_plot()
