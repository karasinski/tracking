from __future__ import division, print_function
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
from array import array
import pygame
import plots


UNINITIALIZED = 0
INITIALIZED = 1
EXIT = 2
FINISHED = 3


class Cursor(object):
    def __init__(self, ax, use_joystick=False):
        self.use_joystick = use_joystick
        self.ax = ax
        self.lx = ax.axhline(xmin=.475, xmax=.525, color='r', animated=True)
        self.ly = ax.axvline(ymin=.475, ymax=.525, color='r', animated=True)

        self.ly.set_xdata(ax.get_xlim()[1]/2)
        self.lx.set_ydata(0)

        # Connect
        if use_joystick:
            try:
                self.joystick = Joystick()
            except Exception:
                print('Joystick initialization failed, falling back to mouse.')
                self.use_joystick = False
                plt.connect('motion_notify_event', self.mouse_move)
        else:
            plt.connect('motion_notify_event', self.mouse_move)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        y = event.ydata
        self.lx.set_ydata(y)

        scale = 2 * self.ax.get_ylim()[1]
        location = y / scale + 0.5
        self.ly.set_ydata([location - .025, location + .025])

    def update(self):
        if self.use_joystick:
            velocity = self.joystick.input()
            location = np.mean(self.ly.get_ydata())
            print('loc: ', location)
            location += velocity / 10.
            self.ly.set_ydata([location - .025, location + .025])


class Joystick(object):
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def input(self):
        '''
        We're working with a 3dconnexion Wireless SpaceMouse.

        Axis 3 is the forward/back tilt axis, axis 1 is the same for push.
        '''

        axis = self.joystick.get_axis(3)
        print(self.joystick.get_name())
        print('ax: ', axis)
        return axis


class StoplightMetric(object):
    def __init__(self, ax, span=60):
        self.ax = ax
        self.errs = []
        self.greens, self.yellows = [], []
        self.span = span  # average over 60 measurements @ 60FPS = 1 second
        self.error = ax.plot(x[:window], np.zeros(window),
                             animated=True, color='k')[0]

    def update(self, new_measurement):
        self.errs.append(new_measurement)

        # Draw error
        recent = np.zeros(window)
        recent[window-len(self.errs[-window:]):] += self.errs[-window:]
        self.error.set_ydata(recent)

        # Update colors
        self.updateColors()
        self.drawColors()

    def updateColors(self):
        t = len(self.errs)
        if t - self.span < 0:
            low = 0
        else:
            low = t - self.span

        green = np.abs(self.errs[low:]) < .05
        self.greens.append(green.mean())
        yellow = np.abs(self.errs[low:]) < .15
        self.yellows.append(yellow.mean())

    def drawColors(self):
        greens = np.array(self.greens)[1:]
        yellows = np.array(self.yellows)[1:]

        # Check the latest value and find the largest--this is inefficient
        try:
            red = 1 - yellows[-1]
            yellow = yellows[-1] - greens[-1]
            green = greens[-1]

            if red > yellow and red > green:
                self.ax.set_axis_bgcolor('red')
            elif yellow > red and yellow > green:
                self.ax.set_axis_bgcolor('yellow')
            else:
                self.ax.set_axis_bgcolor('green')
            # print(red, yellow, green)
        except IndexError:
            self.ax.set_axis_bgcolor('red')


class Tracker(object):
    def __init__(self, fig, ax, statsax,
                 use_joystick=False,
                 left=1., right=1.,
                 span=60, length=20, FPS=60):
        # Set some limits
        ax.set_ylim(-1.1, 1.1)
        statsax.set_ylim(-1.1, 1.1)
        ax.set_xlim(x[0], x[window])

        # Add blockers on left and right
        half_width = ax.get_xlim()[1]/2
        left_covered = half_width - left * half_width
        right_covered = half_width + right * half_width
        self.patchL = patches.Rectangle((0.01, -1.),
                                        left_covered, 2.09,
                                        color='white',
                                        animated=True)
        self.patchR = patches.Rectangle((right_covered, -1.),
                                        half_width, 2.09,
                                        color='white',
                                        animated=True)
        ax.add_patch(self.patchL)
        ax.add_patch(self.patchR)

        # Disable ticks
        ax.set_xticklabels([], visible=False), ax.set_xticks([])
        ax.set_yticklabels([], visible=False), ax.set_yticks([])
        fig.canvas.mpl_connect('key_press_event', self.press)

        # Finally initialize simulation
        self.status = 0
        self.time = 0.
        self.end_time = length * FPS
        self.guidance = ax.plot(x[:window], np.zeros(window), animated=True)[0]
        self.actual = ax.plot(x[:half_w], np.zeros(half_w), animated=True)[0]
        self.cursor = Cursor(ax, use_joystick=use_joystick)
        self.stoplight = StoplightMetric(statsax, span=span)
        self.ys, self.ygs = array('f'), array('f')

    def __call__(self, time):
        self.cursor.update()

        if self.status == INITIALIZED:
            self.time += 1

            # Log cursor position
            self.ys.append(self.cursor.lx.get_ydata())
            self.ygs.append(self.guidance.get_ydata()[half_w])

            err = self.ys[-1] - self.ygs[-1]

            self.stoplight.update(err)

            # Close when the simulation is over
            if self.time >= self.end_time:
                self.status = FINISHED
                plt.close()

        # Update guidance, plot recent data
        curr_range = x[self.time:window + self.time]
        self.guidance.set_ydata(func(curr_range))

        recent = np.zeros(half_w)
        recent[half_w-len(self.ys[-half_w:]):] += self.ys[-half_w:]
        self.actual.set_ydata(recent)

        # List of things to be updated
        return [self.guidance, self.actual,
                self.patchL, self.patchR,
                self.cursor.lx, self.cursor.ly,
                self.stoplight.ax,
                self.stoplight.error,
                ]

    def press(self, event):
        if self.status == UNINITIALIZED:
            self.status = INITIALIZED
        elif self.status == INITIALIZED:
            self.status = EXIT
            plt.close()

    def results(self):
        return np.array(self.ys), np.array(self.ygs)


def func(t):
    frequencyA, frequencyB = 0.6, 1.7
    offsetA, offsetB = 3, 17
    amplitudeA, amplitudeB = .6, .2

    f = draw_sin(t, a=amplitudeA, f=frequencyA, o=offsetA)
    f += draw_sin(t, a=amplitudeB, f=frequencyB, o=offsetB)
    f = f / max(abs(f))
    return f


def draw_sin(t, a=1, f=1, o=0):
    return a * np.sin(f * (t + o))


def RunExperiment():
    # Create a plot
    fig, (ax, statsax) = plt.subplots(nrows=2, figsize=(8, 9), sharex=True)

    # Create cursor and tracker
    kwds = {'use_joystick': True,
            'left': .8,
            'right': .2,
            'span': 60,
            'length': 20,
            'FPS': 60
            }
    # Configure animation
    tracker = Tracker(fig, ax, statsax, **kwds)

    # This needs to be assigned so it can hang around to get called right below
    anim = FuncAnimation(fig, tracker,
                         interval=1000./kwds['FPS'],
                         blit=True, repeat=False)

    # Start animation
    plt.show()

    # Show results
    y, yg = tracker.results()
    t = np.linspace(0, len(y)/kwds['FPS'], len(y))
    d = np.vstack((t, y, yg)).T

    plots.Performance(d)
    plots.ShortLongColor('test', d)

# A couple global parameters
x = np.linspace(0 * np.pi, 40 * np.pi, 10000)
window = 1000
half_w = int(window/2)

# Run the experiment
RunExperiment()
