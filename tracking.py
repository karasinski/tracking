from __future__ import division, print_function
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
from array import array
import pygame
import plots
import time


UNINITIALIZED = 0
INITIALIZED = 1
EXIT = 2
FINISHED = 3


class Cursor(object):
    def __init__(self, ax, use_joystick=False, invert=False):
        self.use_joystick = use_joystick
        self.ax = ax
        self.invert = invert
        self.lx = ax.axhline(xmin=.475, xmax=.525, color='r', animated=True)
        self.ly = ax.axvline(ymin=.475, ymax=.525, color='r', animated=True)

        self.ly.set_xdata(ax.get_xlim()[1]/2)
        self.lx.set_ydata(0)
        self.input = array('f')

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

    def update(self, status):
        if self.use_joystick:
            sensitivity = 35  # larger -> less sensitive
            velocity = self.joystick.input()

            if status == INITIALIZED:
                self.input.append(velocity)

            if self.invert:
                velocity *= -1
            y = self.lx.get_ydata()
            y += velocity / sensitivity

            # Bind position to screen limits
            if y > 1.1:
                y = 1.1
            elif y < -1.1:
                y = -1.1
            self.lx.set_ydata(y)

        y = self.lx.get_ydata()
        scale = 2 * self.ax.get_ylim()[1]
        location = y / scale + 0.5
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

        # Must call this to update values
        pygame.event.get()

        # Get axis value
        axis = self.joystick.get_axis(3)
        return axis


class StoplightMetric(object):
    def __init__(self, ax, span=60, feedback=False):
        self.ax = ax
        self.feedback = feedback
        self.errs = []
        self.greens, self.yellows = [], []
        self.span = span  # average over 60 measurements @ 60FPS = 1 second

        if feedback:
            color = 'k'
        else:
            color = 'white'

        self.error = ax.plot(x[:window], np.zeros(window),
                             animated=True, color=color)[0]

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
        color = ''
        try:
            red = 1 - yellows[-1]
            yellow = yellows[-1] - greens[-1]
            green = greens[-1]

            if red > yellow and red > green:
                color = 'red'
            elif yellow > red and yellow > green:
                color = 'yellow'
            else:
                color = 'green'
            # print(red, yellow, green)
        except IndexError:
            color = 'white'

        if self.feedback:
            self.ax.set_axis_bgcolor(color)
        else:
            self.ax.set_axis_bgcolor('white')


class Tracker(object):
    def __init__(self, fig, ax, statsax,
                 trial=0,
                 use_joystick=False,
                 funckwds={},
                 left=1., right=1.,
                 span=60, length=20, FPS=60,
                 feedback=False, invert=False):
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
        ax.set_title('Trial ' + str(trial))
        ax.set_xticklabels([], visible=False), ax.set_xticks([])
        ax.set_yticklabels([], visible=False), ax.set_yticks([])
        ax.set_yticklabels([], visible=False), statsax.set_yticks([])
        fig.canvas.mpl_connect('key_press_event', self.press)

        # Finally initialize simulation
        self.trial = trial
        self.status = 0
        self.frame = 0.
        self.FPS = FPS
        self.end_frame = length * FPS
        self.funckwds = funckwds
        self.guidance = ax.plot(x[:window], np.zeros(window), animated=True)[0]
        self.actual = ax.plot(x[:half_w], np.zeros(half_w), animated=True)[0]
        self.cursor = Cursor(ax, use_joystick=use_joystick, invert=invert)
        self.stoplight = StoplightMetric(statsax,
                                         span=span * FPS, feedback=feedback)
        self.ys, self.ygs = array('f'), array('f')

    def __call__(self, frame):
        self.cursor.update(self.status)

        if self.status == INITIALIZED:
            self.frame += 1

            # Log cursor position
            self.ys.append(self.cursor.lx.get_ydata())
            self.ygs.append(self.guidance.get_ydata()[half_w])

            err = self.ys[-1] - self.ygs[-1]

            self.stoplight.update(err)

            # Close when the simulation is over
            if self.frame >= self.end_frame:
                self.status = FINISHED
                plt.close()

        # Update guidance, plot recent data
        low = self.frame
        high = window + self.frame
        self.guidance.set_ydata(func(x, **self.funckwds)[low:high])

        recent = np.zeros(half_w)
        recent[half_w-len(self.ys[-half_w:]):] += self.ys[-half_w:]
        self.actual.set_ydata(recent)

        # List of things to be updated
        return [self.guidance, self.actual,
                self.patchL, self.patchR,
                self.cursor.lx, self.cursor.ly,
                self.stoplight.ax,
                self.stoplight.error]

    def press(self, event):
        if self.status == UNINITIALIZED:
            self.status = INITIALIZED
        elif self.status == INITIALIZED:
            self.status = EXIT
            plt.close()

    def results(self):
        inp = self.cursor.input
        y = self.ys
        yg = self.ygs
        t = np.linspace(0, self.frame/self.FPS, len(y))
        d = np.vstack((t, inp, y, yg)).T

        path = 'trials/'
        path += str(self.trial) + ' '
        path += str(int(time.time()))
        np.savetxt(path, d, delimiter=",")
        return d


def func(t,
         frequencyA=0.5, frequencyB=0,
         offsetA=0, offsetB=0,
         amplitudeA=1, amplitudeB=1):

    f = draw_sin(t, a=amplitudeA, f=frequencyA, o=offsetA)
    f += draw_sin(t, a=amplitudeB, f=frequencyB, o=offsetB)
    f = f / max(abs(f))
    return f


def draw_sin(t, a=1, f=1, o=0):
    return a * np.sin(f * (t + o))


def RunTrial(kwds, show=False):
    # Create a plot
    fig, (ax, statsax) = plt.subplots(nrows=2, figsize=(8, 9), sharex=True)

    # Merge input options with defaults
    defaults = {'use_joystick': True,
                'left': 1.,
                'right': 1.,
                'span': 1,
                'funckwds': {},
                'length': 30,
                'FPS': 60,
                'feedback': False,
                'invert': True}
    kwds = dict(defaults.items() + kwds.items())

    # Configure animation
    tracker = Tracker(fig, ax, statsax, **kwds)

    # This needs to be assigned so it can hang around to get called right below
    anim = FuncAnimation(fig, tracker,
                         interval=1000./kwds['FPS'],
                         blit=True, repeat=False)

    # Start animation
    plt.show()
    d = tracker.results()

    if show:
        # Show results
        plots.Performance(d)
        plots.ShortLongColor('test', d)

# A couple global parameters
x = np.linspace(0 * np.pi, 40 * np.pi, 10000)
window = 1000
half_w = int(window/2)

# Experiment parameters
# funckwds = {'frequencyA': 0.6, 'frequencyB': 1.7,
#             'offsetA': 3, 'offsetB': 17,
#             'amplitudeA': 0.6, 'amplitudeB': .2}
# kwds = {'trial': 1,
#         'length': 20,
#         'funckwds': funckwds,
#         'left': .8,
#         'right': .2,
#         'feedback': True}

# Run a trial
# RunTrial(kwds, show=True)

funckwds1 = {'frequencyA': 0.6, 'frequencyB': 1.7,
             'offsetA': 3, 'offsetB': 17,
             'amplitudeA': 0.6, 'amplitudeB': .2}
funckwds2 = {'frequencyA': 0.6, 'frequencyB': 1.,
             'offsetA': 17, 'offsetB': 3,
             'amplitudeA': 0.6, 'amplitudeB': .4}

ks = [  # 'Training' Trials
        {'trial': 1},

        {'trial': 2,
         'feedback': True},

        {'trial': 3,
         'left': .5,
         'right': .2},

        {'trial': 4,
         'left': .5,
         'right': .2,
         'feedback': True},

        {'trial': 5,
         'left': .5,
         'right': 0},

        {'trial': 6,
         'left': .5,
         'right': 0,
         'feedback': True},

        # 'Experiment' Trials
        {'trial': 7,
         'funckwds': funckwds1},

        {'trial': 8,
         'funckwds': funckwds2,
         'left': .5,
         'right': .2,
         'feedback': True},

        {'trial': 9,
         'funckwds': funckwds2,
         'left': .5,
         'right': 0},

        {'trial': 10,
         'funckwds': funckwds1,
         'left': .5,
         'right': .2,
         'feedback': True},

        {'trial': 11,
         'funckwds': funckwds1,
         'left': .5,
         'right': 0},

        {'trial': 12,
         'funckwds': funckwds2},

        {'trial': 13,
         'funckwds': funckwds1,
         'left': .5,
         'right': 0,
         'feedback': True},

        {'trial': 14,
         'funckwds': funckwds1,
         'feedback': True},

        {'trial': 15,
         'funckwds': funckwds2,
         'feedback': True},

        {'trial': 16,
         'funckwds': funckwds1,
         'left': .5,
         'right': .2},

        {'trial': 17,
         'funckwds': funckwds2,
         'left': .5,
         'right': .2},

        {'trial': 18,
         'funckwds': funckwds2,
         'left': .5,
         'right': 0,
         'feedback': True},
        ]

# Run the experiment
for k in ks:
    RunTrial(k)
