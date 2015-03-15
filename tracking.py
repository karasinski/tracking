from __future__ import division, print_function
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
from array import array


class Cursor:
    def __init__(self, ax):
        self.lx = ax.axhline(xmin=.475, xmax=.525, color='r', animated=True)
        self.ly = ax.axvline(color='r', animated=True)

        self.ly.set_xdata(ax.get_xlim()[1]/2)
        self.lx.set_ydata(0)

        # Text location in axes coords
        self.txt = ax.text(0.45, 0.95, '',
                           transform=ax.transAxes, animated=True)

        # Connect
        plt.connect('motion_notify_event', self.mouse_move)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        y = event.ydata
        self.lx.set_ydata(y)


class Tracker(object):
    def __init__(self, ax, left=1., right=1.):
        self.time = 0.
        self.guidance = ax.plot(x[:window], np.zeros(window), animated=True)[0]
        self.actual = ax.plot(x[:half_w], np.zeros(half_w), animated=True)[0]
        self.cursor = Cursor(ax)

        self.ys, self.ygs = array('f'), array('f')

        half_width = ax.get_xlim()[1]/2
        left_covered = half_width - left * half_width
        right_covered = half_width + right * half_width
        self.patchL = patches.Rectangle((0, -1.1),
                                        left_covered, 2.2,
                                        facecolor='black', alpha=1,
                                        animated=True)
        self.patchR = patches.Rectangle((right_covered, -1.1),
                                        half_width, 2.2,
                                        facecolor='black', alpha=1,
                                        animated=True)
        ax.add_patch(self.patchL)
        ax.add_patch(self.patchR)

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        fig.canvas.mpl_connect('key_press_event', self.press)

    def __call__(self, time):
        self.time = time / 100.

        # Log cursor position
        self.ys.append(self.cursor.lx.get_ydata())
        self.ygs.append(self.guidance.get_ydata()[half_w])

        # Update guidance, plot recent data
        curr_range = x[time:window + time]
        self.guidance.set_ydata(func(curr_range))

        recent = np.zeros(half_w)
        recent[half_w-len(self.ys[-half_w:]):] += self.ys[-half_w:]
        self.actual.set_ydata(recent)

        err = self.ys[-1] - self.ygs[-1]
        self.cursor.txt.set_text('y=%1.2f, err=%1.2f' % (self.ys[-1], err))

        # Close when the simulation is over
        if time >= length * FPS:
            plt.close()

        # List of things to be updated
        return [self.guidance, self.actual,
                self.patchL, self.patchR,
                self.cursor.lx, self.cursor.ly,
                # self.cursor.txt
                ]

    def press(self, event):
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


# Create a plot
fig, ax = plt.subplots(figsize=(16, 9))
plt.ylim(-1.1, 1.1)

# Build our guidance to follow
x = np.linspace(0 * np.pi, 40 * np.pi, 10000)

window = 1000
half_w = int(window/2)
plt.xlim(x[0], x[window])

# Create cursor and tracker
tracker = Tracker(ax, left=.8, right=.0)

# Config animation
FPS, length = 60, 20
anim = FuncAnimation(fig, tracker,
                     frames=(length+1)*FPS, interval=1000./FPS,
                     blit=True, repeat=False)

plt.show()

# Show results
y, yg = tracker.results()
t = np.linspace(0, len(y)/FPS, len(y))
plt.plot(t, y, label='Subject')
plt.plot(t, yg, label='Guidance')
plt.xlim(0, t[-1])
plt.legend()
plt.show()
