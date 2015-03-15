from __future__ import division, print_function
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
from array import array
import os


class Cursor(object):
    def __init__(self, ax):
        self.lx = ax.axhline(xmin=.475, xmax=.525, color='r', animated=True)
        self.ly = ax.axvline(ymin=.475, ymax=.525, color='r', animated=True)

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

        scale = 2 * ax.get_ylim()[1]
        location = y / scale + 0.5
        self.ly.set_ydata([location - .025, location + .025])


class StoplightMetric(object):
    def __init__(self, ax):
        self.ax = ax
        self.errs = []
        self.greens, self.yellows = [], []
        self.span = 60  # average over 60 measurements @ 60FPS = 1 second
        self.error = ax.plot(x[:window], np.zeros(window), animated=True, color='k')[0]

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
        # Flip our percentages
        greens = np.array(self.greens)[1:]
        yellows = np.array(self.yellows)[1:]

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
    def __init__(self, ax, statsax, left=1., right=1.):
        self.time = 0.
        self.guidance = ax.plot(x[:window], np.zeros(window), animated=True)[0]
        self.actual = ax.plot(x[:half_w], np.zeros(half_w), animated=True)[0]
        self.cursor = Cursor(ax)
        self.stoplight = StoplightMetric(statsax)

        self.ys, self.ygs = array('f'), array('f')

        # Add blockers on left and right
        half_width = ax.get_xlim()[1]/2
        left_covered = half_width - left * half_width
        right_covered = half_width + right * half_width
        self.patchL = patches.Rectangle((0.01, -1.),
                                        left_covered, 2.1,
                                        color='white',
                                        animated=True)
        self.patchR = patches.Rectangle((right_covered, -1.),
                                        half_width, 2.1,
                                        color='white',
                                        animated=True)
        ax.add_patch(self.patchL)
        ax.add_patch(self.patchR)

        # Disable ticks
        ax.set_xticklabels([], visible=False), ax.set_xticks([])
        ax.set_yticklabels([], visible=False), ax.set_yticks([])
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

        self.stoplight.update(err)

        # Close when the simulation is over
        if time >= length * FPS:
            plt.close()

        # List of things to be updated
        return [self.guidance, self.actual,
                self.patchL, self.patchR,
                self.cursor.lx, self.cursor.ly,
                self.stoplight.ax,
                self.stoplight.error,
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


def GenColors(d, span=60):
    # Find first and last indices
    last = len(d)
    first = 0

    epsilon = d[:, 1] - d[:, 2]

    greens, yellows = [], []
    for t in xrange(first, last + 1):
        if t - span < first:
            low = first
        else:
            low = t - span

        green = abs(epsilon[low:t + 1]) < .05
        green = green.mean()
        yellow = abs(epsilon[low:t + 1]) < .15
        yellow = yellow.mean()

        greens.append(green), yellows.append(yellow)

    return greens, yellows


def ShortLongColors(subj, d, short_span=60, long_span=10000, save=False):
    short_g, short_y = GenColors(d, short_span)
    long_g, long_y = GenColors(d, long_span)

    GenColorPlot(subj, d, short_g, short_y, long_g, long_y, save)


def GenColorPlot(subj, d, short_g, short_y, long_g, long_y, save):
    time = d[:, 0]
    epsilon = d[:, 1] - d[:, 2]

    # Flip our percentages
    short_g = 1 - np.array(short_g)[1:]
    short_y = 1 - np.array(short_y)[1:]
    long_g = 1 - np.array(long_g)[1:]
    long_y = 1 - np.array(long_y)[1:]

    # Plot short duration map
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.fill_between(time, 0, short_y,       color='r', facecolor='red')
    ax1.fill_between(time, short_y, short_g, color='y', facecolor='yellow')
    ax1.fill_between(time, short_g, 1,       color='g', facecolor='green')
    ax1.set_ylim([0, 1])
    plt.setp(ax1.get_yticklabels(), visible=False)

    # Add Error plot on top
    ax1twinx = ax1.twinx()
    # plt.plot(time, abs(pd.rolling_mean(epsilon, span)), 'k--')
    plt.plot(time, abs(epsilon), 'k')
    ax1twinx.set_ylabel('$|Error|$', color='k')
    for tl in ax1twinx.get_yticklabels():
        tl.set_color('k')
    plt.xlim([time[0], time[-1]])
    ax1twinx.set_ylim(bottom=0)
    plt.title(subj)

    # Plot long duration map
    ax2.fill_between(time, 0, long_y,      color='r', facecolor='red')
    ax2.fill_between(time, long_y, long_g, color='y', facecolor='yellow')
    ax2.fill_between(time, long_g, 1,      color='g', facecolor='green')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.xlim([time[0], time[-1]])
    ax2.set_xlabel('Time (s)')
    plt.tight_layout()
    save_fig('Metric X - ' + subj, save)


def save_fig(save_name, save):
    save_name = save_name.replace("$", "") + '.pdf'

    # Show it or save it
    if not save:
        plt.show()
    else:
        try:
            os.mkdir('figures')
        except Exception:
            pass

        plt.savefig('figures/' + save_name, bbox_inches='tight')
    plt.close()


# Create a plot
fig, (ax, statsax) = plt.subplots(nrows=2, figsize=(8, 9), sharex=True)
ax.set_ylim(-1.1, 1.1)
statsax.set_ylim(-1.1, 1.1)

# Build our guidance to follow
x = np.linspace(0 * np.pi, 40 * np.pi, 10000)

window = 1000
half_w = int(window/2)
ax.set_xlim(x[0], x[window])

# Create cursor and tracker
tracker = Tracker(ax, statsax, left=.8, right=.2)
# tracker = Tracker(ax)

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

d = np.vstack((t, y, yg)).T
ShortLongColors('test', d)
