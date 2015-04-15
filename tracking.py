from __future__ import division, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
from array import array
import pygame
import plots
import time
from matplotlib._png import read_png
import seaborn as sns


UNINITIALIZED = 0
INITIALIZED = 1
EXIT = 2
FINISHED = 3

BLUE = -1
GREEN = 1


class Cursor(object):
    def __init__(self, ax, use_joystick=False, invert=False):
        self.use_joystick = use_joystick
        self.ax = ax
        self.invert = invert
        self.lx = ax.axhline(xmin=.49, xmax=.51, color='k', animated=True)
        self.ly = ax.axvline(ymin=0, ymax=0, color='k', animated=True)

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
            sensitivity = 30  # larger -> less sensitive
            velocity = self.joystick.input()

            if status == INITIALIZED:
                self.input.append(velocity)

            if self.invert:
                velocity *= -1
            y = self.lx.get_ydata()
            y += velocity / sensitivity

            # Bind position to screen limits
            if y > 1.2:
                y = 1.2
            elif y < -1.2:
                y = -1.2
            self.lx.set_ydata(y)
        else:
            if status == INITIALIZED:
                # Can't really log your mouse input...
                self.input.append(0)

        y = self.lx.get_ydata()
        scale = 2 * self.ax.get_ylim()[1]
        location = y / scale + 0.5
        self.ly.set_ydata([location - .01, location + .01])


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
        self.errs = array('f')
        self.greens, self.yellows = [], []
        self.span = span  # average over 60 measurements @ 60FPS = 1 second

    def update(self, new_measurement):
        self.errs.append(new_measurement)

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
        except IndexError:
            color = '#EAEAF2'

        if self.feedback:
            self.ax.set_axis_bgcolor(color)
        else:
            self.ax.set_axis_bgcolor((.75, .75, .75, 1))


class Timer(object):
    def __init__(self, fig, ax, timer_start='', has_timer=False):
        if has_timer:
            color = 'black'
        else:
            color = fig.get_facecolor()

        # Text location in axes coords
        self.timer = ax.annotate(str(timer_start), xy=(.5, 0),
                                 xycoords='axes fraction', fontsize=36,
                                 textcoords='offset points', ha='right',
                                 va='bottom', color=color, animated=True)


class Target(object):
    def __init__(self, ax):
        self.x = x[half_w]
        self.target = ax.plot(self.x, 0, 'bo', alpha=0.5, markersize=12, animated=True)[0]

    def update(self, y):
        self.target.set_ydata(y)


class Tracker(object):
    def __init__(self, fig, ax, ax2, statsax,
                 trial=0,
                 use_joystick=False,
                 funckwds={},
                 history=1., preview=1.,
                 span=60, length=20, FPS=60,
                 feedback=False, invert=False,
                 has_timer=False, secondary_task=False):
        # Set some limits
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(x[0], x[window])

        # Add blockers on history and preview
        half_width = ax.get_xlim()[1]/2
        history_covered = half_width - history * half_width
        preview_covered = half_width + preview * half_width
        self.patchL = patches.Rectangle((0, -1.2),
                                        history_covered, 2.4,
                                        color='#EAEAF2',
                                        animated=True)
        self.patchR = patches.Rectangle((preview_covered, -1.2),
                                        half_width, 2.4,
                                        color='#EAEAF2',
                                        animated=True)
        ax.add_patch(self.patchL)
        ax.add_patch(self.patchR)

        # Set up secondary task
        self.secondary_task = secondary_task
        teal = read_png('imgs/TEAL.png')
        blue = read_png('imgs/BLUE.png')
        green = read_png('imgs/GREEN.png')

        self.teal = ax2.imshow(teal, aspect='equal', animated=True, visible=False)
        self.blue = ax2.imshow(blue, aspect='equal', animated=True, visible=False)
        self.green = ax2.imshow(green, aspect='equal', animated=True, visible=False)

        if secondary_task:
            self.teal.set_visible(True)

        color_times = np.arange(5, length, 8, dtype=np.float)
        color_times += + 3 * np.random.rand(len(color_times))
        self.color_times = color_times

        # Disable ticks
        ax.set_title('Trial ' + str(trial))
        fig.canvas.mpl_connect('key_press_event', self.press)

        # Finally initialize simulation
        self.trial = trial
        self.status = 0
        self.frame = 0.
        self.FPS = FPS
        self.timer_start_value = 15
        self.timer, self.secondary_task_color = array('f'), array('f')
        self.end_frame = length * FPS
        self.funckwds = funckwds
        self.guidance = ax.plot(x[:window], np.zeros(window), animated=True)[0]
        self.actual = ax.plot(x[:half_w], np.zeros(half_w), animated=True)[0]
        self.cursor = Cursor(ax,
                             use_joystick=use_joystick, invert=invert)
        self.timer_obj = Timer(fig, ax2, timer_start=self.timer_start_value,
                               has_timer=has_timer)
        self.stoplight = StoplightMetric(statsax,
                                         span=span * FPS, feedback=feedback)
        self.target = Target(ax)
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

            # Set timer value
            try:
                timer = self.timer[-1] - 1./self.FPS
                if timer < 0:
                    timer = 0
            except:
                timer = self.timer_start_value

            self.timer_obj.timer.set_text('%2.0f' % (timer))
            self.timer.append(timer)

            # Set colors value
            if self.secondary_task:
                try:
                    if self.frame/self.FPS > self.color_times[0]:
                        self.color_times = self.color_times[1:]

                        # Randomly select one of the lights to turn on
                        random_choice = np.random.choice((BLUE, GREEN))
                        self.teal.set_visible(False)
                        self.blue.set_visible(False)
                        self.green.set_visible(False)
                        if random_choice == BLUE:
                            self.blue.set_visible(True)
                        elif random_choice == GREEN:
                            self.green.set_visible(True)
                except IndexError:
                    pass

            if self.teal.get_visible():
                val = 0
            elif self.blue.get_visible():
                val = BLUE
            elif self.green.get_visible():
                val = GREEN
            else:
                val = -999
            self.secondary_task_color.append(val)

            # Close when the simulation is over
            if self.frame >= self.end_frame:
                self.status = FINISHED
                plt.close()

        # Update guidance, plot recent data
        low = self.frame
        high = window + self.frame
        f = func(x, **self.funckwds)[low:high]
        self.guidance.set_ydata(f)

        # Send center to target
        self.target.update(f[len(f)/2])

        recent = np.zeros(half_w)
        recent[half_w-len(self.ys[-half_w:]):] += self.ys[-half_w:]
        self.actual.set_ydata(recent)

        # List of things to be updated
        return [self.guidance, self.actual,
                self.patchL, self.patchR,
                self.target.target,
                self.cursor.lx, self.cursor.ly,
                self.stoplight.ax,
                self.timer_obj.timer,
                self.teal, self.blue, self.green]

    def press(self, event):
        # Start the trial when the subject hits the space bar
        if event.key == ' ':
            if self.status == UNINITIALIZED:
                self.status = INITIALIZED
            elif self.status == INITIALIZED:
                self.status = EXIT
                plt.close()

        # If the light is on, turn it off
        visible = self.teal.get_visible()
        if not visible:
            if event.key == 'left' and self.blue.get_visible():
                self.teal.set_visible(True)
                self.blue.set_visible(False)
            elif event.key == 'right' and self.green.get_visible():
                self.teal.set_visible(True)
                self.green.set_visible(False)

        # If the timer has ran out, reset the timer on click
        if event.key is 'left' or event.key is 'right':
            if self.timer[-1] == 0.:
                self.timer[-1] = self.timer_start_value

    def results(self):
        inp = self.cursor.input
        y = self.ys
        yg = self.ygs
        t = np.linspace(0, self.frame/self.FPS, len(y))
        timer = self.timer
        secondary_task_color = self.secondary_task_color
        d = np.vstack((t, inp, y, yg, timer, secondary_task_color)).T

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
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(5, 4)
    ax = plt.subplot(gs[0:4, :])
    ax2 = plt.subplot(gs[4, 0])
    ax3 = plt.subplot(gs[4, 1])
    ax4 = plt.subplot(gs[4, 2])
    statsax = plt.subplot(gs[4, 3])

    for i, ax_i in enumerate([ax, ax2, ax3, ax4, statsax]):
        ax_i.set_xticklabels([], visible=False), ax_i.set_xticks([])
        ax_i.set_yticklabels([], visible=False), ax_i.set_yticks([])
        if i == 0:
            continue
        ax_i.set_axis_bgcolor((.75, .75, .75, 1))

    # Merge input options with defaults
    defaults = {'use_joystick': True,
                'history': 0.,
                'preview': 0.,
                'span': 1,
                'funckwds': {},
                'length': 60,
                'FPS': 60,
                'feedback': False,
                'invert': True}
    kwds = dict(defaults.items() + kwds.items())

    # Configure animation
    tracker = Tracker(fig, ax, ax2, statsax, **kwds)

    # This needs to be assigned so it can hang around to get called below
    anim = FuncAnimation(fig, tracker,
                         interval=1000./kwds['FPS'],
                         blit=True, repeat=False)

    # Start animation
    plt.tight_layout()
    plt.show()

    # Make sure our data logging is working
    d = tracker.results()

    # Show results
    if show:
        plots.Performance(d)
        plots.ShortLongColor('test', d)

# A couple global parameters
x = np.linspace(0 * np.pi, 40 * np.pi, 10000)
window = 1000
half_w = int(window/2)

funckwds1 = {'frequencyA': 0.9, 'frequencyB': 3.3,
             'offsetA': 0, 'offsetB': np.pi/7,
             'amplitudeA': 0.5, 'amplitudeB': .2}
funckwds2 = {'frequencyA': np.log(2), 'frequencyB': np.pi,
             'offsetA': 3, 'offsetB': .17,
             'amplitudeA': 0.6, 'amplitudeB': .2}
funckwds3 = {'frequencyA': np.log(3), 'frequencyB': np.log(7),
             'offsetA': 17, 'offsetB': 3,
             'amplitudeA': 0.6, 'amplitudeB': .4}

ks = [
         # Refresher
         {'trial': 1,
          'history': 0.2},

         {'trial': 2,
          'history': 0.1,
          'funckwds': funckwds1},

         {'trial': 3,
          'funckwds': funckwds1},

         # Experiment
         {'trial': 4,
          'history': 0.2,
          'funckwds': funckwds2},

         {'trial': 5,
          'history': 0.05,
          'funckwds': funckwds3},

         {'trial': 6,
          'funckwds': funckwds3},

         {'trial': 7,
          'history': 0.05,
          'funckwds': funckwds2},

         {'trial': 8,
          'history': 0.2,
          'funckwds': funckwds3},

         {'trial': 9,
          'history': 0.1,
          'funckwds': funckwds3},

         {'trial': 10,
          'funckwds': funckwds2},

         {'trial': 11,
          'history': 0.1,
          'funckwds': funckwds2},

         # {'trial': 0,
         #  'history': 0.2,
         #  'preview': 0.,
         #  'feedback': True,
         #  'funckwds': funckwds2},

         # {'trial': 1,
         #  'history': 0.,
         #  'preview': 0.2,
         #  'has_timer': True,
         #  'feedback': True,
         #  'funckwds': funckwds2},

         # {'trial': 2,
         #  'history': 0.,
         #  'preview': 0.005,
         #  'secondary_task': True,
         #  'funckwds': funckwds1},
        ]

# Run the experiment
for k in ks:
    RunTrial(k)
