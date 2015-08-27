from __future__ import division, print_function
import os
import time
import datetime

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib._png import read_png

import seaborn as sns
import numpy as np
import pygame
import pykov

from trials import *


mpl.rcParams['toolbar'] = 'None'

UNINITIALIZED = 0
INITIALIZED = 1
EXIT = 2
FINISHED = 3

BLUE = -1
GREEN = 1

NO_KEY = np.NaN


class Cursor(object):
    def __init__(self, ax, use_joystick=False, invert=False):
        self.use_joystick = use_joystick
        self.ax = ax
        self.invert = invert

        self.marker = ax.plot(ax.get_xlim()[1]/2, 0,
                              'k', marker=r'$\times$',
                              markersize=12, animated=True)[0]
        self.marker.set_ydata(0)

        self.input = []

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
        self.marker.set_ydata(y)

    def update(self, status):
        if self.use_joystick:
            sensitivity = 25  # larger -> less sensitive
            velocity = self.joystick.input()

            if status == INITIALIZED:
                self.input.append(velocity)

            if self.invert:
                velocity *= -1
            y = self.marker.get_ydata()
            y += velocity / sensitivity

            # Bind position to screen limits
            if y > 1.2:
                y = 1.2
            elif y < -1.2:
                y = -1.2
            self.marker.set_ydata(y)
        else:
            if status == INITIALIZED:
                # Can't really log your mouse input...
                self.input.append(0)


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
        axis = self.joystick.get_axis(0)
        return axis


class Target(object):
    def __init__(self, ax, span=60, feedback=False):
        self.x = x[half_w]
        self.target = ax.plot(self.x, 0, ' o', color='darkgreen',
                              alpha=0.75, markersize=12, animated=True)[0]

        self.ax = ax
        self.feedback = feedback
        self.errs = []
        self.colors, self.fake_colors = [], []
        self.greens, self.yellows = [], []
        self.span = span  # average over 60 measurements @ 60FPS = 1 second
        self.fake = pykov.Chain({( 'green',  'green'): 0.99688064531915932,
                                 ('yellow', 'yellow'): 0.98290598290598286,
                                 ('yellow',  'green'): 0.017094017094017096,
                                 ( 'green', 'yellow'): 0.0031193546808406512})

        self.fake_walk = self.fake.walk(2000)

    def position(self):
        return self.target.get_ydata()

    def update(self, y):
        self.target.set_ydata(y)

    def update_performance(self, new_measurement):
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

        green = np.abs(self.errs[low:]) < .15
        self.greens.append(green.mean())
        yellow = np.abs(self.errs[low:]) > .15
        self.yellows.append(yellow.mean())

    def drawColors(self):
        green = np.array(self.greens)[-1]
        yellow = np.array(self.yellows)[-1]

        color = ''
        try:
            c = {'red': yellow, 'darkgreen': green}
            color = max(c, key=c.get)
        except IndexError:
            color = '#EAEAF2'

        self.colors.append(color)
        fake_color = color
        if self.feedback == FEEDBACK_ON:
            self.target.set_color(color)
        elif self.feedback == FEEDBACK_FALSE:
            step = len(self.greens)
            fake_color = self.fake_walk[step]
            self.target.set_color(fake_color)
        else:
            self.target.set_color('darkgreen')
        self.fake_colors.append(fake_color)


class Tracker(object):
    def __init__(self, fig, ax, ax2, statsax,
                 trial=0, rand_id=None,
                 use_joystick=False,
                 funckwds={},
                 history=1., preview=1.,
                 span=1, length=20, FPS=60,
                 feedback=FEEDBACK_OFF, invert=False,
                 has_timer=False, secondary_task=False):

        # Default rand_id to the trial number
        if rand_id is None:
            rand_id = trial

        # Set some limits
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(x[0], x[window])

        # Set up secondary task
        self.secondary_task = secondary_task
        teal = read_png('imgs/TEAL.png')
        blue = read_png('imgs/BLUE.png')
        green = read_png('imgs/GREEN.png')

        secondary_kwds = dict(aspect='equal', animated=True, visible=False)
        self.teal = ax2.imshow(teal, **secondary_kwds)
        self.blue = ax2.imshow(blue, **secondary_kwds)
        self.green = ax2.imshow(green, **secondary_kwds)

        if secondary_task:
            self.teal.set_visible(True)

        np.random.seed(rand_id)
        color_times = np.arange(5, length, 7, dtype=np.float)
        color_times += 4 * np.random.rand(len(color_times))
        self.color_times = color_times

        # Disable ticks
        ax.set_title('Trial ' + str(trial))
        fig.canvas.mpl_connect('key_press_event', self.press)

        # Finally initialize simulation
        self.trial = trial
        self.status = 0
        self.frame = 0.
        self.FPS = FPS
        self.secondary_task_color = []
        self.end_frame = length * FPS
        self.funckwds = funckwds
        self.guidance_path = generate_path(rand_id)
        self.cursor = Cursor(ax, use_joystick=use_joystick, invert=invert)
        self.target = Target(ax, span=1, feedback=feedback)
        self.ys, self.ygs = [], []
        self.t, self.t_end = [], []
        self.current_key = NO_KEY
        self.key_press = []

    def __call__(self, frame):
        self.cursor.update(self.status)

        if self.status == INITIALIZED:
            t = time.time()
            self.frame += 1

            # Log cursor/target position
            self.ys.append(self.cursor.marker.get_ydata())
            self.ygs.append(self.target.position())
            self.t.append(t)

            err = self.ys[-1] - self.ygs[-1]
            self.target.update_performance(err)

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
                self.secondary_task_color.append(val)
            else:
                self.secondary_task_color.append(np.nan)

        # Update guidance
        self.guidance_path[int(self.frame + window/2)]
        position = self.guidance_path[int(self.frame + window/2)]
        self.target.update(position)

        if self.status == INITIALIZED:
            self.t_end.append(time.time())
            self.key_press.append(self.current_key)
            self.current_key = NO_KEY
            try:
                desired_time = self.frame / self.FPS
                actual_time = self.t_end[-1] - self.t[0]

                time.sleep(desired_time - actual_time)
            except (IndexError, IOError):
                pass

        # Close when the simulation is over
        if self.frame >= self.end_frame:
            self.status = FINISHED
            self.results()
            plt.close()

        # List of things to be updated
        return [self.target.target,
                self.cursor.marker,
                self.teal, self.blue, self.green]

    def press(self, event):
        # Start the trial when the subject hits the space bar
        if event.key in ('down', 'escape'):
            if self.status == UNINITIALIZED:
                self.status = INITIALIZED
                return
        if event.key == 'escape':
            if self.status == INITIALIZED:
                self.status = EXIT
                plt.close()
                return

        # If the light is on, turn it off
        visible = self.teal.get_visible()
        if not visible:
            self.current_key = event.key
            if event.key == 'left' and self.blue.get_visible():
                self.teal.set_visible(True)
                self.blue.set_visible(False)
            elif event.key == 'right' and self.green.get_visible():
                self.teal.set_visible(True)
                self.green.set_visible(False)

    def results(self):
        inp = self.cursor.input
        y = self.ys
        yg = self.ygs
        secondary_task_color = self.secondary_task_color
        feedbackcolor = self.target.colors
        fake_feedbackcolor = self.target.fake_colors
        key_press = self.key_press
        t = self.t
        t_end = self.t_end
        labels = ['Input', 'y', 'yg', 'SecondaryColor',
                  'FeedbackColor', 'FakeFeedbackColor', 'KeyPress',
                  'Time1', 'Time2']

        path = 'trials/Subject' + str(subject_number) + '/'

        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

        path += str(self.trial) + ' '
        path += str(int(time.time()))

        df = pd.DataFrame([inp, y, yg, secondary_task_color,
                           feedbackcolor, fake_feedbackcolor, key_press,
                           t, t_end]).T
        df.columns = labels
        df.to_csv(path, float_format='%.8f')
        return df


def RunTrial(kwds):
    # Some 'trials' are actually rest periods
    if kwds.has_key('wait'):
        now = datetime.datetime.now()
        then = now + datetime.timedelta(seconds=kwds['wait'])
        print('Current time is  {}.\nPlease resume at {}.'.format(now, then))
        time.sleep(kwds['wait'])
        return

    # Create a plot
    fig = plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title('1D Tracking Experiment')
    gs = gridspec.GridSpec(5, 3)
    ax = plt.subplot(gs[0:4, :])
    ax2 = plt.subplot(gs[4, 0])
    ax3 = plt.subplot(gs[4, 1])
    statsax = plt.subplot(gs[4, 2])

    fig.set_facecolor([.75, .75, .75, 1])
    for i, ax_i in enumerate([ax, ax2, ax3, statsax]):
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
                'feedback': FEEDBACK,
                'invert': False,
                'secondary_task': True}
    kwds = dict(defaults.items() + kwds.items())

    # Configure animation
    tracker = Tracker(fig, ax, ax3, statsax, **kwds)

    # This needs to be assigned so that plt.show can start the simulation
    anim = FuncAnimation(fig, tracker,
                         interval=15, blit=True, repeat=False)

    # Start animation
    plt.tight_layout()
    plt.show()


# A couple global parameters
x = np.linspace(0 * np.pi, 40 * np.pi, 10000)
window = 1000
half_w = int(window/2)

subject_number = input('Please input Subject #: ')
if subject_number % 2 == 0:
    FEEDBACK = FEEDBACK_OFF
else:
    FEEDBACK = FEEDBACK_ON

# Run the experiment
for k in ks:
    RunTrial(k)
