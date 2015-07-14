from __future__ import division, print_function
import numpy as np
from numpy import pi
import pandas as pd


FEEDBACK_ON    =  1
FEEDBACK_OFF   =  0
FEEDBACK_FALSE = -1


def f(x):
    return pd.Series((x[0] * 2 * pi * x[1] / 240.) * np.cos(2*pi*x[1] * t / 240. + x[3]))


def rescale(x):
    ''' Rescale array to [-1, 1] '''
    x -= min(x)
    x /= 0.5 * max(abs(x))
    return x - 1


def generate_path(trial_number, length=30, fps=60.):
    # Sweet, Barbara Townsend, and Leonard J. Trejo.
    # "The identification and modeling of visual cue
    # usage in manual control task experiments." (1999).
    data = [[.50,   6,  0.16],
            [.50,  10,  0.26],
            [.50,  15,  0.39],
            [.50,  23,  0.60],
            [.50,  37,  0.97],
            [.50,  59,  1.54],
            [.05, 101,  2.64],
            [.05, 127,  3.32],
            [.05, 149,  3.90],
            [.05, 179,  4.69],
            [.05, 311,  8.14],
            [.05, 521, 13.64]]

    np.random.seed(trial_number)
    d = pd.DataFrame(data)

    offsets = np.random.uniform(-pi, pi, (100, 12))
    d[3] = offsets[trial_number]

    sines = d[:-4].apply(f, axis=1).T
    sines.index = t
    sines = sines.sum(axis=1)
    sines = rescale(sines).tolist()

    return sines

fps = 60.
t = np.linspace(0, 240, 240 * fps)

ks = [
      # Training
      {'trial': 1,
       'rand_id': 1,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 2,
       'rand_id': 2,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 3,
       'rand_id': 3,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 4,
       'rand_id': 4,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 5,
       'rand_id': 5,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 6,
       'rand_id': 1,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 7,
       'rand_id': 2,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 8,
       'rand_id': 3,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 9,
       'rand_id': 4,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 10,
       'rand_id': 5,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},

      # Experiment
      {'trial': 11,  # 28
       'rand_id': 10,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 12,  # 14
       'rand_id': 11,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 13,  # 19
       'rand_id': 10,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 14,  # 15
       'rand_id': 11,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 15,  # 24
       'rand_id': 12,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 16,  # 20
       'rand_id': 13,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 17,  # 11
       'rand_id': 12,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 18,  # 12
       'rand_id': 14,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 19,  # 23
       'rand_id': 13,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 20,  # 22
       'rand_id': 15,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 21,  # 25
       'rand_id': 14,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 22,  # 13
       'rand_id': 15,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 23,  # 18
       'rand_id': 16,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 24,  # 27
       'rand_id': 16,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 25,  # 26
       'rand_id': 17,
       'feedback': FEEDBACK_ON,
       'secondary_task': False},
      {'trial': 26,  # 17
       'rand_id': 17,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 27,  # 21
       'rand_id': 18,
       'feedback': FEEDBACK_ON,
       'secondary_task': True},
      {'trial': 28,  # 16
       'rand_id': 18,
       'feedback': FEEDBACK_ON,
       'secondary_task': False}
      ]

# {'trial': 11,
#  'feedback': FEEDBACK_OFF,
#  'secondary_task': True},
# {'trial': 12,
#  'feedback': FEEDBACK_FALSE,
#  'secondary_task': False},
# {'trial': 13,
#  'feedback': FEEDBACK_ON,
#  'secondary_task': True},
# {'trial': 14,
#  'feedback': FEEDBACK_OFF,
#  'secondary_task': False},
# {'trial': 15,
#  'feedback': FEEDBACK_FALSE,
#  'secondary_task': True},
# {'trial': 16,
#  'feedback': FEEDBACK_ON,
#  'secondary_task': False},
# {'trial': 17,
#  'feedback': FEEDBACK_OFF,
#  'secondary_task': True},
# {'trial': 18,
#  'feedback': FEEDBACK_FALSE,
#  'secondary_task': False},
# {'trial': 19,
#  'feedback': FEEDBACK_ON,
#  'secondary_task': True},
# {'trial': 20,
#  'feedback': FEEDBACK_OFF,
#  'secondary_task': False},
# {'trial': 21,
#  'feedback': FEEDBACK_FALSE,
#  'secondary_task': True},
# {'trial': 22,
#  'feedback': FEEDBACK_ON,
#  'secondary_task': False},
# {'trial': 23,
#  'feedback': FEEDBACK_OFF,
#  'secondary_task': True},
# {'trial': 24,
#  'feedback': FEEDBACK_FALSE,
#  'secondary_task': False},
# {'trial': 25,
#  'feedback': FEEDBACK_ON,
#  'secondary_task': True},
# {'trial': 26,
#  'feedback': FEEDBACK_OFF,
#  'secondary_task': False},
# {'trial': 27,
#  'feedback': FEEDBACK_FALSE,
#  'secondary_task': True},
# {'trial': 28,
#  'feedback': FEEDBACK_ON,
#  'secondary_task': False}
