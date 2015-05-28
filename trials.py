from __future__ import division, print_function
import numpy as np
from numpy import pi
import pandas as pd
# import matplotlib.pyplot as plt


FEEDBACK_ON = 1
FEEDBACK_OFF = 0
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

    # sines = d.apply(f, axis=1)
    sines = d[:-4].apply(f, axis=1).T
    sines.index = t
    sines = sines.sum(axis=1)
    sines = rescale(sines).tolist()

    return sines

fps = 60.
t = np.linspace(0, 240, 240 * fps)
# sines = generate_path(1)
# sines.plot()
# plt.show()

ks = [
      # Refresher
      {'trial': 1,
       'feedback': FEEDBACK_OFF},
      # {'trial': 2,
      #  'feedback': FEEDBACK_FALSE},
      # {'trial': 3,
      #  'feedback': FEEDBACK_ON},
      # # Experiment
      # {'trial': 4,
      #  'feedback': FEEDBACK_OFF},
      # {'trial': 5,
      #  'feedback': FEEDBACK_FALSE},
      # {'trial': 6,
      #  'feedback': FEEDBACK_ON},
      # {'trial': 7,
      #  'feedback': FEEDBACK_OFF},
      # {'trial': 8,
      #  'feedback': FEEDBACK_FALSE},
      # {'trial': 9,
      #  'feedback': FEEDBACK_ON},
      # {'trial': 10,
      #  'feedback': FEEDBACK_OFF},
      # {'trial': 11,
      #  'feedback': FEEDBACK_FALSE},
      # {'trial': 12,
      #  'feedback': FEEDBACK_ON},
      # {'trial': 13,
      #  'feedback': FEEDBACK_OFF},
      # {'trial': 14,
      #  'feedback': FEEDBACK_FALSE},
      # {'trial': 15,
      #  'feedback': FEEDBACK_ON},
      # {'trial': 16,
      #  'feedback': FEEDBACK_OFF},
      # {'trial': 17,
      #  'feedback': FEEDBACK_FALSE},
      # {'trial': 18,
      #  'feedback': FEEDBACK_ON}
      ]
