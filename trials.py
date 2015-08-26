from __future__ import division, print_function
import numpy as np
from numpy import pi
import pandas as pd


FEEDBACK_ON    =  1
FEEDBACK_OFF   =  0
FEEDBACK_FALSE = -1


def f(x, t):
    return pd.Series((x[0] * 2*pi*x[1] / 240.) * np.cos(2*pi*x[1] * t / 240. + x[3]))


def rescale(x):
    ''' Rescale array to [-1, 1] '''
    x -= min(x)
    x /= 0.5 * max(abs(x))
    return x - 1


def generate_path(trial_number, length=240, fps=60):
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

    t = np.linspace(0, length, length * fps)
    sines = d[:-4].apply(f, args=(t,), axis=1).T
    sines.index = t
    sines = sines.sum(axis=1)
    sines = rescale(sines).tolist()

    return sines


ks = [{'trial': 1},
      {'trial': 2},
      {'trial': 3},
      {'trial': 4},
      {'trial': 5},
      {'trial': 6},
      {'trial': 7},
      {'trial': 8},
      {'trial': 9},
      {'trial': 10},
      {'wait': 5*60},
      {'trial': 11},
      {'trial': 12},
      {'trial': 13},
      {'trial': 14},
      {'trial': 15},
      {'trial': 16},
      {'trial': 17},
      {'trial': 18},
      {'trial': 19},
      {'trial': 20}]
