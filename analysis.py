from __future__ import division, print_function
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import pykov

from trials import ks


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    file_paths = natural_sort(file_paths)
    return file_paths


def load_data(file_paths):
    subjs = []
    for path in file_paths:
        subjs.append(path.split('/')[1])
    subjs = list(set(subjs))

    data = pd.DataFrame()
    for subj in subjs:
        for path in file_paths:
            current_subj = path.split('/')[1]
            if subj == current_subj:
                trial_number = path.split('/')[2].split(' ')[0]
                trial = pd.DataFrame.from_csv(path)
                trial['Subject'] = subj.split('Subject')[1]
                trial['Trial'] = trial_number
                trial = trial.reset_index()
                data = pd.concat((data, trial))

    data = data.convert_objects(convert_numeric=True)
    return data


def load_trials(ks):
    trials = pd.DataFrame(ks)
    trials = trials.fillna('0.')

    trials_columns = [el.title() for el in trials.columns.tolist()]
    trials.columns = trials_columns

    return trials


def find_response_times(d):
    d.KeyPress = d.groupby(('Subject', 'Trial')).KeyPress.shift(-1)
    d['Time'] = (d['index'] + 1) / 60.
    d = d.sort(['Subject', 'Trial', 'Time'])

    res = []
    # For each subject and trial with a secondary task
    for subject in d.Subject.unique():
        for trial in d.query('Secondary_Task').Trial.unique():
            # Use the same code as in the experiment to generate start and end times for each light
            np.random.seed(trial)
            color_times = np.arange(5, 30, 5, dtype=np.float)
            color_times += 2 * np.random.rand(len(color_times))
            color_times = np.append(color_times, 30)

            # For each of the five lights
            for i in range(0, 5):
                start_time = color_times[i]
                end_time = color_times[i+1]
                window_time = end_time - start_time

                # Select data for subject and trial within start and end times for comm light
                curr = d.query('Subject == @subject and Trial == @trial')
                curr = curr.query('@start_time < Time <= @end_time')["SecondaryColor"].abs()

                # Find maximum time in window
                n = curr.isnull()
                clusters = (n != n.shift()).cumsum()
                response_time = (curr.groupby(clusters).cumsum() * 1/60.).max()

                # if the response time is approx the available time, they didn't respond
                if abs(window_time - response_time) < 1/60.:
                    response_time = np.NaN

                # append the response time to that light
                res.append(pd.Series([subject, trial, i + 1, response_time]))

    results = pd.DataFrame(res)
    results.columns = ['Subject', 'Trial', 'CommID', 'ResponseTime']
    return results


def transition_matrix(x):
    a = x.FeedbackColor.replace(['green', 'yellow'], [1, 0]).values
    b = np.zeros((2, 2))
    for (x, y), c in Counter(zip(a, a[1:])).iteritems():
        b[x - 1, y - 1] = c

    green_green = b[0][0]
    green_yellow = b[0][1]
    yellow_green = b[1][0]
    yellow_yellow = b[1][1]

    chain = pykov.Chain({( 'green', 'yellow'): green_yellow,
                         ('yellow',  'green'): yellow_green,
                         ( 'green',  'green'): green_green,
                         ('yellow', 'yellow'): yellow_yellow})
    return chain


def find_average_markov(df):
    d = df[['Subject', 'Trial', 'Time', 'FeedbackColor']]
    d = d.drop_duplicates()
    d = d.sort(['Subject', 'Trial', 'Time'])

    res = d.groupby(('Subject', 'Trial')).apply(lambda x: transition_matrix(x))
    r = res.reset_index()[0].sum()
    r.stochastic()
    return r


trials = load_trials(ks)
file_paths = get_filepaths('trials/')
data = load_data(file_paths)
d = data.merge(trials, how='inner').convert_objects(convert_numeric=True)
df = d.merge(find_response_times(d), how='outer')
