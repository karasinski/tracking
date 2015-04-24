from __future__ import division, print_function
import pandas as pd
import os
import re
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


trials = {
    1: {
        'Feedback': False,
        'Visible': 'All',
        'Path': 0,
    },
    2: {
        'Feedback': True,
        'Visible': 'All',
        'Path': 0,
    },
    3: {
        'Feedback': False,
        'Visible': 'Some',
        'Path': 0,
    },
    4: {
        'Feedback': True,
        'Visible': 'Some',
        'Path': 0,
    },
    5: {
        'Feedback': False,
        'Visible': 'None',
        'Path': 0,
    },
    6: {
        'Feedback': True,
        'Visible': 'None',
        'Path': 0,
    },
    7: {
        'Feedback': False,
        'Visible': 'All',
        'Path': 1,
    },
    8: {
        'Feedback': True,
        'Visible': 'Some',
        'Path': 2,
    },
    9: {
        'Feedback': False,
        'Visible': 'None',
        'Path': 2,
    },
    10: {
        'Feedback': True,
        'Visible': 'Some',
        'Path': 1,
    },
    11: {
        'Feedback': False,
        'Visible': 'None',
        'Path': 1,
    },
    12: {
        'Feedback': False,
        'Visible': 'All',
        'Path': 2,
    },
    13: {
        'Feedback': True,
        'Visible': 'None',
        'Path': 1,
    },
    14: {
        'Feedback': True,
        'Visible': 'All',
        'Path': 1,
    },
    15: {
        'Feedback': True,
        'Visible': 'All',
        'Path': 2,
    },
    16: {
        'Feedback': False,
        'Visible': 'Some',
        'Path': 1,
    },
    17: {
        'Feedback': False,
        'Visible': 'Some',
        'Path': 2,
    },
    18: {
        'Feedback': True,
        'Visible': 'None',
        'Path': 2,
    }
}


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

    data = OrderedDict()
    for subj in subjs:
        s = OrderedDict()
        for path in file_paths:
            if subj in path:
                trial_number = path.split('/')[2].split(' ')[0]
                trial = pd.DataFrame.from_csv(path, header=None)
                trial[7] = np.sqrt(np.square(trial[3] - trial[2]))
                trial = trial.reset_index()
                s[trial_number] = trial
        data[subj] = pd.Panel.from_dict(s)
    d = pd.Panel4D.from_dict(data)
    d.minor_axis = ['Timestep', 'Input', 'Subject', 'Guidance', 'Timer', 'Secondary Task', 'Time', 'Error']
    d = d[natural_sort(d.labels)]
    return d


trials = pd.DataFrame.from_dict(trials).T.convert_objects(convert_numeric=True)
file_paths = get_filepaths('trials/')
d = load_data(file_paths)

# feedback = list(trials.query('Feedback').index)
# feedback = [str(trial) for trial in feedback]
# no_feedback = list(trials.query('not Feedback').index)
# no_feedback = [str(trial) for trial in no_feedback]

# none = list(trials.query('Visible == "None"').index)
# none = [str(trial) for trial in none]
# some = list(trials.query('Visible == "Some"').index)
# some = [str(trial) for trial in some]
# allv = list(trials.query('Visible == "All"').index)
# allv = [str(trial) for trial in allv]

# nf = d[:, no_feedback, :, 'Error'].mean().mean()
# f  = d[:,    feedback, :, 'Error'].mean().mean()
# subj_improvement = 100 * (nf - f) / nf

# print("\nRelative average improvement of {0:3.3f}%. \n".format(np.mean(subj_improvement)))
# print("Or by subject:")
# print(subj_improvement)
