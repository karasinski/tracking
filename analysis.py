from __future__ import division, print_function
import pandas as pd
import os
import re
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
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

    data = OrderedDict()
    for subj in subjs:
        s = OrderedDict()
        for path in file_paths:
            if subj in path:
                trial_number = path.split('/')[2].split(' ')[0]
                trial = pd.DataFrame.from_csv(path, header=None)
                trial[6] = trial[6].diff()
                trial[7] = np.sqrt(np.square(trial[3] - trial[2]))
                trial = trial.reset_index()
                s[trial_number] = trial
        data[subj] = pd.Panel.from_dict(s)
    d = pd.Panel4D.from_dict(data)
    d.minor_axis = ['Timestep', 'Input', 'Actual', 'Guidance',
                    'Timer', 'Secondary Task', 'PaceError', 'Error']
    d = d[natural_sort(d.labels)]
    return d


def load_data2(file_paths):
    subjs = []
    for path in file_paths:
        subjs.append(path.split('/')[1])
    subjs = list(set(subjs))

    data = pd.DataFrame()
    for subj in subjs:
        for path in file_paths:
            if subj in path:
                trial_number = path.split('/')[2].split(' ')[0]
                trial = pd.DataFrame.from_csv(path, header=None)
                trial[7] = trial[7] - trial[6]
                trial[6] = trial[6].diff()
                trial[8] = np.sqrt(np.square(trial[3] - trial[2]))
                trial[9] = subj.split('Subject')[1]
                trial[10] = trial_number
                trial = trial.reset_index()
                trial.columns = ['Timestep', 'Input', 'Actual', 'Guidance',
                                 'Timer', 'Secondary Task', 'PaceError', 'LoopTime',
                                 'Error', 'Subject', 'Trial']
                data = pd.concat((data, trial))

    data = data.reset_index().convert_objects(convert_numeric=True)
    return data


def load_trials(ks):
    trials = pd.DataFrame(ks)
    names = []
    for k in ks:
        try:
            names.append(k['funckwds']['name'])
        except:
            names.append(0)

    trials['funckwds'] = names
    trials = trials.fillna('0.')

    trials_columns = [el.title() for el in trials.columns.tolist()]
    trials.columns = trials_columns

    return trials

trials = load_trials(ks)
file_paths = get_filepaths('trials/')
data = load_data2(file_paths)
d = data.merge(trials, how='inner')