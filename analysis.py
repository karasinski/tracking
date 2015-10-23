from __future__ import division, print_function
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from statsmodels.tools.eval_measures import rmse, vare
import pykov
import scipy

from trials import ks
from performance_methods import *




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
    trials = []
    for path in file_paths:
        trial = pd.DataFrame.from_csv(path)
        trial['Day'] = path.split('Day')[1].split('/')[0]
        trial['Subject'] = path.split('Subject')[1].split('/')[0]
        trial['Trial'] = path.split(' ')[0].split('/')[-1]
        trial = trial.reset_index()
        trials.append(trial)
    data = pd.concat(trials).convert_objects(convert_numeric=True)

    return data


def load_trials(ks):
    trials = pd.DataFrame(ks)
    trials = trials.fillna('0.')

    trials_columns = [el.title() for el in trials.columns.tolist()]
    trials.columns = trials_columns

    return trials


def find_response_times(d, t):
    d.KeyPress = d.groupby(('Day', 'Subject', 'Trial')).KeyPress.shift(-1)
    d = d.sort(['Day', 'Subject', 'Trial', 'Time'])

    res = []
    # For each subject and trial with a secondary task
    for day in d.Day.unique():
        for subject in d.Subject.unique():
            for trial in d.Trial.unique():
                # Use the same code as in the experiment to generate start and end times for each light
                rand_id = int(t.query('Trial == @trial').Trial.tolist()[0])
                np.random.seed(rand_id)
                color_times = np.arange(5, 60, 7, dtype=np.float)
                color_times += 4 * np.random.rand(len(color_times))
                color_times = np.append(color_times, 60)

                # For each of the eight lights
                for i in range(0, 8):
                    print(day, subject, trial, i)
                    start_time = color_times[i]
                    end_time = color_times[i+1]
                    window_time = end_time - start_time

                    # Select data for subject and trial within start and end times for comm light
                    curr = d.query('Day == @day and Subject == @subject and Trial == @trial')
                    curr = curr.query('@start_time < Time <= @end_time')["SecondaryColor"].abs()

                    # Find maximum time in window
                    n = curr.isnull()
                    clusters = (n != n.shift()).cumsum()
                    response_time = (curr.groupby(clusters).cumsum() * 1/60.).max()

                    # if the response time is approx the available time, they didn't respond
                    if abs(window_time - response_time) < 1/60.:
                        response_time = np.NaN

                    # append the response time to that light
                    res.append(pd.Series([day, subject, trial, i + 1, response_time]))

    results = pd.DataFrame(res)
    results.columns = ['Day', 'Subject', 'Trial', 'CommID', 'ResponseTime']
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


def gen_results(d):
    #exp = d.query('Trial > 10 and Time > 5')
    exp = d.query('Time > 5')

    error = exp.groupby(('Day', 'Subject', 'Trial')).ae.mean().reset_index()
    error.columns = ['Day', 'Subject', 'Trial', 'AbsoluteError']
    rms = exp.groupby(('Day', 'Subject', 'Trial')).apply(lambda x: rmse(x.y, x.yg)).reset_index()
    rms.columns = ['Day', 'Subject', 'Trial', 'RMSE']
    var = exp.groupby(('Day', 'Subject', 'Trial')).apply(lambda x: vare(x.y, x.yg)).reset_index()
    var.columns = ['Day', 'Subject', 'Trial', 'VARE']
    crossings = exp.groupby(('Day', 'Subject', 'Trial')).apply(lambda x: len(cross(x.e))).reset_index()
    crossings.columns = ['Day', 'Subject', 'Trial', 'Crossings']

    rt = find_response_times(exp, trials)
    response_time = rt.groupby(('Day', 'Subject', 'Trial')).mean().ResponseTime.reset_index()
    td = exp.groupby(('Day', 'Subject', 'Trial')).apply(lambda x: recover_shift(x['Time'], x['y'], x['yg'])).reset_index()
    time_delay = td.groupby(('Day', 'Subject', 'Trial')).mean()[0].reset_index().abs()
    time_delay.columns = ['Day', 'Subject', 'Trial', 'LagTime']
    #entropy = generate_entropy_results(exp)

    #res = error.merge(rms).merge(var).merge(crossings).merge(response_time, how='outer').merge(time_delay).merge(entropy)
    res = error.merge(rms).merge(var).merge(crossings).merge(response_time, how='outer').merge(time_delay)
    res['Feedback'] = res.Subject % 2 == 1
    res = res.merge(trials[['Trial']])
    res = res.sort(['Day', 'Subject', 'Trial'])
    #res['SecondaryTask'] = res['Secondary_Task']
    res = res[['Day', 'Subject', 'Trial', 'AbsoluteError', 'RMSE', 'VARE', 'ResponseTime', 'LagTime', 'Crossings', 'Feedback']]
    res = res.reset_index(drop=True)
    res['ID'] = (res.Day - 1) * res.Trial.max() + res.Trial

    return res


def find_significance(res):
    for col in res.columns[2:-1]:
        z, p = scipy.stats.ranksums(*[data[col].as_matrix() for group, data in res.groupby('Feedback')])
        g1, g2 = [(group, data[col].as_matrix().mean()) for group, data in res.groupby('Feedback')]
        print("{0:15} p: {1:.2f} | {2[0]}: {2[1]:.2f}, {3[0]}: {3[1]:.2f}".format(col, p, g1, g2))


def residuals(p, y, x):
    A = p[0]
    B = p[1]
    C = p[2]
    err = abs(np.array(y - (A * np.exp(-x/B) + C))).mean()
    return err


def peval(x, p):
    return p[0] * np.exp(-x/p[1]) +p[2]


def make_fits(res):
    from scipy.optimize import minimize

    results = []
    for s in res.Subject.unique():
        s = int(s)
        x = res.query('Subject == @s').ID
        y = res.query('Subject == @s').RMSE

        p0 = np.array([0.05, 5, 0.1])
        plsq = minimize(residuals, p0,
                        args=(y, x), method='nelder-mead',
                        options={'maxiter': 1E6, 'maxfev': 1E6,
                                 'xtol': 1e-8, 'disp': True})
        r = plsq.x.tolist()
        r.insert(0, s)
        results.append(r)

        plt.plot(x, peval(x, plsq.x), x, y, 'o')
        ##plt.plot(x, peval(x, plsq[0]), label='Subject ' + str(s))

        plt.title('Subject ' + str(s))
        plt.xlabel('Trial')
        plt.ylabel('RMSE')
        ##plt.legend(loc='best')
        ##plt.show()
        plt.savefig('test_imgs/' + str(s) + '_fit.pdf')
        plt.clf()

    results = pd.DataFrame(results)
    results.columns = ['Subject', 'A', 'B', 'C']
    results['Feedback'] = results.Subject % 2 == 1

    return results


def significance(res):
    from scipy.stats import ttest_ind
    print('Performance')
    for d in res.Day.unique():
        rmse = ttest_ind(res.query('Day == @d and not Feedback').RMSE,
                         res.query('Day == @d and Feedback').RMSE)
        NFB_FB = (res.query('Day == @d and not Feedback').RMSE.mean() -
                  res.query('Day == @d and Feedback').RMSE.mean())
        print('Day {d:02d}: RMSE | NFB-FB: {NFB_FB:+.3e}, t-statistic: {rmse.statistic:+.2f}, p: {rmse.pvalue:.2f}'.format(d=d, NFB_FB=NFB_FB, rmse=rmse))

    print('\nWorkload')
    for d in res.Day.unique():
        rt = ttest_ind(res.query('Day == @d and not Feedback').ResponseTime,
                       res.query('Day == @d and Feedback').ResponseTime)
        NFB_FB = (res.query('Day == @d and not Feedback').ResponseTime.mean() -
                  res.query('Day == @d and Feedback').ResponseTime.mean())
        print('Day {d:02d}:   RT | NFB-FB: {NFB_FB:+.3e}, t-statistic: {rt.statistic:+.2f}, p: {rt.pvalue:.2f}'.format(d=d, NFB_FB=NFB_FB, rt=rt))

    print('\nLearning Parameters')
    r = make_fits(res)
    A = ttest_ind(r.query('not Feedback').A, r.query('Feedback').A)
    B = ttest_ind(r.query('not Feedback').B, r.query('Feedback').B)
    C = ttest_ind(r.query('not Feedback').C, r.query('Feedback').C)

    NFB_FB = (r.query('not Feedback').A.mean() - r.query('Feedback').A.mean())
    print('           A | NFB-FB: {NFB_FB:+.3e}, t-statistic: {A.statistic:+.2f}, p: {A.pvalue:.2f}'.format(d=d, NFB_FB=NFB_FB, A=A))
    NFB_FB = (r.query('not Feedback').B.mean() - r.query('Feedback').B.mean())
    print('           B | NFB-FB: {NFB_FB:+.3e}, t-statistic: {B.statistic:+.2f}, p: {B.pvalue:.2f}'.format(d=d, NFB_FB=NFB_FB, B=B))
    NFB_FB = (r.query('not Feedback').C.mean() - r.query('Feedback').C.mean())
    print('           C | NFB-FB: {NFB_FB:+.3e}, t-statistic: {C.statistic:+.2f}, p: {C.pvalue:.2f}'.format(d=d, NFB_FB=NFB_FB, C=C))



trials = load_trials(ks)
file_paths = get_filepaths('trials/')
data = load_data(file_paths)
d = data.merge(trials, how='inner').convert_objects(convert_numeric=True)
d['e'] = d.y - d.yg
d['ae'] = abs(d.y - d.yg)
d['Time'] = (d['index'] + 1) / 60.

#res = gen_results(d)
#find_significance(res)
#results = make_fits(res)
