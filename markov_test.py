from analysis import *

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pykov


def transition_matrix(data):
    result = np.zeros((2, 2))
    for (x, y), c in Counter(zip(data, data[1:])).iteritems():
        result[x-1, y-1] = c

    return result

fc = d.replace(('green', 'yellow'), (1, 0))
res = fc.groupby(('Subject', 'Trial')).FeedbackColor.apply(transition_matrix)
res_sum = res.sum()
res_dict = {('Good', 'Good'): res_sum[0, 0],
            ('Good', 'Bad'):  res_sum[0, 1],
            ('Bad', 'Good'):  res_sum[1, 0],
            ('Bad', 'Bad'):   res_sum[1, 1]}
res_chain = pykov.Chain(res_dict)
res_chain.stochastic()

for i in fc.Trial.unique():
    # Plot actual
    fc.query('Trial == @i').plot(x='index', y='FeedbackColor')

    # Generate and plot fake
    out = pd.Series(res_chain.walk(1800))
    out = out.replace(('Good', 'Bad'),
                      (1, 0))

    out.plot()
    plt.ylim(-.25, 1.25)
    plt.legend(['A', 'B'], loc='lower right')
    plt.savefig(str(i) + '.pdf')

# res_chain
# Chain([(('Good',  'Bad'), 0.0027697962776008903),
       # (( 'Bad', 'Good'), 0.01670378619153675),
       # (('Good', 'Good'), 0.99723020372239912),
       # (( 'Bad',  'Bad'), 0.98329621380846322)])
