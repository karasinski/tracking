import numpy as np
import matplotlib.pyplot as plt
import os


def ShortLongColor(subj, d, short_span=60, long_span=10000, save=False):
    short_g, short_y = GenColors(d, short_span)
    long_g, long_y = GenColors(d, long_span)

    GenColorPlot(subj, d, short_g, short_y, long_g, long_y, save)


def GenColors(d, span=60):
    # Find first and last indexes
    last = len(d)
    first = 0

    epsilon = d[:, 1] - d[:, 2]

    greens, yellows = [], []
    for t in xrange(first, last + 1):
        if t - span < first:
            low = first
        else:
            low = t - span

        green = abs(epsilon[low:t + 1]) < .05
        green = green.mean()
        yellow = abs(epsilon[low:t + 1]) < .15
        yellow = yellow.mean()

        greens.append(green), yellows.append(yellow)

    return greens, yellows


def GenColorPlot(subj, d, short_g, short_y, long_g, long_y, save):
    time = d[:, 0]
    epsilon = d[:, 1] - d[:, 2]

    # Flip our percentages
    short_g = 1 - np.array(short_g)[1:]
    short_y = 1 - np.array(short_y)[1:]
    long_g = 1 - np.array(long_g)[1:]
    long_y = 1 - np.array(long_y)[1:]

    # Plot short duration map
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.fill_between(time, 0, short_y,       color='r', facecolor='red')
    ax1.fill_between(time, short_y, short_g, color='y', facecolor='yellow')
    ax1.fill_between(time, short_g, 1,       color='g', facecolor='green')
    ax1.set_ylim([0, 1])
    plt.setp(ax1.get_yticklabels(), visible=False)

    # Add Error plot on top
    ax1twinx = ax1.twinx()
    plt.plot(time, abs(epsilon), 'k')
    ax1twinx.set_ylabel('$|Error|$', color='k')
    for tl in ax1twinx.get_yticklabels():
        tl.set_color('k')
    plt.xlim([time[0], time[-1]])
    ax1twinx.set_ylim(bottom=0)
    plt.title(subj)

    # Plot long duration map
    ax2.fill_between(time, 0, long_y,      color='r', facecolor='red')
    ax2.fill_between(time, long_y, long_g, color='y', facecolor='yellow')
    ax2.fill_between(time, long_g, 1,      color='g', facecolor='green')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.xlim([time[0], time[-1]])
    ax2.set_xlabel('Time (s)')
    plt.tight_layout()
    save_fig('Metric X - ' + subj, save)


def Performance(d, save=False):
    t, y, yg = d[:, 0], d[:, 1], d[:, 2]
    plt.plot(t, y, label='Subject')
    plt.plot(t, yg, label='Guidance')
    plt.xlim(0, t[-1])
    plt.legend()
    save_fig('performance', save)


def save_fig(save_name, save):
    save_name = save_name.replace("$", "") + '.pdf'

    # Show it or save it
    if not save:
        plt.show()
    else:
        try:
            os.mkdir('figures')
        except Exception:
            pass

        plt.savefig('figures/' + save_name, bbox_inches='tight')
    plt.close()
