from __future__ import print_function
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (decimate, lfilter, buttord, butter, BadCoefficients,
                          lti, correlate)


warnings.filterwarnings('ignore', category=BadCoefficients)


def cross(series, cross=0, direction='cross'):
    """
    Given a Series returns all the index values where the data values equal
    the 'cross' value.

    Direction can be 'rising' (for rising edge), 'falling' (for only falling
    edge), or 'cross' for both edges
    """
    # Find if values are above or bellow yvalue crossing:
    above = series.values > cross
    below = np.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings = []
    # Find indexes on left side of crossing point
    if direction == 'rising':
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == 'falling':
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]

    # Calculate x crossings with interpolation using formula for a line:
    x1 = series.index.values[idxs]
    x2 = series.index.values[idxs+1]
    y1 = series.values[idxs]
    y2 = series.values[idxs+1]
    x_crossings = (cross-y1)*(x2-x1)/(y2-y1) + x1

    return x_crossings


def recover_shift(t, A, B):
    t = np.array(t)

    # calculate cross correlation of the two signals
    xcorr = correlate(A, B)

    # The peak of the cross-correlation gives the shift between the two signals
    # The xcorr array goes from -nsamples to nsamples
    dt = np.linspace(-t[-1], t[-1], 2*len(t)-1)
    recovered_time_shift = dt[xcorr.argmax()]
    if recovered_time_shift > 1 or recovered_time_shift == 0:
        recovered_time_shift = np.NaN

    return recovered_time_shift


def plot_original_and_shifted(t, A, B, shift=None):
    if not shift:
        shift = recover_shift(t, A, B)
    t = np.array(t)
    dt = t[1] - t[0]

    discrete_shift = int(shift/dt)
    shifted_B = np.roll(B, discrete_shift)
    ads = abs(int(discrete_shift))

    plt.plot(t, A, 'r', t[ads:], shifted_B[ads:], 'b')
    plt.margins(0.1)
    plt.legend(['A', 'Shifted B'])
    plt.title('Shift of: ' + str(shift))
    plt.show()


class CDF:
    """
    cumulative distribution function container object
    Parameters
    ----------
    bin_edges : array_like
        sorted in ascending order. ``(length(percentiles)+1)``
    percentiles : array_like
        percentile values of CDF (0-1)
    """
    def __init__(self, bin_edges, percentiles):
        self.bin_edges = np.array(bin_edges)
        self.percentiles = np.array(percentiles)

    def find(self, percentile):
        """
        Returns the bin value at the given percentile
        Parameters
        ----------
        percentile : float
            Uses np.searchsorted to find closest index larger than percentile
            in self.percentiles. Returns bin value at that index.
            value should be between 0 and 1.
        Returns
        -------
        binatpercentile : float
            Value where CDF is at the supplied percentile
        """
        if percentile < 0. or percentile > 1.:
            raise ValueError('percentile must be between 0 and 1')

        return self.bin_edges[np.searchsorted(self.percentiles, percentile)]

    def plot(self):
        """
        Returns a matplotlib Figure object with CDF plotted
        Parameters
        ----------
        None
        Returns
        -------
        fig : matplotlib.figure.Figure
        See Also
        --------
        percentile : Calculates CDF percentiles for input u
        percentile
        Example
        -------
        >>> import numpy as np
        >>> u = np.random.normal(size=(1000,))
        >>> cdf = percentile(u, numbins=100)
        >>> fig = cdf.plot()
        >>> fig.show()
        """
        import matplotlib.pyplot as plt
        plt.rc('font', family='serif')

        fig = plt.figure(figsize=(5, 5))
        fig.subplots_adjust(left=.12)

        ax = fig.add_subplot(111)
        ax.plot(self.bin_edges[:-1], self.percentiles)
        ax.set_title('Cumumlative Distribution Function')
        ax.set_ylabel('Percentiles')
        ax.set_ylim([0, 1])

        return fig

    def __repr__(self):
        return ('CDF(%s, %s)'%(repr(self.bin_edges), repr(self.percentiles)))\
               .replace('np.','').replace('array', 'np.array')

    def __str__(self):
        st = ['  Bins       Percentiles\n',
               '------------------------\n']
        if len(self.percentiles) > 100:
            indices = range(10) + range(-10,0)
        else:
            indices = range(len(self.percentiles))

        for i in indices:
            if i == -10:
                st.append('      .         .\n'*3)

            _bin, _per = self.bin_edges[i], self.percentiles[i]
            st.append('{: >10}'.format(('{: .3f}'.format(_bin))))
            st.append('{: >10}'.format(('{: .2f}\n'.format(100*_per))))

        return ''.join(st)


def percentile(u, numbins=None):
    """
    Calculates CDF percentiles for input u
    Parameters
    ----------
    u : array_like
        Input data.
    numbins : None or int, optional
        Specifies the number of bins in the CDF object.
        If None than the number of bins become the length
        of u after removing nans. Otherwise numbins specifies
        the number of bins.
    See Also
    --------
    CDF : cumulative distribution function container object
    CDF
    Example
    -------
    >>> import numpy as np
    >>> u = np.random.normal(size=(1000,))
    >>> cdf = percentile(u, numbins=100)
    """
    u = np.array(u)
    u = u[~np.isnan(u)]
    u.sort()

    if numbins is None:
        numbins = len(u)

    stop = u[-1] + (u[-1]-u[0])/float(numbins)
    x = np.linspace(u[0], stop, numbins+1)
    y = np.cumsum(u+u[0])
    y = np.interp(x[:-1], u, y)
    y /= y[-1]

    return CDF(x,y)


def arburg(x, order):
    """
    Autoregressive (AR) all-pole model parameters estimated using Burg method
    Parameters
    ----------
    x : array_like
        Input data. real-valued
    order : int
        model order
    Returns
    -------
    returns AR coefficients
    """
    x = np.array(x)
    N = len(x)
    rho = np.sum(np.abs(x)**2.) / N
    den = rho * 2. * N

    ef = np.array(x)  # forward errors
    eb = np.array(x)  # backward errors

    # AR coeffs
    a = np.ones(1)

    # reflection coeffs
    ref = np.zeros(order)

    E = np.zeros(order+1)
    E[0] = rho

    for m in range(order):
        # reflection coefficient
        efp = ef[1:]
        ebp = eb[0:-1]
        num = -2. * np.dot(ebp.T, efp)
        den = np.dot(efp.T, efp)
        den += np.dot(ebp, ebp.T)
        ref[m] = num / den

        # forward and backward prediction errors
        ef = efp + ref[m] * ebp
        eb = ebp + ref[m].T * efp

        # AR coefficients
        a.resize(len(a)+1)
        a += ref[m] * a[::-1]

        # prediction error
        E[m+1] = (1. - ref[m].T*ref[m]) * E[m]

    return a


class SteeringEntropyModel:
    def __init__(self, **kwds):
        """
        SteeringEntropyModel(**kwds)
        Steering Entropy Model as described by Boer, Rakauskas, Ward, and
        Goodrich (2005).
        Model is calibrated to the unique spectral characteristics of the
        driver. This makes it sensitive to a variety of coping strategies
        employed by drivers in response to increased workload demands.
        A model should be built for each participant. A baseline segment
        of steering angle data is needed to fit the model via the baseline
        method. This segment of data should be at least 2 minutes.
        baseline() will raise an exception if less than 1 minute of data
        is supplied.
        Parameters
        ----------
        resample_fs : float, optional
            Rate in Hz to resample input data. Default is 4. According to
            Borg et al. this rate is optimizes the senstivity of the
            entropy estimates.
        alpha : float, optional
            controls the spacing of the bins used to estimate the
            prediction errors distribution. Must be between 0-1. The
            default value is 0.2. According to Borg et al. this rate
            is optimizes the senstivity of the entropy estimates.
        M : int, optional
            Controls the number of bins used to estimate the prediction
            errors distribution. Number of bins is 2*M + 2. Default is 6
            (resulting in 14 bins). This is set to avoid having an
            excess of empty counts when the data set is small.
        b_pe : array, optional
            AR coefficients
            (initialized when fit_baseline is called)
        pkbas : array, optional
            baseline prediction error density coefficients
            cooresponding to bins (initialized when fit_baseline is called)
        bin_edges : array, optional
            baseline prediction error density bin edges corresponding
            to pkbas (initialized when fit_baseline is called)
            len(bins) == len(pkbas) + 1
        hbas : float, optional
            baseline entropy estimate (initialized when fit_baseline is called)
        cdf : CDF, optional
            Object holding the cumulative density function data of the
            baseline prediction errors. Used to build bins.
            (initialized when fit_baseline is called)
        Example
        -------
        >>> se_model = SteeringEntropyModel()
        >>> baseline_entropy = se_model.fit_baseline(baseline_steering_data)
        >>> task_entropy = se_model.get_entropy(task_steering_data)
        References
        ----------
        Boer, E. R., Rakauskas, M. E., Ward, N. J., and Goodrich. M. A.
            (2005). Steering Entropy Revisited.  Proceedings of the
            Third International Driving Symposium on Human Factors in
            Driver Assessment, Training and Vehicle Design.  June 27-30,
            2005.  Rockport, Maine.
        Boer, E. R., Behavioral entropy as an index of workload
            Proceedings of the IEA 2000 /  HFES 2000 Congress.
        Nakayama, O., Futami, T., Nakamura T., and Boer, E. R.,
            Development of a steering entropy method for evaluating
            driver workload. SAE Technical Paper Series #1999-01-0892.
            Presented at the International Congress and Exposition,
            Detriot, Michigan. March 1-4, 1999.
        Notes
        -----
        Expecting data to be in degrees. It may not matter but small
        values will sometimes cause overflow errors in arburg. If you
        your steering data is in radians it might be better to convert
        it to degrees.
        Loosely based off of Chris' SteeringEntropy.m script in ndaqTools.
        Users familiar with ndaqTools should note that the steeringentropy
        function in ndaqTools (Chris' matlab) uses the first 60 s to find
        the AR coeffs and the second 60 s to find baseline prediction error
        distribution. This module splits the baseline data in half. This
        class applies a 5th order lowpass to 3/7 the resample_fs as
        suggested by Boer et al. before resampling. The ndaqTools algorithm
        just applies an integer factor downsample. Because of these slight
        differences entropy estimates may differ.
        """
        self.resample_fs = kwds.get('resample_fs', 4.)
        self.alpha = kwds.get('alpha', 0.2)
        self.M = kwds.get('M', 6)
        self.b_pe = kwds.get('b_pe', None)
        self.pkbas = kwds.get('pkbas', None)
        self.bin_edges = kwds.get('bin_edges', None)
        self.hbas = kwds.get('hbas', None)
        self.cdf = kwds.get('cdf', None)

    def _build_lpfilter(self, fs):
        """
        builds low-pass filter with a cutoff frequency of 3/7th the resample
        frequency. The filter should be down 40 dB at 1.5 times the cutoff
        frequency (6/7th) the resample frequency.
        Parameters
        ----------
        fs : the base sampling rate
        Returns
        -------
        b, a : array_like
            Numerator (b) and denominator (a) polynomials of the IIR filter.
        """
        nyq = fs/2. # nyquist frequency
        cutoff = (3./7.)*self.resample_fs # cutoff freq defined by Boer
        wp = cutoff * nyq # pass edge freq (pi radians / sample)
        ws = wp*2.        # pass edge freq (pi radians / sample)
        gpass = 1.5       # The maximum loss in the passband (dB)
        gstop = 40        # The minimum attenuation in the stopband (dB)
        n, wn = buttord(wp, ws, gpass, gstop)
        #print('n =',n,'wn =',wn)
        b, a = butter(n, wn, analog=True)

        return b, a

    def _resample(self, x, fs):
        """
        Private method applies 5th low-pass Butterworth filter to x
        and resamples and returns x at self.resample_fs.
        Parameters
        ----------
        x : array_like
            Input data.
        fs : float
            sampling rate of x in Hz
        Returns
        -------
        x_resampled : np.ndarray
            low-passed and resampled copy of x
        """
        b,a = self._build_lpfilter(fs)
        return decimate(lfilter(b, a, x).real, int(fs/self.resample_fs))

    def fit_baseline(self, x, fs=60., ignore_exception=False, _tsplot=False):
        """
        fit_baseline(self, x[, fs=60.][, _tsplot=False])
        Finds AR coefficients to build prediction errors and estimates baseline
        prediction errors distribution. Returns the baseline entropy.
        Parameters
        ----------
        x : array_like
            Input data. Should specify steering wheel angle over at least 120 s.
            x[:N/2] is used to find the AR coefficients.
            x[N/2:] is used to build the baseline prediciton error distribution.
        fs : float
            sampling rate of x in Hz. Default is 60 Hz. Be sure to specify fs if
            x was not sampled at 60 Hz.
        Returns
        -------
        hbas : float
            baseline entropy
        """

        # unpack relevant parameters
        N = len(x)
        M = self.M
        alpha = self.alpha
        resample_fs = self.resample_fs

        # apply LP and downsample signal
        _x = self._resample(x, fs)
        Nx = len(_x)

        if not ignore_exception and len(_x) < resample_fs * 60:
            raise Exception('Need at least 60 seconds of steering '
                            'data to calculate baseline\n(ideally '
                            'need ~ 120 seconds of data).')

        # use first half of x to find coefficients of AR model
        # using Burg's method
        #
        # sign on coeffs returned by arburg is reversed
        b_pe = arburg(_x[:Nx/2], 3)

        # use second half of x to calculate baseline entropy
        # use AR coefficents to build prediction errors
        PE = lfilter(b_pe, 1, _x[Nx/2:])

        # need to build bins to approximate the PE distribution
        cdf = percentile(PE) # cdf is a CDF object
        pe_alpha = 0.5*(abs(cdf.find(alpha)) + abs(cdf.find(1.-alpha)))
        bin_edges = [-10e12]+list(np.linspace(-M, M, 2*M+1)*pe_alpha)+[10e12]

        # now we can calculate the baseline entropy
        # np.histogram returns a tuple of histogram counts and bin_edges
        Pk = np.histogram(PE, bins=bin_edges)[0]/float(len(PE))

        # replace small values to avoid excessively high entropy
        Pk[Pk < 1e-3] = 1e-3

        # and the baseline entropy is
        hbas = np.sum(np.multiply(-Pk, np.log2(Pk)))

        # store the relevant information
        self.bin_edges = bin_edges
        self.b_pe = b_pe
        self.cdf = cdf
        self.pkbas = Pk
        self.hbas = hbas

        if _tsplot:
            return hbas, self._baseline_tsplot(x[N/2:], _x[Nx/2:], PE, fs)

        return hbas

    def get_entropy(self, x, fs=60., _pedistplot=False):
        """
        get_entropy(self, x[, fs=60.][, _pedistplot=False])
        Returns steering entropy from given data vector. Specify fs if not
        collected at 60 Hz.
        Parameters
        ----------
        x : array_like
            Input data. Should specify steering wheel angle over at least 60 s
        fs : float
            sampling rate of x in Hz. Default is 60 Hz.
            Be sure to specify fs if x was not sampled at 60 Hz.
        Returns
        -------
        hbas : float
            baseline entropy
        """
        bin_edges = self.bin_edges
        pkbas = self.pkbas
        b_pe = self.b_pe

        if b_pe is None:
            raise Exception('must first find build baseline AR model')

        # apply LP and downsample signal
        _x = self._resample(x, fs)

        # Use AR coefficents to build prediction errors
        PE = lfilter(self.b_pe, 1, _x)

        # Calculate and entropy
        Pk = np.histogram(PE, bins=bin_edges)[0]/float(len(PE))
        Pk[Pk < 1e-3] = 1e-3
        h = np.sum(np.multiply(-Pk, np.log2(pkbas)))

        if _pedistplot:
            return h, self._pedistplot(Pk)

        return h

    def _lpfilter_bode(self):
        """
        returns bode plot of the lp filter applied before downsampling
        Parameters
        ----------
        None
        Returns
        -------
        fig : matplotlib.figure.Figure
        """

        b, a = self._build_lpfilter(60.)
        w, mag, phase = lti(b,a).bode()

        import matplotlib.pyplot as plt
        plt.rc('font', family='serif')
        fig = plt.figure(figsize=(6,6))
        fig.subplots_adjust(bottom=.125, top=.92, right=.95, hspace=0.1)

        xticks = [.01, .03, .1, .3, 1., 3., 10.]
        xlim = [.01, 10.]

        # magnitude
        ax1 = fig.add_subplot(211)
        ax1.semilogx(w, mag)
        ax1.set_xlim(xlim)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(['' for t in xticks])
        ax1.set_ylabel('Gain (db)')
##        ax1.set_title('n = %i, wn = %.6f'%(n, wn) )
        ax1.grid()

        # phase
        ax2 = fig.add_subplot(212)
        ax2.semilogx(w, phase)
        ax2.set_xlim(xlim)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(['%.3f'%t for t in xticks], rotation=30)
        ax2.set_ylabel('Phase (deg)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.grid()

        return fig

    def _baseline_tsplot(self, x, _x, PE, fs):
        """
        returns ts plot to fit_baseline so it can be
        returned indirectly to user
        Parameters
        ----------
        x : array_like
            unfiltered data
        _x : array_like
            lowpassed and downsampled data
        PE : array_like
            prediction errors
        fs : float
            base sampling rate
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        pe_alpha = self.bin_edges[8] - self.bin_edges[7]
        bin_edges = self.bin_edges

        import matplotlib.pyplot as plt
        plt.rc('font', family='serif')
        fig = plt.figure(figsize=(12, 12*(9/16.)))
        fig.subplots_adjust(left=.07, right=.97, bottom=.03,
                            top=.96, hspace=0.35)

        # unfiltered data
        ax1 = fig.add_subplot(311)
        ax1.plot(x)
        ax1.set_xlim([0, len(x)])
        ax1.set_title('Unfiltered Data')
        ax1.grid()

        # unfiltered data
        ax2 = fig.add_subplot(312)
        ax2.plot(-1*_x)
        ax2.set_xlim([0, len(_x)])
        ax2.set_title('Lowpassed and Downsampled Data')
        ax2.grid()

        # phase
        ax3 = fig.add_subplot(313)
        ax3.plot(PE)
        ax3.set_xlim([0, len(PE)])
        ax3.set_yticks(bin_edges[1:-1])
        ax3.set_ylim([bin_edges[2], bin_edges[-2]])
        ax3.set_yticklabels([r'%i$PE(\alpha)$'%i for i in range(-6,7)])
        ax3.set_title('Prediction Errors')
        ax3.grid()

        return fig

    def _pedistplot(self, Pk):
        """
        returns plot to get_entropy so it can be
        returned indirectly to user
        """
        pkbas = self.pkbas
        M = self.M
        x = np.linspace(-(M+1), (M+1), 2*(M+1))
        h = np.multiply(-Pk, np.log2(pkbas))

        import matplotlib.pyplot as plt
        plt.rc('font', family='serif')

        fig = plt.figure(figsize=(5,5))
        fig.subplots_adjust(left=.12)

        ax = fig.add_subplot(111)
        ax.plot(x, pkbas, color= 'b', alpha=.3, label='baseline')
        ax.plot(x, Pk, color= 'g', alpha=.3, label='task')
        ax.plot(x, h, color='r', alpha=.8, label='entropy')
        ax.scatter(x, pkbas, color= 'b', marker='*')
        ax.scatter(x, Pk, color='g', marker='+')
        ax.legend()


        return fig

    def bode(self):
        """
        returns bode plot of AR coeffs filter
        Parameters
        ----------
        None
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        b_pe = self.b_pe
        w, mag, phase = lti(b_pe, [1,1,1,1]).bode()
        w *= (.5*np.pi)
        iend = np.searchsorted(w, self.resample_fs)

        import matplotlib.pyplot as plt
        plt.rc('font', family='serif')
        fig = plt.figure(figsize=(6,6))
        fig.subplots_adjust(bottom=.125, top=.95, right=.95, hspace=0.1)

        xticks = [.1, .3, 1., 3., 10.]
        xlim = [.1, 10.]

        # magnitude
        ax1 = fig.add_subplot(211)
        ax1.semilogx(w[:iend], mag[:iend])
        ax1.set_xlim(xlim)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(['' for t in xticks])
        ax1.set_ylabel('Gain (db)')
        ax1.set_title('AR coeffs = [%s]'%' '.join('%.3f'%c for c in b_pe))
        ax1.grid()

        # phase
        ax2 = fig.add_subplot(212)
        ax2.semilogx(w[:iend], phase[:iend])  # Bode phase plot
        ax2.set_xlim(xlim)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(['%.3f'%t for t in xticks], rotation=30)
        ax2.set_ylabel('Phase (deg)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.grid()

##        # Nakamura et al. (1999) Taylor series coeffs
##        wt, magt, phaset = lti([1,-2.5,2,-.5], [1,1,1,1]).bode()
##        ax1.semilogx(w[:iend], magt[:iend])
##        ax2.semilogx(w[:iend], phaset[:iend])
##
##        fig.savefig('bode.png')
##        plt.close('all')

        return fig

    def __str__(self):
        return '\n'.join(['b_pe = %s,'%str(self.b_pe),
                          'hbas = %s,'%str(self.hbas),
                          'pkbas = %s'%str(self.pkbas)])

    def __repr__(self):
        st = ('SteeringEntropyModel(resample_fs=%s, alpha=%s, M=%s,'\
                'b_pe=%s, pkbas=%s, bin_edges=%s, hbas=%s, '\
                %(repr(self.resample_fs), repr(self.alpha), repr(self.M),
                  repr(self.b_pe), repr(self.pkbas), repr(self.bin_edges),
                  repr(self.hbas) )).replace('array(','np.array(')

        st += 'cdf=%s)'%repr(self.cdf)

        return st


def generate_entropy_results(df):
    entropy_res = []

    # For each day
    for day_num in df.Day.unique():
        day = df.query('Day == @day_num')
        # For each subject
        for subject_num in df.Subject.unique():
            subject = day.query('Subject == @subject_num')

            # For each random id used to generate paths
            for rand_id_num in subject.Rand_Id.unique():
                rand_ids = subject.query('Rand_Id == @rand_id_num')

                # Find the two trials that match this rand_id
                lookup_table = rand_ids[['Trial', 'Secondary_Task']].drop_duplicates()

                # Identify the trial ids w and w/o secondary task
                st_id  = lookup_table.query('Secondary_Task == True').Trial.values
                nst_id = lookup_table.query('Secondary_Task == False').Trial.values

                # Grab the trial data, remove duplicates on st trial
                st  = rand_ids.query('Trial == @st_id')[['Time', 'Input']].drop_duplicates()
                nst = rand_ids.query('Trial == @nst_id')[['Time', 'Input']]

                try:
                    sem = SteeringEntropyModel()
                    baseline = sem.fit_baseline(nst.Input.as_matrix(), ignore_exception=True)
                    entropy = sem.get_entropy(st.Input.as_matrix())
                    our_entropy = entropy

                    entropy_res.append({'Day': day_num,
                                        'Subject': subject_num,
                                        'Rand_Id': rand_id_num,
                                        'Entropy': our_entropy,
                                        'Baseline': baseline
                                        })
                except Exception as e:
                    print(e)
    entropy_res = pd.DataFrame(entropy_res)
    res = entropy_res.groupby(('Day', 'Subject')).mean()[['Baseline', 'Entropy']].reset_index()

    return res
