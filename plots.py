import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


res = pd.DataFrame.from_csv('res.csv')
day_subject = res.groupby(('Day', 'Subject')).mean().reset_index()
day_subject.Feedback = day_subject.Feedback.astype(bool)
day_subject = day_subject.sort('Learner', ascending=False).query('Learner')

sns.set(style="ticks")

sns.lmplot(x="Day", y="ResponseTime", hue="Feedback", data=day_subject, x_estimator=np.mean, ci=68)
plt.grid()
plt.xlim(0.5, 5.5)
sns.despine()
#plt.show()
plt.title('Category B ($n_{FB}$ = ' + '{}'.format(len(day_subject.query('Feedback').Subject.unique())) + ', $n_{NFB}$ = ' + '{})'.format(len(day_subject.query('not Feedback').Subject.unique())))
plt.savefig('test_imgs/day_vs_rt_l.pdf')
plt.close('all')

sns.lmplot(x="Day", y="RMSE", hue="Feedback", data=day_subject, x_estimator=np.mean, ci=68)
plt.grid()
plt.xlim(0.5, 5.5)
sns.despine()
#plt.show()
plt.title('Category B ($n_{FB}$ = ' + '{}'.format(len(day_subject.query('Feedback').Subject.unique())) + ', $n_{NFB}$ = ' + '{})'.format(len(day_subject.query('not Feedback').Subject.unique())))
plt.savefig('test_imgs/day_vs_rmse_l.pdf')
plt.close('all')


#sns.lmplot(x="Day", y="ResponseTime", hue="Learner", data=day_subject, x_estimator=np.mean, ci=68)
#plt.grid()
#plt.xlim(0.5, 5.5)
#sns.despine()
##plt.show()
#plt.savefig('test_imgs/day_vs_rt_learner.pdf')
#plt.close('all')

#sns.lmplot(x="Day", y="RMSE", hue="Learner", data=day_subject, x_estimator=np.mean, ci=68)
#plt.grid()
#plt.xlim(0.5, 5.5)
#sns.despine()
##plt.show()
#plt.savefig('test_imgs/day_vs_rmse_learner.pdf')
#plt.close('all')


def make_fits(res):
    def residuals(p, y, x):
        A = p[0]
        B = p[1]
        C = p[2]
        err = abs(np.array(y - (A * np.exp(-x/B) + C))).mean()
        return err

    def peval(x, p):
        return p[0] * np.exp(-x/p[1]) + p[2]

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
        plt.title('Subject ' + str(s))
        plt.xlabel('Trial')
        plt.ylabel('RMSE')
        plt.ylim(0.05, .2)
        plt.xlim(0, 100)
        plt.grid()
        sns.despine()
        plt.savefig('test_imgs/' + str(s) + '_fit.pdf')
        plt.clf()

    results = pd.DataFrame(results)
    results.columns = ['Subject', 'A', 'B', 'C']
    results['Feedback'] = results.Subject % 2 == 1

    return results

make_fits(res)

#sns.lmplot(x="Trial", y="RMSE", hue="Feedback", data=res.query('Day == 1'), x_estimator=np.mean, ci=68)
#plt.grid()
#plt.xlim(0, 21)
#plt.ylim(0.05, 0.25)
#sns.despine()
##plt.show()
#plt.savefig('test_imgs/day1_trial_vs_rmse.pdf')
#plt.close('all')

#sns.lmplot(x="Trial", y="RMSE", hue="Feedback", data=res.query('Day == 2'), x_estimator=np.mean, ci=68)
#plt.xlim(0, 21)
#plt.ylim(0.05, 0.25)
#plt.grid()
#sns.despine()
##plt.show()
#plt.savefig('test_imgs/day2_trial_vs_rmse.pdf')
#plt.close('all')

#sns.lmplot(x="Trial", y="RMSE", hue="Feedback", data=res.query('Day == 3'), x_estimator=np.mean, ci=68)
#plt.xlim(0, 21)
#plt.ylim(0.05, 0.25)
#plt.grid()
#sns.despine()
##plt.show()
#plt.savefig('test_imgs/day3_trial_vs_rmse.pdf')
#plt.close('all')

#sns.lmplot(x="Trial", y="RMSE", hue="Feedback", data=res.query('Day == 4'), x_estimator=np.mean, ci=68)
#plt.xlim(0, 21)
#plt.ylim(0.05, 0.25)
#plt.grid()
#sns.despine()
##plt.show()
#plt.savefig('test_imgs/day4_trial_vs_rmse.pdf')
#plt.close('all')

#sns.lmplot(x="Trial", y="RMSE", hue="Feedback", data=res.query('Day == 5'), x_estimator=np.mean, ci=68)
#plt.xlim(0, 21)
#plt.ylim(0.05, 0.25)
#plt.grid()
#sns.despine()
##plt.show()
#plt.savefig('test_imgs/day5_trial_vs_rmse.pdf')
#plt.close('all')
