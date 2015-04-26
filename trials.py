import numpy as np


funckwds1 = {'name': 1,
             'frequencyA': 0.9, 'frequencyB': 3.3,
             'offsetA': 0, 'offsetB': np.pi/7,
             'amplitudeA': 0.5, 'amplitudeB': .2}
funckwds2 = {'name': 2,
             'frequencyA': np.log(2), 'frequencyB': np.pi,
             'offsetA': 3, 'offsetB': .17,
             'amplitudeA': 0.6, 'amplitudeB': .2}
funckwds3 = {'name': 3,
             'frequencyA': np.log(3), 'frequencyB': np.log(7),
             'offsetA': 17, 'offsetB': 3,
             'amplitudeA': 0.6, 'amplitudeB': .4}
ks = [
         # Refresher
         {'trial': 1,
          'preview': 0.2},

         {'trial': 2,
          'preview': 0.1,
          'funckwds': funckwds1},

         {'trial': 3,
          'funckwds': funckwds1},

         # Experiment
         {'trial': 4,
          'preview': 0.2,
          'funckwds': funckwds2},

         {'trial': 5,
          'preview': 0.05,
          'funckwds': funckwds3},

         {'trial': 6,
          'funckwds': funckwds3},

         {'trial': 7,
          'preview': 0.05,
          'funckwds': funckwds2},

         {'trial': 8,
          'preview': 0.2,
          'funckwds': funckwds3},

         {'trial': 9,
          'preview': 0.1,
          'funckwds': funckwds3},

         {'trial': 10,
          'funckwds': funckwds2},

         {'trial': 11,
          'preview': 0.1,
          'funckwds': funckwds2},

         {'trial': 12,
          'preview': 0.1,
          'funckwds': funckwds3},

         {'trial': 13,
          'funckwds': funckwds3},

         {'trial': 14,
          'preview': 0.1,
          'funckwds': funckwds2},

         {'trial': 15,
          'preview': 0.2,
          'funckwds': funckwds2},

         {'trial': 16,
          'preview': 0.05,
          'funckwds': funckwds3},

         {'trial': 17,
          'funckwds': funckwds2},

         {'trial': 18,
          'preview': 0.05,
          'funckwds': funckwds2},

         {'trial': 19,
          'preview': 0.2,
          'funckwds': funckwds3},
        ]
