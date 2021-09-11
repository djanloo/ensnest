import sys
import numpy as np
import model
import test
from utils  import *
from matplotlib import pyplot as plt
from numpy.random import uniform as U
from numpy.lib import recfunctions as rfn
from bisect import bisect_left
from tqdm import trange

import samplers

np.seterr(divide = 'ignore')

class MyModel(model.Model):
    def __init__(self):
        self.bounds = (-np.ones(2)*10 ,np.ones(2)*10 )
        super().__init__()

    @model.Model.auto_bound
    def log_prior(self,x):
        return 0

    def log_likelihood(self,x):
        return -0.5*np.sum(x**2,axis = -1)


my_model = MyModel()

nlive = 100
npoints = 1000
evosteps = 70

evo     = samplers.AIESampler(my_model, evosteps , nwalkers = nlive).sample_prior().tail_to_head()
points  = np.sort(evo.chain[0], order='logL')

for i in trange(npoints):
    new_point     = evo.get_new(points['logL'][i])
    insert_index  = bisect_left(points['logL'],new_point['logL'])
    points        = np.insert(points, insert_index, new_point)
    evo.chain[0]  = points[i+1 :]

    evo.elapsed_time_index = 0

logX = -np.linspace(0,1,nlive+npoints)/(nlive+npoints)
plt.plot(logX, points['logL'])
plt.show()
