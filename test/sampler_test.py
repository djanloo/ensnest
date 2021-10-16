from samplers import AIESampler
from model import Model, Gaussian
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as time

class abs_x(Model):

    def set_parameters(self):
        self.bounds = (-5,5)
        self.names = ['x']

    @Model.auto_bound
    @Model.varenv
    def log_prior(self, vars):
        return np.log(np.abs(vars['x']))

    @Model.auto_bound
    def log_likelihood(self,vars):
        return 0

M       = abs_x()
samp    = AIESampler(M,200, nwalkers = 10000)
t = time()
x       = samp.sample_prior().join_chains()
t = time() - t
plt.hist(x['position'], bins = 100, histtype = 'step', color = 'k' )
print(f'{len(x)/t:e} samp/s')
plt.show()
