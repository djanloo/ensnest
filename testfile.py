import numpy as np
import samplers
import model
import test
from utils  import *
from matplotlib import pyplot as plt
from numpy.random import uniform as U


bounds = np.ones(2)*5
bounds = (-bounds , bounds)

def log_prior(x):
    x1, x2 = model.unpack_variables(x)
    return np.log(np.sin(x1) + 2)

def log_likelihood(x):
    return np.log(np.abs(np.sum(x, axis = -1)))


my_model = model.Model(log_prior, log_likelihood, bounds)
sampler = samplers.AIESampler(my_model, 500, nwalkers = 10000)

x = sampler.sample_prior().join_chains(burn_in = 0.1)
plt.hist(x,bins = 100, histtype = 'step', density = True)
plt.show()
