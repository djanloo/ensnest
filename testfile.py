import numpy as np
import samplers
import model
import test
from utils  import *
from matplotlib import pyplot as plt
from numpy.random import uniform as U


bounds = np.ones(2)
bounds = (bounds , 100*bounds)

def log_prior(x):
    x0, x1 = model.unpack_variables(x)
    return np.log(1/x0)

def log_likelihood(x):
    return np.log(np.abs(np.sum(x, axis = -1)))

#generate some fucking samples and watch how stupidly fast it is
my_model = model.Model(log_prior, log_likelihood, bounds)
sampler = samplers.AIESampler(my_model, 100, nwalkers = 5000)

x = sampler.sample_prior()
plt.hist(x.join_chains(burn_in = 0.1),bins = 100, histtype = 'step', density = True)

plt.show()
