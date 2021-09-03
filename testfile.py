import numpy as np
import samplers
import model
import test
from utils  import *
from matplotlib import pyplot as plt
from numpy.random import uniform as U
from bisect import bisect_left
from tqdm import trange

np.seterr(divide = 'ignore')

bounds = np.ones(2)
bounds = (bounds , 50*bounds)

def likelihood_constraint(x,worstL):
    result = np.log((log_likelihood(x) > worstL).astype(int))
    return result

def log_prior(x):
    x1,x2 = model.unpack_variables(x)
    return np.log(np.sin(x1))

def log_likelihood(x):
    x = model.unpack_variables(x)
    return -0.5*np.sum(x**2,axis = 0)

nlive = 1000

my_model = model.Model(log_prior, log_likelihood, bounds)
init_sampler = samplers.AIESampler(my_model,10000,nwalkers = nlive, space_scale = 10)
all_points = init_sampler.sample_function(log_prior).join_chains()

plt.hist(all_points, bins = 50, histtype = 'step',  density = True)




plt.show()
