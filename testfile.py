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
bounds = (bounds*0.1 , 5*bounds)


def log_prior(x):
    x1,x2 = model.unpack_variables(x)
    return np.log(1/x1)

def log_likelihood(x):
    x = model.unpack_variables(x)
    return -0.5*np.sum((x - 2)**2,axis = 0)

nlive = 100
npoints = 100

my_model  = model.Model(log_prior, log_likelihood, bounds)

points = samplers.AIESampler(my_model, 100, nwalkers=nlive ).sample_function(log_prior).chain[99]
logLs  = log_likelihood(points)

points  = points[np.argsort(logLs)]
logLs   = np.sort(logLs)

evo_sampler  = samplers.AIESampler(my_model, 100, nwalkers = nlive-1)


#start nesting
for n_generated in trange(npoints):

    evo_sampler.chain[evo_sampler.elapsed_time_index] = points[n_generated+1:].copy()
    new_point = evo_sampler.sample_over_threshold(logLs[n_generated])

    #inserts the point in the right place of the ordered list
    replace_index = bisect_left(logLs, log_likelihood(new_point))
    logLs         = np.insert(logLs, replace_index , log_likelihood(new_point))
    points        = np.insert(points,  replace_index,  new_point, axis = 0)

    #reset the sampler
    evo_sampler.reset()

n_generated = npoints

print(f'Now sampling over logL = {logLs[n_generated]}')

npoints = 1000
sample_over = np.zeros((npoints,my_model.space_dim))
#at this point it has npoints values stored, last likelihood is greatest
for i in trange(0, npoints):
    evo_sampler.chain[evo_sampler.elapsed_time_index] = points[n_generated+1:].copy()
    new_point = evo_sampler.sample_over_threshold(logLs[n_generated])

    #inserts the point in the sample
    sample_over[i] = new_point

    #reset the sampler
    evo_sampler.reset()

plt.scatter(sample_over[:,0], sample_over[:,1],alpha = 0.5)



plt.show()
