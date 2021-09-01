import numpy as np
import samplers
import model
import test
from utils  import *
from matplotlib import pyplot as plt
from numpy.random import uniform as U

bounds = np.ones(2)*5
bounds = (-bounds , bounds)

def a_stupid_log_f(x):
    likeconstr = np.log((log_likelihood(x) > -0.1).astype(int))
    return -0.5*np.sum(x**2,axis = -1) + likeconstr + my_model.log_chi(x)

def log_prior(x):
    return 0

def log_likelihood(x):
    x = model.unpack_variables(x)
    return -0.5*np.sum(x**2,axis = 0)


my_model = model.Model(log_prior, log_likelihood, bounds)
sampler = samplers.AIESampler(my_model, 100, nwalkers = 100, space_scale = 10)
run = sampler.sample_function(a_stupid_log_f)
x = run.chain
fig, ax = plt.subplots()

for i in range(sampler.nwalkers):
    ax.plot(x[:,i,0], alpha = 0.5, lw = 1.25)

fig, ax = plt.subplots()
ax.plot(run.join_chains(0)[:,0],alpha = 0.8)

fig, ax = plt.subplots()
for i in range(sampler.nwalkers):
    ax.hist(x[-100:-1,i,0], histtype = 'step', bins = 100)


plt.show()
