import numpy as np
import samplers
import model
import test
from utils  import *
from matplotlib import pyplot as plt
from numpy.random import uniform as U

bounds = np.ones(2)*5
bounds = (-bounds , bounds)

def likelihood_constraint(x,worstL):
    return np.log((log_likelihood(x) > worstL).astype(int))


def log_prior(x):
    return 0

def log_likelihood(x):
    x = model.unpack_variables(x)
    return -0.5*np.sum(x**2,axis = 0)

my_model = model.Model(log_prior, log_likelihood, bounds)

nlive = 10


#main loop
x = samplers.AIESampler(my_model,1,nwalkers = nlive, space_scale = 10).chain.squeeze()

x  = x[np.argsort(log_likelihood(x))]
print(log_likelihood(x))
print(f'worst is {x[0]} and has loglikelihood {log_likelihood(x[0])}')

sampler = samplers.AIESampler(my_model, 5 , nwalkers = nlive - 1 , space_scale = 10)
#overwrite uniform initialisation
sampler.chain[sampler.elapsed_time_index] = x[1:].copy()
print(f'setting threshold for point generation logL= { log_likelihood(x[0])}')
LCP = lambda x: log_prior(x) + likelihood_constraint(x, log_likelihood(x[0]))

print(f'LCP -> {LCP(sampler.chain[sampler.elapsed_time_index])}')
print(f'log_likelihood -> {log_likelihood(sampler.chain[sampler.elapsed_time_index])}')

sampler.AIEStep(LCP)
a_point = sampler.chain[sampler.elapsed_time_index, np.random.randint(nlive-1), :]
print(f'selected {a_point} with loglikelihood {log_likelihood(a_point)}')


exit()



plt.show()
