import model
import test
import samplers
import numpy as np
import cProfile

bounds = np.ones(5)*5
bounds = (-bounds , bounds)

def log_prior(x):
    x = np.array(model.to_variables(x))
    return -0.5*np.sum(x**2, axis = 0)

def log_likelihood(x):
    return log_prior(x)

def log_posterior(x):
    return log_prior(x) + log_likelihood(x)


my_model = model.Model(log_prior, log_likelihood, bounds)
#sampler = samplers.AIESampler(my_model,500, nwalkers = 10000)
point = np.zeros((50,50,50,my_model.space_dim))
def test():
    for i in range(1000):
        my_model.log_chi(point)

cProfile.run('test()')
