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

bounds = np.ones(2)*10
bounds = (-bounds , bounds)

def likelihood_constraint(x,worstL):
    result = np.log((log_likelihood(x) > worstL).astype(int))
    return result


def log_prior(x):
    return 0

def log_likelihood(x):
    x = model.unpack_variables(x)
    return -0.5*np.sum(x**2,axis = 0)

my_model = model.Model(log_prior, log_likelihood, bounds)

nlive = 100
evolution_steps = 70
npoints = 1000

all_points = samplers.AIESampler(my_model,1,nwalkers = nlive, space_scale = 10).chain.squeeze()
all_points = all_points[np.argsort(log_likelihood(all_points))]
likelihoods = np.sort(log_likelihood(all_points))

########### begin main loop ##############
for n_generated in trange(npoints):

    worstL = likelihoods[n_generated]

    #takes all points but the worse
    sampler = samplers.AIESampler(my_model, evolution_steps + 1 , nwalkers=nlive-1 , space_scale=10)

    #overwrite uniform initialisation
    #print(f"sampler chain: \n{sampler.chain[sampler.elapsed_time_index]}\n>>> {len(sampler.chain[sampler.elapsed_time_index])}")
    #print(f"all_points: \n{all_points[n_generated+1:]}\n>>> {len(all_points[n_generated+1:])}")

    sampler.chain[sampler.elapsed_time_index] = all_points[n_generated+1:].copy()

    #defines the Likelihood constrained pdf
    LCP = lambda var: log_prior(var) + likelihood_constraint(var, worstL)

    #generates nlive - 1 points over L>Lmin
    for i in range(evolution_steps):
        sampler.AIEStep(LCP)

    #selects one of this point (again checks that L>Lmin)
    #print(f'old:{likelihoods[n_generated:]}')
    #print(f'new: {log_likelihood(sampler.chain[sampler.elapsed_time_index])}')

    is_duplicate = (log_likelihood(sampler.chain[sampler.elapsed_time_index]) == likelihoods[n_generated:,None]).any(axis = 0)
    n_duplicate = np.sum(is_duplicate.astype(int))
    if is_duplicate.any(): print(f'>>>>>>>>>>> WARNING: {n_duplicate} duplicate(s) found')

    correct     = sampler.chain[sampler.elapsed_time_index, np.logical_not(is_duplicate), :]
    new_point   = correct[ np.random.randint(nlive - 1 - n_duplicate), :]

    #inserts the point in the right place of the ordered list
    replace_index   = bisect_left(likelihoods, log_likelihood(new_point))
    likelihoods     = np.insert(likelihoods, replace_index , log_likelihood(new_point))
    all_points      = np.insert(all_points,  replace_index,  new_point, axis = 0)
    #print(f'--------round {n_generated}----------- added L = {log_likelihood(new_point)}')
print(likelihoods)

X = np.exp( - np.arange(len(likelihoods))/nlive )
plt.plot(np.log(X), likelihoods)

Z = np.trapz(- np.exp(likelihoods), x = X)
print(Z)
plt.show()



exit()
