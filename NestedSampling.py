import numpy as np
import model
import test
from utils  import *
from matplotlib import pyplot as plt
from numpy.random import uniform as U
from bisect import bisect_left
from tqdm import trange

import pyximport; pyximport.install()
import cy_samplers

import cProfile, pstats

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

def main():
    my_model = MyModel()
    nlive = 100
    npoints = 1000


    #initialisation of the first nlive points and sorting
    points = cy_samplers.AIESampler(my_model, 100, nwalkers=nlive ).sample_function(my_model.log_prior).chain[99]
    logLs  = my_model.log_likelihood(points)

    points  = points[np.argsort(logLs)]
    logLs   = np.sort(logLs)


    #initialise evolver sampler
    evolve_sampler = cy_samplers.AIESampler(my_model, 70 ,nwalkers=nlive-1)

    for n_generated in trange(npoints):

        evolve_sampler.chain[evolve_sampler.elapsed_time_index] = points[n_generated+1:].copy()
        new_point = evolve_sampler.sample_over_threshold(logLs[n_generated])

        #inserts the point in the right place of the ordered list
        replace_index = bisect_left(logLs, my_model.log_likelihood(new_point))
        logLs         = np.insert(logLs, replace_index , my_model.log_likelihood(new_point))
        points        = np.insert(points,  replace_index,  new_point, axis = 0)

        #reset the sampler
        evolve_sampler.reset()

profiler = cProfile.Profile()
profiler.enable()
main()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats()
