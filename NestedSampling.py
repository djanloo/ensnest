import sys
import numpy as np
import model
import test
from utils  import *
from matplotlib import pyplot as plt
from numpy.random import uniform as U
from numpy.lib import recfunctions as rfn
from bisect import bisect_left
from tqdm import trange,tqdm

import samplers

np.seterr(divide = 'ignore')

class MyModel(model.Model):
    def __init__(self):
        self.bounds = (-np.ones(2)*10 ,np.ones(2)*10 )
        self.names  = ['a','b']
        super().__init__()

    #@model.Model.varenv
    @model.Model.auto_bound
    def log_prior(self,x):
        return 0#np.log(x['a'])

    def log_likelihood(self,x):
        return -0.5*np.sum(x**2,axis = -1)


class NestedSampler:

    def __init__(self,model, nlive = 1000, npoints = None, evosteps = 70):

        self.model      = model
        self.nlive      = 1000
        self.evosteps   = 70
        self.logZ       = -np.inf
        self.logX       = None
        self.logL       = None

        #initialises the sampler (AIES is the only option currently)
        self.evo    = samplers.AIESampler(self.model, evosteps , nwalkers = nlive).sample_prior().tail_to_head()
        self.generated  = 0
        self.npoints    = npoints
        self.points     = np.sort(self.evo.chain[0], order = 'logL')

        #takes trace of how efficient is the sampling
        self.mean_duplicates_fraction= 0

    def run(self):
        ng = [0]
        with tqdm(total = self.npoints) as pbar:
            while self.generated + self.nlive < self.npoints:
                _, all , rel_duplicates = self.evo.get_new(self.points['logL'][self.generated])
                insert_index    = np.searchsorted(self.points['logL'],all['logL'])
                self.points     = np.insert(self.points, insert_index, all)
                self.points          = np.sort(self.points, order = 'logL')
                self.evo.chain[0]    = self.points[-self.nlive:]
                ng.append(len(all))
                self.generated += len(all)
                self.mean_duplicates_fraction += rel_duplicates
                self.evo.elapsed_time_index = 0
                pbar.update(len(all))
        self.logL = self.points['logL']
        ng = np.array(ng)
        #generate the logX values
        jumps = np.zeros(len(self.points))
        N     = np.zeros(len(self.points))
        current_index = 0
        for ng_i in ng:
            jumps[current_index] = ng_i
            current_index += ng_i

        N[0] = self.nlive
        for i in range(1,len(N)):
            N[i] = N[i-1] - 1+ jumps[i-1]

        logX = np.zeros(len(self.points))
        for i in  tqdm(range(1,len(self.points)), desc = 'calculating logX'):
            logX[i] = logX[i-1] - 1/N[i]
        self.logX = logX
        self.logZ = np.log(np.trapz(-np.exp(self.points['logL']), x = np.exp(logX)))




def main():
    my_model = MyModel()
    ns = NestedSampler(my_model, nlive = 1000,  npoints = 10000, evosteps = 70)
    ns.run()
    plt.plot(ns.logX,ns.logL)
    plt.show()


if __name__ == '__main__':
    main()
