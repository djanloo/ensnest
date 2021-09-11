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

def main():
    my_model = MyModel()

    nlive = 1000
    npoints = 1000000
    generated = 0
    evosteps = 70

    evo     = samplers.AIESampler(my_model, evosteps , nwalkers = nlive).sample_prior().tail_to_head()
    points  = np.sort(evo.chain[0], order='logL')
    plt.ion()
    plt.show()
    with tqdm(total = npoints) as pbar:
        while generated < npoints:
            _, all          = evo.get_new(points['logL'][generated])
            plt.figure(1)
            plt.scatter(all['position'][:,0], all['position'][:,1])
            plt.title(f"sampling on logL > {points['logL'][generated]} ({generated})")
            insert_index    = np.searchsorted(points['logL'],all['logL'])
            points          = np.insert(points, insert_index, all)
            points          = np.sort(points, order = 'logL')

            plt.figure(2)
            plt.plot(points['logL'])

            evo.chain[0]    = points[-nlive:]
            breakpoint()
            generated += len(all)
            evo.elapsed_time_index = 0
            pbar.update(len(all))

    logX = -np.linspace(0,1,nlive+generated)/(nlive+generated)
    plt.plot(logX, points['logL'])
    plt.show()

if __name__ == '__main__':
    main()
