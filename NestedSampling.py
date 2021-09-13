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

    nlive = 10000
    npoints = 100000
    ng = [0]
    generated = 0
    evosteps = 50
    mean_duplicates_percentage = 0
    evo     = samplers.AIESampler(my_model, evosteps , nwalkers = nlive).sample_prior().tail_to_head()
    points  = np.sort(evo.chain[0], order='logL')

    with tqdm(total = npoints) as pbar:
        while generated + nlive < npoints:
            _, all , rel_duplicates = evo.get_new(points['logL'][generated])
            insert_index    = np.searchsorted(points['logL'],all['logL'])
            points          = np.insert(points, insert_index, all)
            points          = np.sort(points, order = 'logL')
            evo.chain[0]    = points[-nlive:]
            ng.append(len(all))
            generated += len(all)
            mean_duplicates_percentage += rel_duplicates
            evo.elapsed_time_index = 0
            pbar.update(len(all))

    mean_duplicates_percentage /= (len(ng)-1)
    print(f'mean duplicates {mean_duplicates_percentage*100} %')
    ng = np.array(ng)

    #generate the logX values
    jumps = np.zeros(len(points))
    N     = np.zeros(len(points))
    current_index = 0
    for ng_i in ng:
        jumps[current_index] = ng_i
        current_index += ng_i

    N[0] = nlive
    for i in range(1,len(N)):
        N[i] = N[i-1] - 1+ jumps[i-1]

    logX = np.zeros(len(points))
    for i in  tqdm(range(1,len(points)), desc = 'calculating logX'):
        logX[i] = logX[i-1] - 1/N[i]

    plt.figure(2)
    Z = np.trapz(-np.exp(points['logL']), x = np.exp(logX))
    print(f'Z = {Z} , integral = {my_model.volume*Z}')
    print(f'Vol = {my_model.volume}')
    plt.plot(logX, points['logL'])
    plt.figure(3)
    plt.plot(logX, np.exp(logX + points['logL']))

    plt.show()


if __name__ == '__main__':
    main()
