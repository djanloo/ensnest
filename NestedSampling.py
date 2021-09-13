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
    evosteps = 70

    evo     = samplers.AIESampler(my_model, evosteps , nwalkers = nlive).sample_prior().tail_to_head()
    points  = np.sort(evo.chain[0], order='logL')

    with tqdm(total = npoints) as pbar:
        while generated < npoints:
            _, all          = evo.get_new(points['logL'][generated])
            insert_index    = np.searchsorted(points['logL'],all['logL'])
            points          = np.insert(points, insert_index, all)
            points          = np.sort(points, order = 'logL')
            evo.chain[0]    = points[-nlive:]
            ng.append(len(all))
            generated += len(all)
            evo.elapsed_time_index = 0
            pbar.update(len(all))

    ng.append(nlive)
    ng = np.array(ng)

    #generate the logX values
    DeltaN = []
    for i in tqdm(range(len(ng)), desc = 'calculating numbers'):
        c = ng[i]
        while c >= 0:
            DeltaN.append(c)
            c -= 1
    DeltaN = np.array(DeltaN)
    N = DeltaN + nlive
    logX = np.zeros(len(points))
    print(f'len DeltaN = {len(DeltaN)} --- len points = {len(points)} ')
    for i in  tqdm(range(1,len(points)), desc = 'calculating logX'):
        logX[i] = logX[i-1] - 1/N[i]
    Z = np.trapz(-np.exp(points['logL']), x = np.exp(logX))
    print(f'logZ = {Z} , integral = {400*Z}')
    plt.plot(logX, points['logL'], ls ='' ,marker = '.')
    plt.show()


if __name__ == '__main__':
    main()
