import numpy as np

import model
import samplers

from matplotlib import pyplot as plt
from tqdm import trange,tqdm
from timeit import default_timer as time



np.seterr(divide = 'ignore')


class NestedSampler:

    def __init__(self,model, nlive = 1000, npoints = None, evosteps = 70):

        self.model      = model
        self.nlive      = nlive
        self.evosteps   = evosteps
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
        with tqdm(total = self.npoints, desc = 'nested sampling') as pbar:
            while self.generated + self.nlive < self.npoints:
                _, all = self.evo.get_new(self.points['logL'][self.generated])
                insert_index    = np.searchsorted(self.points['logL'],all['logL'])
                self.points     = np.insert(self.points, insert_index, all)
                self.points          = np.sort(self.points, order = 'logL')
                self.evo.chain[0]    = self.points[-self.nlive:]
                ng.append(len(all))
                self.generated += len(all)
                self.mean_duplicates_fraction += self.evo.duplicate_ratio
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
        logX = np.append(logX, [-np.inf])
        self.logL = np.append(self.logL, [self.logL[-1]])
        self.logX = logX
        self.logZ = np.log(np.trapz(-np.exp(self.logL), x = np.exp(logX)))


def main():
    my_model = model.RosenBrock()
    nlive, npoints = 1_000, 100_000
    ns = NestedSampler(my_model, nlive = nlive,  npoints = npoints, evosteps = 100)
    start = time()
    ns.run()
    T = time() - start
    plt.plot(ns.logX,np.exp(ns.logL + ns.logX))

    plt.figure(2)
    plt.scatter(ns.points['position'][:,0], ns.points['position'][:,1], c = np.exp(ns.points['logL']), cmap = 'plasma')
    plt.rc('font', size = 11)
    plt.rc('font', family = 'serif')
    plt.title(f'Rosenbrock model: {len(ns.points)} samples in {T:.1f} seconds')
    plt.show()


if __name__ == '__main__':
    main()
