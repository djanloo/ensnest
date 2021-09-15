import numpy as np

import model
import samplers

from matplotlib import pyplot as plt
from tqdm import trange,tqdm
from timeit import default_timer as time

from scipy.stats import kstest



np.seterr(divide = 'ignore')


class NestedSampler:

    def __init__(self,model, nlive = 1000, npoints = None, evosteps = 70):

        self.model      = model
        self.nlive      = nlive
        self.evosteps   = evosteps
        self.logZ       = -np.inf
        self.logX       = None
        self.logL       = None

        #since variable nlive is faster:
        self.N          = None # ~N(t)
        self.ngen       = None  # number of live points pumped in at each iteration
        self.generated  = 0     # cumulant for ngen

        #initialises the sampler (AIES is the only option currently)
        self.evo    = samplers.AIESampler(self.model, evosteps , nwalkers = nlive).sample_prior().tail_to_head()
        self.npoints    = npoints
        self.points     = np.sort(self.evo.chain[0], order = 'logL')

        #takes trace of how efficient is the sampling
        self.mean_duplicates_fraction= 0

        self.run_time = np.inf

    def run(self):
        start = time()

        #takes trace of the generated points in case duplicates are excluded
        self.ngen = [0]

        with tqdm(total = self.npoints, desc='nested sampling', unit_scale=True , colour = 'blue') as pbar:

            while self.generated + self.nlive < self.npoints:

                _, all          = self.evo.get_new(self.points['logL'][self.generated])
                insert_index    = np.searchsorted(self.points['logL'],all['logL'])
                self.points     = np.insert(self.points, insert_index, all)
                self.points          = np.sort(self.points, order = 'logL') #because searchsorted fails sometimes
                self.evo.chain[0]    = self.points[-self.nlive:]
                self.ngen.append(len(all))
                self.generated += len(all)
                self.mean_duplicates_fraction   += self.evo.duplicate_ratio
                self.evo.elapsed_time_index     = 0
                pbar.update(len(all))

        self.ngen = np.array(self.ngen)

        #generate the logX values
        jumps  = np.zeros(len(self.points), dtype=np.int)
        self.N = np.zeros(len(self.points), dtype=np.int)
        current_index = 0
        for ng_i in self.ngen:
            jumps[current_index] = ng_i
            current_index += ng_i

        self.N[0] = self.nlive
        for i in range(1,len(self.N)):
            self.N[i] = self.N[i-1] - 1 + jumps[i-1]

        # X = [1,X0,..., X(n-1), 0]  -> n points + 2 'artificial'
        # with X0 = worst among N[0] in (0,1)   ~ exp(-1/N[0])
        #      X1 = worst among N[1] in (0,X0)  ~ exp(-1/N[0] - 1/N[1])
        #      ecc
        self.logX = np.zeros(len(self.points)+2)
        self.logL = np.zeros(len(self.points)+2)

        self.logX[1:-1] = -np.cumsum(1/self.N)
        self.logX[-1] = -np.inf

        # L =[0,L0, ... , L(n-1), L(n-1)] -> fills last block by duplicating the last L
        self.logL[1:-1] = self.points['logL']
        self.logL[-1]   = self.logL[-2]
        self.logL[0]    = -np.inf

        self.logZ = np.log(np.trapz(-np.exp(self.logL), x = np.exp(self.logX)))

        self.run_time = time() - start
        self.estimate_Zerror()

    @np.vectorize
    def log_worst_t_among(N):
        '''Helper function to generate a shrink factors'''
        return np.log(np.min(np.random.uniform(0,1, size = N)))

    def estimate_Zerror(self):
        '''Estimates the error sampling the shrink variables
        '''
        # X[i] = worst among(N[i]) ~(det) exp(-1/N[i])
        # for each array of X calc Z
        # do this many times then avg Z over sample

        # actually, since log_worst_t_among is vestorized
        # creates copies of N then do all together
        Nexpanded = np.repeat([self.N], 10, axis = 0)
        logt = self.log_worst_t_among(Nexpanded)

        # ti = t(N[i])
        #   X0 = t0*1
        #   X1 = t1*X0 = t1*t0*1
        #   X2 = t2*t1*t0 ecc.

        #so logX0 = logt0
        #   logX1 = logt0 + logt1
        #   logX2 = logt0 + logt1 + logt2   ecc.
        logX = np.cumsum(logt, axis = -1)
        logX = np.insert(logX,[0,-1],[0, -np.inf], axis = -1)
        breakpoint()




    def check_prior_sampling(self, logL, evosteps, nsamples):
        """Samples the prior over a given likelihood threshold with different steps of evolution.

        Can be useful in estimating the steps necessary for convergence to the real distribution.

        At first it brings the sample from being uniform to being over threshold,
        then it evolves sampling over the threshold for ``evosteps[i]`` times,
        then takes the last generated points (``sampler.chain[sampler.elapsed_time_index]``)
        and adds them to the sample to be returned.

        Args
        ----
            logL : float
                the log of likelihood threshold
            evosteps : ``int`` or array of ``int``
                the steps of evolution performed for sampling
            nsamples : int
                the number of samples taken from ``sampler.chain[sampler.elapsed_time_index]``

        Returns:
            np.ndarray : samples
        """

        evosteps = np.array(evosteps)
        samp     = samplers.AIESampler(self.model, 100, nwalkers = self.nlive)

        samp.bring_over_threshold(logL)

        samples      = np.zeros((len(evosteps),nsamples, self.model.space_dim))
        microtimes   = np.zeros(len(evosteps))

        for i in range(len(evosteps)):
            samp.set_length(evosteps[i])
            points  = samp.chain[0]
            start   = time()
            with tqdm(total = nsamples, desc = f'[test] sampling prior (evosteps = {evosteps[i]:4})', colour = 'green') as pbar:
                while len(points) < nsamples:
                    _ , new = samp.get_new(logL)
                    points = np.append(points,new)
                    samp.elapsed_time_index = 0
                    pbar.update(len(new))

            microtimes[i]   = (time() - start)/len(points)*1e6
            samples[i,:]    = points['position'][:nsamples]

        ks_stats = np.zeros((len(evosteps)-1,self.model.space_dim))
        for run_i in range(len(evosteps)-1):
            for axis in range(self.model.space_dim):
                pval = list(kstest(samples[run_i,:,axis] , samples[run_i+1,:,axis]))[1]
                ks_stats[run_i,axis] = pval

        return samples, microtimes, ks_stats
