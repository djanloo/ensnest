import numpy as np

import model
import samplers

from matplotlib import pyplot as plt
from tqdm import trange,tqdm
from timeit import default_timer as time

from scipy.stats import kstest



np.seterr(divide = 'ignore')


class NestedSampler:

    def __init__(self,model, nlive = 1000, npoints = np.inf, evosteps = 70):

        self.model      = model
        self.nlive      = nlive
        self.evosteps   = evosteps
        self.logZ       = -np.inf
        self.logX       = np.array([0., -1./nlive],        dtype=np.float64)
        self.logL       = np.array([-np.inf],   dtype=np.float64)
        self.dlogZ      = None

        #since variable nlive is faster:
        self.N          = np.array([self.nlive],dtype=np.int) # ~N(t)
        self.ngen       = np.array([0],         dtype=np.int) # number of live points pumped in at each iteration could be non-constant
        self.generated  = 1    # cumulant for ngen

        #initialises the sampler (AIES is the only option currently)
        self.evo        = samplers.AIESampler(self.model, evosteps , nwalkers = nlive).sample_prior().tail_to_head()
        self.npoints    = npoints
        self.points     = np.sort(self.evo.chain[0], order = 'logL')

        self.run_time   = np.inf
        self.run_again  = True
        self.logZ_error = None
        self.relative_precision = 1e-2

        #opening Z increment: takes care of the first point as dZ = (1-X0)*L0
        self.logZ += (1 - np.exp(-1/nlive))*np.exp(self.points['logL'][0])

    def run(self):
        start = time()
        plt.ion()
        plt.show()
        with tqdm(total = self.npoints, desc='nested sampling', unit_scale=True , colour = 'blue') as pbar:

            #main loop
            while self.run_again:

                new             = self.evo.get_new(self.points['logL'][self.generated])
                insert_index    = np.searchsorted(self.points['logL'],new['logL'])
                self.points     = np.insert(self.points, insert_index, new)
                self.points     = np.sort(self.points, order = 'logL') #because searchsorted fails sometimes

                self.evo.reset(start = self.points[-self.nlive:])      #restarts the sampler giving last live points as initial ensemble
                self.ngen       = np.append(self.ngen, [len(new)] )
                self.generated += len(new)

                self.update()
                pbar.update(len(new))

        self.logZ = np.log(np.trapz(-np.exp(self.logL), x = np.exp(self.logX)))
        print(f'det integral {np.exp(self.logZ)*self.model.volume}')

        self.run_time = time() - start
        breakpoint()
        self.estimate_Zerror()
        print(f'Run finished: logZ = {self.logZ} +- {self.dlogZ}')
        print(f'stoch integral {np.exp(self.logZ)*self.model.volume} +- {np.exp(self.logZ)*self.model.volume*self.dlogZ}')

    def update(self):
        '''updates the value of Z given the current state.

        The number of live points is like:

        nlive,(jump) ~2nlive, 2nlive-1, ... ,nlive, (jump) ~2nlive, ecc.

        This function is called between each pair of jumps. Uses the last ngen value
        and appends N values.
        '''
        #checks if it is a normal update or a closure update
        print(f"log_max_fut_inc - logZ = {self.points['logL'][-1]+ self.logX[-1] - self.logZ}")
        relative_increment_condition =  (self.points['logL'][-1]+ self.logX[-1] > np.log(self.relative_precision) + self.logZ)
        n_points_condition           =  (len(self.points) < self.npoints )
        self.run_again               =  relative_increment_condition and n_points_condition

        #if it is a closure includes also the last live points in the Z computation
        Nmax        = self.N[-1]     + self.ngen[-1]
        inf_index   = self.generated - self.ngen[-1]
        if self.run_again:
            Nmin      = self.nlive
            sup_index = self.generated
        else:
            Nmin      = 1
            sup_index = self.generated + self.nlive

        local_N     = np.flip(np.arange( Nmin , Nmax, dtype = np.int))
        local_logX  = self.logX[-1] - np.cumsum(1./local_N)
        local_logL  = self.points['logL'][inf_index:sup_index]
        self.dlogZ  = np.log(np.trapz( - np.exp(local_logL), x = np.exp(local_logX) ))
        self.logZ   = np.logaddexp(self.logZ, self.dlogZ)
        self.logX   = np.append(self.logX, local_logX)
        self.logL   = np.append(self.logL, local_logL)
        self.N      = np.append(self.N   , local_N)

        plt.fill_between(local_logX, local_logL , -200 + 0*local_logL)


    def _log_worst_t_among(self,N):
        '''Helper function to generate shrink factors'''
        return np.log(np.max(np.random.uniform(0,1, size = N)))

    def estimate_Zerror(self):
        '''Estimates the error sampling t.

        Very slow.
        '''
        # X[i] = worst among(N[i]) ~(det) exp(-1/N[i])
        # for each array of X calc Z
        # do this many times then avg Z over sample

        Ntimes = 10
        #-------------------------------
        start = time()
        Nexpanded = np.tile(self.N, Ntimes)
        breakpoint()
        logt = np.array([self._log_worst_t_among(n) for _ , n in tqdm(np.ndenumerate(Nexpanded), total = Ntimes*len(self.N) ,desc = 'estimating error on logZ')])
        logt = logt.reshape(-1,len(self.N))
        #----------------------------------
        # Nexpanded = np.repeat([self.N],Ntimes, axis = 0)
        #
        # logt = self.log_worst_t_among_vec("I have no clue why this has to be put here", Nexpanded)
        print(f'required {time()-start}')
        logX = np.cumsum(logt, axis = -1)
        logX = np.insert(logX,[0,-1],[0, -np.inf], axis = -1)

        logZ_samp  = np.log(-np.trapz(np.exp(self.logL), x = np.exp(logX)))
        self.logZ  = np.mean(logZ_samp)
        self.logZ_error = np.std (logZ_samp)


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
