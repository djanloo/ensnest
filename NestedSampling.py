import numpy as np

import model
import samplers
import utils

from matplotlib import pyplot as plt
from tqdm import trange,tqdm
from timeit import default_timer as time

from scipy.stats import kstest

import os

BAR_FMT= "{desc:<25.25}:{percentage:3.0f}%|{bar}|"
u = '{r_bar}'

BAR_FMT_ZSAMP= "{desc:<25.25}:{percentage:3.0f}%|{bar}|{r_bar}"

np.seterr(divide = 'ignore')


class NestedSampler:

    def __init__(self,model, nlive = 1000, npoints = np.inf, evosteps = 100, load_old = None):

        self.model      = model
        self.nlive      = nlive
        self.evosteps   = evosteps

        self.logZ       = -np.inf
        self.logX       = np.array([0.],        dtype=np.float64)
        self.logL       = np.array([-np.inf],   dtype=np.float64)
        self.N          = np.array([],          dtype=np.int)
        self.logdZ      = None
        self.npoints    = npoints

        self.run_again  = True
        self.logZ_error = None
        self.logZ_samples  = None
        self.relative_precision = 1e-4
        self.run_time            = None
        self.error_estimate_time = None

        self.elapsed_clusters    =   0
        self.N_continue          =   np.flip( np.append( np.arange(self.nlive, 2*self.nlive , dtype=np.int) , [self.nlive] ))
        self.delta_logX_continue = - np.cumsum(1./self.N_continue)
        self.N_closure           =   np.flip( np.arange(1, self.nlive+1, dtype=np.int) )
        self.delta_logX_closure  = - np.cumsum(1./self.N_closure)

        #checks for an already saved run
        self.loaded   = False
        self.load_old = load_old
        self.check_saved()

        if not self.loaded:
            #initialises the sampler (AIES is the only option currently)
            self.evo        = samplers.AIEevolver(self.model, evosteps , nwalkers = nlive).init()
            self.points     = np.sort(self.evo.chain[self.evo.elapsed_time_index], order = 'logL')

            #integrate the first zone: (1-X0)*L0
            self.logZ = utils.logsubexp(0,-1./self.nlive) + self.points['logL'][0]
            self.logdZ_max_estimate      = self.points['logL'][-1]+ self.logX[-1]

            self.progress_offset     = self.logZ - self.logdZ_max_estimate + np.log(self.relative_precision)


    def run(self):
        if self.loaded:
            return
        start = time()
        with tqdm(total = 1., desc='nested sampling', unit_scale=True , colour = 'blue', bar_format = BAR_FMT) as pbar:

            #main loop
            while self.run_again:
                new             = self.evo.get_new(self.points['logL'][ -self.nlive]) #counts nlive points from maxL, takes the L that contains all them
                insert_index    = np.searchsorted(self.points['logL'],new['logL'])
                self.points     = np.insert(self.points, insert_index, new)
                self.points     = np.sort(self.points, order = 'logL') #because searchsorted fails sometimes

                self.evo.reset(start = self.points[-self.nlive:])      #restarts the sampler giving last live points as initial ensemble

                self.update()
                pbar.n = self._compute_progress()
                pbar.refresh()

        plt.show()
        self.run_time = time() - start
        self.estimate_Zerror()
        self.save()

    def _compute_progress(self):
        progress = self.logZ - self.logdZ_max_estimate + np.log(self.relative_precision)
        if progress < self.progress_offset:
            self.progress_offset = progress
        return min((progress - self.progress_offset)/(-self.progress_offset),1.)

    def update(self):
        '''updates the value of Z given the current state.

        The number of live points is like:

        nlive,(jump) 2nlive, 2nlive-1, ... ,nlive, (jump) 2nlive, ecc.

        This function is called between each pair of jumps. Uses the last ngen value
        and appends N values.

        Integration is performed between the two successive times at which N = nlive (extrema included)
        so

        cluster_N = [nlive, 2*nlive -1, ... , nlive]

        cluster_N.shape = (nlive + 1)
        '''
        #checks if it is a normal update or a closure update
        self.logdZ_max_estimate      = self.points['logL'][-1]+ self.logX[-1]
        relative_increment_condition =  (self.logdZ_max_estimate - self.logZ > np.log(self.relative_precision))
        n_points_condition           =  (len(self.points) < self.npoints )
        self.run_again               =  relative_increment_condition and n_points_condition

        if self.run_again:
            #integration values for logX, logL
            #first point of cluster included
            #last  point of cluster included
            _logX = self.logX[-1] + self.delta_logX_continue
            _logL = self.points['logL'][self.elapsed_clusters*self.nlive: (self.elapsed_clusters+1)*self.nlive + 1]

            #storage values for logX, logL (to prevent from redundance)
            #first point of cluster included
            #last  point of cluster excluded, will be stored at next iter
            self.logX = np.append(self.logX, _logX[:-1])
            self.logL = np.append(self.logL, _logL[:-1])
            self.N    = np.append(self.N,    self.N_continue[:-1])
        else:
            _logX = self.logX[-1] + self.delta_logX_closure
            _logX = np.append(_logX, [-np.inf] )

            _logL = self.points['logL'][-self.nlive:]
            _logL = np.append(_logL, [_logL[-1]])

            #stores everything, extrema included
            self.logX = np.append(self.logX, _logX)
            self.logL = np.append(self.logL, _logL)
            self.N    = np.append(self.N,    self.N_closure )

        self.logdZ  = np.log(np.trapz( - np.exp(_logL), x = np.exp(_logX) ))
        self.logZ   = np.logaddexp(self.logZ, self.logdZ)

        self.elapsed_clusters += 1

    def _log_worst_t_among_N(self):
        '''Helper function to generate shrink factors

        Since max({t}) with t in [0,1], len({t}) = N
        is distributed as Nt**(N-1), the cumulative function is y = t**(N)
        and sampling uniformly over y gives the desired sample.

        Therefore, max({t}) is equiv to (unif)**(1/N)

        and log(unif**(1/N)) = 1/N*log(unif)
        '''
        return 1./self.N*np.log(np.random.uniform(0,1, size = len(self.N)))

    def estimate_Zerror(self):
        '''Estimates the error sampling t.
        '''
        Ntimes = 1000
        start = time()
        self.logZ_samples = np.zeros(Ntimes)
        logt      = np.zeros(len(self.N))
        for i in tqdm(range(Ntimes), 'computing Z samples', bar_format = BAR_FMT_ZSAMP):
            logt = self._log_worst_t_among_N()

            logX = np.cumsum(logt)
            logX = np.insert(logX,len(logX), -np.inf)
            logX = np.insert(logX,0,0)

            self.logZ_samples[i]  = np.log(-np.trapz(np.exp(self.logL), x = np.exp(logX)))

        self.logZ  = np.mean(self.logZ_samples)
        self.logZ_error = np.std(self.logZ_samples)
        self.error_estimate_time = time() - start


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

    def save(self):
        if not os.path.exists('_inknest_'):
            os.makedirs('_inknest_')
        stats = np.array( [self.points, self.logZ, self.logZ_error,self.logX, self.logL, self.N, self.logZ_samples, self.elapsed_clusters,self.relative_precision, self.run_time, self.error_estimate_time], dtype=object)
        np.save(os.path.join('_inknest_','INKNEST_NS_STATS' + str(hash(self))),stats)


    def load(self,filename = None):
        '''Loads an already performed run which has the same hashcode'''
        if filename is None:
            filename = os.path.join('_inknest_','INKNEST_NS_STATS' + str(hash(self)) + '.npy'  )
        self.points, self.logZ, self.logZ_error,self.logX, self.logL, self.N, self.logZ_samples,self.elapsed_clusters,self.relative_precision , self.run_time, self.error_estimate_time = np.load(filename, allow_pickle = True)

    def check_saved(self):
        '''Checks wether an already performed run exists and asks wether to load it or not.'''
        try:
            with open(os.path.join('_inknest_','INKNEST_NS_STATS' + str(hash(self)) + '.npy'  )):
                if self.load_old is None:
                    reply = str(input('An execution of this run has been found. Do you want to load it? (y/n): ')).lower().strip()
                    if reply[0] == 'y':
                        self.load()
                        self.loaded = True
                elif self.load_old:
                    self.load()
                    self.loaded = True
        except IOError:
            pass

    def __hash__(self):
        '''Gives the (almost) unique code for the run'''
        return hash((self.model, self.nlive, self.npoints, self.evosteps, self.relative_precision))
