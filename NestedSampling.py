'''
The nested sampling module. In the first (and probably only) version it is mainly
tailored onto the AIEsampler class, so there's no choice for the sampler.
'''

import numpy as np

import model
import samplers
import utils

from matplotlib import pyplot as plt
from tqdm import trange,tqdm
from timeit import default_timer as time

from scipy.stats import kstest

import os
import pickle
import multiprocessing as mp
from time import sleep

BAR_FMT= "{desc:<25.25}:{percentage:3.0f}%|{bar}|"
BAR_FMT_ZSAMP= "{desc:<25.25}:{percentage:3.0f}%|{bar}|{r_bar}"

np.seterr(divide = 'ignore')


class NestedSampler(mp.Process):
    '''Class performing nested sampling
    '''
    def __init__(self,model,
                nlive = 1000, npoints = np.inf,
                evosteps = 100, relative_precision = 1e-4,
                load_old = None, filename = None, evo_progress = True):

        #run fundamentals
        self.model      = model
        self.nlive      = nlive
        self.evosteps   = evosteps
        self.relative_precision = relative_precision

        #main variables
        self.logZ       = -np.inf
        self.Z          = None
        self.logX       = np.array([0.],        dtype=np.float64)
        self.logL       = np.array([-np.inf],   dtype=np.float64)
        self.N          = np.array([],          dtype=np.int)
        self.logdZ      = None
        self.npoints    = npoints

        #errors and time log
        self.run_again  = True
        self.logZ_error = None
        self.Z_error    = None
        self.logZ_samples  = None
        self.run_time            = None
        self.error_estimate_time = None

        #utils
        self.elapsed_clusters    =   0
        self.N_continue          =   np.flip( np.append( np.arange(self.nlive, 2*self.nlive , dtype=np.int) , [self.nlive] ))
        self.delta_logX_continue = - np.cumsum(1./self.N_continue)
        self.N_closure           =   np.flip( np.arange(1, self.nlive+1, dtype=np.int) )
        self.delta_logX_closure  = - np.cumsum(1./self.N_closure)

        #save/load
        self.loaded   = False
        self.load_old = load_old
        self.filename = 'inknest_NS' + str(hash(self)) + '.nkn' if filename is None else filename

        self.path = os.path.join('__inknest__',self.filename)
        self.check_saved()

        self.evo_progress = evo_progress

        if not self.loaded:
            #initialises the sampler (AIES is the only option currently)
            self.evo        = samplers.AIEevolver(self.model, evosteps , nwalkers = nlive).init()
            self.points     = np.sort(self.evo.chain[self.evo.elapsed_time_index], order = 'logL')

            #integrate the first zone: (1-X0)*L0
            self.logZ = utils.logsubexp(0,-1./self.nlive) + self.points['logL'][0]
            self.logdZ_max_estimate      = self.points['logL'][-1]+ self.logX[-1]

            self.progress_offset     = self.logZ - self.logdZ_max_estimate + np.log(self.relative_precision)

    def run(self):
        '''Performs nested sampling.'''
        
        if self.loaded:
            print('Run loaded from file')
            return
        else:
            print(f'Starting run with code: {hash(self)}')
        start = time()
        with tqdm(total = 1., desc='nested sampling', unit_scale=True , colour = 'blue', bar_format = BAR_FMT) as pbar:

            #main loop
            while self.run_again:
                new             = self.evo.get_new(self.points['logL'][ -self.nlive], progress = self.evo_progress) #counts nlive points from maxL, takes the logL that contains all them
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
        self.varenv_points()
        self.save()

    def _compute_progress(self):
        progress = self.logZ - self.logdZ_max_estimate + np.log(self.relative_precision)
        if progress < self.progress_offset:
            self.progress_offset = progress
        return min((progress - self.progress_offset)/(-self.progress_offset),1.)

    def update(self):
        '''Updates the value of Z given the current state.

        The number of live points is like:

        ``nlive``,(jump) ``2nlive``, ``2nlive-1``, ... ,``nlive``, (jump) ``2nlive``, ecc.

        This function is called between each pair of jumps. Uses the last ngen value
        and appends N values.

        Integration is performed between the two successive times at which ``N = nlive`` (extrema included)
        so

            >>> cluster_N = [nlive, 2*nlive -1, ... , nlive]
            >>> cluster_N.shape = (nlive + 1)
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

    def log_worst_t_among_N(self):
        '''Helper function to generate shrink factors

        Since max({t}) with t in [0,1], len({t}) = N
        is distributed as Nt**(N-1), the cumulative function is y = t**(N)
        and sampling uniformly over y gives the desired sample.

        Therefore, max({t}) is equivalent to (unif)**(1/N)

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
            logt = self.log_worst_t_among_N()

            logX = np.cumsum(logt)
            logX = np.insert(logX,len(logX), -np.inf)
            logX = np.insert(logX,0,0)

            self.logZ_samples[i]  = np.log(-np.trapz(np.exp(self.logL), x = np.exp(logX)))

        self.logZ       = np.mean(self.logZ_samples)
        self.logZ_error = np.std(self.logZ_samples)
        self.Z          = np.exp(self.logZ)
        self.Z_error    = self.Z*self.logZ_error
        self.error_estimate_time = time() - start

    def varenv_points(self):
        '''Gives usable fields to ``self.points['position']`` based on ``model.names``
        '''
        var_names_specified_t = np.dtype([ ('position', self.model.position_t), ('logL',np.float64), ('logP', np.float64) ])
        self.points = self.points.view(var_names_specified_t)

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
        samp     = samplers.AIEevolver(self.model, 100, nwalkers = self.nlive)

        samp.bring_over_threshold(logL)
        starting_points = samp.chain[samp.elapsed_time_index]

        samples      = np.zeros((len(evosteps),nsamples, self.model.space_dim))
        microtimes   = np.zeros(len(evosteps))

        for i in tqdm(range(len(evosteps)), desc='checking LCPS', bar_format = BAR_FMT):
            samp._force_steps_number(evosteps[i])
            points  = starting_points
            start   = time()
            with tqdm(total = nsamples, desc = f'[test] sampling prior (evosteps = {evosteps[i]:4})', colour = 'green', leave = False) as pbar:
                while len(points) < nsamples:
                    new     = samp.get_new(logL, progress = False, allow_resize = False)
                    points  = np.append(points,new)
                    samp.reset(start = new) #check what happens if you say start = last sorted points
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
        try:
            os.mkdir('__inknest__')
        except FileExistsError:
            pass
        out_file = open(self.path, 'wb')
        pickle.dump(self.__dict__, out_file, -1)

    def load(self):
        with open(self.path, 'rb') as in_file:
            tmp_dict = pickle.load(in_file)
            self.__dict__.update(tmp_dict)

    def check_saved(self):
        try:
            with open(self.path):
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

class mpNestedSampler:

    def __init__(self,*args, **kwargs):

        self.nproc  = mp.cpu_count()
        self.queue = mp.Queue()

        self.processes = []
        self.nested_samplers  = []

        for i in range(self.nproc):
            ns = NestedSampler(*args,**kwargs)
            p  = mp.Process(target=self.execute_from_queue, args = (self.queue,))
            self.nested_samplers.append(ns)
            self.processes.append(p)

    def execute_from_queue(self, queue):
        ns_instance = queue.get()
        ns_instance.run()
        print('nested run')
        queue.put(ns_instance.points)
        print('put stuff in queue')

    def run(self):
        for p in self.processes:
            p.start()

        for ns in self.nested_samplers:
            self.queue.put(ns)

        for p in self.processes:
            p.join()

        while not self.queue.empty():
            print('fetching from queue')
            print(self.queue.get())
