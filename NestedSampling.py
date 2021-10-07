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
N_Z_SAMPLES=1000

np.seterr(divide = 'ignore')

def log_worst_t_among(N):
    '''Helper function to generate shrink factors

    Since max({t}) with t in [0,1], len({t}) = N
    is distributed as Nt**(N-1), the cumulative function is y = t**(N)
    and sampling uniformly over y gives the desired sample.

    Therefore, max({t}) is equivalent to (unif)**(1/N)

    and log(unif**(1/N)) = 1/N*log(unif)
    '''
    return 1./N*np.log(np.random.uniform(0,1, size = len(N)))

class NestedSampler:
    '''Class performing nested sampling
    '''
    def __init__(self,model,
                nlive = 1000, npoints = np.inf,
                evosteps = 150, relative_precision = 1e-4,
                load_old = None, filename = None, evo_progress = True):
        ''' NS initialisation.

        Args
        ----
            model : :class:`~model.Model~
            nlive : int
            npoints : int
            evosteps : int
                the steps for which the ensemble is evolved
            relative_precision: float
            load_old : bool
                specify whether to load an existing run

                if not specified, the user will be asked in runtime in case an old run is found
        '''
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
        self.weights    = None
        self.npoints    = npoints

        #errors and time log
        self.run_again  = True
        self.logZ_error = None
        self.Z_error    = None
        self.logZ_samples  = None
        self.run_time            = None
        self.error_estimate_time = None
        self.initialised = False

        #mpNestedSampler compatibility
        self.seed           = 1234
        self.process_number = 0

        #utils
        self.elapsed_clusters    =   0
        self.N_continue          =   np.flip( np.append( np.arange(self.nlive, 2*self.nlive , dtype=np.int) , [self.nlive] ))
        self.delta_logX_continue = - np.cumsum(1./self.N_continue)
        self.N_closure           =   np.flip( np.append( np.arange(1 , 2*self.nlive , dtype=np.int) , [self.nlive] ))
        self.delta_logX_closure  = - np.cumsum(1./self.N_closure)

        #save/load
        self.loaded   = False
        self.load_old = load_old
        self.run_code = hash(self)
        self.filename = str(self.run_code) if filename is None else filename
        self.path     = os.path.join('__inknest__',self.filename)

        self.check_saved()
        self.evo_progress = evo_progress

    def initialise(self):
        '''Initialises the evolver and Z value'''
        #for subprocesses: force seed update to prevent identical copies after process fork
        np.random.seed(self.seed)
        if not self.loaded:
            #initialises the sampler (AIES is the only option currently)
            self.evo        = samplers.AIEevolver(self.model, self.evosteps , nwalkers = self.nlive).init( progress_position=self.process_number)
            self.points     = np.sort(self.evo.chain[self.evo.elapsed_time_index], order = 'logL')

            #integrate the first zone: (1-<X0>)*L0
            self.logZ = utils.logsubexp(0,-1./self.nlive) + self.points['logL'][0]
            self.logdZ_max_estimate      = self.points['logL'][-1]+ self.logX[-1]

            self.progress_offset     = self.logZ - self.logdZ_max_estimate + np.log(self.relative_precision)
        self.initialised = True

    def run(self):
        '''Performs nested sampling.'''
        if not self.initialised:
            self.initialise()
        if self.loaded:
            print('Run loaded from file')
            return
        start = time()
        with tqdm(total = 1., desc='nested sampling', unit_scale=True , colour = 'blue', bar_format = BAR_FMT, position = self.process_number) as pbar:

            #main loop
            while self.run_again:
                new             = self.evo.get_new(self.points['logL'][ -self.nlive], progress = self.evo_progress) #counts nlive points from maxL, takes the logL that contains all them
                #insert_index    = np.searchsorted(self.points['logL'],new['logL'])
                #self.points     = np.insert(self.points, insert_index, new)        
                self.points     = np.append(self.points,new)
                self.points     = np.sort(self.points, order = 'logL') #because searchsorted fails sometimes
                self.evo.reset(start = self.points[-self.nlive:])      #restarts the sampler giving last live points as initial ensemble

                self.update()
                pbar.n = self._compute_progress()
                pbar.refresh()

        self.run_time = time() - start
        self.mean_over_t()
        self.get_ew_samples()
        self.varenv_points()
        self.save()

    def _compute_progress(self):
        progress = self.logZ - self.logdZ_max_estimate + np.log(self.relative_precision)
        if progress < self.progress_offset:
            self.progress_offset = progress
        return min((progress - self.progress_offset)/(-self.progress_offset),1.)

    def update(self):
        '''Updates the value of Z given the current state.

        The number of live points is of the form:

        ``nlive``,(jump) ``2nlive-1``, ``2nlive-2``, ... , ``nlive`` , (jump) ``2nlive-1``, ecc.

        Integration is performed between the two successive times at which ``N = nlive`` (extrema included),
        then one extremum is excluded when saving to ``self.N``.
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

            _logL = self.points['logL'][self.elapsed_clusters*self.nlive:]
            _logL = np.append(_logL, [_logL[-1]])

            #stores everything, extrema included
            self.logX = np.append(self.logX, _logX)
            self.logL = np.append(self.logL, _logL)
            self.N    = np.append(self.N,    self.N_closure )

        self.logdZ  = np.log(np.trapz( - np.exp(_logL), x = np.exp(_logX) ))
        self.logZ   = np.logaddexp(self.logZ, self.logdZ)

        self.elapsed_clusters += 1

    def mean_over_t(self):
        '''Computes the mean and std of logZ and weights over t.

        The process of finding the worst among n values is simulated and
        the mean over N_Z_SAMPLES is computed.
        '''
        start = time()
        self.logZ_samples = np.zeros(N_Z_SAMPLES)
        self.weights      = np.zeros(len(self.points))
        logt      = np.zeros(len(self.N))
        for i in tqdm(range(N_Z_SAMPLES), 'computing Z samples', bar_format = BAR_FMT_ZSAMP, position = self.process_number):
            logt = log_worst_t_among(self.N)

            logX = np.cumsum(logt)
            logX = np.insert(logX,len(logX), -np.inf)
            logX = np.insert(logX,0,0)

            self.logZ_samples[i]  = np.log(-np.trapz(np.exp(self.logL), x = np.exp(logX)))
            self.weights          -= np.exp(self.logL[1:-1])*np.diff(np.exp(logX[1:]))/np.exp(self.logZ_samples[i])/N_Z_SAMPLES # <L_i * w_i / Z >t

        self.logZ       = np.mean(self.logZ_samples)
        self.logZ_error = np.std(self.logZ_samples)
        self.Z          = np.exp(self.logZ)
        self.Z_error    = self.Z*self.logZ_error

        self.error_estimate_time = time() - start

    def get_ew_samples(self):
        '''Generates equally weghted samples by accept/reject strategy.
        '''
        K = np.max(self.weights)
        accepted = (self.weights/K > np.random.uniform(0,1,size = len(self.points)))
        self.ew_samples = self.points[accepted]


    def varenv_points(self):
        '''Gives usable fields to ``self.points['position']`` based on ``model.names``
        '''
        var_names_specified_t = np.dtype([ ('position', self.model.position_t), ('logL',np.float64), ('logP', np.float64) ])
        self.points     = self.points.view(var_names_specified_t)
        self.ew_samples = self.ew_samples.view(var_names_specified_t)

    def save(self):
        # print(f'saving on {self.path}')
        try:
            os.mkdir(os.path.dirname(self.path))
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

class mpNestedSampler(NestedSampler):
    '''Multiprocess version of nested sampling.

    Runs ``multiprocess.cpu_count()`` instances of :class:`~NestedSampler` and joins them.

    Attributes
    ----------

        logX : np.ndarray(dtype=np.float64)
        logL : np.ndarray(dtype=np.float64)
        N    : np.ndarray(dtype=np.int)
        logZ : np.float64
        Z    : np.float64
        logZ_error  : np.float64
        Z_error     : np.float64
        logZ_samples: np.ndarray(dtype=np.float64)
        nested_samplers : list of :class:`~NestedSampler`
            The individual runs. Each nested sampler has completely defined attributes.
        run_time : np.float64
            The time required to perform the runs and merge them.
        error_estimate_time: np.float64
            The time required to perform error estimate on ``logZ``
    '''

    def __init__(self,*args, **kwargs):
        '''
        Takes the same arguments of ``NestedSampler``.
        '''

        #NestedSampler arguments
        self.args   = args
        self.kwargs = kwargs

        #samplers
        self.nproc              = mp.cpu_count()
        self.processes          = []
        self.nested_samplers    = []
        self.process_number     = 0 # for subclass compatibility

        #merged run attributes
        self.model  = args[0]
        self.logX   = None
        self.logL   = None
        self.N      = None
        self.points = None
        self.means  = None
        self.stds   = None
        self.ew_samples = None

        #save/load an time log
        self.run_time = None
        self.load_old = False if 'load_old' not in kwargs else kwargs['load_old']
        self.loaded   = False
        self.filename = str(hash( NestedSampler(*args,**kwargs))) if not 'filename' in kwargs else kwargs['filename']
        self.path     = os.path.join('__inknest__',self.filename,'merged')

        #shuts down evo_progress
        kwargs['evo_progress'] = False
        self.check_saved()

        if not self.loaded:
            for i in range(self.nproc):

                kwargs['filename'] = os.path.join(self.filename, f'run_{i}')

                ns = NestedSampler(*args,**kwargs)
                ns.process_number = i
                ns.seed           = i #solves identical runs bug
                p  = mp.Process(target=self.execute_and_save, args = (ns,) )

                self.nested_samplers.append(ns)
                self.processes.append(p)

    def execute_and_save(self,ns):
        ns.run()
        ns.save() #clumsy way to communicate between parent/child process

    def run(self):
        if self.loaded:
            print('Merge loaded from file')
            return

        self.run_time = time()
        for p in self.processes:
            p.start()

        for p in self.processes:
            p.join()

        del self.processes  #as they are unsavable

        #recovers nested samplers from save file
        for ns in self.nested_samplers:
            with open(ns.path, 'rb') as in_file:
                tmp_dict = pickle.load(in_file)     #clumsy way to communicate between parent/child process
                ns.__dict__.update(tmp_dict)

        self.merge_all()
        self.mean_over_t()
        self.param_stats()
        self.run_time = time() - self.run_time
        self.save()
        print(f'Executed in {utils.hms(self.run_time)} ({len(self.points)} samples - {len(self.ew_samples)} e.w. samples)')

    def how_many_at_given_logL(self,N,logLs,givenlogL):
        '''Helper function that does what the name says.

        See `dynamic nested sampling <https://arxiv.org/abs/1704.03459>`_.
        '''
        index = np.searchsorted(logLs,givenlogL)
        if index == len(logLs):
            return 1
        return N[index]

    def merge_two(self, logLa, Na, logLb, Nb):
        logL = np.append(logLa, logLb)
        logL = np.sort(logL)
        N    = np.zeros(len(logL))

        for i in range(len(logL)):
            N[i] = self.how_many_at_given_logL(Na, logLa, logL[i]) + self.how_many_at_given_logL(Nb, logLb, logL[i])
        return logL, N

    def merge_all(self):
        '''Merges all the runs'''
        self.logL = self.nested_samplers[0].logL[1:-1]
        self.N    = self.nested_samplers[0].N
        for i in tqdm(range(1,self.nproc), desc='merging runs', bar_format=BAR_FMT):
            self.logL,self.N = self.merge_two(self.logL,self.N,
                                                self.nested_samplers[i].logL[1:-1], self.nested_samplers[i].N )
        # set logX as its deterministic estimate
        self.logX = -np.cumsum(1./self.N)

        # re-insert values at boundary
        self.logX = np.insert(self.logX, [0,len(self.logX)], [0, - np.inf])
        self.logL = np.insert(self.logL, [0,len(self.logL)], [-np.inf, self.logL[-1]])

        # merges the points and equally weighted samples
        self.points     = np.concatenate(tuple([ns.points     for ns in self.nested_samplers]))
        self.points     = self.points[np.argsort(self.points['logL'])]
        self.ew_samples = np.concatenate(tuple([ns.ew_samples for ns in self.nested_samplers]))
        self.ew_samples = self.ew_samples[np.argsort(self.ew_samples['logL'])]


    def param_stats(self):
        '''Estimates the mean and standard deviation of the parameters'''

        self.means  = np.sum(self.weights[:,None]*self.points['position'].copy().view((np.float64, self.model.space_dim)), axis = 0)
        self.stds   = np.sum(self.weights[:,None]*(
                                (self.points['position'].copy().view((np.float64, self.model.space_dim)) - self.means)**2
                                ), axis = 0)
        self.stds   = np.sqrt(self.stds)

        self.means  = self.means.view(self.model.position_t)
        self.stds   = self.stds.view(self.model.position_t)
