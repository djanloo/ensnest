"""
Module containing the samplers used in main calculations.

Since almost every sampler is defined by a markov chain, basic attributes are
the model and the length of the chain.

Each sampler shoud be capable of tackling with discontinuous functions.

Since is intended to be used in nested sampling, each sampler should support likelihood constrained prior sampling (LCPS).

"""
from sys import exit
import numpy as np
from numpy.random import uniform as U
from numpy.random import randint
import model
from tqdm import tqdm, trange

from timeit import default_timer as timer

class Sampler:
    """Produces samples from model.

    It is intended as a base class that has to be further defined.
    For generality the attribute `nwalkers` is present, but it can be one for not ensamble-based samplers.

    Attributes
    ----------
        model : model.Model
            Model defined as the set of (log_prior, log_likelihood , bounds)
        mcmc_lenght : int
            the lenght of the single markov chain
        nwalkers : int
            the number of walkers the ensamble is made of


    """
    def __init__(self, model , mcmc_length , nwalkers , verbosity=0 ):
        """Initialise the chain uniformly over the space bounds.
        """
        self.model      = model
        self.length     = mcmc_length
        self.nwalkers   = nwalkers
        self.verbosity  = verbosity
        self.elapsed_time_index = 0     #'time' index up to which self.chain has been developed


        #the squeeze method is for mantaining generality. Single-walker methods reduce to normal definition.
        self.chain      = np.zeros((self.length, self.nwalkers) , dtype=self.model.livepoint_t).squeeze()

        #uniform initialisation
        for walker in range(self.nwalkers):

            self.chain['position'][0, walker]  = U(*self.model.bounds)
            self.chain['logP'] [0, walker]      = self.model.log_prior(self.chain[0, walker]['position'])
            self.chain['logL'][0, walker]       = self.model.log_likelihood(self.chain[0, walker]['position'])

class AIESampler(Sampler):
    '''The Affine-Invariant Ensemble sampler (Goodman, Weare, 2010).

    After a uniform initialisation step, for each particle k selects a *pivot* particle an then proposes

    .. math::
        j = k + random(0 \\rightarrow n)

        z \\char`~ g(z)

        y = x_j + z (x_k - x_j)

    and then executes a MH-acceptance over y (more information at <https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf>).

    '''

    def __init__(self, model, mcmc_length, nwalkers=10, space_scale = None, verbosity=0):

        super().__init__(model, mcmc_length, nwalkers, verbosity=verbosity)
        #if space_scale is not defined takes the 'diameter' of the space
        self.space_scale = space_scale
        if self.space_scale is None:
            self.space_scale = 0.5*np.sqrt(np.sum(self.model.bounds[0]**2)) + 0.5*np.sqrt(np.sum(self.model.bounds[1]**2))
        if self.space_scale <= 1:
            print('space scale parameter must be > 1: set 2x')
            self.space_scale *= 2

        self.duplicate_ratio = None

    def get_stretch(self, size = 1):
        '''
        Generates the stretch values given the scale_parameter ``a``.

        Output is distibuted as :math:`\\frac{1}{\\sqrt{z}}`  in :math:`[1/a,a]``.
        Uses inverse transform sampling
        '''
        return (U(0,1, size = size )*(self.space_scale**(1/2) - self.space_scale**(-1/2) ) + self.space_scale**(-1/2) )**2

    def AIEStep(self, Lthreshold = None):
        '''Single step of AIESampler.

            Args
            ----
                Lthreshold : float, optional
                    The threshold of likelihood below which a point is set as impossible to reach
        '''
        #considers the whole ensamble at at time
        current_walker_position = self.chain[self.elapsed_time_index,:]['position']

        #OPTIMIZATION: np.random.randint is really slow
        #generate a number from 1 to self.nwalkers-1
        delta_index = ((self.nwalkers-2)*np.random.rand(self.nwalkers)+1).astype(int)

        #for each walker selects randomly another walker as a pivot for the stretch move
        pivot_index     = (np.arange(self.nwalkers) + delta_index   ) % self.nwalkers
        pivot_position  = self.chain[self.elapsed_time_index, pivot_index]['position']

        z        = self.get_stretch(size = self.nwalkers)
        proposal = pivot_position + z[:,None] * (current_walker_position - pivot_position)

        log_prior_proposal      = self.model.log_prior(proposal)
        log_likelihood_proposal = self.model.log_likelihood(proposal)
        log_prior_current       = self.chain[self.elapsed_time_index, :]['logP']

        if not np.isfinite(log_prior_current).all():
            breakpoint()
            print(f'FATAL: past point is in impossible position')
            exit()

        #if a threshold Lmin is set, sets as 'impossible' the proposals outside
        if Lthreshold is not None:
            log_prior_proposal[log_likelihood_proposal < Lthreshold] = -np.inf

        log_accept_prob = ( self.model.space_dim - 1) * np.log(z) + log_prior_proposal - log_prior_current

        #if point is out of function domain, sets rejection
        log_accept_prob[np.isnan(log_prior_proposal)] = -np.inf

        accepted = (log_accept_prob > np.log(U(0,1,size = self.nwalkers)))

        #assigns accepted values
        self.chain['position'][self.elapsed_time_index+1, accepted] = proposal[accepted]
        self.chain['logP'][self.elapsed_time_index+1, accepted]     = log_prior_proposal[accepted]
        self.chain['logL'][self.elapsed_time_index+1, accepted]     = log_likelihood_proposal[accepted]
        # copies rejected values
        self.chain[self.elapsed_time_index+1, np.logical_not(accepted)] = self.chain[self.elapsed_time_index, np.logical_not(accepted)]
        self.elapsed_time_index += 1


    def sample_prior(self, Lthreshold = None, progress = False):
        """Fills the chain by sampling the prior.
        """
        if progress:
            desc = 'sampling prior'
            if Lthreshold is not None:
                desc += f' over logL > {Lthreshold:.2f}'
            for t in tqdm(range(self.length - 1), desc = desc):
                self.AIEStep(Lthreshold = Lthreshold)
        else:
            for t in range(self.length - 1):
                self.AIEStep(Lthreshold = Lthreshold)
        return self

    def get_new(self,Lmin):
        '''Returns a new different point from prior given likelihood threshold

        As for AIEStep, needs that every point is in a valid region (the border is included).

        args
        ----
            Lmin : float
                the threshold likelihood that a point must have to be accepted

        Returns:
            tuple : (new , correct) one of the evolved points and all the generated points
        '''
        #generates nlive - 1 points over L>Lmin
        for t in range(1,self.length):
            self.AIEStep(Lthreshold = Lmin)

        #selects one of this point give it's different from the given ones
        is_duplicate    = (self.chain['logL'][self.elapsed_time_index] == self.chain['logL'][0][:,None]).any(axis = 0)
        n_duplicate     = np.sum(is_duplicate.astype(int))

        self.duplicate_ratio = n_duplicate/self.nwalkers
        if self.duplicate_ratio > 0.8: print(f'>>>>>>>>>>> WARNING: {int(self.duplicate_ratio*100)}% of duplicate(s) found')

        correct_ones = self.chain[self.elapsed_time_index, np.logical_not(is_duplicate)]
        new_point    = correct_ones[np.random.randint(self.nwalkers - n_duplicate)]
        return new_point, correct_ones

    def reset(self):
        self.elapsed_time_index = 0
        self.chain      = np.zeros((self.length, self.nwalkers) , dtype=self.model.livepoint_t).squeeze()

    def tail_to_head(self):
        '''Helper function for doing continuous sampling.

        Sets the end of the chain as the head and restarts elapsed time.
        '''
        self.chain[0] = self.chain[self.elapsed_time_index]
        self.elapsed_time_index = 0
        return self

    def join_chains(self, burn_in = 0.02):
        '''Joins the chains for the ensemble after removing  ``burn_in`` \% of each single_particle chain.

        Args
        ----
            burn_in : float, optional
                the burn_in percentage.

                Must be ``burn_in`` > 0 and ``burn_in`` < 1.
        '''
        return self.chain[int(burn_in*self.length):].flatten()

    def bring_over_threshold(self, logLthreshold):
        '''Brings the sampler over threshold.

        It is necessary to initialise the sampler before sampling over threshold.

        args
        ----
            Lthreshold : float
                the logarithm of the likelihood.
        '''
        logLmin = np.min(self.chain['logL'][0])
        old_progress = logLthreshold/logLmin
        with tqdm(total = 1, desc = 'bringing over threshold') as pbar:
            while logLmin < logLthreshold:
                sorted = np.sort(self.chain[0], order='logL')
                logLmin   = sorted['logL'][0]
                _, new = self.get_new(logLmin)
                sorted = np.append(sorted, new)
                sorted = np.sort(sorted, order='logL')
                self.chain[0] = sorted[-self.nwalkers:]
                self.elapsed_time_index = 0
                pbar.update(logLthreshold/logLmin - old_progress)

    def set_length(self, length):
        old_length  = self.length
        self.length = length
        new_chain   = np.zeros((self.length, self.nwalkers) , dtype=self.model.livepoint_t).squeeze()
        new_chain[:min(old_length, length)] = self.chain[:min(old_length, length)]
        self.chain = new_chain

if __name__ == '__main__':

    def test1():
        np.seterr(divide = 'ignore')
        from scipy.stats import kstest
        from  matplotlib import cm
        mymodel  = model.UniformJeffreys()
        logL = -3.6
        nwalkers = 1_000
        totsamp = 100_000
        samp     = AIESampler(mymodel, 100, nwalkers = nwalkers )
        samp.bring_over_threshold(logL)

        plt.rc('font', size = 8)
        plt.rc('font', family = 'serif')

        fig,(ax1,ax2)= plt.subplots(2)

        nsteps = [150,100,50,5, 2]
        samples = np.zeros((len(nsteps),totsamp))
        colors = cm.get_cmap('plasma')(np.linspace(1,0,len(nsteps)))

        for i,l in enumerate(nsteps):
            samp.set_length(l)
            points = np.array([], dtype = mymodel.livepoint_t)
            duplicate_ratio = []
            start = timer()
            with tqdm(total = totsamp) as pbar:
                while len(points) < totsamp:
                    _ , new = samp.get_new(logL)
                    duplicate_ratio.append(samp.duplicate_ratio*100)
                    points = np.append(points,new)
                    samp.elapsed_time_index = 0
                    pbar.update(len(new))
            time = timer() - start
            # plt.scatter(points['position'][:,0],points['position'][:,1],alpha = 0.05)
            # plt.xlim(mymodel.bounds[0][0], mymodel.bounds[1][0])
            # plt.ylim(mymodel.bounds[0][1], mymodel.bounds[1][1])
            micros_per_sample = time/len(points)*1e6
            samples[i,:] = points['position'][:totsamp,0]
            ax1.hist(samples[i], bins=64, histtype = 'step', color = colors[i], density = True, label =f'{l:5}  ({micros_per_sample:5.1f} $\mu$s/samp)')

        x_ = np.linspace(mymodel.center[0] - (-2*logL)**(1/2) + 0.01 ,mymodel.center[0] + (-2*logL)**(1/2) - 0.01  ,1000)
        def analytical(x):
            p = 1./x*np.sqrt(-2*logL -(x- mymodel.center[0])**2)
            N = np.trapz(p, x = x)
            return p/N

        ax1.plot(x_,analytical(x_),label = 'analytical', color = 'k' ,ls = ":")
        fig.legend(loc = 'upper right', bbox_to_anchor = (0.95,0.9))
        ax1.set_title ('Live points update distribution for various n_update')

        #cumulants
        samples = np.sort(samples, axis = -1)
        x_ = np.linspace(0,1,totsamp)
        for i in range(len(nsteps)):
            plt.step(samples[i], x_, color = colors[i])
        x_ = np.linspace(mymodel.center[0] - (-2*logL)**(1/2) + 0.01 ,mymodel.center[0] + (-2*logL)**(1/2) - 0.01  ,1000)
        plt.plot(x_, np.cumsum(analytical(x_) * np.diff(x_)[0]),color = 'k' ,ls = ":")

        fig.tight_layout()
        plt.show()
        print(kstest(samples[0],samples[1]))



    ####tests if sampling over threshold is correct
    import matplotlib.pyplot as plt
    test1()
