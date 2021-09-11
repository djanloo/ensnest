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

        log_prior_proposal = self.model.log_prior(proposal)
        log_prior_current  = self.chain[self.elapsed_time_index, :]['logP']

        if not np.isfinite(log_prior_current).all():
            print(f'FATAL: past point is in impossible position')
            exit()

        #if a threshold Lmin is set, sets as 'impossible' the proposals outside
        if Lthreshold is not None:
            log_prior_proposal[self.model.log_likelihood(proposal) < Lthreshold] = -np.inf

        log_accept_prob = ( self.model.space_dim - 1) * np.log(z) + log_prior_proposal - log_prior_current

        #if point is out of function domain, sets rejection
        log_accept_prob[np.isnan(log_prior_proposal)] = -np.inf

        accepted = (log_accept_prob > np.log(U(0,1,size = self.nwalkers)))

        #assigns accepted values
        self.chain['position'][self.elapsed_time_index+1, accepted] = proposal[accepted]
        self.chain['logP'][self.elapsed_time_index+1, accepted]     = log_prior_proposal[accepted]
        self.chain['logL'][self.elapsed_time_index+1, accepted]     = self.model.log_likelihood(proposal[accepted])
        # copies rejected values
        self.chain[self.elapsed_time_index+1, np.logical_not(accepted)] = self.chain[self.elapsed_time_index, np.logical_not(accepted)]
        self.elapsed_time_index += 1

    def sample_prior(self, Lthreshold = None):
        """Samples function.

        The real problem for being used in NS is that it is not clear how
        to treat an Ensemble of particles.

        In vanilla NS one has a set of points {x1,---,xn}, chooses the worse, replace with another.

        Here the evolution of the single particle itself depends on what others are doing.

        One option could be (must be confirmed by theoretical calculations) taking the currentlive points and consider them as the ensemble,
        then generate a new point like so.

        The problem is that this sampler produces ``nwalker`` particles at a time, which means that the process would be:

            * generate nlive from prior (can use this func)
            * take worst
            * generate OTHER (nlive - 1) points
            * pick one of theese at random

        which doesn't seem a reasonable way to follow.

        Well, i could by the way proceed like this:

            * generate nlive from prior (can use this func)
            * take worst -> do stuff
            * generate a bunch of points (say M)
            * take M worst point -> do stuff

        but i don't think it is how the vanilla NS should work, because
        nlive is variable throughout the process. Check dynamic NS.

        returns:
            np.ndarray : the chain obtained

        """
        for t in range(self.length - 1):
            self.AIEStep(Lthreshold = Lthreshold)
        return self

    def get_new(self,Lmin):
        '''Returns a new different point from prior given likelihood threshold
        '''
        #generates nlive - 1 points over L>Lmin
        for t in range(1,self.length):
            self.AIEStep(Lthreshold = Lmin)

        #selects one of this point give it's different from the given ones
        is_duplicate    = (self.chain['logL'][self.elapsed_time_index] == self.chain['logL'][0][:,None]).any(axis = 0)
        n_duplicate     = np.sum(is_duplicate.astype(int))

        if n_duplicate/self.nwalkers > 0.8: print(f'>>>>>>>>>>> WARNING: {int(n_duplicate/self.nwalkers*100)}% of duplicate(s) found')

        correct_ones = self.chain[self.elapsed_time_index, np.logical_not(is_duplicate)]
        new_point    = correct_ones[np.random.randint(self.nwalkers - n_duplicate)]
        return new_point

    def reset(self):
        self.elapsed_time_index = 0
        self.chain      = np.zeros((self.length, self.nwalkers) , dtype=self.model.livepoint_t).squeeze()

    def tail_to_head(self):
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
