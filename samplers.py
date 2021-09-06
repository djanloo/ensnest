"""
Module containing the samplers used in main calculations.

Since almost every sampler is defined by a markov chain, basic attributes are
the model and the length of the chain.

Each sampler shoud be capable of tackling with discontinuous functions.

Since is intended to be used in nested sampling, each sampler should support likelihood constrained prior sampling (LCPS).

"""

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

        self.chain      = np.zeros((self.length, self.nwalkers, self.model.space_dim))

        #uniform initialisation
        for walker in range(self.nwalkers):
            self.chain[0, walker, : ] = U(*self.model.bounds)


class AIESampler(Sampler):
    '''The Affine-Invariant Ensemble sampler (Goodman, Weare, 2010).

    After a uniform initialisation step, for each particle k selects a *pivot* particle an then proposes

    .. math::
        j = k + random(0 \\rightarrow n)

        z \\char`~ g(z)

        y = x_j + z (x_k - x_j)

    and then executes a MH-acceptance over y (more information at <https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf>).

    '''

    def __init__(self, model, mcmc_length, nwalkers=10, space_scale = 4, verbosity=0):

        super().__init__(model, mcmc_length, nwalkers, verbosity=verbosity)

        if space_scale <= 1:
            print('space scale parameter must be > 1')
            exit()
        self.space_scale = space_scale

    def get_stretch(self, size = 1):
        '''
        Generates the stretch values given the scale_parameter ``a``.

        Output is distibuted as :math:`\\frac{1}{\\sqrt{z}}`  in :math:`[1/a,a]``.
        Uses inverse transform sampling
        '''

        return (U(0,1, size = size )*(self.space_scale**(1/2) - self.space_scale**(-1/2) ) + self.space_scale**(-1/2) )**2

    def AIEStep(self, log_function):
        '''Single step of AIESampler

            Args
            ----
                log_function : function
        '''
        #considers the whole ensamble at at time
        current_walker_index    = np.arange(self.nwalkers)
        current_walker_position = self.chain[self.elapsed_time_index,current_walker_index, :]

        #for each walker selects randomly another walker as a pivot for the stretch move
        pivot_index     = (current_walker_index + randint(1,self.nwalkers, size = self.nwalkers)) % self.nwalkers
        pivot_position  = self.chain[self.elapsed_time_index, pivot_index, : ]

        if (pivot_index == current_walker_index).any():
            print('pivot index is the same as current')
            exit()

        z        = self.get_stretch(size = self.nwalkers)
        proposal = pivot_position + z[:,None] * (current_walker_position - pivot_position)

        log_function_proposal = log_function(proposal)
        log_function_current  = log_function(current_walker_position)

        log_accept_prob = ( self.model.space_dim - 1) * np.log(z) + log_function_proposal - log_function_current

        #if point is out of function domain, sets rejection
        log_accept_prob[np.isnan(log_function_proposal)] = -np.inf

        if not np.isfinite(log_function_current).all():
            print(f'FATAL: past point is in impossible position')
            exit()

        accepted = (log_accept_prob > np.log(U(0,1,size = self.nwalkers)))

        self.chain[self.elapsed_time_index+1, accepted,:]                 = proposal[accepted]
        self.chain[self.elapsed_time_index+1, np.logical_not(accepted),:] = self.chain[self.elapsed_time_index, np.logical_not(accepted), :]
        self.elapsed_time_index += 1

    def sample_function(self,log_function):
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
        boxed_log_function = lambda x: log_function(x) + self.model.log_chi(x)
        for t in trange(self.length - 1):
            self.AIEStep(boxed_log_function)
        return self

    def likelihood_constraint(self,x,worstL):
        result = np.log((self.model.log_likelihood(x) > worstL).astype(int))
        return result

    def sample_over_threshold(self,Lmin):
        '''Performs likelihood-constrained prior sampling.

        The NS algorithm starts with a set :math:`{\theta_i}` of points distributed as :math:`\pi(\theta)`

        After excluding the worst (:math:`L_{w}`), a new point of likelihood greater than :math:`L_{w}`
        has to be generated. One has freedom to choose the way this new point is yielded,
        as long as the new point has pdf:

            .. math::
                p(\theta_{new}) d\theta_{new} = \pi(\theta_{new}) (L(\theta_{new}) > L_w)
                p(\theta_{new}) d\theta_{new} = 0                 (L(\theta_{new}) > L_w)

        This function uses the AIE sampler on the current live points (nlive -1) and
        evolves them in the likelihood-constrained prior to generate other (nlive - 1)
        points, then takes one at random.

        Furthermore, the newly generated point is forced to have likelihood different from all the initial ones.

        note
        ----
            (at the moment) sampler has to be initialised to points already inside bounds

            To solve: conflict nlive -> nlive - 1
        '''
        LCprior = lambda x: self.model.log_prior(x) + self.likelihood_constraint(x, Lmin)

        if not np.isfinite(LCprior(self.chain[self.elapsed_time_index])).all():
            print('WARNING: at least one point is out of L-bounds')

        #TODO: this line costs a lot of lik-evaluation. solve
        initial_log_likelihoods = self.model.log_likelihood(self.chain[self.elapsed_time_index])

        #generates nlive - 1 points over L>Lmin
        for t in range(self.length - 1):
            self.AIEStep(LCprior)

        #selects one of this point give it's different from the given ones
        is_duplicate    = (self.model.log_likelihood(self.chain[self.elapsed_time_index]) == initial_log_likelihoods[:,None]).any(axis = 0)
        n_duplicate     = np.sum(is_duplicate.astype(int))

        if is_duplicate.any(): print(f'>>>>>>>>>>> WARNING: {n_duplicate} duplicate(s) found')

        correct_ones = self.chain[self.elapsed_time_index, np.logical_not(is_duplicate), :]
        new_point    = correct_ones[np.random.randint(self.nwalkers - n_duplicate), :]
        return new_point

    def reset(self):
        self.elapsed_time_index = 0
        self.chain = np.zeros((self.length, self.nwalkers, self.model.space_dim))


    def join_chains(self, burn_in = 0.02):
        '''Joins the chains for the ensemble after removing  ``burn_in`` \% of each single_particle chain.

        Args
        ----
            burn_in : float, optional
                the burn_in percentage.

                Must be ``burn_in`` > 0 and ``burn_in`` < 1.
        '''
        joined_chain = self.chain[ int(burn_in*self.length) :].reshape(-1, self.model.space_dim)
        return joined_chain
