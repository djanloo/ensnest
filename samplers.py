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
        self.verbosity  = 0

        self.chain      = np.zeros((self.length, self.nwalkers, self.model.space_dim))
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

    def __init__(self, model, mcmc_length, nwalkers=10,space_scale=4, verbosity=0):

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

    def sample_prior(self):
        """Samples prior.

        returns:
            np.ndarray : the chain obtained
        """
        for t in tqdm(range(self.length - 1), desc = f'Sampling log_prior' ):

            current_index = np.arange(self.nwalkers)
            current_value = self.chain[t,current_index, :]

            pivot_index = (current_index + randint(1,self.nwalkers, size = self.nwalkers)) % self.nwalkers
            pivot_value = self.chain[t,pivot_index, :]

            if (pivot_index == current_index).any():
                print('pivot index is the same as current')
                exit()

            z = self.get_stretch(size = self.nwalkers)
            proposal = pivot_value + z[:,None] * (current_value - pivot_value)

            log_accept_prob = ( self.model.space_dim - 1) * np.log(z) + self.model.log_prior(proposal) - self.model.log_prior(current_value)
            accepted = (log_accept_prob > np.log(U(0,1,size = self.nwalkers)))

            self.chain[t+1, accepted,:]                 = proposal[accepted]
            self.chain[t+1, np.logical_not(accepted),:] = self.chain[t, np.logical_not(accepted), :]
        return self

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
