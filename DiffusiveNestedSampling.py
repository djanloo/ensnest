import numpy as np
import samplers

class mixtureAIESampler(samplers.AIESampler):
    '''An AIE sampler for mixtures of pdf.

    It is really problem-specific, forked from AIESampler class because of the great conceptual differences.
    '''

    def __init__(self, model, mcmc_length, nwalkers=10, space_scale = None, verbosity=0):

        super().__init__(model, mcmc_length, nwalkers, verbosity=verbosity, space_scale = space_scale)

        self.chain_j = np.zeros((mcmc_length, nwalkers))    # for each point, memorize the j of the pdf sampled
        self.level_logL = np.array([-np.inf])               # the array of the likelihoods for each level
        self.level_logX = np.array([0.])                    # estimated log(prior mass) for each level
        self.J      = 0                                     # max level number
        self.Lambda = 1

    def AIEStep(self, continuous=False):
        '''Single step of mixtureAIESampler. Overrides the parent class method.

            Args
            ----
                continuous : ``bool``, optional
                    If true use modular index assignment, overwriting past values as
                    ``self.elapsed_time_index`` > ``self.length``

            Note
            ----
                The diference between the AIEStep function of :class:`~samplers.AIESampler` is that every
                single-particle distribution is chosen randomly between the ones in the mixture.

                This is equivalent to choosing a logL_threshold and rejecting a proposal point if its logL < logL_threshold
        '''
        t_now  = self.elapsed_time_index
        t_next = self.elapsed_time_index + 1

        if continuous:
            t_now  = t_now  % self.length
            t_next = t_next % self.length

        # MIXTURE: chosing j
        j     = self.chain_j[t_now]
        new_j = j + np.random.choice([-1,1], size = self.nwalkers)

        #automatically refuse if the level does not exist yet
        new_j[new_j > self.J] -=1

        log_accept_prob_j = (new_j-j)/self.Lambda - self.level_logX[new_j] + sel.level_logX[j] #exponential weights

        accepted_j  = ( log_accept_prob > np.log(U(0,1,size = self.nwalkers)) )
        self.chain_j[t_next, accepted_j]                    = new_j[accepted_j]
        self.chain_j[t_next, np.logical_not(accepted_j)]    = j[np.logical_not(accepted_j)]

        logL_thresholds = self.levels_logL[self.chain_j[t_next]]

        # STANDARD AIEStep -----------------------------------------------
        #considers the whole ensamble at a time
        current_walker_position = self.chain[t_now,:]['position']

        #generate a number from 1 to self.nwalkers-1
        delta_index = ((self.nwalkers-2)*np.random.rand(self.nwalkers)+1).astype(int)

        #for each walker selects randomly another walker as a pivot for the stretch move
        pivot_index     = (np.arange(self.nwalkers) + delta_index   ) % self.nwalkers
        pivot_position  = self.chain[t_now, pivot_index]['position']

        z        = self.get_stretch(size = self.nwalkers)
        proposal = pivot_position + z[:,None] * (current_walker_position - pivot_position)

        log_prior_proposal      = self.model.log_prior(proposal)
        log_likelihood_proposal = self.model.log_likelihood(proposal)
        log_prior_current       = self.chain[t_now, :]['logP']

        if not np.isfinite(log_prior_current).all():
            breakpoint()
            print(f'FATAL: past point is in impossible position')
            exit()

        #if proposal is outside its level, it is not accepted
        #this line is mainly the one that makes diffusive NS
        log_prior_proposal[log_likelihood_proposal < logL_thresholds] = -np.inf

        log_accept_prob = ( self.model.space_dim - 1) * np.log(z) + log_prior_proposal - log_prior_current

        #if point is out of function domain, sets rejection
        log_accept_prob[np.isnan(log_prior_proposal)] = -np.inf

        accepted = (log_accept_prob > np.log(U(0,1,size = self.nwalkers)))

        #assigns accepted values
        self.chain['position'][t_next, accepted] = proposal[accepted]
        self.chain['logP'][t_next, accepted] = log_prior_proposal[accepted]
        self.chain['logL'][t_next, accepted] = log_likelihood_proposal[accepted]

        # copies rejected values
        self.chain[t_next, np.logical_not(accepted)] = self.chain[t_now, np.logical_not(accepted)]
        self.elapsed_time_index = t_next

    def sample_prior(self, progress = False):
        """Fills the chain by sampling the mixture.
        """
        if progress:
            desc = 'sampling prior'
            if Lthreshold is not None:
                desc += f' over logL > {Lthreshold:.2f}'
            for t in tqdm(range(self.elapsed_time_index, self.length - 1), desc = desc):
                self.AIEStep()
        else:
            for t in range(self.elapsed_time_index, self.length - 1):
                self.AIEStep()
        return self
