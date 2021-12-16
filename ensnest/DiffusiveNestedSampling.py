import numpy as np
from numpy.random import uniform as U

from . import samplers
from . import model
from .NestedSampling import mpNestedSampler as mns
from .NestedSampling import NestedSampler

from tqdm import tqdm, trange
from timeit import default_timer as time

import sys
import os


class mixtureAIESampler(samplers.AIESampler):
    """An AIE sampler for mixtures of pdf.

    It is really problem-specific, forked from AIESampler class because of the great conceptual differences.

    Allws only exponential and uniform weighting of the levels.

    Attibutes
    ---------

        chain_j : np.ndarray
            the chain of j values for each walker for each time
        level_logL : np.ndarray
            likelihood levels
        level_logX : np.ndarray
            prior mass levels
        Lambda : float
            exploration factor
        chain : np.ndarray
            the array of position, likelihood and prior of each point

    """

    def __init__(self, model, mcmc_length, nwalkers=10, space_scale=None, verbosity=0):

        super().__init__(
            model, mcmc_length, nwalkers, verbosity=verbosity, space_scale=space_scale
        )

        # for each point, memorize the j of the pdf sampled
        self.chain_j = np.zeros((mcmc_length, nwalkers), dtype=np.int)
        # the array of the likelihoods for each level
        self.level_logL = np.array([-np.inf])
        # estimated log(prior mass) for each level
        self.level_logX = np.array([0.0])
        self.Lambda = 0.1  # exp weighting constant

    def AIEStep(self, continuous=False, uniform_weights=False):
        """Single step of mixtureAIESampler. Overrides the parent class method.

        Args
        ----
            continuous : ``bool``, optional
                If true use modular index assignment, overwriting past values as
                ``self.elapsed_time_index`` > ``self.length``
            uniform_weights : ``bool``, optional
                If ``True`` sets uniform weighting between levels instead of exponential.

        Note
        ----
            The diference between the AIEStep function of :class:`~samplers.AIESampler` is that every
            single-particle distribution is chosen randomly between the ones in the mixture.

            This is equivalent to choosing a logL_threshold and rejecting a proposal point if its logL < logL_threshold
        """
        t_now = self.elapsed_time_index
        t_next = self.elapsed_time_index + 1

        if continuous:
            t_now = t_now % self.length
            t_next = t_next % self.length

        # MIXTURE: chosing j
        j = self.chain_j[t_now]
        new_j = j + np.random.choice([-1, 1], size=self.nwalkers)

        # automatically refuse if the level does not exist (yet)
        new_j[new_j >= len(self.level_logX)] = len(self.level_logX) - 1
        new_j[new_j < 0] = 0

        if uniform_weights:
            W = 0
        else:
            W = (new_j - j) / self.Lambda - self.level_logX[new_j] + self.level_logX[j]

        # exponential weights
        log_accept_prob_j = W

        accepted_j = log_accept_prob_j > np.log(U(0, 1, size=self.nwalkers))
        self.chain_j[t_next, accepted_j] = new_j[accepted_j]
        self.chain_j[t_next, np.logical_not(accepted_j)] = j[np.logical_not(accepted_j)]

        logL_thresholds = self.level_logL[self.chain_j[t_next]]

        # STANDARD AIEStep -----------------------------------------------
        # considers the whole ensamble at a time
        current_walker_position = self.chain[t_now, :]["position"]

        # generate a number from 1 to self.nwalkers-1
        delta_index = ((self.nwalkers - 2) * np.random.rand(self.nwalkers) + 1).astype(
            int
        )

        # for each walker selects randomly another walker as a pivot for the
        # stretch move
        pivot_index = (np.arange(self.nwalkers) + delta_index) % self.nwalkers
        pivot_position = self.chain[t_now, pivot_index]["position"]

        z = self.get_stretch(size=self.nwalkers)
        proposal = pivot_position + z[:, None] * (
            current_walker_position - pivot_position
        )

        log_prior_proposal = self.model.log_prior(proposal)
        log_likelihood_proposal = self.model.log_likelihood(proposal)
        log_prior_current = self.chain[t_now, :]["logP"]

        if not np.isfinite(log_prior_current).all():
            breakpoint()
            print(f"FATAL: past point is in impossible position")
            exit()

        # if proposal is outside its level, it is not accepted
        # this line is mainly the one that makes diffusive NS
        log_prior_proposal[log_likelihood_proposal < logL_thresholds] = -np.inf

        log_accept_prob = (
            (self.model.space_dim - 1) * np.log(z)
            + log_prior_proposal
            - log_prior_current
        )

        # if point is out of function domain, sets rejection
        log_accept_prob[np.isnan(log_prior_proposal)] = -np.inf

        accepted = log_accept_prob > np.log(U(0, 1, size=self.nwalkers))

        # assigns accepted values
        self.chain["position"][t_next, accepted] = proposal[accepted]
        self.chain["logP"][t_next, accepted] = log_prior_proposal[accepted]
        self.chain["logL"][t_next, accepted] = log_likelihood_proposal[accepted]

        # copies rejected values
        self.chain[t_next, np.logical_not(accepted)] = self.chain[
            t_now, np.logical_not(accepted)
        ]
        self.elapsed_time_index = t_next

    def sample_prior(self, progress=False, **kwargs):
        """Fills the chain by sampling the mixture.

        Args
        ----
            progress : bool
                Displays a progress bar. Default: ``False``
            uniform_weights : ``bool``, optional
                If ``True`` sets uniform weighting between levels instead of exponential.
        """
        if progress:
            desc = "sampling mixture"
            for t in tqdm(range(self.elapsed_time_index, self.length - 1), desc=desc):
                self.AIEStep(**kwargs)
        else:
            for t in range(self.elapsed_time_index, self.length - 1):
                self.AIEStep(**kwargs)
        return self

    def join_chains(self, burn_in=0.2, clean=True):
        """Joins the walkers, joins chain_j and also clean samples"""
        j_ = self.chain_j.flatten().astype(int)
        samps_ = self.chain.flatten()

        j_ = j_[int(burn_in * len(j_)) :]
        samps_ = samps_[int(burn_in * len(samps_)) :]

        if clean:
            correct = samps_["logL"] > self.level_logL[j_]
            samps_ = samps_[correct]
            j_ = j_[correct]
        return (samps_, j_)


class DiffusiveNestedSampler(NestedSampler):
    def __init__(
        self,
        model,
        max_n_levels=100,
        nlive=100,
        chain_length=1000,
        clean_samples=True,
        filename=None,
        load_old=None,
    ):

        self.model = model
        self.nlive = nlive

        self.max_n_levels = max_n_levels
        self.level_logL = np.array([-np.inf])
        self.level_logX = np.array([0.0])

        self.logZ = -np.inf
        self.logZ_error = None
        self.Z = None
        self.Z_error = None

        self.Xratio = 0.95  # np.e**(-1)

        self.sampler = mixtureAIESampler(self.model, chain_length, nwalkers=nlive)
        self.clean_samples = clean_samples

        self.n_j = np.array([])
        self.n_exceeded = np.array([])
        self.Ncontinue = 50

        # load&save
        self.run_code = hash(self)
        self.filename = str(self.run_code) if filename is None else filename
        self.load_old = load_old
        self.prepare_save_load()
        self.path = os.path.join(self.path, "diffusive")
        self.check_saved()

    def G(self):
        """Theorethical cumulative density function for X,
        given that X is smaller than the last level.
        It is reasonable that the major source of error in assessing the X values
        is originated from this assumption, so for generality G has its own
        function in case of future theoretical refinements.
        """
        q = 1.0 - self.Xratio
        return q

    def update_sampler(self):
        self.sampler.level_logL = self.level_logL
        self.sampler.level_logX = self.level_logX

    def run(self):
        """Runs the code"""
        if self.loaded:
            print("Diffusive run loaded from file")
            return
        current_logL_array = np.array([])
        self.run_time = time()
        with tqdm(total=self.max_n_levels, desc="generating new levels") as pbar:
            while len(self.level_logX) < self.max_n_levels:

                new, js = self.sampler.sample_prior().join_chains(
                    burn_in=0.0, clean=self.clean_samples
                )
                current_logL_array = np.append(current_logL_array, new["logL"])
                new_level_logL = np.quantile(current_logL_array, self.G())
                current_logL_array = np.delete(
                    current_logL_array, current_logL_array < new_level_logL
                )
                self.level_logL = np.append(self.level_logL, new_level_logL)
                self.level_logX = np.append(
                    self.level_logX, self.level_logX[-1] + np.log(self.Xratio)
                )
                self.update_sampler()

                # reset sampler on last state
                self.sampler.reset(start=self.sampler.chain[-1])
                self.sampler.chain_j[0] = self.sampler.chain_j[-1]

                self.n_j = np.append(self.n_j, [0])
                self.n_exceeded = np.append(self.n_exceeded, [0])

                self.revise_X(new["logL"], js)
                pbar.update(1)
        plt.show()
        # before-revise quantities
        self.level_logX_before_revise = self.level_logX
        self.level_logL_before_revise = self.level_logL
        self.level_logL_before_revise = np.append(
            self.level_logL_before_revise, [self.level_logL_before_revise[-1]]
        )
        self.level_logX_before_revise = np.append(
            self.level_logX_before_revise, [-np.inf]
        )
        self.Z_before_revise = np.trapz(
            np.exp(self.level_logL_before_revise),
            x=-np.exp(self.level_logX_before_revise),
        )
        self.logZ_before_revise = np.log(self.Z_before_revise)
        self.run_time_before_revise = time() - self.run_time

        self.continue_exploration()
        self.close()
        self.run_time = time() - self.run_time
        self.save()

    def revise_X(self, points_logL, current_level_index):
        """Revises the X values of the levels.

        Args
        ----
            points_logL : np.ndarray
                the points generated while sampling the mixture

        """
        # when executed after a new level is created max(j) = nlvl - 2 so max(j + 1) = nlvl -1
        # when executed in continue mode max(j)  = nlvl - 1 and max(j+1) = nlvl, so level_logL[j+1] is out of bounds sometimes
        # this is solved by deleting j and logL where j = nlvl - 1

        filter = current_level_index != len(self.level_logL) - 1
        filtered_logLs = points_logL[filter]
        current_level_index = current_level_index[filter]

        self.n_j += np.bincount(current_level_index, minlength=len(self.n_j))[
            np.arange(len(self.n_j))
        ]
        where_exceeded = filtered_logLs > self.level_logL[current_level_index + 1]
        np.add.at(self.n_exceeded, current_level_index[where_exceeded], 1)

        delta_logX = np.log(self.n_exceeded + 5.0 * self.Xratio) - np.log(
            self.n_j + 5.0
        )
        logX = np.cumsum(delta_logX)
        logX = np.insert(logX, 0, 0)
        self.level_logX = logX

    def continue_exploration(self):
        """Continues the exploration using uniform weighting."""
        for i in tqdm(range(self.Ncontinue), desc="uniform exploration"):
            new, js = self.sampler.sample_prior(uniform_weights=True).join_chains(
                burn_in=0.0, clean=self.clean_samples
            )

            # reset sampler on last state
            self.sampler.reset(start=self.sampler.chain[-1])
            self.sampler.chain_j[0] = self.sampler.chain_j[-1]

            self.revise_X(new["logL"], js)

    def close(self):

        self.level_logL = np.append(self.level_logL, [self.level_logL[-1]])
        self.level_logX = np.append(self.level_logX, [-np.inf])
        self.Z = np.trapz(np.exp(self.level_logL), x=-np.exp(self.level_logX))
        self.logZ = np.log(self.Z)
        del self.sampler  # reduce useless save disk space

    def __hash__(self):
        return hash(
            (self.model, self.nlive, self.max_n_levels, self.Xratio, self.Ncontinue)
        )
