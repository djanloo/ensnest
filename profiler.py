import model
import numpy as np
import cy_samplers as samplers


class MyModel(model.Model):
    def __init__(self):
        self.bounds = (-np.ones(2)*10 ,np.ones(2)*10 )
        super().__init__()

    @model.Model.auto_bound
    def log_prior(self,x):
        return 0

    def log_likelihood(self,x):
        return -0.5*np.sum(x**2,axis = -1)


my_model = MyModel()

#initialise evolver sampler
evolve_sampler = samplers.AIESampler(my_model, 20 ,nwalkers=10_000)

def lol():
    evolve_sampler.AIEStep(my_model.log_prior)
