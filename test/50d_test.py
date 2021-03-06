import numpy as np
import matplotlib.pyplot as plt

from ensnest import model
from ensnest import mpNestedSampler as mpns
from ensnest import stdplots
from ensnest import utils

class mGaussian(model.Model):

    def set_parameters(self):
        self.bounds = np.array([-5,5]*2).reshape(-1,2)

    @model.Model.auto_bound
    def log_prior(self,x):
        return 0

    def log_likelihood(self,x):
        return -0.5*np.sum(x**2,axis = -1)

model_ = mGaussian()

result         = mpns(model_, nlive = 2000, evosteps = 2000, load_old=False, evo_progress = False, filename = '50d')
result.run()
for ns in result.nested_samplers:
    print(f'single integral = {ns.Z*model_.volume} +- {ns.Z_error*model_.volume}')
    plt.step(ns.logX, ns.logL)

print(f'merged integral = {result.Z*model_.volume} +- {result.Z_error*model_.volume}')
plt.step(result.logX, result.logL, color = 'k')
print(f'run executed in {utils.hms(result.run_time)} ({len(result.logL)} samples)')

plt.figure(2)
for ns in result.nested_samplers:
    plt.plot(ns.logX, np.exp(ns.logL+ns.logX))
plt.plot(result.logX, np.exp(result.logL+result.logX), color = 'k')

plt.show()
