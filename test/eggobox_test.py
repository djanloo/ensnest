import model
import NestedSampling
import numpy as np
import matplotlib.pyplot as plt
import stdplots


class eggbox(model.Model):

    def set_parameters(self):
        self.bounds = [[0,10*np.pi],[0,10*np.pi]]
        self.names = ['a','b']

    @model.Model.auto_bound
    def log_prior(self,x):
        return 0

    @model.Model.varenv
    def log_likelihood(self,x):
        return (2 + np.cos(x['a']/2.)*np.cos(x['b']/2.))**5.

model_ = eggbox()

mpns = NestedSampling.mpNestedSampler(model_, nlive=500, evosteps=500, filename='eggobox', load_old = True)
mpns.run()
print(f'run_time = {mpns.run_time}')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for ns in mpns.nested_samplers:
    ax.scatter(ns.points['position']['a'], ns.points['position']['b'], ns.points['logL'], c = ns.points['logL'] , cmap = 'plasma')

stdplots.XLplot(mpns)
plt.show()
