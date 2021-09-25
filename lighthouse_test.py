import numpy as np
import model
from NestedSampling import NestedSampler, mpNestedSampler
import matplotlib.pyplot as plt
import stdplots

class lighthouse_model(model.Model):

      def set_parameters(self,data):
          self.bounds = [[-10,10],[0,10]]
          self.names  = ['a','b']
          self.data   = data

      @model.Model.auto_bound
      def log_prior(self,vars):
          return 0

      @model.Model.varenv
      def log_likelihood(self,vars):
          u = np.zeros(vars.shape)
          for i in range(len(self.data)):
              u += np.log(vars['b']) - np.log(vars['b']**2 + (self.data[i] - vars['a'])**2)
          return u

x_observations = [-9.8,-8.5,9.1,9.9,7.4,-6.]
model_        = lighthouse_model(x_observations)
ns            = mpNestedSampler(model_, nlive = 5000, evosteps = 500, load_old=True, filename='lighthouse')

ns.run()
# stdplots.XLplot(ns)
# stdplots.hist_points(ns)
# stdplots.scat(ns)
plt.scatter(ns.points['position']['a'],ns.points['position']['b'], c = ns.weigths, cmap = 'plasma',s = 10)
plt.show()
