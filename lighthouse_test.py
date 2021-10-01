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
      @model.Model.varenv
      def log_prior(self,vars):
          # return 0
          return -0.5*((vars['a']+1)**2/(1**2))

      @model.Model.varenv
      def log_likelihood(self,vars):
          u = np.zeros(vars.shape)
          for i in range(len(self.data)):
              u += np.log(vars['b']) - np.log(vars['b']**2 + (self.data[i] - vars['a'])**2)
          return u

x_observations = np.array([-9.,-8.,5.,6.,7.])
model_        = lighthouse_model(x_observations)
ns            = mpNestedSampler(model_, nlive = 1000, evosteps = 1000, load_old=False, filename='lighthouse')

ns.run()
#stdplots.XLplot(ns)
stdplots.hist_points(ns)
#stdplots.scat(ns)
stdplots.weightscat(ns)
#plt.scatter(ns.points['position']['a'],ns.points['position']['b'], c = ns.weigths, cmap = 'plasma',s = 10)

# t = np.linspace(0,1,len(ns.points))
# plt.scatter(ns.points['position']['a'], ns.points['position']['b'], c = t, cmap = 'plasma')
plt.show()
