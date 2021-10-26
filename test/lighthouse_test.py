import numpy as np
import matplotlib.pyplot as plt

from ensnest import model
from ensnest import NestedSampler, mpNestedSampler
from ensnest import stdplots


class lighthouse_model(model.Model):

      def set_parameters(self,data):
          self.bounds = [[-10,10],[0,10]]
          self.names  = ['a','b']
          self.data   = data

      @model.Model.auto_bound
      @model.Model.varenv
      def log_prior(self,vars):
          return 0

      @model.Model.varenv
      def log_likelihood(self,vars):
          u = np.zeros(vars.shape)
          for i in range(len(self.data)):
              u += np.log(vars['b']) - np.log(vars['b']**2 + (self.data[i] - vars['a'])**2)
          return u

# x_observations = np.array([-9.,-8.,6.,7.])
x_observations = np.array([-9.1,8.,6])

M  = lighthouse_model(x_observations)
ns = mpNestedSampler(M,
                    nlive=500,
                    evosteps=500,
                    load_old=True,
                    filename='lighthouse',
                    a=3.
                    )

ns.run()

# stdplots.XLplot(ns)
# stdplots.hist_points(ns)
stdplots.scat(ns)
# stdplots.weightscat(ns)
stdplots.contour(ns, levels=20, cmap='plasma')

plt.show()
