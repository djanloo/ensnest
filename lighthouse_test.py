import numpy as np
import model
from NestedSampling import NestedSampler
import matplotlib.pyplot as plt

class lighthouse_model(model.Model):

      def __init__(self,data):
          self.bounds = ([-10,0], [10,10])
          self.names  = ['a', 'b']
          self.data   = data
          super().__init__()

      @model.Model.auto_bound
      def log_prior(self,vars):
          return 0

      @model.Model.varenv
      def log_likelihood(self,vars):
          u = np.ones(vars.shape)
          for i in range(len(data)):
              u += np.log(vars['b']) - np.log(vars['b']**2 + (data[i] - vars['a'])**2)
          return u

data = np.array([-10,-5,-2,-2,-5,-1,-2,6,5,3])
mod = lighthouse_model(data)
ns  = NestedSampler(mod, nlive = 10, evosteps = 100, load_old=False, filename = 'lighthouse.nkn')
ns.run()
print(ns.Z, ns.Z_error)
fig, ax = plt.subplots()
ax.plot(ns.logX, ns.logL)

fig, scat = plt.subplots()
scat.scatter(ns.points['position'][:,0],ns.points['position'][:,1], c = np.exp(ns.points['logL']), cmap='plasma')
plt.show()
