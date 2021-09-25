import model
import numpy as np
from NestedSampling import mpNestedSampler
import matplotlib.pyplot as plt

M  = model.Gaussian(dim = 2)
ns = mpNestedSampler(M, nlive = 1000, evosteps = 10000, load_old=False, filename='2dgaussian')

ns.run()
weigthed = ns.weigths[:,None]*ns.points['position'].copy().view((np.float64, 2))
means = np.sum(weigthed, axis = 0)
weigthed_sq = ns.weigths[:,None]*ns.points['position'].copy().view((np.float64, 2))**2
std = np.sum(weigthed_sq, axis = 0) - means**2
print(means)
print(std)

plt.scatter(ns.points['position']['a'], ns.points['position']['var0'], c = ns.weigths, cmap = 'plasma')
plt.show()
