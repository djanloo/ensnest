import model
import numpy as np
from NestedSampling import mpNestedSampler
import matplotlib.pyplot as plt
means = []
stds = []
for dim in np.arange(1,40):
    M  = model.Gaussian(dim = dim)
    ns = mpNestedSampler(M, nlive = 500, evosteps = 1500, load_old=False, filename='2dgaussian')

    ns.run()

    means.append(np.mean(ns.means.copy().view(np.float64)))
    stds.append(np.mean(ns.stds.copy().view(np.float64)))

print(means, stds)
np.savetxt('stats3.txt',(means, stds))
