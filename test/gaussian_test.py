import numpy as np
import matplotlib.pyplot as plt

from ensnest import model
from ensnest import mpNestedSampler

means = []
stds = []
logZ = []
dlogZ = []
T = []
for dim in np.arange(50,51):
    M  = model.Gaussian(dim = dim)
    ns = mpNestedSampler(M, nlive = 1500, evosteps = 2500, load_old=False, filename='2dgaussian')

    ns.run()
    print(np.exp(ns.logZ)*M.volume)
    means.append(np.mean(ns.means.copy().view(np.float64)))
    stds.append(np.mean(ns.stds.copy().view(np.float64)))
    logZ.append(ns.logZ)
    dlogZ.append(ns.logZ_error)
    T.append(ns.run_time)
    print(means, stds,logZ,dlogZ,T)
#np.savetxt('stats1500-2500.txt',(means, stds,logZ,dlogZ,T))
