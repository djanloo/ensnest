import numpy as np
import matplotlib.pyplot as plt

from ensnest import model
from ensnest import mpNestedSampler
from ensnest import stdplots

means = []
stds = []
logZ = []
dlogZ = []
T = []
for dim in np.arange(10,11):
    M  = model.nGaussian(dim = dim)
    ns = mpNestedSampler(M, nlive = 700, evosteps = 700, load_old=False, filename='gaussian', relative_precision=1e-8)

    ns.run()
    print(np.exp(ns.logZ)*M.volume)
    means.append(np.mean(ns.means.copy().view(np.float64)))
    stds.append(np.mean(ns.stds.copy().view(np.float64)))
    logZ.append(ns.logZ)
    dlogZ.append(ns.logZ_error)
    T.append(ns.run_time)
    print(means, stds,logZ,dlogZ,T)
    print(f'dim = {dim} H = {ns.H}')

plt.plot(ns.logX, np.exp(ns.logL))
plt.show()
#np.savetxt('stats1500-2500.txt',(means, stds,logZ,dlogZ,T))
