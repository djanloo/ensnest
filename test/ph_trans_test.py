import numpy as np
import matplotlib.pyplot as plt

import model
from NestedSampling import mpNestedSampler as mpns
import stdplots

M  = model.PhaseTransition()
ns = mpns(M, nlive = 500, evosteps = 500, filename = 'phasetransition', load_old = False)
ns.run()
stdplots.XLplot(ns)
stdplots.scat3D(ns)
print(ns.run_time)
plt.show()
