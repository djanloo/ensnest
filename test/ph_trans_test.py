import numpy as np
import matplotlib.pyplot as plt

from ensnest import model
from ensnest import mpNestedSampler as mpns
from ensnest import stdplots

M  = model.PhaseTransition()
ns = mpns(M, nlive=500, evosteps=500, filename='phasetransition', load_old=True)
ns.run()
stdplots.XLplot(ns)
print(ns.run_time)
plt.show()
