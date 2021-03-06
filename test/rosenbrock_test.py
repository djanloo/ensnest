import matplotlib.pyplot as plt
import numpy as np

from ensnest import model
from ensnest import mpNestedSampler
from ensnest import stdplots

my_model = model.Rosenbrock()

mpns = mpNestedSampler(my_model, nlive = 700,  evosteps = 700, load_old = False, filename = 'rosenbrock')

mpns.run()
print(f'run_time = {mpns.run_time}')
print(f'logZ = {mpns.logZ} +- {mpns.logZ_error}')
stdplots.XLplot(mpns)
stdplots.scat3D(mpns)

plt.show()
