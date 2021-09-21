import numpy as np
import model
from NestedSampling import mpNestedSampler as mpns
import matplotlib.pyplot as plt
import utils

model_ = model.Gaussian(dim=5)

result         = mpns(model_, nlive = 5000, evosteps = 500, load_old=False, evo_progress = False)
result.run()
for ns in result.nested_samplers:
    print(f'single integral = {ns.Z*model_.volume} +- {ns.Z_error*model_.volume}')
    plt.plot(ns.logX, ns.logL)

print(f'merged integral = {result.Z*model_.volume} +- {result.Z_error*model_.volume}')
plt.plot(result.logX, result.logL, color = 'k')
print(f'run executed in {utils.hms(result.run_time)} min ({len(result.logL)} samples)')

plt.figure(2)
for ns in result.nested_samplers:
    plt.plot(ns.logX, np.exp(ns.logL+ns.logX))
plt.plot(result.logX, np.exp(result.logL+result.logX), color = 'k')
plt.show()
