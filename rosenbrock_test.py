import model
from NestedSampling import mpNestedSampler
import stdplots
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ

def rosenbrock(x,y):
    return np.exp(- .5*(y - x**2)**2 - 1./20.*(1.- x)**2)/110.

my_model = model.Rosenbrock()
x1,x2 = (-5.,5.)
y1,y2 = (-1.,10.)

Z,err = integ.dblquad(rosenbrock,x1,x2,y1,y2)
print(np.log(Z))

x = np.linspace(x1,x2, 1000)
y = np.linspace(y1,y2, 1000)
X,Y = np.meshgrid(x,y)
plt.contourf(X,Y, rosenbrock(X,Y))
plt.xlim(x1,x2)
plt.ylim(y1,y2)

mpns = mpNestedSampler(my_model, nlive = 500,  evosteps = 500, load_old = False, filename = 'rosenbrock')


mpns.run()
print(f'run_time = {mpns.run_time}')
print(f'logZ = {mpns.logZ} +- {mpns.logZ_error}')
stdplots.XLplot(mpns)
stdplots.scat3D(mpns)


plt.show()
