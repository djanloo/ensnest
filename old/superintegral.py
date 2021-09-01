import my_NS as ns
import numpy as np
import matplotlib.pyplot as plt

L = 5
dim = 1
upper = np.ones(dim)*L/2
Volume = L**dim

def log_likelihood(x):
    return -0.5*np.sum(x**2,axis = -1)

def log_prior(x):
    if (x > upper).any() or (x < -upper).any():
        return -np.inf
    return 0#-np.log(Volume)

bounds = (-upper,upper)

logX,logL,logZ, stats = ns.NS( log_likelihood,log_prior,
                            bounds,
                            Nlive = 100,
                            X_assessment = 'stochastic',
                            shrink_scale = False,
                            Npoints = 10000,
                            stop_log_relative_increment = -5,
                            verbose_search = False)

print('Integral = {} (should be {})'.format(Volume*np.exp(logZ), (2*np.pi)**(dim/2) ))

maincol = np.array([0.,0.,1.])
colors = [i*maincol for i in np.linspace(0.3,1,len(stats['points']))]
#plt.scatter(stats['points'][:,0],stats['points'][:,1],marker = '.' ,c = colors)

plt.figure(2)
plt.plot(logX,logL[:len(logX)])
plt.figure(3)
plt.plot(stats['points'][:,0],ls = '',marker = '.')
print(stats['points'][:,0])
plt.show()
