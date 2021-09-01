import my_NS as ns
import numpy as np
import matplotlib.pyplot as plt
import unittest

L = 30
dim = 5
upper = np.ones(dim)*L/2
bounds = (-upper,upper)
Volume = L**dim

def log_likelihood(x):
    return -0.5*np.sum((x)**2,axis = -1)

def log_prior(x):
    if (x > upper).any() or (x < -upper).any():
        return -np.inf
    return 0#-np.log(Volume)

class GaussianTestCase(unittest.TestCase):

    def setUp(self):
        self.bounds = bounds

    def test_run(self):
        logX,logL,logZ, stats = ns.NS( log_likelihood,log_prior,
                                    self.bounds,
                                    Nlive = 500,
                                    explore = ns.metro_gibbs,
                                    X_assessment = 'stochastic',
                                    shrink_scale = False,
                                    Npoints = 100,
                                    stop_log_relative_increment = -np.inf,
                                    verbose_search = False,
                                    display_progress = True,
                                    chain_L = 100)
        dLogZ = stats['dLogZ']
        print(f'Integral = {Volume*np.exp(logZ)} +- {Volume*np.exp(logZ)*dLogZ} (should be {(2*np.pi)**(dim/2)})')
        colors = [[0,0,i/len(stats['points'])] for i in range(len(stats['points']))]
        plt.scatter(stats['points'][:,0],stats['points'][:,1],c = colors)
        plt.show()

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
