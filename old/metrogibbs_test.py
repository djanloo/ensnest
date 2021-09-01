import numpy as np
import unittest
import my_NS as ns
import matplotlib.pyplot as plt
from tqdm import trange

L = 20
dim = 2
upper = np.ones(dim)*L/2
bounds = (-upper,upper)
Volume = L**dim

def log_likelihood(x):
    return -0.5*np.sum(x**2 + (x - 1)**2,axis = -1)

def log_prior(x):
    if (x > upper).any() or (x < -upper).any():
        return -np.inf
    return 0#-np.log(Volume)


class MGTest(unittest.TestCase):

    def setUp(self):
        self.N = 10000
        self.logLmin = -0.5 #say
        init_attempts = 0
        self.start_point = np.zeros(dim)
        while True:
            dots = (int(init_attempts/10000)%5)
            print('SetUp finding initial point' + dots*'.' + (5-dots)*' ' + f'{self.start_point}' , end = '\r')
            if log_likelihood(self.start_point) > self.logLmin:
                break
            self.start_point = np.random.uniform(-upper,upper)
            init_attempts += 1
        self.chain = np.array([self.start_point])

    def test_run(self):
        print(f'start from {self.start_point}')

        for i in trange(self.N):
            new_point, _ = ns.metro_gibbs(log_likelihood, log_prior,
                                self.logLmin,#logLmin,
                                self.start_point,
                                [0.1]*dim,#sigma,
                                chain_L = 10,
                                verbose = False
                                )
            self.start_point = new_point
            self.chain = np.append(self.chain,[new_point], axis = 0)
        print(self.chain)
        plt.plot(self.chain[:,0],ls = '',marker  = '.')
        plt.plot(self.chain[:,1],ls = '', marker = '.')
        plt.figure(2)
        plt.hist(self.chain[:,0], bins = 50, histtype = 'step')
        plt.hist(self.chain[:,1], bins = 50, histtype = 'step')
        plt.figure(3)
        plt.scatter(self.chain[:,0],self.chain[:,1],alpha = 0.1)
        plt.show()

if __name__ == '__main__':
    unittest.main(verbosity = 2)
