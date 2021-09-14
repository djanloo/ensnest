import numpy as np
import unittest

import NestedSampling
import model
import samplers

class rosenbrockTest(unittest.TestCase):
        my_model = model.RosenBrock()
        nlive, npoints = 1_000, 50_000
        ns = NestedSampler(my_model, nlive = nlive,  npoints = npoints, evosteps = 200)

        # samp,times,ks_stats = ns.check_prior_sampling(-0.5, [5,5], 100_000)
        # plt.hist(samp[0,:,0], bins = 50, histtype = 'step')
        # plt.hist(samp[1,:,0], bins = 50, histtype = 'step')
        # plt.show()
        # print(ks_stats)

        ns.run()
        print(f'logZ = {ns.logZ}')

        fig,ax = plt.subplots(1)
        plt.rc('font', size = 11)
        plt.rc('font', family = 'serif')
        ax.set_title(f'Rosenbrock model: {len(ns.points)} samples in {ns.run_time:.1f} seconds')

        ax.scatter(ns.points['position'][:,0], ns.points['position'][:,1], c = np.exp(ns.points['logL']), cmap = 'plasma')

        fig3d = plt.figure(3)
        ax    = fig3d.add_subplot(projection = '3d')
        ax.scatter(ns.points['position'][:,0],ns.points['position'][:,1],np.exp(ns.points['logL']), c = np.exp(ns.points['logL']), cmap = 'plasma')

        plt.show()
