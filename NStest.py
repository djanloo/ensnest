import numpy as np
import unittest
from matplotlib import pyplot as plt
from matplotlib import cm
from NestedSampling import NestedSampler
import model
import samplers

def check_constrained_distrib():
    target = model.RosenBrock()
    nlive, npoints = 1_000, 80_000
    ns = NestedSampler(target, nlive = nlive,  npoints = npoints, evosteps = 200)

    test_step = np.arange(50,500,10)
    samp,times,ks_stats = ns.check_prior_sampling(-0.5, test_step , 100_000)

    xcol = cm.get_cmap('plasma') (np.linspace(1,0,len(test_step)))
    ycol = cm.get_cmap('viridis')(np.linspace(0,1,len(test_step)))

    fig, (axx,axy) = plt.subplots(2)
    for i, st in enumerate(test_step):
        axx.hist(samp[i,:,0], bins = 50, histtype = 'step', color = xcol[i], density = True,label = f'x - {st:4}')
        axy.hist(samp[i,:,1], bins = 50, histtype = 'step', color = ycol[i], density = True,label = f'y - {st:4}')
    print(ks_stats)
    plt.rc('font', size = 10)
    axx.legend()
    axy.legend()
    plt.show()


def rosenbrocktest():
    my_model = model.RosenBrock()
    nlive, npoints = 1_000, 50_000
    ns = NestedSampler(my_model, nlive = nlive,  npoints = npoints, evosteps = 1000)

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


def gaussiantest():
    nlive, npoints = 1_000,5_000
    ns = NestedSampler(model.Gaussian(5), nlive = nlive, npoints = npoints, evosteps = 1000)
    ns.run()
    plt.plot(ns.logX , ns.logL)
    print(f'integral = {np.exp(ns.logZ)*model.Gaussian(5).volume}')
    print(f'logX = {ns.logX}')
    print(f'logL = {ns.logL}')
    plt.show()
