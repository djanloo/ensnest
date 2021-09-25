'''
standard plots module
'''

import matplotlib.pyplot as plt
import numpy as np
from NestedSampling import NestedSampler,mpNestedSampler
import os
plt.rc('font', size = 11)
plt.rc('font', family = 'serif')


def XLplot(NS):
    XLfig, XLax = plt.subplots(1)

    if isinstance(NS, mpNestedSampler):

        for ns in NS.nested_samplers:
            path = ns.path
            XLax.plot(ns.logX, ns.logL)

        XLax.plot(NS.logX, NS.logL, color = 'k', ls =':',alpha = 0.5)

    elif isinstance(NS, NestedSampler):

        XLax.plot(NS.logX, NS.logL, color = 'k')

    XLax.set_xlabel('logX')
    XLax.set_ylabel('logL')
    x = min(NS.logX[10:-10])
    y = max(NS.logL[10:-10])/2 + min(NS.logL[10:-10])/2
    XLax.annotate(f'logZ = {NS.logZ:.3f} +- {NS.logZ_error:.3f} ', xy = (x,y)  )
    savepath = os.path.join(os.path.dirname(NS.path), 'logXlogL.pdf')
    plt.savefig(savepath)


def hist_points(NS):
    for name in NS.model.names:
        fig,ax = plt.subplots(1)
        ax.hist(NS.points['position'][name], bins = len(NS.points)//100 ,histtype = 'step', density = True, color = 'k')
        ax.set_xlabel(name)

        savepath = os.path.join(os.path.dirname(NS.path), f'hist-{name}.pdf')
        plt.savefig(savepath)

def scat(NS):
    if NS.model.space_dim != 2:
        raise ValueError('Space dimension is not 2')

    fig,ax = plt.subplots(1)
    scat = ax.scatter(NS.points['position'][NS.model.names[0]], NS.points['position'][NS.model.names[1]], c = np.exp(NS.points['logL']) , cmap = 'plasma', s = 10)

    x_ = NS.model.data[:,0]
    y_ =  NS.model.data[:,1]
    ax.scatter(x_,y_, color = 'green', marker = '^')

    weigthed = NS.weigths[:,None]*NS.points['position'].copy().view((np.float64, 2))
    means = np.sum(weigthed, axis = 0)
    ax.scatter(means[0],means[1],color = 'cyan')

    maxpost = NS.points[np.argmax(NS.points['logL'])]['position']
    ax.scatter(maxpost[NS.model.names[0]], maxpost[NS.model.names[0]], color = 'r')

    ax.set_xlabel(NS.model.names[0])
    ax.set_ylabel(NS.model.names[1])
    plt.colorbar(scat)
    savepath = os.path.join(os.path.dirname(NS.path), f'scat.pdf')
    plt.savefig(savepath)

def scat3D(NS):
    if NS.model.space_dim != 2:
        raise ValueError('Space dimension is not 2')

    fig= plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(NS.points['position'][NS.model.names[0]], NS.points['position'][NS.model.names[1]], np.exp(NS.points['logL']), c = np.exp(NS.points['logL']) , cmap = 'plasma')
    ax.set_xlabel(NS.model.names[0])
    ax.set_ylabel(NS.model.names[1])
    ax.set_zlabel('L')

    savepath = os.path.join(os.path.dirname(NS.path), f'3Dscat.pdf')
    plt.savefig(savepath)

def weigthscat(NS):
    if NS.model.space_dim != 2:
        raise ValueError('Space dimension is not 2')
    fig, ax = plt.subplots(1)
    ax.scatter(NS.points['position'][NS.model.names[0]], NS.points['position'][NS.model.names[1]], c = NS.weigths , cmap = 'plasma')
    ax.set_xlabel(NS.model.names[0])
    ax.set_ylabel(NS.model.names[1])

    savepath = os.path.join(os.path.dirname(NS.path), f'weigthscat.pdf')
    plt.savefig(savepath)
