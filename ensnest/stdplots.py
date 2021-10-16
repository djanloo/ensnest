'''
standard plots module
'''

import matplotlib.pyplot as plt
import numpy as np
from .NestedSampling import NestedSampler, mpNestedSampler
import os
plt.rc('font', size=11)
plt.rc('font', family='serif')


def XLplot(NS, fig_ax=None):
    '''Does the (logX, logL) plot.

    Args
    ----
        NS : ``NS/mpNS/DNS sampler``

    '''
    XLfig, XLax = plt.subplots(1) if fig_ax is None else fig_ax

    if isinstance(NS, mpNestedSampler):

        for ns in NS.nested_samplers:
            path = ns.path
            XLax.plot(ns.logX, ns.logL, color='k', alpha=0.5)

        XLax.plot(NS.logX, NS.logL, color='r')

    elif isinstance(NS, NestedSampler):

        XLax.plot(NS.logX, NS.logL, color='k')

    XLax.set_xlabel('logX')
    XLax.set_ylabel('logL')
    x = min(NS.logX[10:-10])
    y = max(NS.logL[10:-10]) / 2 + min(NS.logL[10:-10]) / 2
    XLax.annotate(f'logZ = {NS.logZ:.3f} +- {NS.logZ_error:.3f} ', xy=(x, y))
    savepath = os.path.join(os.path.dirname(NS.path), 'logXlogL.pdf')
    plt.savefig(savepath)


def hist_points(NS):
    '''Plots the histogram of the equally weighted points

    Args
    ----
        NS : ``NS/mpNS sampler``

    '''
    Nbins = int(1 + 4 * np.log(len(NS.ew_samples))
                )  # Sturge's rule for bins' number
    for name in NS.model.names:
        fig, ax = plt.subplots(1)
        ax.hist(
            NS.ew_samples['position'][name],
            bins=Nbins,
            histtype='step',
            density=True,
            color='k',
            label='posterior')
        ax.set_xlabel(name)

        ax.axvline(NS.means[name], ls=':', color='k', label='mean')

        savepath = os.path.join(os.path.dirname(NS.path), f'hist-{name}.pdf')
        ax.legend()
        plt.savefig(savepath)


def scat(NS, fig_ax=None):
    '''Does a 2D scatter plot.

    Args
    ----
        NS : ``NS/mpNS sampler``

    '''
    fig, ax = plt.subplots(1) if fig_ax is None else fig_ax
    if NS.model.space_dim != 2:
        raise ValueError('Space dimension is not 2')

    scat = ax.scatter(NS.points['position'][NS.model.names[0]],
                      NS.points['position'][NS.model.names[1]],
                      c=np.exp(NS.points['logL']),
                      cmap='plasma',
                      s=10)

    weighted = NS.weights[:, None] * \
        NS.points['position'].copy().view((np.float64, 2))
    means = np.sum(weighted, axis=0)
    ax.scatter(means[0], means[1], color='cyan')

    maxpost = NS.points[np.argmax(NS.points['logL'])]['position']
    ax.scatter(maxpost[NS.model.names[0]],
               maxpost[NS.model.names[1]], color='r')

    ax.set_xlabel(NS.model.names[0])
    ax.set_ylabel(NS.model.names[1])
    plt.colorbar(scat)
    savepath = os.path.join(os.path.dirname(NS.path), f'scat.pdf')
    plt.savefig(savepath)


def scat3D(NS):
    '''Does a 3D scatter plot (x,y,L).

    Args
    ----
        NS : ``NS/mpNS sampler``

    '''
    if NS.model.space_dim != 2:
        raise ValueError('Space dimension is not 2')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(NS.points['position'][NS.model.names[0]],
               NS.points['position'][NS.model.names[1]],
               np.exp(NS.points['logL']),
               c=np.exp(NS.points['logL']),
               cmap='plasma')
    
    L = tuple([s[1] - s[0] for s in np.transpose(NS.model.bounds)])
    L += (8,)
    ax.set_box_aspect(L)  # aspect ratio is 1:1:1 in data space

    ax.set_xlabel(NS.model.names[0])
    ax.set_ylabel(NS.model.names[1])
    ax.set_zlabel('L')

    savepath = os.path.join(os.path.dirname(NS.path), f'3Dscat.pdf')
    plt.savefig(savepath)


def weightscat(NS, fig_ax=None):
    '''Does a 2D scatter plot using a colormap to display weighting.

    Args
    ----
        NS : ``NS/mpNS sampler``

    '''
    fig, ax = plt.subplots(1) if fig_ax is None else fig_ax
    if NS.model.space_dim != 2:
        raise ValueError('Space dimension is not 2')
    ax.scatter(NS.points['position'][NS.model.names[0]],
               NS.points['position'][NS.model.names[1]],
               c=NS.weights,
               cmap='plasma',
               alpha=0.9)
    ax.set_xlabel(NS.model.names[0])
    ax.set_ylabel(NS.model.names[1])

    savepath = os.path.join(os.path.dirname(NS.path), f'weightscat.pdf')
    plt.savefig(savepath)
