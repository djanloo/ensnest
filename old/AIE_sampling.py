'''
implementation of affine invariant ensemble MCMC
'''

import numpy as np
from  numpy.random import uniform as U
from  numpy.random import randint

import matplotlib.pyplot as plt
from tqdm import trange

def get_stretch(a):
    '''
    Generator of numbers distibuted as :math:`\\frac{1}{\\sqrt{z}} in [1/a,a]``
    Uses Inverse transform sampling
    '''
    assert a > 1, "Stretch interval parameter must be > 1"
    return (U(0,1)*(a**(1/2) - a**(-1/2) ) + a**(-1/2) )**2

def dummyfunc(u):
    """This is a dummy function.

    Args:
        x : ``bool``
            A true goal rosp
        y
            Variable with type unspecified

            .. note:: y can be anything

            Note:
                napoleon-notes can be everywhere tho

    Returns:
        bool: True if successful, False otherwise.

        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    .. note:: You asshole

    .. code-block:: python

        >>>>> pipo is log_likelihood
        >>>>> i don't like pipo

    Example:
        Display text is okay, math is :math:`\\sqrt{z + 1}`

        >>> numpy.buy(22) = 5


    .. math::
        \\sqrt{z}


    """
    b = 0
    return b

def AIE_sampling(log_f, bounds,space_scale, nwalkers = 10, nsteps = 100, verbose = False):
    '''Affine invariant ensemble sampling algorithm

    :param log_f: Function to be sampled
    :type log_f: function

    :param bounds: bounds of the space
    :type bounds: 2-tuple of arrays

    :param space_scale: stretch values are generated between space_scale and (space_scale)**(-1)
    :type space_scale: float

    :param nwalkers: the number of walkers of the ensemble
    :type nwalkers: int

    :return: the chain evolved organized as (time, walker, space_variables)

    '''
    #chain is (iteration (t), walker (k), axis (j))
    x = np.zeros((nsteps , nwalkers, len(bounds[0])))
    #initialise start point for each walker
    for k in range(nwalkers):
        x[0,k,:] = U(*bounds)
        print(f"init log_f for walker {k} ({x[0,k,:]}) is {log_f(x[0,k,:])}")
    for t in trange(nsteps - 1):
        for k in range(nwalkers):
            #instead of removing k from range(nwalkers) selects a delta in mod nwalkers
            selected_index = (k + randint(1,nwalkers)) % nwalkers
            if selected_index == k:
                print('selected is the same as current')
                exit()
            selected = x[t, selected_index , : ]
            z = get_stretch(space_scale)
            if verbose: print(f't {t}:\twalker {k} selected walker {selected_index} and stretch {z} ',end = ' ')
            proposal = selected + z*(x[t,k,:] - selected)
            log_accept_prob = ( len(bounds[0]) )*np.log(z) + log_f(proposal) - log_f(x[t,k,:])
            if np.isnan(log_accept_prob):
                print('Invalid value for AIE acceptance probability')
                exit(130)
            if verbose: print(f'(p = {log_accept_prob})')
            if log_accept_prob > np.log(U(0,1)):
                x[t + 1, k , :] = proposal
            else:
                x[t + 1, k , :] = x[t , k , : ]
    return x

#################################################################

if __name__ == '__main__':
    bounds = np.ones(2)*10
    bounds = ([-4,0],[6,30])
    NWALKERS = 100
    NSTEPS = 1000
    def log_prior(x,bounds):
        if (x > bounds[1]).any() or (x < bounds[0]).any():
            return -np.inf
        return np.sum(-0.5*x**2, axis = -1)

    def rosenbrock(x,bounds):
        if (x > bounds[1]).any() or (x < bounds[0]).any():
            return -np.inf
        x = x.reshape(-1,2)
        return -(1/20*np.sum( 100.*(x[:,1] - x[:,0]**2.)** 2. + (1. - x[:,0])**2.))

    scale_parameter = 100
    x = AIE_sampling(lambda x: rosenbrock(x,bounds), bounds, scale_parameter, nwalkers = NWALKERS, nsteps = NSTEPS, verbose = False)
    for walker in range(NWALKERS):
        plt.figure(1)
        plt.plot(x[:,walker,0])
        plt.figure(2)
        plt.hist(x[:,walker,0], bins = NSTEPS//500 ,histtype = 'step')

    plt.figure(3)
    plt.hist(x[NSTEPS//10:,:,0].flatten(), bins = 50 ,histtype = 'step')

    plt.figure(4)
    plt.scatter(x[:,:,0].flatten(),x[:,:,1].flatten(), alpha = 0.01)

    plt.show()
