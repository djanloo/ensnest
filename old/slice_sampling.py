import numpy as np
from matplotlib import pyplot as plt
from numpy.random import uniform as u
from tqdm import tqdm,trange

bounds = (-3,5)

def log_p(x):
    return np.exp(-0.5*x**2)

def find_I(log_func, bounds, point,log_value, window_size, verbose = False):
    iterations = 0
    I = list(bounds)
    while True:
        L = point - window_size/2
        R = point + window_size/2
        if  (log_func(R) < log_value and log_func(L) < log_value
            or (R > bounds[1] and L < bounds[0])):
            if L < bounds[0]:
                L = bounds[0]
            if R > bounds[1]:
                R = bounds[1]
            return (L,R), iterations
        else:
            window_size *= 2
            if verbose: print(f'w = {window_size}', end = '\r')
            iterations += 1

def test_find_I():
    log_f = lambda x: -x**2 + 5
    x = 0.01
    y = log_f(x) + np.log(u(0,1))
    I, iter = find_I(log_f, x, y,0.001)
    print(f'Found interval {I} in {iter} iterations')
    x_ = np.linspace(*bounds)
    plt.plot(x_, log_f(x_))
    plt.plot(list(I),[y,y])
    plt.axvline(x)

def slice_sampling(log_f, bounds, window_size, N = 100, verbose = False):
    x = np.zeros(N)
    x[0] = u(*bounds)
    for i in trange(N-1):
        y = log_f(x[i]) + np.log(u(0,1))
        I,iter = find_I(log_f,bounds, x[i] , y , window_size, verbose = verbose)
        if verbose: print(f'i:{i}\ty: {y}\tI: {I}({iter} iter) ', end = '')
        while True:
            proposed_x = u(*I)
            if verbose: print('#', end = '')
            if log_f(proposed_x) > y: break
        x[i+1] = proposed_x
        if verbose: print('Done')
    return x


if __name__ == '__main__':
    samples = slice_sampling(lambda x: x, bounds, 0.01, N = 100, verbose = True)
    plt.hist(samples, bins = 50)
    plt.show()
