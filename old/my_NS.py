'''
Nested Sampling implementation
author: G.B.
'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

def logsubexp(x1,x2):
    assert x1 > x2, 'Logsubexp got wrong arguments (log(-c))'
    return  x1 + np.log1p(-np.exp(x2-x1))

def logsumexp(arg):
    '''Given a  vector [a1,a2,a3, ... ] returns log(e**a1 + e**a2 + ...)'''
    if isinstance(arg, np.ndarray):
        if len(arg) == 1:
            return arg
        elif len(arg) == 2:
            return np.logaddexp(arg[0],arg[1])
        else:
            result =  -np.inf
            for a in arg:
                result = np.logaddexp(result,a)
            return result
    else:
        raise ValueError('argument is not an array')

def metropolis(log_likelihood, log_prior, logLmin, p0, sigma, Narrest = 1000000, verbose = False, chain_L = 10):
    attempts = 1
    while True:
        new_point = np.random.normal(p0,sigma)
        if new_point.shape != p0.shape:
            print('Something went wrong in metropolis: shape mismatch')
            exit()
        logL_new = log_likelihood(new_point)
        if verbose:
            print("logL_min = {: 7.3}\tlogL_proposed = {: 7.3}\t(attempt {: 8d})".format(logLmin, logL_new,attempts),end = '\r',flush = True)
        if logL_new >= logLmin:
            if log_prior(new_point) - log_prior(p0) > np.log(np.random.uniform(0,1)):
                if verbose:
                    print('>> metro-accepted')
                chain_L -= 1
                p0 = new_point
                #return new_point, attempts
        if chain_L == 0:
            return new_point, attempts
        if attempts > Narrest:
            return None, attempts
        attempts += 1

def metro_gibbs(log_likelihood, log_prior, logLmin, p0, sigma, chain_L = 10 ,Narrest = 1000000, verbose = False):
    if verbose: print(f'mhg start at >> {p0}')
    attempts = 1
    p0_copy = p0.copy()
    last_accepted = p0.copy()
    new_point = p0_copy.copy()
    while True:
        for i, (p0_i,sigma_i) in enumerate(zip(p0_copy,sigma)):
            if verbose: print(f'coord{i}:')
            new_coord    =  np.random.normal(p0_i,sigma_i)
            if verbose: print(f'coord {i} update proposal: {new_point[i]} -> {new_coord}')
            new_point[i] = new_coord
            if new_point.shape != p0.shape:
                print('Something went wrong in metropolis: shape mismatch')
                exit()
            logL_new = log_likelihood(new_point)

            if logL_new >= logLmin:
                r = log_prior(new_point) - log_prior(p0_copy)
                if verbose: print(f'suggested transition: {p0_copy} -> {new_point}\t logp_new = {log_prior(new_point)} , logp_old = {log_prior(p0_copy)}, r = {r}')
                if np.isnan(r):
                    print('Fatal error: NaN log_prior subtraction')
                    print(f'Current points: old = {p0_copy}\t new = {new_point}')
                    exit()
                if  r > np.log(np.random.uniform(0,1)):
                    if verbose:
                        print(f'>> accepted')
                        print(f'now >> {new_point} (L = {logL_new}, prior = {log_prior(new_point)})')
                    p0_copy = new_point.copy()
                    last_accepted = new_point.copy()
                    #return new_point, attempts
                else:
                    if verbose: print('#', end = '')
            else:
                if verbose: print('L',end = '')
        chain_L -= 1
        if verbose: print('')
        if chain_L == 0:
            if verbose: print(f'----------- chain terminated, returned {new_point}')
            return last_accepted, attempts
        if attempts > Narrest:
            return None, attempts
        attempts += 1

def get_shrinking(N):
    return np.max(np.random.uniform(0,1,size = N))

def get_log_shrinking(N):
    return np.max(
                    np.log(
                            np.random.uniform(0,1,size = N)
                    ))

def get_Z_error(L,Nlive, N_t_vec = 50):
    '''Estimates the uncertainty of the value of Z given stochastic sample of X'''
    Z = np.zeros(N_t_vec)
    for sample in tqdm(range(N_t_vec),desc = "Estimating Z error"):
        X = np.cumprod([get_shrinking(Nlive) for _ in range(len(L)-1 - Nlive)])
        X = np.concatenate(([1],X))
        Z[sample] = np.trapz(-L[0:-Nlive],x = X) #reverse sign to account for reversed direction of integration
        Z[sample] += np.mean(L[-Nlive:])*np.min(X)
    print('z_err ---> Z = ', np.mean(Z), ' +- ', np.std(Z))
    return np.mean(np.log(Z)), np.std(np.log(Z))


def NS(log_likelihood,log_prior,
        bounds,
        explore = metro_gibbs,
        Nlive = 100,
        Npoints = np.inf,
        X_assessment = 'deterministic',
        shrink_scale = False,
        scale_factor = None,
        stop_log_relative_increment = -10,
        max_attempts = 1000000,
        verbose_search = False,
        display_progress = True,
        chain_L = 100):
    '''NestedSampling algorithm

        :param log_likelihood:  function of a SINGLE ARGUMENT (parameters vector(s) Np - dimensional)
                                    must be able to handle (M, Np) shaped arrays
                                    returning a (M,) vector
                                    Note: 1D case -> (M,) is wrong, use instead (M,1)
        :type log_likelihood: function
        :param bounds:  Indicate the searching bounds of a Nd-dimensional cube giving the upper and lower bounds
                        len(bounds) must be 2
                        each element a Np-dimensional vector
                        Note: for 1D case the shape must be (\*,1) instead of (\*,)
                                use reshape(-1,1)
        :type bounds: tuple of lists
        :param Nlive: Number of live points
        :type Nlive: int

        :return: LogX, LogL, LogZ, stats

        :param logX: X values used
        :param logL: L values used
        :param logZ: the estimated value of logZ (using square integration)
        :param stats:
            'param points' points in variable space
            'log_posterior_weights':(np.ndarray)    log posterior weights,
            'attempts':             (np.ndarray)    attempts history,
            'log_weights':          (np.ndarray)    logw,
            'clustering_warning':   (boolean)       label (work in progress) to detect clustering
            'dLogZ':                (float)         error on LogZ
        :type stats: dict(array,array,array, array, bool, float)

    '''
    assert len(bounds) == 2, "The bounds of search have len(bounds) == %d instead of 2"%len(bounds)
    assert stop_log_relative_increment <= -1, 'Invalid vaue for stop threshold log(dZ/Z) : %f (should be <= -1)'%stop_log_relative_increment
    assert X_assessment == 'deterministic' or X_assessment == 'stochastic', 'Unrecognized X assessment (%s)'%(X_assessment)
    space_dimension = bounds[0].shape[0]

    #given likelihood test
    dummy = np.zeros((2,space_dimension))
    if log_likelihood(dummy).shape != (2,):
        print('likelihood function can\'t handle array of points?')
    dummy = np.zeros((2,1))
    if log_likelihood(dummy).shape != (2,):
        print('likelihood can\'t handle single points?')

    #initilisation
    print('Initialisation')
    live_points = np.zeros((Nlive, space_dimension))
    i = 1 #iteration counter
    logZ = float(64)
    logZ = -np.inf
    force_arrest = False
    #proposal sigma initially set to the characteristic dimension of the space
    sigma_search = np.sqrt(bounds[0]**2 + bounds[1]**2)

    #number of attempts that generating the new point required (efficiency monitoring)
    attempts = np.inf
    #average of attempts over a temporal window
    #if this number is too close to one -> clustering
    attempts_windown_length = 10
    attempts_history = np.zeros(0)
    attempts_head = 0
    clustering_warning = False

    #first Nlive are generated in the whole X = 0 to X = 1 domain
    start = np.random.uniform(bounds[0], bounds[1])
    for k in tqdm(range(Nlive),desc = 'Generating initial points'):
        live_points[k], _ = explore(log_likelihood, log_prior,
                                        -np.inf,
                                        start,
                                        sigma_search,
                                        chain_L = chain_L,
                                        verbose = False)#np.random.uniform(bounds[0], bounds[1])
        start = live_points[k]
    print('\r')
    print('> Done')

    #tracks the history including dead points
    all_points = np.zeros((1,space_dimension))- np.inf  #initialised to ((-inf,-inf)) for shape keeping
    logL = np.ones( 1, dtype = np.float64) -    np.inf  #logarithm of likelihoods
    logX = np.zeros(1, dtype = np.float64)              #points in prior mass
    logw = np.ones (1, dtype = np.float64) -    np.inf  #weights for each interval


    #set shrink factor to default
    if scale_factor is None:
        scale_factor = (Nlive/(Nlive+1))**(1/space_dimension)

    #main loop
    with tqdm(total = Npoints) as bar:
        while True:
            worst = np.argmin(log_likelihood(live_points))
            logL_worst = log_likelihood(live_points[worst].reshape(1,-1))[0]

            #assessment for X
            if X_assessment == 'deterministic':
                #deterministic X assessment
                logX_new = -i/Nlive
            else:
                #stochastic X assessment
                logX_new = logX[i-1] + get_log_shrinking(Nlive)

            logw_current = logsubexp(logX[i-1], logX_new)
            logdZ = logw_current + logL_worst

            #progress output
            if logZ > -np.inf:
                #calculate the max relative increment in logZ
                # MRI = (max likelihood of live points)*(full remaining X-width)/Z
                max_log_relative_increment = np.max(log_likelihood(live_points)) + logX_new - logZ
                #if display_progress: print(' NS >> log(L) = {: 8.3}\tlog(Z) = {: 8.3}\tlog_max_rel_inc = {: 8.3}\tattempts = {:6} (iteration {: 8d}/{: 8d})'.format(logL_worst ,logZ, max_log_relative_increment , attempts, i, Npoints), end = '\r',flush = True)

            #CLOSING OPERATIONS: before making history

            if logZ != -np.inf and max_log_relative_increment < stop_log_relative_increment or i > Npoints or force_arrest:
                print('End criterion abided')

                #fill the gap - Skilling mode (mean L)
                sorting_order = np.argsort(log_likelihood(live_points))
                log_last_Ls = log_likelihood(live_points)[sorting_order ]#sort the last points (w is equal for everyone)
                log_last_w = (logX_new - np.log(Nlive)) * np.ones(Nlive) #last weight(s) is divided by Nlive

                #since the assessment fo the various last Xs is not possible (questionable)
                #L.shape ~ X.shape + Nlive

                #increment evidence
                logZ  = np.logaddexp(logZ, logsumexp(log_last_Ls + log_last_w))

                #end history
                logX = np.append(logX, -np.inf)
                logL = np.append(logL, log_last_Ls)
                logw = np.append(logw, log_last_w)
                all_points = np.append(all_points, live_points[sorting_order])

                if logsumexp(logw) != 1.:
                    print('WARNING: w-sum != 1 (1 + %e)'%(np.exp(logsumexp(logw)) - 1))

                #remove shape-setting first points
                all_points = np.delete(all_points,range(len(bounds[0]))).reshape((-1,space_dimension))
                logw = np.delete(logw,0)
                logL = np.delete(logL,0)

                #generate posterior weights
                log_posterior_weights = np.zeros(len(logL), dtype = np.float64)
                log_posterior_weights = logL + logw
                log_posterior_weights -= logsumexp(log_posterior_weights)

                print('sum posterior = ', np.exp(logsumexp(log_posterior_weights)))
                print('Squares logZ estimate = ', logZ)
                LogZ,dLogZ = get_Z_error(np.exp(logL), Nlive, N_t_vec = 10)
                print(f'Refined integration (trapz) logZ = {LogZ} +- {dLogZ}')

                #return the control stats
                stats = dict({
                            'points':                   all_points,
                            'log_posterior_weights':    log_posterior_weights,
                            'attempts':                 attempts_history,
                            'log_weights':              logw,
                            'clustering_warning':       clustering_warning,
                            'dLogZ':                    dLogZ
                            })

                return logX,logL,logZ, stats

            #(since the routine has to continue)
            #increment Z
            logZ = np.logaddexp(logZ,logdZ)

            #append the worst
            all_points = np.append(all_points, [live_points[worst]], axis = 0)

            #make history
            logL = np.append(logL,logL_worst)
            logX = np.append(logX,logX_new)
            logw = np.append(logw,logw_current)

            #kill the worst - get a new one
            #shrink the box where to search new samples
            if shrink_scale:
                sigma_search *= scale_factor
                #print(f'sigma search{sigma_search}')

            #take random live point and use it for MS search
            selected = np.random.randint(Nlive)
            new_point, attempts = explore(log_likelihood, log_prior,
                                            logL_worst,
                                            live_points[selected],
                                            sigma_search,
                                            chain_L = chain_L,
                                            verbose = verbose_search)
            if isinstance(new_point, np.ndarray) and np.isfinite(new_point).all():
                live_points[worst] = new_point
            else:
                print('ERROR: search got stuck')
                force_arrest = True

            #clustering detection
            attempts_history = np.append(attempts_history, attempts)
            if i > attempts_windown_length:
                mean_attempts = np.mean(attempts_history[-attempts_windown_length:])
                if not clustering_warning and np.mean(attempts_history) < 1.2:
                    print('Clustering detected')
                    clustering_warning = True
            i+=1
            bar.update(1)


if __name__ == '__main__':

    bounds = (np.array([-5,-5]), np.array([5,5]))

    logX,logL,logZ, stats = NS( log_likelihood,
                                bounds,
                                Nlive = 100,
                                X_assessment = 'stochastic',
                                shrink_parameters_box = True,
                                stop_log_relative_increment = -10)
    points = stats['points']
    maincol = np.array([0.,0.,1.])
    colors = [i*maincol for i in np.linspace(0.2,1,len(points))]
    plt.scatter(points[:,0],points[:,1],c = colors)

    plt.figure(2)
    plt.plot(stats['attempts'])
    plt.figure(3)
    plt.plot(stats['log_weights'])
    plt.figure(4)
    p_i = np.exp(stats['log_posterior_weights'])
    plt.plot(p_i)
    mean = p_i.dot(points)
    std = p_i.dot(points**2) - mean**2
    std = np.sqrt(std)
    print('p = ',mean,' pm ', std )
    plt.show()
