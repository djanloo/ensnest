import numpy as np

def logsubexp(x1,x2):
    '''Helper function to execute :math:`\\log{(e^{x_1} - e^{x_2})}`

        Args
        ----
            x1 : float
            x2 : float
    '''

    assert x1 > x2, 'Logsubexp got wrong arguments (log(-c))'
    return  x1 + np.log1p(-np.exp(x2-x1))

def logsumexp(arg):
    '''Utility to sum over log_values.
    Given a  vector [a1,a2,a3, ... ] returns :math:`\\log{(e^{a1} + e^{a2} + ...)}`

    Args
    ----
        arg : np.ndarray
            the array of values to be log-sum-exponentiated
    Returns:
        float : :math:`\\log{(e^{a1} + e^{a2} + ...)}`

    '''
    if type(arg) == list:
        arg = np.array(arg)
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

def hms(secs):
    '''Returns time in ``hour minute seconds`` fromat given time in seconds.'''
    h = secs//3600
    m = secs//60 - 60*h
    s = (secs - 60*m - 3600*h)//1
    return f'{h} h {m} m {s} s'
