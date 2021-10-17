'''Model module
Defines the model class and some test models
'''
import sys
from timeit import default_timer as timer
import numpy as np

N_TIMES_EVAL = 1000

MULTIWALKER_TEST_SHAPE = (4, 3)
TEST_SHAPE = (3,)


class Model:
    r'''Class to describe models

    Attributes
    ----------
        log_prior : function
            the logarithm of the prior pdf
        log_likelihood : function
            the logarithm of the likelihood function
        space_bounds : 2-tuple of ``np.ndarray``
            the coordinate of the two vertices of the hyperrectangle
            defining the bounds of the parameters

    note
    ----
        The ``log_prior`` and ``log_likelihood`` functions are user defined and
        must have **one argument only**.

        They also must be capable of managing (\*,\*, .., space_dimension )-shaped arrays,
        so make sure every operation is done on the **-1 axis of input** or use
        :func:`~model.Model.varenv`.

        If input is a single point of shape (space_dimension,) both the functions
        must return a float ( not a (1,)-shaped array )

    '''
    # pylint: disable=too-many-instance-attributes

    def __init__(self, *args):
        '''Initialise and checks the model
        '''
        # call the child function to set parameters
        self.set_parameters(*args)

        # switchs between ([min1, max], [min2,max2], ..) and
        # ([min1,min2,..],[max1,max2, ..]
        self.bounds = np.array(self.bounds).transpose()
        if len(self.bounds.shape) == 1:  # checks (a,b) case
            self.bounds = self.bounds[:, None]
        self.bounds = (
            self.bounds[0].astype(
                np.float64),
            self.bounds[1].astype(float))

        if not hasattr(self, 'names'):
            self.names = []

        try:
            dim1 = self.bounds[0].shape[0]
            dim2 = self.bounds[1].shape[0]
            if dim1 == dim2:
                self.space_dim = dim1
            else:
                print('Different space dimensions in bounds')
                sys.exit()
        except IndexError:  # in case bounds is given of the form (n,m)
            self.space_dim = 1

        self.volume = np.prod(self.bounds[1] - self.bounds[0])
        # defines the structure of the data used
        # this semplifies the usage
        # creates a list of dummy names to give to the variables
        for i in range(self.space_dim - len(self.names)):
            self.names.append('var%d' % i)

        # defines the datatype used in variable environment (varenv)
        # and output results
        self.position_t = np.dtype([(name, np.float64) for name in self.names])

        # defines the datatype used inside code
        self.livepoint_t = np.dtype([
            ('position', np.float64, (self.space_dim,)),
            ('logL', np.float64),
            ('logP', np.float64)
        ])

        self._check()

    def set_parameters(self):
        '''Model parameters such bound, names, additional data should be defined here'''
        raise NotImplementedError('set_parameters must be user-defined')

    def log_likelihood(self, var):
        '''the log_likelihood function'''
        raise NotImplementedError('log_likelihood must be user-defined')

    def log_prior(self, var):
        '''the log_prior function'''
        raise NotImplementedError('log_prior not must be user-defined')

    # pylint: disable=no-member
    def _check(self):
        '''Checks if log_prior and log_likelihood are well behaved in return shape
        '''

        # checks for (time, walker, position)-like evaluation
        testshape = MULTIWALKER_TEST_SHAPE
        dummy1 = np.random.random(testshape + (self.space_dim,))

        log_prior_result = self.log_prior(dummy1)

        if not isinstance(
                log_prior_result,
                np.ndarray) or log_prior_result.shape != testshape:
            raise ValueError(
                'Bad-behaving log_prior:\n'
                f'input shape: {dummy1.shape} \n'
                f'output shape: {self.log_prior(dummy1).shape} (should be {testshape})')

        if not isinstance(self.log_likelihood(dummy1), np.ndarray) or\
                self.log_likelihood(dummy1).shape != testshape:

            raise ValueError(
                'Bad-behaving log_likelihood:\n'
                f'input shape: {dummy1.shape}\n'
                f'output shape: {self.log_likelihood(dummy1).shape} (should be {testshape})')

        # checks for (time, position)-like evaluation
        # and iteration order differences
        testshape = TEST_SHAPE
        dummy2 = np.random.random(testshape + (self.space_dim,))

        result1 = np.array([self.log_prior(_) for _ in dummy2])
        result2 = self.log_prior(dummy2)

        if not (result1 == result2).any():
            raise ValueError('Bad-behaving log_prior:'
                             ' different results for different iteration order')

        result1 = np.array([self.log_likelihood(_) for _ in dummy2])
        result2 = self.log_likelihood(dummy2)

        if not (result1 == result2).any():
            op1 = [f'logL{_}' for _ in dummy2]
            op2 = f'logL({dummy2})'
            compare = f'\n{op1} = {result1}\n{op2} = {result2}'
            raise ValueError(
                'Bad-behaving log_likelihood:'
                ' different results for different iteration order' +
                compare)

        # estimates the time required for the evaluation of log_likelihood and
        # log_prior
        testshape = MULTIWALKER_TEST_SHAPE
        dummy_input = np.random.random(testshape + (self.space_dim,))

        start = timer()
        for _ in range(N_TIMES_EVAL):
            dummy_result = self.log_prior(dummy_input)
        end = timer()
        self.log_prior_execution_time_estimate = (end - start) / N_TIMES_EVAL

        start = timer()
        for _ in range(N_TIMES_EVAL):
            dummy_result = self.log_prior(dummy_input)
        end = timer()
        self.log_likelihood_execution_time_estimate = (
            end - start) / N_TIMES_EVAL

        print(
            f'Correctly initialised a {self.space_dim}-D model with \n'
            f'\tT_prior      ~ {self.log_prior_execution_time_estimate*1e6:.2f} us\n'
            f'\tT_likelihood ~ {self.log_likelihood_execution_time_estimate*1e6:.2f} us')

    @classmethod
    def varenv(cls, func):
        '''
        Helper function to index the variables by name inside user-defined functions.

        Uses the names defined in the constructor of the model + var0,var1, ... for the one
        which are left unspecified.

        warning
        -------
            When using with ``@auto_bound``, it must be first:

            >>> @auto_bound
            >>> @varenv
            >>> def f(self,var):
            >>>      u = var['A']+var['mu']
            >>>      ... do stuff
        '''

        def _wrap(self, var, *args, **kwargs):
            var = var.view(self.position_t).squeeze()
            return func(self, var, *args, **kwargs)
        return _wrap

    def is_inside_bounds(self, points):
        r'''Checks if a point is inside the space bounds.

        Args
        ----
            points : np.ndarray
                point to be checked. Must have shape (\*,space_dim,).
        Returns:
            np.ndarray : True if all the coordinates lie between bounds

                        False if at least one is outside.
        '''
        shape = np.array(points.shape)[:-1]
        is_coordinate_inside = np.logical_and(points > self.bounds[0],
                                              points < self.bounds[1])
        return is_coordinate_inside.all(axis=-1).reshape(shape)

    def log_chi(self, points):
        r''' Logarithm of the characteristic function of the domain.
        Is equivalent to

        >>> np.log(model.is_inside_bounds(point).astype(float))

        Args
        ----
            points : np.ndarray
                point to be checked. Must have shape (\*,space_dim,).
        Returns:
            np.ndarray : 0 if all the coordinates lie between bounds

                    -``np.inf`` if at least one is outside
        '''
        is_inside = self.is_inside_bounds(points)
        value = np.zeros(is_inside.shape)
        value[np.logical_not(is_inside)] = -np.inf
        return value

    @classmethod
    def auto_bound(cls, log_func):
        '''Decorator to bound functions.

        args
        ----
            log_func : function
                A function for which ``self.log_func(var)`` is valid.

        Returns:
            function : the bounded function ``log_func(var) + log_chi(var)``

        Example
        -------

            >>> class MyModel(model.Model):
            >>>
            >>>     @model.Model.auto_bound
            >>>     def log_prior(var):
            >>>         return var


        '''

        def _autobound_wrapper(self, *args):
            return log_func(self, *args) + self.log_chi(*args)
        return _autobound_wrapper

    def __hash__(self):
        """Generates a (almost) unique code for model"""
        # since function by themselves are variably hashable
        # takes 10 points over the diagonal of the space
        points = np.linspace(0, 1, 10)[
            :, None] * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        results_on_diag = tuple(
            [*self.log_prior(points), *self.log_likelihood(points)])
        inf = tuple(self.bounds[0])
        sup = tuple(self.bounds[1])
        return hash(results_on_diag + inf + sup)


# some simple models useful for testing

class Gaussian(Model):
    '''MVN likelihood, uniform prior.'''

    def __init__(self, dim=1):
        self.dim = dim
        super().__init__()

    def set_parameters(self):
        self.bounds = np.array([-5, 5] * self.dim).reshape(-1, 2)
        self.names = ['a']

    @Model.auto_bound
    def log_prior(self, var):
        return 0

    def log_likelihood(self, var):
        return -0.5 * np.sum(var**2, axis=-1)


class UniformJeffreys(Model):
    '''Gaussian likelihood, 1/y prior'''

    def set_parameters(self):
        self.bounds = ([0.1, 10], [.1, 5])
        self.names = ['a', 'b']
        self.center = np.array([3, 0])

    @Model.auto_bound
    @Model.varenv
    def log_prior(self, var):
        return np.log(1. / var['b'])

    def log_likelihood(self, var):
        return -0.5 * np.sum((var - self.center)**2, axis=-1)


class Rosenbrock(Model):
    '''Rosenbrock likelihood, uniform prior'''

    def set_parameters(self):
        self.bounds = ([-5., 5.], [-1, 10.])
        self.names = ['x', 'y']

    @Model.auto_bound
    def log_prior(self, var):
        return 0

    @Model.varenv
    def log_likelihood(self, var):
        return - .5 * (var['y'] - var['x']**2)**2 - \
            1. / 20. * (1. - var['x'])**2


class PhaseTransition(Model):
    '''The model that exhibits phase transition by Skilling (2006)'''

    def set_parameters(self):
        self.bounds = np.array([-0.5, 0.5] * 10).reshape(-1, 2)
        self.par_v = 0.1
        self.par_u = 0.01
        self.center = 0.031

    @Model.auto_bound
    def log_prior(self, var):
        return 0

    def log_likelihood(self, var):
        partial_1 = np.prod(
            np.exp(-0.5 * (var / self.par_v)**2) / (self.par_v * np.sqrt(2 * np.pi)),
            axis=-1)
        partial_2 = 100 * np.prod(
            np.exp(-0.5 * ((var - self.center) / self.par_u)**2) /
            (self.par_u * np.sqrt(2 * np.pi)),
            axis=-1)
        return np.log(partial_1 + partial_2)
