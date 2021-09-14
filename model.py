import numpy as np
from timeit import default_timer as timer

class Model:
    '''Class to describe models

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
        The log_prior and logg_likelihood functions are user defined and must have **one argument only**.

        They also must be capable of managing (\*,\*, .., space_dimension )-shaped arrays,
        so make sure every operation is done on the **-1 axis of input** or use ``Model.unpack_variables()``.

        If input is a single point of shape (space_dimension,) both the functions
        must return a float ( not a (1,)-shaped array )

    '''
    def __init__(self):
        '''Initialise and checks the model
        '''
        self.bounds = (np.array(self.bounds[0]).astype(float), np.array(self.bounds[1]).astype(float))
        if not hasattr(self, 'names'):
            self.names = []

        try:
            dim1 = self.bounds[0].shape[0]
            dim2 = self.bounds[1].shape[0]
            if dim1 == dim2:
                self.space_dim = dim1
            else:
                print('Different space dimensions in bounds')
                exit()
        except IndexError: #in case bounds is given of the form (n,m)
            self.space_dim   = 1

        self.volume = np.prod(self.bounds[1] - self.bounds[0])
        #defines the structure of the data used
        #this semplifies the usage
        #creates a list of dummy names to give to the variables
        for i in range(self.space_dim - len(self.names)):
            self.names.append('var%d'%i)

        #defines the datatype used in variable environment (varenv)
        #and output results
        self.position_t  = np.dtype([ (name, np.float64) for name in self.names])

        #defines the datatype used inside code
        self.livepoint_t = np.dtype([
                                ('position' , np.float64, (self.space_dim,)), #using the dtype position_t makes the rest of the code really silly
                                ('logL'     , np.float64),
                                ('logP'     , np.float64)
                                ])

        self._check()

    def log_likelihood(self,x):
        raise NotImplementedError('log_likelihood not defined')

    def log_prior(self, x):
        raise NotImplementedError('log_prior not defined')

    def _check(self):
        '''Checks if log_prior and log_likelihood are well behaved in return shape
        '''

        #checks for (time, walker, position)-like evaluation
        testshape = (4,3)
        dummy1 = np.random.random(testshape + (self.space_dim,))

        log_prior_result = self.log_prior(dummy1)

        if not isinstance( log_prior_result, np.ndarray) or log_prior_result.shape != testshape:
            raise ValueError(f'Bad-behaving log_prior:\ninput shape: {dummy1.shape} \noutput shape: {self.log_prior(dummy1).shape} (should be {testshape})')

        if not isinstance(self.log_likelihood(dummy1), np.ndarray) or self.log_likelihood(dummy1).shape != testshape:
            raise ValueError(f'Bad-behaving log_likelihood:\ninput shape: {dummy1.shape}\noutput shape: {self.log_likelihood(dummy1).shape} (should be {testshape})')

        #checks for (time, position)-like evaluation
        #and iteration order differences
        testshape = (3,)
        dummy2 = np.random.random(testshape + (self.space_dim,))

        result1 = np.array([self.log_prior(_) for _ in dummy2])
        result2 = self.log_prior(dummy2)

        if not (result1 == result2).any():
            raise ValueError('Bad-behaving log_prior: different results for different iteration order')

        result1 = np.array([self.log_likelihood(_) for _ in dummy2])
        result2 = self.log_likelihood(dummy2)

        if not (result1 == result2).any():
            raise ValueError('Bad-behaving log_likelihood: different results for different iteration order')

        #estimates the time required for the evaluation of log_likelihood and log_prior
        testshape = (4,3)
        dummy_input = np.random.random(testshape + (self.space_dim,))

        start = timer()
        for i in range(100):
            dummy_result = self.log_prior(dummy_input)
        end = timer()
        self.log_prior_execution_time_estimate = (end - start)/100.

        start = timer()
        for i in range(100):
            dummy_result = self.log_prior(dummy_input)
        end = timer()
        self.log_likelihood_execution_time_estimate = (end - start)/100.

        print(f'Correctly initialised a {self.space_dim}-D model with \n\tT_prior      ~ {self.log_prior_execution_time_estimate*1e6:.2f} us\n\tT_likelihood ~ {self.log_likelihood_execution_time_estimate*1e6:.2f} us')


    def varenv(func):
        '''
        Helper function to index the variables by name inside user-defined functions.

        Uses the names defined in the constructor of the model + var0,var1, ... for the one
        which are left unspecified.

        warning
        -------
            When using with ``@auto_bound``, it must be first:

            >>> @auto_bound
            >>> @varenv
            >>> def f(self,x):
            >>>      u = x['A']+x['mu']
            >>>      ... do stuff
        '''
        def _wrap(self,x,*args, **kwargs):
            x = x.view(self.position_t).squeeze()
            return func(self, x)
        return _wrap

    def is_inside_bounds(self,points):
            '''Checks if a point is inside the space bounds.

            Args
            ----
                points : np.ndarray
                    point to be checked. Must have shape (\*,space_dim,).
            Returns:
                np.ndarray : True if all the coordinates lie between bounds

                            False if at least one is outside.
            '''
            shape = np.array(points.shape)[:-1]
            is_coordinate_inside = np.logical_and(  points > self.bounds[0],
                                                    points < self.bounds[1])
            return is_coordinate_inside.all(axis = -1).reshape(shape)

    def log_chi(self, points):
        ''' Logarithm of the characteristic function of the domain.
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
        is_inside   = self.is_inside_bounds(points)
        value       = np.zeros(is_inside.shape)
        value[np.logical_not(is_inside)] = -np.inf
        return value

    def auto_bound(log_func):
        '''Decorator to bound functions.

        args
        ----
            log_func : function
                A function for which ``self.log_func(x)`` is valid.

        Returns:
            function : the bounded function ``log_func(x) + log_chi(x)``

        Example
        -------

            >>> class MyModel(model.Model):
            >>>
            >>>     @model.Model.auto_bound
            >>>     def log_prior(x):
            >>>         return x


        '''
        def _autobound_wrapper(self,*args):
            return log_func(self,*args) + self.log_chi(*args)
        return _autobound_wrapper


#some simple models useful for testing

class ToyGaussian(Model):
    def __init__(self,dim = 1):
        self.bounds = (-np.ones(dim)*10 ,np.ones(dim)*10 )
        self.names  = ['a']
        super().__init__()

    #@model.Model.varenv
    @Model.auto_bound
    def log_prior(self,x):
        return 0#np.log(x['a'])

    def log_likelihood(self,x):
        return -0.5*np.sum(x**2,axis = -1)

class UniformJeffreys(Model):

    def __init__(self):
        self.bounds = ([0.1,-10],[10,10])
        self.names  = ['a','b']
        self.center = np.array([3,0])
        super().__init__()

    @Model.auto_bound
    @Model.varenv
    def log_prior(self,x):
        return np.log(1./x['a'])

    def log_likelihood(self,x):
        return -0.5*np.sum((x-self.center)**2,axis = -1)

class RosenBrock(Model):

    def __init__(self):
        self.bounds = ([-10,-1],[10,10])
        self.names  = ['1','2']
        self.center = np.array([3,0])
        super().__init__()

    @Model.auto_bound
    def log_prior(self,x):
        return 0

    @Model.varenv
    def log_likelihood(self,x):
        return - 5* (x['2'] - x['1']**2)**2 - 1./20*(1-x['1'])**2
