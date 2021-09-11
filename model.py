import numpy as np
import utils
import functools
from numpy.lib import recfunctions as rfn

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

        #defines the structure of the data used
        #this semplifies the usage
        #creates a list of dummy names to give to the variables
        for i in range(self.space_dim - len(self.names)):
            self.names.append('var%d'%i)

        self.position_t  = np.dtype([ (name, np.float64) for name in self.names])
        self.livepoint_t = np.dtype([
                                ('position' , self.position_t),
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
        dummy = np.random.random(testshape + (self.space_dim,)).view(self.position_t).squeeze()
        # for some unknown reasons, the view method in the above line gives an extra 1 in the shape -> use squeeze
        # may bring problem in unusual situations

        log_prior_result = self.log_prior(dummy)

        if not isinstance( log_prior_result, np.ndarray) or log_prior_result.shape != testshape:
            raise ValueError(f'Bad-behaving log_prior:\ninput shape: {dummy.shape} \noutput shape: {self.log_prior(dummy).shape} (should be {testshape})')

        if not isinstance(self.log_likelihood(dummy), np.ndarray) or self.log_likelihood(dummy).shape != testshape:
            raise ValueError(f'Bad-behaving log_likelihood:\ninput shape: {dummy.shape}\noutput shape: {self.log_likelihood(dummy).shape} (should be {testshape})')

        #checks for (time, position)-like evaluation
        #and iteration order differences
        testshape = (3,self.space_dim)
        dummy = np.random.random(testshape).view(self.position_t)

        result1 = np.array([self.log_prior(_) for _ in dummy])
        result2 = self.log_prior(dummy)

        if not (result1 == result2).any():
            raise ValueError('Bad-behaving log_prior: different results for different iteration order')

        result1 = np.array([self.log_likelihood(_) for _ in dummy])
        result2 = self.log_likelihood(dummy)

        if not (result1 == result2).any():
            raise ValueError('Bad-behaving log_likelihood: different results for different iteration order')

    def destructure(func):
        '''Helper function to manipulate data inside user-defined functins.

            Reduces the internal structure of variables to a standard np.ndarray.
            Useful when iterative operations have to be made upon the input.
        '''
        def _destructure_wrap(self, x):
            return func(self, rfn.structured_to_unstructured(x))
        return _destructure_wrap    

    @destructure
    def is_inside_bounds(self,points):
            '''Checks if a point is inside the space bounds.

            Args
            ----
                points : np.ndarray
                    point to be checked. Must have shape (\*,space_dim,).
            Returns:
                np.ndarray : True if all the coordinates lie between bounds

                            False if at least one is outside.

                            The returned array has shape (\*,) = ``utils.pointshape(point)``
            '''
            shape = points.shape

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

                    The returned array has shape (\*,) = ``utils.pointshape(point)``
        '''
        is_inside   = self.is_inside_bounds(points)
        value       = np.zeros(is_inside.shape)
        value[np.logical_not(is_inside)] = -np.inf
        return value

    def new_is_inside_bounds(self,points):
        '''Same as ``is_inside_bounds``.

        Shorter but slower (allegedly due to high processing time of numpy broadcasting).
        '''
        return np.logical_and((points > self.bounds[0]).all(axis = -1),(points < self.bounds[1]).all(axis = -1))

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
