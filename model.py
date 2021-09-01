import numpy as np
import utils

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
        so make sure every operation is done on the **-1 axis of input**.

        If input is a single point of shape (space_dimension,) both the functions
        must return a float ( not a (1,)-shaped array )

    '''
    def __init__(self, log_prior, log_likelihood, space_bounds):
        '''Initialise the sampler.

        By default the starting point of the markov chain are uniformly distributed over all space.
        '''
        space_bounds = (np.array(space_bounds[0]).astype(float), np.array(space_bounds[1]).astype(float))
        assert space_bounds[0].shape == space_bounds[1].shape, "Uncoherent space dimensions"

        self.bounds          = space_bounds

        self.log_prior       = lambda x: log_prior(x)       + self.log_chi(x)
        self.log_likelihood  = lambda x: log_likelihood(x)  + self.log_chi(x)

        try:
            self.space_dim   = space_bounds[0].shape[0]
        except IndexError: #in case bounds is given of the form (n,m)
            self.space_dim   = 1

        self._check()

    def _check(self):
        '''Checks if log_prior and log_likelihood are well behaved in return shape
        '''

        #checks for (time, walker, position)-like evaluation
        testshape = (4,3)
        dummy = np.random.random(testshape + (self.space_dim,))

        log_prior_result = self.log_prior(dummy)

        if not isinstance( log_prior_result, np.ndarray) or log_prior_result.shape != testshape:
            raise ValueError(f'Bad-behaving log_prior:\ninput shape: {dummy.shape} \noutput shape: {self.log_prior(dummy).shape} (should be {testshape})')

        if not isinstance(self.log_likelihood(dummy), np.ndarray) or self.log_likelihood(dummy).shape != testshape:
            raise ValueError(f'Bad-behaving log_likelihood:\ninput shape: {dummy.shape}\noutput shape: {self.log_likelihood(dummy).shape} (should be {testshape})')

        #checks for (time, position)-like evaluation
        #and iteration order differences
        testshape = (3,)
        dummy = np.random.random(testshape + (self.space_dim,))

        result1 = np.array([self.log_prior(_) for _ in dummy])
        result2 = self.log_prior(dummy)

        if not (result1 == result2).any():
            raise ValueError('Bad-behaving log_prior: different results for different iteration order')

        result1 = np.array([self.log_likelihood(_) for _ in dummy])
        result2 = self.log_likelihood(dummy)

        if not (result1 == result2).any():
            raise ValueError('Bad-behaving log_likelihood: different results for different iteration order')

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
            shape = list(points.shape)

            if shape.pop() != self.space_dim:
                raise IndexError('Last axis must have len == space_dim')

            shape = tuple(shape)
            list_of_points       = points.reshape(-1,self.space_dim)
            is_coordinate_inside = np.logical_and(  list_of_points > self.bounds[0],
                                                    list_of_points < self.bounds[1])
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


    def pointshape(self,x):
        """ ``self`` shorthand for ``utils.pointshape(x, dim = self.space_dim)``
        """
        return utils.pointshape(x, dim = self.space_dim)


def unpack_variables(x):
    '''Helper function that performs values shapecasting.

    Given a ``np.ndarray`` of shape ``(n1,n2,--, space_dim)``
    returns an unpackable array of shape ``(space_dim, n1,n2, --)``.

    note
    ----
        if any of the n1, n2, -- other dimension is equal to one,
        it gets squeezed as it is an unnecessary nesting.

    Args
    ----
        x : np.ndarray
            the array to be casted
    Returns:
        tuple : an unpackable array

    Example
    -------
        It can be used to define models:

        >>> def log_prior(x):
        >>>     x1,x2,x3 = unpack_variables(x)
        >>>     return x1/x2*x3

    warning
    -------
        It may be computationally expensive. Check for improvements.
    '''
    x = np.array(x)
    vars = x if not x.shape else np.moveaxis(x,-1,0).squeeze()
    return vars
