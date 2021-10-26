Example
-------

.. code-block:: python


    import numpy as np
    import matplotlib.pyplot as plt

    from ensnest.model import Model
    from ensnest import mpNestedSampler
    from ensnest import stdplots

    class AckleyModel(Model):

        def set_parameters(self):
            self.names=['x','y']
            self.bounds=[[-5,5],[-5,5]]

        @Model.varenv
        def log_likelihood(self,var):
            partial_1 = -20.*np.exp(-.2*np.sqrt(0.5*(var['x']**2 + var['y']**2)))
            partial_2 = -np.exp(0.5*(np.cos(2*np.pi*var['x']) + np.cos(2*np.pi*var['y'])))
            offset = np.e + 20.
            return np.log(partial_1 + partial_2 + offset)

        @Model.auto_bound
        def log_prior(self,var):
            return 0

    M = AckleyModel()
    ns = mpNestedSampler(M, nlive=500, evosteps=200, filename='ackley', load_old=False)

    ns.run()

    stdplots.XLplot(ns)
    stdplots.scat3D(ns)

    # plt.plot(ns.logX, np.exp(ns.logX + ns.logL), color='k')
    # for nns in ns.nested_samplers:
    #     plt.plot(nns.logX, np.exp(nns.logX + nns.logL))


    plt.show()
