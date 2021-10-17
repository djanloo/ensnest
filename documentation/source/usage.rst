Usage
-----

Model Definition
````````````````
	To define a model create a class based on :class:`model.Model` .

	The ``set_parameters()`` method must specify the space bounds as a list ``[ [inf1,sup1] , [inf2,sup2], .. ]``.
	If a list of names is specified for certain variables, they can be accessed by name indexing (see :func:`~ensnest.model.Model.varenv`).
	The parameters for which the name is not specified are called automatically ``var<n>``.
	After the model initialization ``model.names`` will contain all the names.

	.. code-block:: python

		from ensnest import model

		class MyModel(model.Model):

			def set_parameters(self,*args):
				self.bounds = [[0,1], [0,1], [0,42]]
				self.names  = ['A','mu','sigma']

	Other args passed when the instance is created are automatically passed to set_parameters, so

	.. code-block:: python

			class MyModel(model.Model):

				def set_parameters(self, data, flag=False ):
					self.bounds = [[0,1], [0,1], [0,42]]
					self.names = ['A', 'mu', 'sigma']
					self.data = data
					self.flag = flag

				...
			M = MyModel([0.1, 0.2, 0.4], flag=True)

	is fine.

	The logarithm of likelihood and prior have to be specified as:

	.. code-block:: python

			# ...
			def log_prior(self, var):
				return

	The :func:`~ensnest.model.Model.log_prior` and :func:`~ensnest.model.Model.log_likelihood` methods must be capable to manage (\*, space_dim)-shaped arrays and return a (\*)-shaped array.

 	If names are not specified, all operations must be preformed over the last axis.

	.. code-block:: python

			def log_prior(self,var):
				return -0.5*np.sum(var**2, axis = -1)

	If names are specified, it is possible to use :func:`ensnest.model.Model.varenv`:

	.. code-block:: python

			@model.Model.varenv
			def log_prior(self,var):
				return -(var['A'] - self.data[0])**2 - var['mu']**2

	Finally, to automatically bound a function inside the model domain use the :func:`~ensnest.model.Model.auto_bound` decorator:

	.. code-block:: python

			@model.Model.auto_bound
			@model.Model.varenv
			def log_prior(self,var):
				return -(var['A'] - self.data[0])**2 - var['mu']**2

	.. warning::
		``varenv`` must be the first decorator applied

The data type used in the models is ``['position', 'logL', 'logP']``

	>>> var['position']['A'][time,walker]
	>>> var['logL'][time,walker]

in case it is necessary to reduce the data structure use ``numpy.lib.recfunctions.structured_to_unstructured`` or ``numpy.view()``.

Samplers usage
``````````````

	The available samplers are contained in :py:mod:`~ensnest.samplers` module. The first argument is a :class:`model.Model` subclass instance.
	The second argument is the chain length.

		>>> from ensnest import sampler
		>>> sampler = sampler.AIESampler(MyModel(), 500 , nwalkers=100)

	To sample a function, define it as a ``log_prior`` and use ``sample_prior`` method of a ``Sampler`` subclass.
	After the chain is filled it is accessible as an attribute:

		>>> samples = sampler.chain

	To join the chains of each particle after removing a ``burn_in`` use:

		>>> samples = sampler.join_chains(burn_in = 0.3)

Nested Sampling usage
`````````````````````

After having defined a model, create an instance of :class:`ensnest.NestedSampling.NestedSampler` specifying:

	#. the model
	#. the number of live points
	#. the number of sampling steps the live points undergo before getting accepted

Other options are:

	* ``npoints`` stops the computation after having generated a fixed number of points
	* ``relative_precision``
	* ``load_old`` loads the save of the same run (if it exists). If ``filename`` is not specified, an *almost* unique code for the run is generated based on the features of the model and the NSampler run
	* ``filename`` to specify a save file
	* ``evo_progress`` to display the progress bar for the evolution process

The run is performed by ``ns.run()``, after that every computed feature is stored as an attribute of the nested sampler:

.. code-block:: python

	ns = NestedSampler(M, nlive=1000, evosteps=1000, load_old=False)
	ns.run()
	print(ns.Z, ns.Z_error, ns.points)

Multiprocess Nested Sampling
````````````````````````````
It is performed by :class:`~ensnest.NestedSampling.mpNestedSampler`. The arguments are the same of :class:`~ensnest.NestedSampling.NestedSampler`.

Runs ``multiprocessing.cpu_count()`` copies of nested sampling, then merges them using the `dynamic nested sampling <https://arxiv.org/abs/1704.03459>`_ merge algorithm.

After running, the instance contains the merged computed variables (``logX``, ``logZ``, ecc.) and the single run variables through ``nested_samplers`` attribute:

.. code-block:: python

	mpns = mpNestedSampler(model_, nlive=500, evosteps=1200, load_old=False)
	mpns.run()
	print(f'Z = {mpns.Z} +- {mpns.Z_error})
	single_runs = mpns.nested_samplers
	for ns in single_runs:
		print(f'Z = {ns.Z} +- {ns.Z_error})

Diffusive Nested Sampling
`````````````````````````
It is performed by :class:`~ensnest.DiffusiveNestedSampling.DiffusiveNestedSampler`.
The main parameters are the :class:`~ensnest.model.Model` ``chain_length`` before a level is added,
``nlive`` of points the ensemble is made of and ``max_n_levels``.

	>>> dns = DiffusiveNestedSampler(M, nlive=200, max_n_levels=100, chain_length=200)


The resolution in prior mass can be adjusted specifying ``dns.Xratio`` after the sampler is initialised.


Plotting
````````
In :py:mod:`~ensnest.stdplots` are contained some shorthands for plotting the results for ``NS/mpNS/DNS`` runs.
