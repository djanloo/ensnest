Usage
-----

Model Definition
````````````````
	To define a model create a class based on :class:`model.Model` .
	
	The ``__init__`` method must specify the space bounds as a 2-tuple ``(inf_bound , sup_bound)``.
	
	After that call the superclass initialisation method to check the model.
	
		>>> import model 
		>>> class MyModel(model.Model):
		>>> 
		>>> 	def __init__(self):
		>>> 		self.bounds = ([0,0], [1,1])
		>>>		super().__init__()
	
	The logarithm of likelihood and prior have to be specified as 
	class methods.
	
		>>> ...
		>>> 	def log_prior(self, x):
		>>>		return
	
	The :func:`~model.Model.log_prior` and :func:`~model.Model.log_likelihood` methods must be capable to manage (\*, space_dim)-shaped arrays and return a (\*)-shaped array.
	
 	All operation must be performed onto the last axis. To separate the variables use :func:`~model.unpack_variables`
	
		>>>	def log_prior(self,x):
		>>>		x0,x1 = model.unpack_variables(x)
		
	Finally, to automatically bound a function inside the model domain use the :func:`~model.Model.auto_bound` decorator:
	
		>>>	@model.Model.auto_bound
		>>>	def log_prior(self,x):
		>>>		x0,x1 = model.unpack_variables(x)
		>>>		return -0.5*x0**2
		
Samplers usage
``````````````
	
	The available samplers are contained in :py:mod:`~samplers` module. The first argument is a :class:`model.Model` user-defined subclass instance.
	
	The second argument is the chain lentgth. Once defined, a sampler has a definite length.
	
		>>> import sampler
		>>> sampler = sampler.AIESampler(MyModel(), 500 , nwalkers=100)
		
	To sample a function use the ``sample_function`` method of a ``Sampler`` subclass. The function is not necessarily a ``Model.log_prior`` or a ``Model.log_likelihood``, but the sampling bounds are inherited from the model onto which the sampler is instantiated.
	
		>>> def log_foo(x):
		>>>	...
		>>> sampler.sample_function(log_foo)
		
	At this point the sampler fills its chain. For the ensamble samplers the chain has shape ``(Niter, Nwalkers, Model.space_dim)``.
	
		>>> x = sampler.chain
	
	To join the chains of each particle after removing a ``burn_in`` use:
		
		>>> x = sampler.join_chains(burn_in = 0.3)
		
	
 		
		
	
	
