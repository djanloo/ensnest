import numpy as np
import matplotlib.pyplot as plt
from ensnest import mpNestedSampler, model
from ensnest import stdplots

class lineModel(model.Model):

    def set_parameters(self,data):
        self.names  = ['A', 'mu', 'line_sigma', 'noise_sigma']
        self.bounds = [[0.05, 10.], [0., 64.], [0.05, 10.], [0.5, 5.]]
        self.data = data
        self.x = np.arange(0,64, step=1)

    def f(self, A ,mu,sigma, x):
        return A*np.exp(- ((x - mu)/sigma)**2)

    def log_errs(self, t):
        return -0.5*t**2

    @model.Model.varenv
    def log_likelihood(self, vars):
        '''L is the product of the error function calculated in the residual (data - model)'''
        logl = np.zeros(vars.shape)
        for x_,d_ in zip(self.x, self.data):
            logl += self.log_errs( (self.f(vars['A'], vars['mu'], vars['line_sigma'], x_) - d_)/vars['noise_sigma'])\
                    - np.log(vars['noise_sigma']) - 0.5*np.log(2*np.pi)
        return logl

    @model.Model.auto_bound
    @model.Model.varenv
    def log_prior(self, vars):
        return -np.log(vars['A']*vars['line_sigma'])


data = np.array([1.42, 0.468, 0.762,
        -1.312, 2.029, 0.086,
        1.249, -0.368, -0.657,
        -1.294, 0.235, -0.192,
        -0.269,0.827,-0.685,
        -0.702,-0.937,1.331,
        -1.772,-0.530,0.330,
        1.205,1.613,0.3,
        -0.046,-0.026,-0.519,
        0.924,0.230,0.877,
        -0.650,-1.004,0.248,
        -1.169,0.915,1.113,
        1.463,2.732,0.571,
        0.865,-0.849,-0.171,
        1.031,1.105,-0.344,
        -0.087,-0.351,1.248,
        0.001,0.360,-0.497,
        -0.072,1.094,-1.425,
        0.283,-1.526,-1.174,
        -0.558,1.282,-0.384,
        -0.120,-0.187,0.646,0.399])

M = lineModel(data)
ns = mpNestedSampler(M, nlive=1000, evosteps=400, filename='line', load_old=False)
ns.run()

stdplots.hist_points(ns)
stdplots.XLplot(ns)

fig, ax = plt.subplots(1)
plt.plot(M.x, M.data)

x_ = np.linspace(min(M.x), max(M.x), 1000)
samp = ns.ew_samples['position'].copy().astype(M.position_t)
m = np.zeros((len(samp), len(x_)))
for i in range(len(samp)):
    m[i] = M.f(samp['A'][i], samp['mu'][i], samp['line_sigma'][i], x_)

plt.plot(x_, np.mean(m, axis=0))
p20,p80 = np.percentile(m,[20,80], axis=0)

plt.fill_between(x_,p20, p80, zorder = -100, color = 'turquoise', alpha=0.5)
plt.savefig('line.png')
