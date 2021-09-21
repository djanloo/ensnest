import model
import NestedSampling
import numpy as np
import matplotlib.pyplot as plt

def circ(x,c,w,r):
    return  - (np.sum((x[...,:] - c)**2) - r**2)/2./w**2 + 0.5*np.log(2*np.pi*w**2)


class gshell(model.Model):

    def set_parameters(self):
        self.bounds = [[-7,7],[-4,4]]
        self.names  = ['a','b']
        self.c1     = [-2,0]
        self.c2     = [2,0]
        self.r1     = 3
        self.r2     = 3
        self.w1     = 0.1
        self.w2     = 0.1

    @model.Model.auto_bound
    def log_prior(self,x):
        return 0

    @model.Model.varenv
    def log_likelihood(self,x):
        sum1 = np.zeros(x.shape)
        sum2 = np.zeros(x.shape)

        for i,name in enumerate(self.names):
            sum1 += (x[name] - self.c1[i])**2
            sum2 += (x[name] - self.c2[i])**2

        sum1 = np.sqrt(sum1)
        sum2 =  np.sqrt(sum2)

        L = np.exp(-.5*(sum1 - self.r1)**2/self.w1**2)/(np.sqrt(2*np.pi*self.w1**2)) + np.exp(-.5*(sum2 - self.r2)**2/self.w2**2)/(np.sqrt(2*np.pi*self.w2**2))
        return np.log(L)

model_ = gshell()

mpns = NestedSampling.mpNestedSampler(model_, nlive=1000, evosteps=4000, evo_progress=False)
mpns.run()
print(f'run_time = {mpns.run_time}')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for ns in mpns.nested_samplers:
    ax.scatter(ns.points['position']['a'], ns.points['position']['b'], np.exp(ns.points['logL']), c = np.exp(ns.points['logL']) , cmap = 'plasma')

plt.figure(2)
plt.plot(mpns.logX, mpns.logL)

print(mpns.logZ, mpns.logZ_error)
plt.show()
