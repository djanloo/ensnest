import model
import samplers
import numpy as np

class caucau(model.Model):
    def __init__(self):
        self.bounds = ([0,0],[1.2,1.2])
        super().__init__()

    def log_likelihood(self,x):
        x1,x2 = model.unpack_variables(x)
        return 1/(x1 +x2)

    @model.Model.auto_bound
    def log_prior(self,x):
        x1,x2 = model.unpack_variables(x)
        return -0.5*x1**2

point = np.array([1.5,1.1])

print(caucau().log_prior(point))
