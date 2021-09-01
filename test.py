import numpy as np
import model
import utils

def rosenbrock(x):
    x1,x2 = model.to_variables(x)
    return  - 5* (x2 - x1**2)**2 - 1./20*(1-x1)**2

def uniform(x):
    return np.zeros(utils.pointshape(x))
