import numpy as np

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)

from scipy import io

scale = 5
stretch = 1

#Basic sanity checks to establish parameter recovery
sanitycheck_0 = ConstantKernel(constant_value=1.0)

"""
horrible:
[[['RBF'], 'RBF', 'RBF'], 'COS']
"""

sanitycheck_1 = ConstantKernel(constant_value=1.0) + WhiteKernel(noise_level=0.01)

"""
['RBF']
final loss: -0.8737539649009705
WORKEDS
"""


sanitycheck = RBF(length_scale=0.5*scale) + ConstantKernel(constant_value=2.0) + WhiteKernel(noise_level=0.005)

"""
N=300
['RBF', 'LIN']
final loss: -1.182499885559082
Parameter name: covar_module.kernels.0.base_kernel.raw_lengthscale                               value = 0.482

RECOVERED length scale
"""

test1 = ExpSineSquared(length_scale=0.2*scale, periodicity=0.1*scale) + RBF(length_scale=0.3*scale) + WhiteKernel(noise_level=0.1) #Periodic + SE

"""
N=300
[[['RBF'], 'COS'], 'PERIODIC']
final loss: 0.5518975257873535
Parameter name: covar_module.kernels.0.kernels.0.base_kernel.raw_lengthscale                     value = 0.962
Parameter name: covar_module.kernels.0.kernels.1.base_kernel.raw_period_length                   value = 0.614
Parameter name: covar_module.kernels.1.base_kernel.raw_lengthscale                               value = 0.056
Parameter name: covar_module.kernels.1.base_kernel.raw_period_length                             value = 0.506

EFFECTIVE prediction
"""

kernel1 = ExpSineSquared(length_scale=0.2*scale, periodicity=0.2*scale) + RBF(length_scale=0.6*scale) + WhiteKernel(noise_level=0.1) #Periodic + SE
"""
N=100

['RBF', 'PERIODIC']
final loss: 0.6256566643714905

Parameter name: covar_module.kernels.0.base_kernel.raw_lengthscale                               value = 0.673
Parameter name: covar_module.kernels.1.base_kernel.raw_lengthscale                               value = 0.131
Parameter name: covar_module.kernels.1.base_kernel.raw_period_length                             value = 0.6

"""

###Attempting to duplicate some of the values found on https://arxiv.org/pdf/1302.4922.pdf page 8


paper1 = RBF(length_scale=0.5*scale) + RationalQuadratic(length_scale=0.8*scale) + WhiteKernel(noise_level=0.01)
"""
['RBF', 'LIN']
final loss: -0.8393036723136902

covar_module.kernels.0.base_kernel.raw_lengthscale 0.38062992691993713
"""
paper2 = ConstantKernel(constant_value=-1.0) * ExpSineSquared(length_scale=0.3*scale, periodicity=0.3*scale) + WhiteKernel(noise_level=0.01)

"""
['PERIODIC']
final loss: -0.5422026515007019

covar_module.base_kernel.raw_lengthscale 0.3568160831928253
covar_module.base_kernel.raw_period_length 0.5963536500930786
"""
paper3 = 3*RBF(length_scale=0.05*scale) * RBF(length_scale=0.8*scale) + WhiteKernel(noise_level=0.1)

"""
['RBF', 'PERIODIC', 'PERIODIC', 'PERIODIC']
final loss: 0.43419793248176575

weird model - good prediction
"""

test2 = 0.1*RBF(length_scale=0.01*scale) * 5*RBF(length_scale=0.8*scale) + 2*ExpSineSquared(length_scale=0.5*scale, periodicity=0.5*scale) + WhiteKernel(noise_level=0.01)
test3 = 0.2*RBF(length_scale=0.01*scale) * 5*RBF(length_scale=0.8*scale) + 2*ExpSineSquared(length_scale=0.5*scale, periodicity=0.5*scale) + WhiteKernel(noise_level=0.1)

#set TEST kernel


kernel0 = ExpSineSquared(length_scale=0.2*scale, periodicity=0.2*scale) + RBF(length_scale=0.6*scale) + WhiteKernel(noise_level=0.03) #Periodic + SE

test4 = kernel0 + 0.3*test2

TEST=test4

def kernel_draw(N,Lim,kernel=TEST):
    #ASL=1
        # Specify Gaussian Process

    gp = GaussianProcessRegressor(kernel=kernel)

        # Plot prior

    X_ = np.linspace(0, scale*Lim, N)

    y_samples = gp.sample_y(X_[:, np.newaxis], 1)

    print(y_samples.shape)
    return stretch*np.squeeze(y_samples.T)

def function_draw(N,Lim):
    X = np.linspace(0, scale*Lim, N)

    Y = (0.3*np.sin(X*30) + 0.5*np.cos(X*5)**2 + X**0.53)**2.3 + np.random.rand(N)

    return stretch*np.squeeze(Y.T)

def load_data(filename):
    data = io.loadmat(filename)

    X = np.squeeze(data['X'])
    #X = X / np.max(X)
    Y = np.squeeze(data['y'])

    return X, Y

if __name__ == "__main__":
    Lim,N = 1,1000
    plt.figure(figsize=(8, 5))

    X_ = np.linspace(0, Lim, N)
    X_ = np.squeeze(X_.T)
    Y = kernel_draw(N,Lim)

    plt.plot(X_,Y)
    plt.show()
