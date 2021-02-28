import math
import torch
import gpytorch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from KernelGen import kernel_draw

N=300
mode = "NLL"
#mode = "BIC"

task = "best"
print("Utility measure?", mode)

train_x = torch.linspace(0, 1, N)
train_y = torch.tensor(kernel_draw(N,1), dtype = torch.float32)


"""
Kernels from Autostat:

C
(constant) [handled by mean_module]
Lin
(linear), [LIN]
SE
(squared exponential), [RBF]
Per
(periodic), and [REPEAT]
WN
(white noise)
#WhiteNoiseKernel is now hard deprecated. Use a FixedNoiseLikelihood instead.
"""

kernels = ['RBF','PERIODIC','COS','LIN','MATERN','SPECTRAL'] #The Full list

kernels = ['RBF','PERIODIC','LIN'] #PURE AUTOSTAT

kernels = ['RBF','PERIODIC','LIN','COS'] #ACTUALLY WORKS

def BIC(loss,model,n):
    k = len([p for p in model.parameters()])
    #print(loss)
    return np.log(n)*k + 2*loss

"""
def BIC(loss,model,n):
    k = float(len([p for p in model.parameters()]))
    return 2*k + 2*loss
"""
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='RBF', kernel2=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if not kernel2:
            self.covar_module = self.selectkernel(kernel)
        if kernel2:
            
            """
            The kernel operators are: follows: + (addition), × (multiplication),
            
            and CP (a change point operator) #Difficult to implement - not in by default
            """
            
            self.covar_module = gpytorch.kernels.ScaleKernel(self.selectkernel(kernel) + self.selectkernel(kernel2))
            #EITHER WORKS??
            #self.covar_module = gpytorch.kernels.ScaleKernel(self.selectkernel(kernel) * self.selectkernel(kernel2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def selectkernel(self, kernel):
        if kernel == 'RBF':
            kern_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel == 'PERIODIC':
            kern_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel == 'COS':
            kern_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        elif kernel == 'LIN':
            kern_module = gpytorch.kernels.LinearKernel(num_dimensions=1)
        elif kernel == 'MATERN':
            kern_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        elif kernel == 'SPECTRAL':
            #BIC score don't work for this
            kern_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
            kern_module.initialize_from_data(train_x, train_y)
        elif kernel == 'WN':
            kern_module = gpytorch.kernels.WhiteNoiseKernel()

        else:
            print("NO!")
        return kern_module

def evaluate_kernel(kind, kind2=None,plot=False):
    # initialize likelihood and model
    #NP=1
    #likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.NormalPrior(0, NP))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = ExactGPModel(train_x, train_y, likelihood, kind, kind2)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()


    # Use the adam optimizer
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    
    """
    optimizer = torch.optim.SGD([
                {'params': model.parameters()},
            ], lr=0.1, momentum=0.2)
    """
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in tqdm(range(training_iter)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Output from model
        output = model(train_x)

        # Calc loss and backprop gradients
        loss = -mll(output, train_y) #loss is negative MLL
        loss.backward()
        optimizer.step()
    
    print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
    i + 1, training_iter, loss.item(),
    model.likelihood.noise.item()
    ))

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        params = model.state_dict() #saves parameters for recovery
        if plot == True:
            #to plot fitted model and save figure if required
            test_x = torch.linspace(0, 1, 101)
            observed_pred = likelihood(model(test_x))

            f, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
        #BROKEN FOR SOME REASON

        # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='1')
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            if not kind2:
                kind2 = ""
            plt.title(kind+" "+kind2+" (new)")
            plt.savefig(kind+" "+kind2+" (new)"+".png")
            return


    goodness = BIC(float(loss.item()),model,N)


    return goodness, float(loss.item()), params #BIC score (or whatever else is used), -NLL, model state dict

def rank(lst):
    seq = sorted(lst)
    index = [lst.index(v) for v in seq]
    return index

def maximise_kernel(last_round=False, mode=mode):
    BICs = []
    losses = []
    params = []
    
    #finds best kernel and stores for next stage of search
    
    for kernel in kernels:
        if not last_round:
            g,l,p = evaluate_kernel(kernel)
        else:
            g,l,p = evaluate_kernel(last_round[0],kernel)
        BICs.append(g)
        params.append(p)
        losses.append(l)
        
    #for first level of search
    if not last_round:
        best = [kernels[np.argmin(losses)],losses[np.argmin(losses)]]
        if mode=="BIC":
            best = [kernels[np.argmin(BICs)],BICs[np.argmin(BICs)]]
    
    #for second level of search
    else:
        best = [last_round[0], kernels[np.argmin(losses)],losses[np.argmin(losses)]]
        if mode=="BIC":
            best = [last_round[0], kernels[np.argmin(BICs)],BICs[np.argmin(BICs)]]

        improved = (last_round[1]-best[2]) >0.01 #if there has been a significant increase in (whatever the score is)
        evaluate_kernel(last_round[0], plot=True)
        #BIC apparently doesn't work - look into it

    print("Winning Score:", tuple(best))

    print("NLL:")
    if not last_round:
        print([(kernels[i],losses[i]) for i in rank(losses)])
    else:
        print([(last_round[0], kernels[i], losses[i]) for i in rank(losses)])


    print("BICs:")
    if not last_round:
        print([(kernels[i],BICs[i]) for i in rank(BICs)])
    else:
        print([(last_round[0], kernels[i], BICs[i]) for i in rank(BICs)])


    store = params[np.argmin(losses)] #will need later
    print("Number of params:",len(store))

    #print("\nWinning Params:", [val.item() for val in store])
    if not last_round:
        return best
    else:
        print("")
        print("Search improved", mode, "?", improved)
    
        #return best[:2], store

    
        if improved == False:
            plt.show()
            #Don't display the second level fit if there is no improvement
            return False, False
        else:
            return best[:2], store

def untransform(model):
    rawparams = [[rawparam_name, rawparam] for rawparam_name, rawparam in model.named_parameters()]
    constraints = [[const_name, const] for const_name, const in model.named_constraints()]

    #print(rawparams)
    #print(constraints)

    for param_entry in rawparams:
        transformed = False
        for constraint_entry in constraints:
            if param_entry[0] in constraint_entry[0]:
                true_param =constraint_entry[1].transform(param_entry[1])
                print()
                print(param_entry[0],true_param.item()," (extracted)", constraint_entry[0])
                transformed = True
        if transformed == False:
            print()
            print(param_entry[0],param_entry[1].item()," (raw)")


def exhaustive_maximise_kernel():

    #Exhaustive (depth first) search of kernels
    
    BICs = {}
    losses = {}

    for kernel2 in kernels:
        for kernel in kernels:
            g,l,p = evaluate_kernel(kernel,kernel2)

            BICs[(kernel2,kernel)] = g
            losses[(kernel2,kernel)] = l


    print("NLL:")

    for key, value in sorted(losses.items(), key=lambda item: item[1]):
        print("%s: %s" % (key, value))

    print("BICs:")

    for key, value in sorted(BICs.items(), key=lambda item: item[1]):
        print("%s: %s" % (key, value))

###plotfn

def plotfn(kernels,params):
    
    #Plots using actual recovered model parameters
    
    kernel = kernels[0]
    kernel2 = kernels[1]


    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = ExactGPModel(train_x, train_y, likelihood, kernel, kernel2)

    l1 = [p for p in model.parameters()]
    l0 = deepcopy(l1)
    model.load_state_dict(params, strict=False) #V IMPORTANT - LOADS STATE
    print("Params load succeded?", str(len([l0[i]!=l1[i] for i in range(len(l0))])) + "/" +str(len(l0)))
    #print(l0)
    #print(l1)
    # Get into evaluation (predictive posterior) mode


    model.eval()
    likelihood.eval()
    untransform(model)
    #Print true parameters



    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1.5, 101)
        
        observed_pred = likelihood(model(test_x))

        f, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        
        #BROKEN FOR SOME REASON if too many test data points

        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='1')
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        plt.title(kernel+" "+kernel2)
        plt.savefig(kernel+" "+kernel2+".png")
        plt.show()
    # Shade between the lower and upper confidence bounds

#task = input("Exhaustive (e), one-level(o) or best-first? ")
if __name__ == "__main__":
    if task == "e":
        exhaustive_maximise_kernel()
    elif task == "o":
        maximise_kernel()
    else:
        kernels,params = maximise_kernel(maximise_kernel())
        if params:
            plotfn(kernels,params)
        #evaluate_kernel(kernels[0],kernels[1],True)
