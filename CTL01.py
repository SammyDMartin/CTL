import math
import torch
import gpytorch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from KernelGen import kernel_draw
from KernelGen import function_draw
from KernelGen import load_data
import time
import os

#2 possible model scoring heuristics
def BIC(loss,model,n):
    k = len([p for p in model.parameters()])
    #print(loss)
    return np.log(n)*k + 2*loss

def AIC(loss,model,n):
    k = float(len([p for p in model.parameters()]))
    return 2*k + 2*loss


class CompositionalGPModel(gpytorch.models.ExactGP):
    """
    A standard GPtorch kernel model, but modified to automatically generate a kernel composition if fed a list

    Args:
        kernel_pattern: a list of kernel descriptors (given in code below), where [[X,Y],Z] = (X+Y)*Z
        new_kernel: kernel to be appended to kernel_pattern
        add=boolean, True means add the kernel, False means multiply it

    """

    def __init__(self, likelihood, new_kernel, kernel_pattern=None,add=True):
        super(CompositionalGPModel, self).__init__(train_x, train_y, likelihood)

        grid_size = gpytorch.utils.grid.choose_grid_size(x_values,1.0)

        self.mean_module = gpytorch.means.ConstantMean()
        if new_kernel:
            if not kernel_pattern:

            #Starting case - First Kernel

                kernel_structure = self.selectkernel(new_kernel)
                self.new_kernel_pattern = [new_kernel]
            else:

            #Add or multiply new kernel to list, then generate new list

                if add:
                    self.new_kernel_pattern = kernel_pattern + [new_kernel]
                else:
                    self.new_kernel_pattern = [kernel_pattern, new_kernel]

                kernel_structure = self.extract_kernel(self.new_kernel_pattern)
        else:
            self.new_kernel_pattern = kernel_pattern
            kernel_structure = self.extract_kernel(kernel_pattern)
            """
            The kernel operators are: follows: + (addition), * (multiplication),

            and CP (a change point operator) #Difficult to implement - not in by default
            """

        #self.covar_module = gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.ScaleKernel(kernel_structure),grid_size=grid_size, num_dims=1)
        if grid:
            self.covar_module = gpytorch.kernels.GridInterpolationKernel((kernel_structure),grid_size=grid_size, num_dims=1)
            #cannot project past training data but gives working confidence regions
            #also not significantly faster
        else:
            self.covar_module = kernel_structure

       # self.covar_module = kernel_structure
        self.describe_kernel(self.new_kernel_pattern)

        #print(self.covar_module)
        print(self.new_kernel_pattern)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def selectkernel(self, kernel):

        """
        Pulls out the relevant Kernel function from GPytorch based on descriptor.
        """

        if kernel == 'RBF':
            kern_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel == 'PERIODIC':
            kern_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel == 'COS':
            kern_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        elif kernel == 'LIN':
            kern_module = gpytorch.kernels.LinearKernel(num_dimensions=1)
            #GONNA GET SCREWED UP BY NO SCALEKERNEL
        elif kernel == 'MATERN':
            kern_module = gpytorch.kernels.MaternKernel(nu=0.5)
        elif kernel == 'SPECTRAL':
            #BIC score don't work for this
            kern_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
            kern_module.initialize_from_data(train_x, train_y)
        elif kernel == 'WN':
            kern_module = gpytorch.kernels.WhiteNoiseKernel()

        else:
            print("NO!")

        return kern_module

    def extract_kernel(self, kernlist):

        """
        Builds kernel composition from an input list of kernel descriptions. [[X,Y],Z] = (X+Y)*Z
        """

        out = None
        for element in kernlist:
            if type(element) is str:
                kern = self.selectkernel(element)
                if out:
                    out = out + kern
                else:
                    out = kern
            elif type(element) is list:
                branch = self.extract_kernel(element)
                if out:
                    out = out * branch
                else:
                    out = branch
        return out

    def describe_kernel(self, kernlist, printl = True):
        """
        Mirrors extract_kernel - makes sure that the correct pattern of Kernel is generated as expected
        """
        out = None
        for idx, element in enumerate(kernlist):
            if type(element) is str:
                if idx < len(kernlist)-1:
                    if out:
                        out = out + element + " + "
                    else:
                        out = element + " + "
                else:
                    if out:
                        out = out + element
                    else:
                        out = element

            elif type(element) is list:
                branch = self.describe_kernel(element, False)
                if out:
                    out = out + "(" + branch + ")"
                else:
                    out = "(" + branch + ")"
        if printl:
            print()
            print(kernlist)
            print("+++ STRUCTURE INITIALIZED +++")
            print(out)
            print()
        return(out)

    def untransform(self):

        """
        Extracts and prints model raw parameters, then removes transformations given by the optimiser to print true model parameters.

        BROKEN in latest version of GPytorch
        """

        rawparams = [[rawparam_name, rawparam] for rawparam_name, rawparam in self.named_parameters()]
        #print([[param[0],param[1].item()] for param in rawparams])
        constraints = [[const_name, const] for const_name, const in self.named_constraints()]

        true_params = {}

        for param_entry in rawparams:
            transformed = False
            for constraint_entry in constraints:
                if param_entry[0] in constraint_entry[0]:
                    true_param =constraint_entry[1].transform(param_entry[1])

                    true_params[param_entry[0]] = true_param.item()

                    transformed = True
            if transformed == False:
                true_params[param_entry[0]] = param_entry[1].item()
        return true_params

def evaluate_model(model, likelihood, orig_loss):

    """
    A function to be run within score_model that evaluates the fitted model
    Args:
    model - a fitted gpytorch model
    likelihood - gpytorch likelihood function
    orig_loss - the loss function output by the function generator
    """

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    params = model.state_dict() #saves parameters for recovery

    try:
        output_train = model(train_x)
        output_validation = model(validation_x)
        output_test = model(test_x)
    except Exception as e:
        print(e)
        print(model.covar_module)
        print("Model cannot be evaluated")
        return model.new_kernel_pattern, math.inf, None

    try:
        loss_val = -mll(output_validation, validation_y).item() #loss is negative MLL
    except Exception as r:
        print(r)
        print(model.covar_module)
        print("Validation Failed")
        return model.new_kernel_pattern, math.inf, None

    #Plotting whole distribution BROKEN for large model!
    if grid:
        x_base = train_x
    else:
        x_base = x_values

    #x_base = x_values - this doesn't work for large models

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    #PLOT predictive distribution

    observed_pred = likelihood(model(x_base)) #generates the entire predictive distribution

    MSE_Train = np.mean((observed_pred.mean.numpy()[:N_train] - train_y.numpy())**2)
    MSE_Val = np.mean((observed_pred.mean.numpy()[N_train:(N_train+N_val)] - validation_y.numpy())**2)
    MSE_Test = np.mean((observed_pred.mean.numpy()[(N_train+N_val):] - test_y.numpy())**2)

    prediction = ax.plot(x_base.numpy(), observed_pred.mean.numpy(), 'b', label="Predictive Mean")
    try:
        lower, upper = observed_pred.confidence_region()
        ax.fill_between(x_base.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    except RuntimeError:
        print("LazyEvaluatedKernelTensor failure")

    # Plot training data as black
    trainplot = ax.plot(train_x.numpy(), train_y.numpy(), 'k',label="Training Data, loss = "+str(round(orig_loss,3))  + ", MSE = "+ str(round(MSE_Train,3)))
    # Plot validation data as green
    valplot = ax.plot(validation_x.numpy(), validation_y.numpy(), 'g', label="Validation Data, loss = "+str(round(loss_val,3)) + ", MSE = "+ str(round(MSE_Val,3)))
    # Plot test data as red
    try:
        testplot = ax.plot(test_x.numpy(), test_y.numpy(), 'r', label="Test Data, super_secret_loss = "+str(round(-mll(output_test, test_x).item(),3)) + ", MSE = "+ str(round(MSE_Test,3)))
    except Exception as E:
        print("Test cannot be evaluated")

    # Plot predictive means as blue line
    ax.legend(fontsize="x-large",markerscale=3)

    """
    BIC/AIC lead to underfitting for small data sets
    raw loss lead to overfitting for small data sets
    validation loss seems to work best
    validation MSE leads to overfitting on the validation set
    train MSE seems to avoid some odd behaviour that we get with raw loss
    (may be do do with approximation methods)
    """

    if mode == "loss":
        goodness = orig_loss
    elif mode == "BIC":
        goodness = BIC(float(loss.item()),model,N)
    elif mode == "AIC":
        goodness = AIC(float(loss.item()),model,N)
    elif mode == "validation_loss":
        goodness = loss_val
    elif mode == "validation_MSE":
        goodness = MSE_Val
    elif mode == "train_MSE":
        goodness = MSE_Train

    goodstring = "%.3f" % goodness

    plt.title(str(model.describe_kernel(model.new_kernel_pattern, False)) + " " + str(mode) + " = " + goodstring + ", N = " + str(N) + ", approximate (CUDA/Grid) = " + str(approximate)+str(grid) + " pass_params? = " + str(pass_thru) + str(", lr: ") + str(learning_rate))

    plt.savefig(str(the_time)+"/"+goodstring+" "+str(model.describe_kernel(model.new_kernel_pattern, False))+".png")

    return model.new_kernel_pattern, goodness, params


def score_model(base_kernels, new_kernel,toadd,approx_parameters):

    """
    Appends one kernel to the current frontier of the search and evaluates its model score. Returns the new list, model score and parameters

    Args:
        base_kernels: list representing kernel structure in correct format for CompositionalGPModel
        new_kernel: new kernel to test out
        toadd: True for add, False for multiply
        approx_parameters: model parameters from the current tree frontier to be fed in to the new kernel compositon (Non-Strict)
        mode: what is used to score model?
    """

    #likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-10), noise_prior=gpytorch.priors.NormalPrior(0, 1e-2)) #Standard likelihood
    #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(learn_additional_noise=False,noise=0.001*torch.ones([600,1], dtype=torch.float32)) #Standard likelihood

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = CompositionalGPModel(likelihood,new_kernel,base_kernels, add=toadd)

    if approx_parameters and pass_thru:
        #approx_parameters to be fed in from the last go round right now
        #later to be saved and recovered from time series library
        model.load_state_dict(approx_parameters, strict=False) #V IMPORTANT - LOADS STATE

        l1 = [p.item() for p in model.parameters()]
        l0 = [p.item() for v,p in approx_parameters.items()]
        #print(l0)
        #print(l1)
        sumcheck = lambda list1,list2: sum([list1[i] in list2 for i in range(len(list1))])
        print("Params load succeded?", str(sumcheck(l0,l1)) + "/" +str(len(l0)))
        
        #PARAMS LOAD does not work!

        #print(l0)
        #print(l1)
    else:
        if randomize:
            for p in model.parameters():
                p = (np.random.randn() - 1)

    model.train()
    likelihood.train()

    if superlocal and approx_parameters:
        for k,v in model.covar_module.named_parameters():
            if v.item() != 0.0:
                if v.item() in l0:
                    v.requires_grad = False
                    print(k,v.requires_grad,v.item())
        #print([p for p in model.named_parameters()])


    # Use different optimisers
    if optim == "Adam":
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=learning_rate)
    elif optim == "SGD":
        optimizer = torch.optim.SGD([
                    {'params': model.parameters()}
                ], lr=learning_rate, momentum=0.9)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in tqdm(range(steps)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Output from model
        output = model(train_x)

        # Calc loss and backprop gradients
        try:
            loss = -mll(output, train_y) #loss is negative MLL
        except Exception as e:
            print(model.covar_module)
            print(e)
            print("Kernel cannot be evaluated!")
            #If you're getting this all the time - probably because you're using unwrapped kernels on too large a data set
            #Or it could be tyring to put linear*linear or linear+linear in a scalekernel
            return(model.new_kernel_pattern, math.inf, None)

        loss.backward()
        optimizer.step()

    #Print final loss and noise values - sanity check during testing
    print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
    i + 1, steps, loss.item(),
    model.likelihood.noise.item()
    ))

    orig_loss = deepcopy(loss.item())

    # Get into evaluation (predictive posterior) mode
    # Make predictions by feeding model through likelihood

    model.eval()
    likelihood.eval()

    if approximate:
        #switch over to the accelerations required for the Scalable GP Regression (CUDA) with Fast Predictive Distributions
        #change root decomp size??
        #significantly faster
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(20):
            return evaluate_model(model,likelihood,orig_loss)
    else:
        with torch.no_grad():
            return evaluate_model(model,likelihood,orig_loss)

def explore_add_multiply(base_kernels,approx_parameters):
    """
    Expands out the current frontier of the tree search, multiplies and adds every kernel and chooses the best.
    Implements the next step of the greedy best-first search.

    Args:
        train_x,train_y: training data
        base_kernels: list representing kernel structure in correct format for CompositionalGPModel
        approx_parameters: model parameters from the current tree frontier to be fed in to the new kernel compositon (Non-Strict)
        mode1: what is used to score model?
            BIC/AIC lead to underfitting for small data sets
            raw mll lead to overfitting for small data sets
    """

    structures = []
    params = []
    losses = []

    #finds best kernel and stores for next stage of search

    for new_kernel in kernels:
        #cycles through all kernels and scores them
        s, l, p = score_model(base_kernels, new_kernel,True,approx_parameters)
        s1, l1, p1 = score_model(base_kernels, new_kernel,False,approx_parameters)


        structures.append(s)
        params.append(p)
        losses.append(l)

        structures.append(s1)
        params.append(p1)
        losses.append(l1)

    best = [structures[np.argmin(losses)],params[np.argmin(losses)],losses[np.argmin(losses)]]

    return(best)

#Create Folder for this run-through of the Search
the_time = time.strftime("%m %d %Y, %H %M %S")
os.mkdir(the_time)
use_synth = True # use synthetic data from Kernelgen
pass_thru = True #whether to pass parameters on or not
randomize = False #If NOT passing through, randomize starting parameters?

steps = 50 #number of gradient descent steps
threshold = 1e-2 #when to ignore differences in model score
optim = "Adam"
superlocal = False #attempt to stop refitting old kernels. BUGGY!
learning_rate = 1e-1

#loss, BIC, AIC, validation_loss, validation_MSE, train_MSE
mode = "loss"

approximate = False #faster but less accurate, also doesn't handle linear kernels so well
grid = False #grid interpolation - unable to plot predictions

#Number of data points in the range (0,1)

N_train=500
N_val=100
N_test=100

N = N_train+N_val+N_test


if use_synth:
    x_values = torch.linspace(0, 1, N)
    #y_values = torch.tensor(function_draw(N,1), dtype = torch.float32)
    #can only use Kernel_draw for small N<2000
    y_values = torch.tensor(kernel_draw(N,1), dtype = torch.float32) #Draws out test kernel
else:
    x_values,y_values = load_data('Examples/03-mauna2003-res')
    x_values = torch.tensor(x_values, dtype = torch.float32)
    y_values = torch.tensor(y_values, dtype = torch.float32)

    N = (x_values.shape)[0]

    N_train = int(N*0.8)
    N_val = int(N*0.1)
    N_test = N - N_train - N_val
    print(N_train,N_val,N_test)


train_x = x_values[:N_train]
validation_x = x_values[N_train:(N_train+N_val)]
test_x = x_values[(N_train+N_val):]

train_y = y_values[:N_train]
validation_y = y_values[N_train:(N_train+N_val)]
test_y = y_values[(N_train+N_val):]

#kernels = ['LIN','RBF','PERIODIC','COS'] #ACTUALLY WORKS
kernels = ['RBF','PERIODIC','LIN']
#kernels = ['RBF','PERIODIC']



#Execute main search
if __name__ == "__main__":
    "Sanity Checks"
    #score_model(None,'PERIODIC',True,None)
    score_model(['LIN'],'SPECTRAL',False,None) #THis is the baseline

    last_loss = loss = None
    last_structure = structure = None
    last_params = params = None
    loss = math.inf

    for depth in range(10):
        last_loss = deepcopy(loss)
        last_structure = deepcopy(structure)
        last_params = deepcopy(params)
        print(last_structure)
        print(last_loss)

        #Key line that does the recursion
        structure, params, loss = explore_add_multiply(structure,params)
        if loss>(last_loss-float(threshold)): #if loss is unchanged (more-or-less)
            print("+++SEARCH TERMINATED+++")
            break

    test = CompositionalGPModel(gpytorch.likelihoods.GaussianLikelihood(),None,last_structure)
    test.describe_kernel(test.new_kernel_pattern)
    l1 = [p for p in test.parameters()]
    l0 = deepcopy(l1)
    test.load_state_dict(last_params, strict=False) #V IMPORTANT - LOADS STATE
    print("Params load succeded?", str(len([l0[i]!=l1[i] for i in range(len(l0))])) + "/" +str(len(l0)))

    print(test.new_kernel_pattern)
    print("final loss: "+str(loss))

    args = test.untransform()
    print()
    for key, value in args.items():
        #print(key,value)
        "print(f'Parameter name: {key:80} value = {round(value,3)}')"
