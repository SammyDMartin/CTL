import GPy
from IPython.display import display
import numpy as np
import time
import os
from sklearn.model_selection import RepeatedKFold
import io 

REST = 10 #how many random restarts to do (do more for larger models)
adj = False #switch between mean (true) and sum (false) loss values
MESS = True #whether to print messages
PLOT = False
splits,repeats = 3,1 #if CV loss is used
mkdir = True #must be true unless running CTL04. then must be false
PARA = False

#Create Folder for this run-through of the Search
the_time = time.strftime("%m %d %Y, %H %M %S")
import matplotlib
import matplotlib.pyplot as plt
if mkdir:
    try:
        os.mkdir(the_time)
    except Exception:
        pass

class CompositionalGPyModel():
    """
    A compositional kernel using the more stable (but slower) GPy. Has additional methods for optimizing.
    Args:
        inputs: Dictionary of x and y values and Numbers in special format made by CTL02 function create_inputs
        new_kernel/add: use if you want to do amend kernel without CTL02, otherwise leave it blank
        kernel_pattern = list describing kernel structure in format where [x,[y]] means x*y and [x,y] means x+y
            parsed left-to-right
    """

    def __init__(self,inputs, new_kernel, kernel_pattern=None,add=True):
        self.x_values,self.y_values = inputs['x_values'],inputs['y_values']
        self.train_x,self.train_y = inputs['train_x'], inputs['train_y']
        self.validation_x, self.validation_y = inputs['val_x'], inputs['val_y']
        self.test_x, self.test_y = inputs['test_x'], inputs['test_y']
        self.N, self.N_train, self.N_val, self.N_test = inputs['N'], inputs['N_train'],inputs['N_val'],inputs['N_test']
        self.name = inputs['name'][:20]
        self.name = "[" + self.name +"] "
        self.mode = inputs['mode']
        self.fold = ''
        self.figure = None

        if self.mode in ['loss','BIC']:
            #no train/test split
            self.train_x = self.x_values
            self.train_y = self.y_values
            self.N_train = self.N
            self.validation_x,self.validation_y,self.test_x,self.test_y,self.N_val,self.N_test = None,None,None,None,None,None

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
        self.model = GPy.models.GPRegression(self.train_x,self.train_y,kernel_structure) #employ standard GPy model
       # self.covar_module = kernel_structure
        self.describe_kernel(kernel_pattern)

        #print(self.covar_module)
        #print(self.new_kernel_pattern)

    def selectkernel(self, kernel):
        """
        Pulls out the relevant Kernel function from GPy based on descriptor. Used to construct compositonal kernel
        """
        #see https://www.cs.toronto.edu/~duvenaud/cookbook/ for info on kernel choices
        #http://mlg.eng.cam.ac.uk/tutorials/06/es.pdf
        
        if kernel == 'RBF':
            kern_module = GPy.kern.RBF(1)
        elif kernel == 'PERIODIC':
            kern_module = GPy.kern.PeriodicExponential(1)
        elif kernel == 'COS':
            kern_module = GPy.kern.Cosine(1)
        elif kernel == 'LIN':
            kern_module = GPy.kern.Linear(1) + GPy.kern.Bias(1)
        elif kernel == 'MATERN':
            kern_module = GPy.kern.Matern32(1)
        elif kernel == 'SPECTRAL':
            #BIC score don't work for this
            raise NotImplementedError
        elif kernel == 'WN':
            kern_module = GPy.kern.White(1)
        elif kernel == 'CONST':
            kern_module = GPy.kern.Bias(1)
        elif kernel == 'RQ':
            kern_module = GPy.kern.RatQuad(1)
        else:
            print("NO!")

        return kern_module

    def extract_kernel(self, kernlist):

        """
        Builds kernel composition from an input list of kernel descriptions. [[X,Y],Z] = (X+Y)*Z
        """
        backwards = lambda l: (backwards (l[1:]) + l[:1] if l else [])

        out = None
        for element in backwards(kernlist): #oh god don't ask
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
        returns str description of kernel to be parsed by grammar reader
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
            print(self.name.ljust(20)+ out)
            print()
        return(out)

    def load_params(self,Params):
        """
        used in initialization - loads numpy array of all model parameters and prints params before and after
        fills out remaining parameters with no equivalent in Params with ones
        """
        modelparams = self.model[:]

        if not Params:
            return
        Params = np.array(Params)
        gap = np.shape(modelparams)[0] - np.shape(Params)[0]
        Params = np.hstack((Params,np.ones(gap)))

        print([x for x in self.model])
        self.model[:] = Params
        print([x for x in self.model])

    def BIC(self):
        """
        uses sum marginal log likelihood and returns model BIC score
        """
        k = len([p for p in self.model])
        #print(loss)
        n = np.shape(self.train_x)[0]
        mll = float(self.model._log_marginal_likelihood)

        return float(np.log(n)*k - 2*mll)

    def validate(self):
        """
        Creates model validation loss - gets called after training in optimize
        """

        #https://link.springer.com/article/10.1007/s11222-013-9416-2#Sec2
        #https://arxiv.org/pdf/1307.5928.pdf


        self.validation_loss = -1*np.sum(self.model.log_predictive_density(self.validation_x,self.validation_y))

    def goodness(self):
        """
        Determines what is returned as the actual model loss depending on mode (file global variable)
        loss, BIC, validation_MSE,validation_loss
        """
        if self.mode == "loss":
            if adj:
                return -1*float(self.model._log_marginal_likelihood)/self.N_train
            return -1*float(self.model._log_marginal_likelihood)
        elif self.mode == "BIC":
            return self.BIC()
        elif self.mode == "validation_MSE":
            return self.MSE_Val
        elif self.mode == "validation_loss":
            if adj:
                return self.validation_loss/self.N_val
            return self.validation_loss
        elif self.mode == "cross_validation_loss":
            if adj:
                return self.validation_loss/(np.shape(self.validation_x)[0])
            return self.validation_loss

    def optimize(self):
        """
        To be called after params loaded, if there are any. Returns nothing but runs model optimization and plots the big graph.
        Returns the kernel pattern, loss value and new parameters

        """
        print()
        print("+++OPTIMIZING+++")
        try:
            self.model.optimize(messages=False)
            self.model.optimize_restarts(num_restarts = REST,parallel=PARA,verbose=MESS)
        except Exception as E:
            print(E)
            print("Cannot Evaluate!")
            return self.new_kernel_pattern,np.inf,None
        #mean,cov = self.model.predict(validation_x)
        #[mean,cov,lower,upper] = self.model.predict(test_x)

        #plt.plot(mean,validation_x)

        display(self.model)
        if self.validation_x is not None:
            self.validate()
            #might not be a validation set

        if PLOT:
            plt.close('all')
            self.model.plot(plot_density=True,figsize=(20,10),samples=0,resolution=1000,plot_limits=np.array([np.min(self.x_values),np.max(self.x_values)]))
            #fig_conf = self.model.plot_confidence(resolution = 1000)

        simY, simMse = self.model.predict(self.x_values)

        self.MSE_Train = np.mean((simY[:self.N_train] - self.train_y)**2)
        if self.validation_x is not None:
            self.MSE_Val = np.mean((simY[self.N_train:(self.N_train+self.N_val)] - self.validation_y)**2)
            if self.N_train + self.N_val == self.N:
                MSE_Test = np.nan
            else:
                MSE_Test = np.mean((simY[(self.N_train+self.N_val):] - self.test_y)**2)

        if PLOT:
            plt.plot(self.x_values, simY, '-k', label = "Prediction")
            plt.fill_between(np.squeeze(self.x_values), np.squeeze(simY - 3 * simMse ** 0.5), np.squeeze(simY + 3 * simMse ** 0.5), alpha=0.1)

        params = [x for x in self.model]
        goodstring = "%.3f" % self.goodness()
        goodstring = "{} = {}".format(self.mode, goodstring)
        namestr = str(self.name) + " " + str(self.describe_kernel(self.new_kernel_pattern, False)) + " " + goodstring + ", N = " + str(self.N) + ", BIC = " + str(round(self.BIC(),3)) + self.fold
        if PLOT:
            plt.plot(self.train_x, self.train_y, ',',label="Training Data, loss = "+str(round(-1*float(self.model._log_marginal_likelihood),3))  + ", MSE = "+ str(round(self.MSE_Train,3)))
            
            if self.validation_x is not None:
                plt.plot(self.validation_x, self.validation_y, 'g.', label="Validation Data, loss 'LPD' = "+str(round(self.validation_loss,3)) + ", MSE = "+ str(round(self.MSE_Val,3)))
                plt.plot(self.test_x,self.test_y, 'r.', label= "TEST MSE = "+ str(round(MSE_Test,3)))
            
            plt.legend(fontsize="x-large",markerscale=3)
            plt.title(namestr)
            self.savestr = str(the_time)+"/" +str(self.name) + " " + goodstring+" "+str(self.describe_kernel(self.new_kernel_pattern, False))+self.fold+".png"
        print()
        print("++++++STATS++++++")
        display(namestr)
        #display(self.savestr)
        print("+++++++++++++++++")
        print()


        if PLOT:
            out_fig = [manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()][0] #gets all figures and adds this one
            self.figure = (self.savestr,out_fig)

        return self.new_kernel_pattern, self.goodness(), params
    
    def fit(self):
        """
        Wrapper around optimize to enable k-fold cross validation. Returns the cross-validation score by re-running multiple times if cross-validation is chosen.
        """
        if self.mode == "cross_validation_loss":
            k_score = []
            np.random.seed()
            rkf = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=np.random.randint(1e8))
            for train_index, test_index in rkf.split(self.x_values):
                self.train_x, self.validation_x = self.x_values[train_index], self.x_values[test_index]
                self.train_y, self.validation_y = self.y_values[train_index], self.y_values[test_index]
                self.test_x,self.test_y = self.validation_x,self.validation_y
                self.N_train, self.N_val, self.N_test = np.shape(self.train_x)[0],np.shape(self.validation_x)[0],np.shape(self.test_x)[0]
                try:
                    k_model = GPy.models.GPRegression(self.train_x,self.train_y,self.model.kern)
                except Exception as E:
                    print(E)
                    print("Evaluation fail")
                    return self.new_kernel_pattern,np.inf,None
                self.fold = str(", fold {}".format(str(len(k_score)+1)))
                self.model = k_model
                
                print(self.name+self.fold)
                _, goodness, _ = self.optimize()
                k_score.append(goodness)

            score = np.mean(np.array(k_score))*(self.N/self.N_val) #correct to full 'mll'
            print(self.N/self.N_val)
            print("+++++++++++++++++")
            print("Cross-Validation Score = {}".format(round(score,4)))
            print("+++++++++++++++++")
            return self.new_kernel_pattern, score, None
        else:
            return self.optimize()


                
    def breakdown_params(self):
        out = io.StringIO()
        #could write out extra params here
        out.write(self.name)
        out.write("N_train {}, N_val {}, N_test {}".format(self.N_train,self.N_val,self.N_test))
        out.write("{}: Splits, {}, Repeats, {}, Restarts {}".format(self.mode,splits,repeats,REST))
        out.write(str(self.new_kernel_pattern)+"\n")
        print(self.model, file=out)
        return out
        
        #partial and not functional heirachical params breakdown
        """
        def depth_regex(tup):
            kern,depth = tup[0],tup[1]
            depthstr = str('[a-z]+\.')
            
            if kern == 'LIN':
                out = "linear"
            elif kern == 'PERIODIC':
                out = "periodic_exponential"
            elif kern == 'RBF':
                out = "rbf"
            return (depthstr*depth)+out

        def extract(tup):
            kern,depth = tup[0],tup[1]
            try:
                out = self.model[depth_regex((kern,depth))].values()
            except Exception:
                out = None
            else:
                print("{0} at depth {1}".format(kern,depth))
            return out

        params_heirachy = {}

        for kern in ['RBF','PERIODIC','LIN']:
            for depth in range(10):
                node = extract((kern,depth))
                if node is not None:
                    params_heirachy[(kern,depth)] = node
            
        for k,v in params_heirachy.items():
            print(str(k).ljust(8), str(v))
        return params_heirachy
        """
