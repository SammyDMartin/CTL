#scl enable rh-python36 bash
from anytree import NodeMixin, RenderTree
import anytree
import math
import numpy as np
import pickle
import time
import KernelGen
from IPython.display import display

from KernelGen import kernel_draw
from KernelGen import function_draw
from KernelGen import load_data
import time
import os
import sys
import io
import CTLGPy

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

use_old = False
if use_old:
    import CTL01

def create_inputs(filename=None,m=None,folder="training_data_real",Train_f = 0.8, Val_f = None,cut=None):
    """
    creates the global variable inputs, which is used in class ScoredKernelNode
    filename = the name of the mat file in training_data/
    simulate = CTL02.[kernel name] (found in KernelGen)
    fill out either to simulate/draw real data
    """
    x_values,y_values = load_data(folder+'/'+str(filename))
    if cut:
        N = (x_values.shape)[0] * cut
        x_values = x_values[:int(N)]
        y_values = y_values[:int(N)]

    N = (x_values.shape)[0]

    N_train = int(N*Train_f)
    if Val_f is not None:
        N_val = int(N*Val_f) #reasonable normal split 80/10/10
        N_test = N - N_train - N_val

        train_x = x_values[:N_train][:,np.newaxis]
        validation_x = x_values[N_train:(N_train+N_val)][:,np.newaxis]
        test_x = x_values[(N_train+N_val):][:,np.newaxis]

        train_y = y_values[:N_train][:,np.newaxis]
        validation_y = y_values[N_train:(N_train+N_val)][:,np.newaxis]
        test_y = y_values[(N_train+N_val):][:,np.newaxis]
    else:
        #No test set

        N_test = N_val = N - N_train

        test_y = validation_y = y_values[N_train:][:,np.newaxis]
        test_x = validation_x = x_values[N_train:][:,np.newaxis]

        train_x = x_values[:N_train][:,np.newaxis]
        train_y = y_values[:N_train][:,np.newaxis]
    



    x_values,y_values = x_values[:,np.newaxis], y_values[:,np.newaxis]

    inputs={}
    inputs['x_values'], inputs['y_values'] = x_values,y_values
    inputs['train_x'], inputs['train_y'] = train_x,train_y
    inputs['val_x'], inputs['val_y'] = validation_x,validation_y
    inputs['test_x'], inputs['test_y'] = test_x,test_y
    inputs['N'], inputs['N_train'],inputs['N_val'],inputs['N_test'] = N,N_train,N_val,N_test
    inputs['name'] = filename + " "
    inputs['mode'] = m
    return inputs


def render(root):
    "pretty prints the entire tree for viewing purposes, just loss and best-or-not"

    print(str("").ljust(90),str("-mll").ljust(20), str("best").ljust(8))
    for pre, _, node in RenderTree(root):
        treestr = u"%s%s" % (pre, node.name)
        loss_string = str(round(node.loss,3))
        delta_string = str(node.best)
        print(treestr.ljust(90).encode("utf-16"), loss_string.ljust(8), delta_string.ljust(8))

def recursive_len(item):
    "helper function for searching kernel structures"
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1

class ScoredKernel():
    """
    Goes in ScoredKernelNode - more or less a wrapper around model scoring for gpy or gpytorch
    """
    def __init__(self,Kernels, Params):
        if not Kernels:
            self.structure = []
            self.loss = np.inf
            self.params = None
        else:
            if use_old:
                self.structure, self.loss, self.params = CTL01.score_model(Kernels,None,True,Params)
            else:
                #try:
                model = CTLGPy.CompositionalGPyModel(Inputs,None,Kernels,True)
                if LOAD and Params:
                    model.load_params(Params)
                self.structure, self.loss, self.params = model.fit()
                if model.figure:
                    self.plot = model.figure
            #except Exception as E:
                #    print(E)
                #    print("Model Failure!")
                #    self.structure,self.loss,self.params = Kernels, math.inf, Params
                self.params_breakdown = model.breakdown_params()
        self.allkernels = ['RBF','PERIODIC','LIN']

class ScoredKernelNode(ScoredKernel, NodeMixin):  # Add Node feature
    """
    An anytree object that serves as a node containing all parameters and loss function values for a kernel search

    """
    def __init__(self,Kernels=None,parent=None, children=None,Params=None):
        super(ScoredKernelNode, self).__init__(Kernels, Params)
        self.name = self.describe_kernel(self.structure)
        self.parent = parent
        self.LIM = LIM
        self.best = False
        #self.MAP = 0
        if children:
            self.children = children
        if Kernels == None:
            #is a root node
            self.name = "root"
            self.counts = self.combinate() #gonna get used later
            self.priors = {}
            self.mode = mode

    def describe_kernel(self, kernlist):
        """
        Mirrors extract_kernel - makes sure that the correct pattern of Kernel is generated as expected.
        It is also used to parse the kernel, returns kernel str description
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
                branch = self.describe_kernel(element)
                if out:
                    out = out + "(" + branch + ")"
                else:
                    out = "(" + branch + ")"
        return(out)

    def combinate(self):
        """
        returns the dictionary of grammar rule usage counts used by the root node in the search (and potentially by others)
        """
        grammar = {}
        for kern1 in self.allkernels:
            grammar[str('+'+kern1)] = 0 #start condition
            grammar[str(kern1+'-')] = 0 #stop condition
            for kern2 in self.allkernels:
                grammar[self.describe_kernel([kern1,kern2])] = 0
                multiply = self.describe_kernel([[kern1],kern2])
                multiply = multiply[1:]
                grammar[multiply] = 0
        return grammar

    def list_children(self):
        """
        returns a list of structure-lists that inherit from this structure either by adding or multiplying a new kernel.
        Used to construct the next level of search.
        """

        kernlist = self.structure
        if self.structure == []:
            return [[kernel] for kernel in self.allkernels]
        children_list = []
        for new_kernel in self.allkernels:
            new_composition_1 = kernlist + [new_kernel]
            new_composition_2 = [kernlist, new_kernel]
            children_list.append(new_composition_1)
            children_list.append(new_composition_2)

        return children_list

    def children_from_parent(self):
        """
        produces actual child nodes from list of child structures.
        """
        children = []
        loc = "Depth " + str(recursive_len(self.structure)+1) +"/" +str(LIM)
        for child_structure in tqdm(self.list_children(),desc = loc,miniters = 1):
            child = ScoredKernelNode(child_structure, Params=self.params)
            children.append(child)
        self.children = children

    def update_at_end(self, noprint = False):
        """
        This function parses a description of a kernel and breaks it down into component transtion rules using advanced space magic.
        own_counts = False will search out the best node in the whole tree and update the final count tally (for best-first)
        own_counts = True will break down this nodes' own kernel and deduce the transition rules that were used to reach it

        the counts for root (own_counts=False) are fed into CTL03 to build up the grammar priors. On its own this does not affect search.
        """

        for key in self.counts.keys():
            self.counts[key] = 0
            #wipe counts if re-running in CTL03

        #for updating self with root node
        if self.parent:
            #this is only supposed to be run on root
            return
        moves = anytree.search.findall(self, filter_ = lambda node : node.best is True)
        #gives a list sequence of all the moves best-first search took
        final = moves[-1]
        final_structure = self.describe_kernel(final.structure)

        #UPDATE the root node with all the search parameters
        self.name = final_structure + " " + str(round(final.loss,4))
        self.params = final.params
        self.params_breakdown = final.params_breakdown
        #https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb
        #find the finishing kernel and update its counts

        started,finished = False, False
        try:
            for idx,_ in enumerate(final_structure):
                partial = final_structure[-idx-1:]
                for start in self.allkernels:
                    if start in partial:
                        k = str(start)+'-'
                        self.counts[k] = self.counts[k] + 1
                        finished = True
                if finished == True:
                    break

            #find the starting kernel and update its counts
            for idx,_ in enumerate(final_structure):
                partial = final_structure[:idx+1]
                for start in self.allkernels:
                    if start in partial:
                        k = '+'+str(start)
                        self.counts[k] = self.counts[k] + 1
                        started = True
                if started == True:
                    break

            #count all the non-starting transitions til the terminal state
            for key in self.counts.keys():
                if key[1:] not in self.allkernels and key[:-1] not in self.allkernels:
                    #non-starting
                    self.counts[key] = self.counts[key] + sum(1 for i in range(len(final_structure)) if final_structure.startswith(key, i))
        except Exception as E:
            print(E)
        #print verbose message about count updating

        try:
            try:
                sys.stdout = open(u_time+'/{}_textree.txt'.format(the_time), 'a')
            except Exception:
                sys.stdout = open(u_time+'/{}_textree.txt'.format(the_time), 'w')
            print("LIM: {}, MODE: {}, thresh: {}\n\n".format(LIM,mode,thresh))
        except Exception:
            pass
        
        print("Final, Loss: {}".format(self.name))
        
        print("Search Seq: ")
        seq = [(x.name, str(round(x.loss,4))) for x in moves]
        for tup in seq:
            print(tup[0].ljust(30),tup[1])
        print()
        try:
            print("Counts:")
            for key,value in self.counts.items():
                if value != 0:
                    print(key.ljust(30),str(round(value,4)))
        except Exception as E:
            pass

        print("\n\n")
        print(self.params_breakdown.getvalue())
        print("\n\n")

        


    def best_first(self):
        """
        Performs breadth first search given current data and CTLGPy parameters
        """
        self.children_from_parent()
        scores = [child.loss for child in self.children]

        best_child = self.children[np.argmin(scores)]
        
        threshold = np.abs(best_child.loss) * float(thresh)

        if best_child.loss + threshold < best_child.parent.loss:
            print("+++CURRENT BEST+++")
            print(best_child.name, str(round(best_child.loss,4)))
            if hasattr(best_child,"plot"):
                fig = best_child.plot[1]
                savestr = best_child.plot[0]
                fig.savefig(savestr)
            best_child.best = True
        else:
            print("+++LOSS OPTIMUM REACHED+++")
            best_child.best = False

        if self.name is "root":
            best_child.best = True

        if (recursive_len(best_child.structure) < self.LIM) and best_child.best is True:
            best_child.best_first()
        else:
            print("+++SEARCH TERMINATED+++")

def operate():
    """
    convenience function that runs all ScoredKernelNode methods in the right order and saves the resulting tree
    """
    root=ScoredKernelNode()
    root.best_first()
    root.update_at_end()
    render(root)
    sys.stdout = sys.__stdout__#have to go after update_at_end

    print()
    
    time.sleep(2)
    the_name = str(u_time+'/' + name + ' tree.pickle')

    with open(the_name, 'wb') as f:
        pickle.dump(root, f, pickle.HIGHEST_PROTOCOL)
        #need to save whole tree somehow, when actually making use of grammar file
    
    #update log file at end

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search Hyperparams')

    parser.add_argument('-mode', action="store", type = str)
    parser.add_argument('-lim', action="store", type=int)
    parser.add_argument('-th', action="store", type=float)
    parser.add_argument('-F', action="store", type=str)
    parser.add_argument('-plot', action="store", type=int)

    args = vars(parser.parse_args())
    print(args)

    LIM = args['lim'] #depth limit
    LOAD = True #whether to pass through parameters
    mode = args['mode']
    CTLGPy.PLOT = bool(args['plot'])
    #if REAL will run through all files in training_data
    thresh = args['th'] #when to stop search at minimum as fraction of total loss
    FOLDER = args['F']
    CTLGPy.REST = 10
    CTLGPy.PARA = False

    u_time = CTLGPy.the_time

    CTLGPy.MESS = True

    files = os.listdir(FOLDER)
    for name in files:
        print(name)
        Inputs = create_inputs(filename = name,m=args['mode'],folder=FOLDER)
        the_time = time.strftime("%m %d %Y, %H %M %S ") + str(name)
        print(the_time)
        operate()