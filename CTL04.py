#scl enable rh-python36 bash
from anytree import NodeMixin, RenderTree
import anytree
import math
import numpy as np
import pickle
import time
import KernelGen
from IPython.display import display
import CTL03
import CTL03C
from CTL02 import create_inputs

import time
import os

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import sys
from tqdm import tqdm
from copy import deepcopy

import multiprocessing
import CTLGPy

CTLGPy.PLOT = False
CTLGPy.REST = 0
CTLGPy.splits,CTLGPy.repeats = 3,1
the_time = CTLGPy.the_time
CTLGPy.mkdir = False
CTLGPy.PARA = False
CTLGPy.MESS = False

mode = "cross_validation_loss" #Must be this to work correctly
NJNEW = True #whether to append to visit sequence only when its new

def render_visits(root):
    #print(str("").ljust(90),str("-mll").ljust(20), str("best").ljust(8))
    for pre, _, node in RenderTree(root):
        onpath = lambda node : bool(sum([n.visits for n in node.descendants]))
        if node.name is not 'root':
            if node.visits is not 0:
                treestr = u"%s%s" % (pre, node.name)
                loss_string = str(round(node.pos,3))
                visits = str(node.visits)
                accepts = str(node.accepts)
                print(treestr.ljust(90), loss_string.ljust(10), visits.ljust(5), accepts.ljust(5))
            elif onpath(node):
                treestr = u"%s%s" % (pre, node.name)
                print(treestr.ljust(90))
        else:
            treestr = u"%s%s" % (pre, node.name)
            print(treestr)

def recursive_len(item):
    "helper function for searching kernel structures"
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1

class MarkovKernel():
    """
    Goes in MarkovKernelNode - more or less a wrapper around model scoring for gpy
    By default, the model is not fitted. Call activate to fit
    """
    def __init__(self,Kernels):
        self.loss = None
        self.allkernels = ['RBF','PERIODIC','LIN']
        self.visits = 0
        self.accepts = 0
        if Kernels:
            self.structure = Kernels
        else:
            self.structure = []
        self.Kernels = deepcopy(self.structure)
    
    def describe_kernel(self, kernlist):
        """
        Mirrors extract_kernel - makes sure that the correct pattern of Kernel is generated as expected.
        It is also used to parse the kernel, returns kernel str description
        """
        if not kernlist:
            return ""
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

    def activate(self,Inputs,Cache = None):
        """
        Triggers the model fitting if the model has not already been fitted. Sets the loss if it has been fitted.
        To be called when the metropolis-hastings search hits the model.
        """
        if self.visits == 0:
            if Cache is not None:
                name = self.describe_kernel(self.structure)
                if name in Cache:
                    if Cache[name] is not None:
                        self.structure = self.Kernels
                        self.loss = Cache[name]
                        self.params = None
                        self.visits += 1
    
                        print("+++ Loaded from Cache +++")
                        print(name.ljust(30),str(self.loss))
                        print("+++++++++++++++++++++++++")
                        return
    
            model = CTLGPy.CompositionalGPyModel(Inputs,None,self.Kernels,True)
            self.structure, self.loss, self.params = model.fit()
            if model.figure:
                self.plot = model.figure
            self.visits += 1
        else:
            self.visits += 1
    def reactivate(self,Inputs,R_P):
        if self.visits > 0:
            if np.random.rand() < R_P:
                rf_model = CTLGPy.CompositionalGPyModel(Inputs,None,self.Kernels,True)
                _, newloss, _ = rf_model.fit()
                if newloss < self.loss:
                    self.loss = newloss
                    print("Loss Updated")


class MarkovKernelNode(MarkovKernel, NodeMixin):  # Add Node feature
    """
    An anytree object that serves as a node containing all parameters and loss function values for a kernel search
    """
    def __init__(self,Kernels=None,parent=None, children=None,Params=None):
        super(MarkovKernelNode, self).__init__(Kernels)
        if Kernels:
            self.name = self.describe_kernel(self.structure)
            self.parent = parent
            self.best = False
            #self.pos = 0
        if children:
            self.children = children
        if Kernels == None:
            #is a root node
            self.name = "root"
            self.mode = "cross_validation_loss"
            self.loss = np.inf

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

    def prior(self,grammar,use_counts,nt=False):
        """
        Generates the prior probability of a given kernel, either informed or uninformed.
        If informed, the prior is based on the counts and if uninformed the prior is uniform.
        Args:
            nt: boolean. If nt is true the prior does not include the model terminating as is
            use nt=True to calculate the conditional probability of all child models, or of terminating on this model
            use nt=False to calculate the prior probability of just this model.
        """
        if use_counts is True:
            transition_probabilities = grammar
        elif use_counts is False:
            transition_probabilities = {}
            for key in grammar.keys():
                if key[0] is '+':
                    transition_probabilities[key] = 1/float(len(self.allkernels))
                else:
                    transition_probabilities[key] = 1/float(1+len(self.allkernels)*2)

        counts = CTL03.generate_transitions(self.describe_kernel(self.structure), grammar,self.allkernels)
        prior = 1.0
        if nt is True:
            trans = [k for k in counts.keys() if '-' not in k]
        elif nt is False:
            trans = counts.keys()

        for k in trans:
            prior = prior * (transition_probabilities[k] ** counts[k])
        return prior

    def obtain_posterior(self,g, informed):
        "Sets posterior from prior and loss"
        #self.validation_loss = -1*np.sum(self.model.log_predictive_density(self.validation_x,self.validation_y))

        #If the model is informed, the posterior is given by the counts prior
        self.pos = np.log(self.prior(g,use_counts = informed)) -1*self.loss

    def children_from_parent(self):
        """
        produces actual child nodes from list of child structures.
        """
        for child_structure in self.list_children():
            MarkovKernelNode(child_structure,parent=self)

    def initialize(self,depth,overallkernels=False):
        """
        Without activating any kernels, produces entire tree up to any depth.
        Must be run before starting search
        """
        if overallkernels is True:
            self.allkernels.append("RQ")
        self.children_from_parent()
        for child in self.children:
            child.allkernels = self.allkernels
            if recursive_len(child.structure) < depth:
                child.initialize(depth-1)

    def greedy_search_cache(self,depth):
        """
        Performs breadth first search given current data and CTLGPy parameters.
        Might want to run to fill out the tree map.
        """
        for child in self.children:
            child.activate()
            if recursive_len(child.structure) < depth:
                child.greedy_search(depth-1)

def step_back(node):
    """
    Steps back along the current kernel structure a uniform random number of times.
    """
    if not node.parent:
        return node
    descend = node.ancestors
    n = len(descend)
    if n == 1:
        return node.parent
    steps = np.random.randint(1,n)
    return descend[n-steps]

def step_forward(g,node,counts_policy):
    """
    Steps forward according to a stochastic policy that is either counts-based or uniform random,
    depending on the value of global bool counts_policy
    """
    while True:
        options = list(node.children)

        if node.children:
            trans_probs = [child.prior(g,counts_policy, nt=True) for child in options] #The prior probability (at start) of each child being reached (not of stopping there)

            if node.parent:
                options.append(node)
                trans_probs.append(node.prior(g,counts_policy, nt=False)) #if not at root, there is a chance of terminating at node. prior probability (at start) of that

            trans_probs = np.array(trans_probs)/node.prior(g,counts_policy, nt=True) #everything is conditional on reaching the current node (nonterminal probability of getting where we already are)

            #if DIAG:
            #    print("\nTrans Probs:")
            #    print("{} -> \n{}{}\n{}".format(node.name,[node.name for node in options],[round(x,3) for x in trans_probs],np.sum(trans_probs)))
            #    print("\n{}'s children priors: {}\n".format(node.name,[round(n.prior(use_counts = counts_policy),3) for n in options]))

            new_node = np.random.choice(options,p=trans_probs)

            if new_node is node:
                #terminal
                return node
            else:
                node = new_node
        else:
            print("Hit limit!\n\n")
            return node


def random_walk(g):
    "Runs the same as the actual Metropolis Hastings but doesn't fit any models"
    root=MarkovKernelNode()
    root.initialize(shot_range)
    sequence = []
    acceptance = []

    while True:
        if sequence == []:
            s = root
        else:
            s = sequence[-1]

        s_ = step_forward(g,step_back(s),counts_policy=True)
        while s_ is s:
            s_ = step_forward(g,step_back(s),counts_policy=True) #This is the model to be evaluated

        A = np.random.rand()

        if A > np.random.rand():
            print("Accepted!")
            s_.accepts += 1
            print(s_.name)

            sequence.append(s_)
            acceptance.append(1)
        else:
            print("Rejected!")
            acceptance.append(0)

        time.sleep(0.2)

def accept(s_,s,counts_policy,g):
    """
    Calculates the metropolis-hastings acceptance probability from posterior (which may either be informed or uninformed) and counts prior
    calculation from https://web.mit.edu/cocosci/Papers/RRfinal3.pdf#page=26
    """
    transitions = lambda s: recursive_len(s.structure) + 1

    #To handle the case where the policy takes you to the edge of the explored tree
    NT_s, NT_s_ = False,False
    if not s_.children:
        NT_s_ = True
    
    if not s.children:
        NT_s = True

    return (s_.pos - s.pos) + \
    (np.log(transitions(s)*s.prior(g,use_counts=counts_policy,nt = NT_s)) - np.log(transitions(s_)*s_.prior(g,use_counts=counts_policy,nt = NT_s_)))

def save_progress(sequence,counts_policy,informed,ts):
    name = "model" + str(informed)+str(counts_policy)+str(ts)
    with open('Summary/{}.pickle'.format(name), 'wb') as f:
        pickle.dump(sequence, f, pickle.HIGHEST_PROTOCOL)

def plot_model(model):
    fig = model.plot[1]
    savestr = model.plot[0]

    stats = "pos = {}, CV_l = {}".format(str(round(model.pos,4)), str(round(-1*model.loss)))
    print(stats)

    out = savestr.split("]")
    out.insert(1,stats)
    savestr = "]".join(out)

    fig.savefig(savestr)

def load_caches(ts):
    cache = {}
    caches = []

    files = os.listdir("Trees/Caches/")
    for filename in files:
        if "cache" in filename:
            if ts in filename:
                with open("Trees/Caches/"+filename, 'rb') as f:
                    data = pickle.load(f)            
                caches.append(data)
    
    if caches is []:
        return {}
    
    for c in caches:
        for (kern,loss) in c.items():
            if kern in cache:
                if loss < cache[kern]:
                    cache[kern] = loss
            else:
                cache[kern] = loss

    print("Cache Up!",ts,len(cache.items()))
    return cache

def save_cache(cache,rootnode,ts):
    for kernel in rootnode.descendants:
        if kernel.loss is not None:
            if kernel.name in cache:
                if kernel.loss < cache[kernel.name]:
                    cache[kernel.name] = kernel.loss
            cache[str(kernel.name)] = float(kernel.loss)
        
    cname = "Trees/Caches/cache_{}_{}".format(ts,str(len(cache.items())))
    with open(cname, 'wb') as f:
        pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)
            

def metropolis_hastings(informed,employ_policy,shot_range,inst,grammar, R_P,N,Inputs,the_time,ts):
    """
    Actually performs the metropolis_hastings search starting at root.
    """
    if informed and employ_policy:
        counts_policy = True
    else:
        counts_policy = False

    interval = 20
    
    cache = load_caches(ts)

    root=MarkovKernelNode()
    root.initialize(shot_range)
    root.pos = -1*np.inf #log probability is - infinity
    sequence = []
    acceptance = []

    pbar = tqdm(total=N)

    np.random.seed()
    
    for _ in range(N):
        if sequence == []:
            s = root
        else:
            s = sequence[-1]

        s_ = step_forward(grammar,step_back(s),counts_policy)
        #while s_ is s:
        #    s_ = step_forward(grammar,step_back(s),counts_policy) #Still not sure about this

        s_.activate(Inputs,cache)

        s.reactivate(Inputs,R_P), s_.reactivate(Inputs,R_P)
        s.obtain_posterior(grammar,informed), s_.obtain_posterior(grammar,informed)
        
        if s_ is s:
            A = 1.00
        else:
            A = np.exp(accept(s_,s,counts_policy,grammar))

        if A > np.random.rand():
            print("Accepted!")
            s_.accepts += 1
            try:
                plot_model(s_)
            except Exception:
                s.plot = None
            sequence.append(s_)
            acceptance.append(1)
        else:
            if NJNEW:
                sequence.append(s)
            print("Rejected!")
            acceptance.append(0)

        ds = " {}: Accepted: {} Total: {}".format(str(inst),str(sum(acceptance)),str(len(acceptance)))
        
        print("\n\n")
        pbar.set_description(ds)
        pbar.update(1)
        print(ts,inst,multiprocessing.current_process())
        print("\n"+ds+"\n\n")
        
        if len(acceptance) % interval is 0:
            save_cache(cache,root,ts)
            save_progress(sequence,counts_policy,informed,ts)
            name = "tree"+str(informed)+str(counts_policy)+str(ts)
            sys.stdout = open('Summary/{}.txt'.format(name), 'w')
            print("\n Grammar \n\n")

            for k,v in grammar.items():
                print(k.ljust(20), v)

            print("\nAccepted: {} Total: {}\n".format(str(len(sequence)),str(len(acceptance))))
            print("Sequence:\n")
            print("Name".ljust(70),"log_Posterior".ljust(15),"log_Loss".ljust(15),
                "log Prior (final)".ljust(15))
            for visit in sequence:
                print(visit.name.ljust(70),str(round(visit.pos,4)).ljust(15),str(round(visit.loss,4)).ljust(15),
                    str(round(np.log(visit.prior(grammar,use_counts = informed)),4)).ljust(15))
            print("\n\n Tree: \n\n")
            try:
                render_visits(root)
            except Exception:
                pass
            sys.stdout = sys.__stdout__

    save_cache(cache,root,ts)

def create_grammar(G):
    tsst = MarkovKernel([])
    if informed is False:
        G_uninformed = {}
        for key in G.keys():
            if key[0] is '+':
                G_uninformed[key] = 1/float(len(tsst.allkernels))
            else:
                G_uninformed[key] = 1/float(1+len(tsst.allkernels)*2)
            grammar = G_uninformed
    else:
        grammar = G

    for k,v in grammar.items():
        print(k.ljust(20), v)
    return grammar

if __name__ == "__main__":
    jobs = []
    #pool = multiprocessing.Pool(processes=32)

    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('-g', action="store", type = str)
    parser.add_argument('-F', action="store", type = str)

    args = vars(parser.parse_args())

    mode = "cross_validation_loss"

    F = args['F']

    afiles = os.listdir(F)
    files = []

    grammars = os.listdir(args['g'])

    for f in afiles:
        for g in grammars:
            if f in g:
                files.append(f)

    print(files)
    print(grammars)

    for where,ts in enumerate(files):
        if '.mat' in ts:
            print(ts)
            ts = ts.split('.')[0]

        for gn in grammars:
            if ts in gn:
                with open(args['g']+"/{}".format(gn), 'rb') as f:
                    G0 = pickle.load(f)
                    G0 = CTL03C.normalize(G0,['RBF','PERIODIC','LIN'])
                break
        
        print("++ MATCHED COUNTS TO TIMESERIES ++")
        print(ts,gn)

        N = 10000 #number of iters
        LIM = 5
        Inputs = create_inputs(filename = ts,folder = F,m=mode,cut=0.9)
        #DIAG = False #print diagnostics
        
        R_P = 0.01 #repeats probability
        
        
        informed, employ_policy, shot_range = False,False,LIM
        grammar = create_grammar(G0)

        argslist = [informed,employ_policy,shot_range,(2*where+1),grammar, R_P,N,Inputs,the_time,ts]

        #pool.apply_async(metropolis_hastings, args=argslist)

        p1 = multiprocessing.Process(target=metropolis_hastings,args=argslist)
        jobs.append(p1)
        p1.start()
        
        #informed, employ_policy, shot_range,inst = True,False,5,2
        #grammar = create_grammar(G0)
        #metropolis_hastings()
        
        
        informed, employ_policy, shot_range = True,False,LIM
        grammar = create_grammar(G0)

        argslist2 = [informed,employ_policy,shot_range,(2*where+2),grammar, R_P,N,Inputs,the_time,ts]
        
        p2 = multiprocessing.Process(target=metropolis_hastings,args = argslist2)
        jobs.append(p2)
        p2.start()
    
    for i,process in enumerate(jobs):
        print("++++ Initiation Complete ++++")
        print(i)
        process.join()

        #pool.apply_async(metropolis_hastings, args=argslist2)
