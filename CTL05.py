from CTL02 import create_inputs
from CTL04 import MarkovKernelNode
from CTL02 import ScoredKernelNode
import time
import CTL02
CTL02.u_time = CTL02.the_time = time.strftime("%m %d %Y, %H %M %S")
import anytree
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import os
import pickle

def read(the_name):
    with open(the_name, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_MH(folder):
    a = []
    MH_sequences = {}

    files = os.listdir(folder)
    for name in tqdm(files):
        if 'model' in name:
            if 'pickle' in name[-8:]:
                data = read(folder+'/'+name)
                a.append(len(data))
    a = min(a)
    for name in tqdm(files):
        if 'model' in name:
            if 'pickle' in name[-8:]:
                data = read(folder+'/'+name)
                MH_sequences[name] = data[:a]
    return MH_sequences

def extract_best(MH):
    acc = {}
    for model in MH:
        if model.name not in acc:
            acc[model.name] = 1
        else:
            acc[model.name] += 1   
    best = sorted(acc.items(), key = lambda x : x[1], reverse=True)
    
    print(str(best[:4]))
    best = best[0]
    return best

def baseline(Greedy_seqs):
    files = os.listdir(Greedy_seqs)
    print(files)
    bests = {}
    for name in tqdm(files):
        if '.pickle' in name:
            best = read(Greedy_seqs+"/"+name)

            n = best.name
            m = best.mode
            a = n.split(" ")
            n = (" ".join(a[:-1]))
            bests[name] = (n,0.0)

    return bests

#####stats

MH_seqs = r"CTL04 Full Sweeps\New RBF - Fail\Summary_Adv\RBF"
test_data = r"training_data_comp\Test"
Greedy_seqs = r"Trees_comp"

MH = extract_MH(MH_seqs)

for k,v in MH.items():
    MH[k] = extract_best(v)

B = baseline(Greedy_seqs)

all_models = {**B, **MH}

init = MarkovKernelNode()
init.initialize(10,overallkernels=True)
print("Tree Up")

"""
for pre, _, node in anytree.RenderTree(init):
    treestr = u"%s%s" % (pre, node.name)
    print(treestr.ljust(90))
"""
def pattern_from_desc(kernelstr):
    correct = anytree.find_by_attr(init, kernelstr)
    return correct.Kernels

def fit_model(kern,filename,name):
    CTL02.mode = mode
    in_dict = create_inputs(filename,m=mode,Train_f=TF,Val_f=None,folder=test_data)
    N_test = in_dict['N_test']
    CTL02.Inputs = in_dict
    kernel = CTL02.ScoredKernelNode(kern)
    kernel.counts = {}
    kernel.best = True
    kernel.update_at_end()
    print("\n ABOVE IS: {} \n\n\n\n".format(name))
    sys.stdout = sys.__stdout__
    if hasattr(kernel,"plot"):
        fig = kernel.plot[1]

        savestr = kernel.plot[0]
        s = savestr.split("/")
        s.insert(1, "/"+name)

        fig.savefig(''.join(s))
    return float(kernel.loss)/N_test


CTL02.LOAD, CTL02.Params = False, False
CTL02.LIM = 10
CTL02.CTLGPy.REST = 0
CTL02.thresh = 0.0
CTL02.FOLDER = test_data
CTL02.CTLGPy.PLOT = True
CTL02.CTLGPy.MESS = True

mode = 'validation_loss'
TF = (1/1.20)
tests = os.listdir(test_data)

Informed = []
Uninformed = []
Original = []

deltas = []
deltas_init = []

for t in tqdm(tests):
    try:
        delta_ts = {}
        for best_model in all_models.keys():
            if t in best_model:
                print(best_model,t)
                print(all_models[best_model])
                loss = fit_model(pattern_from_desc(all_models[best_model][0]),t, str(best_model))
                if "True" in best_model:
                    Informed.append(loss)
                    delta_ts["True"]=loss
                elif "model" in best_model:
                    Uninformed.append(loss)
                    delta_ts["False"]=loss
                else:
                    Original.append(loss)
                    delta_ts["Init"] = loss
        deltas.append(delta_ts["True"]-delta_ts["False"])
        deltas_init.append(delta_ts["True"]-delta_ts["Init"])
    except Exception:
        pass
    finally:
        print(delta_ts)


print(Informed)
print(Uninformed)
import numpy as np
print("Informed")
print(np.mean(Informed),np.std(Informed))
print("Uninformed")
print(np.mean(Uninformed),np.std(Uninformed))
print("Original")
print(np.mean(Original),np.std(Original))

print("Deltas")
print(deltas)
print(np.mean(deltas),np.std(deltas))

print("Deltas from original")
print(deltas_init)
print(np.mean(deltas_init),np.std(deltas_init))
