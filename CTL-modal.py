import CTL03
import pickle
from tqdm import tqdm
from CTL02 import ScoredKernelNode
from CTL04 import MarkovKernelNode
from CTL04 import plot_model
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys

#actual = ['RBF','RBF + PERIODIC']

def actual(name):
    if "RBF_PER(" in name:
        return "RBF + PERIODIC"
    elif "RBF(" in name:
        return "RBF"
    elif "Linear" in name:
        return "LIN"
    else:
        return ""


Greedy_seqs = 'Trees'
Grammar_files = 'Grammars'
MH_seqs = 'Summary'

actual = 'PERIODIC'
#Greedy_seqs = r'CTL04 Experiments/RBF + RBF Recovery/RBFRBFRecoveryLong/Trees_RBFRBF'
#Grammar_files = r'CTL04 Experiments/RBF + RBF Recovery/RBFRBFRecoveryLong/Grammars'
#MH_seqs = r'CTL04 Experiments/RBF + RBF Recovery/RBFRBFRecoveryLong'


allkernels = ['RBF','PERIODIC,','LIN']

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

    
def return_MAP(seq,name):
    acc = {}
    for model in seq:
        if model.name not in acc:
            acc[model.name] = model.pos   
    best = sorted(acc.items(), key = lambda x : x[1], reverse=True)
    
    best = best[0]
    try:
        there = acc[actual(name).strip()]/len(seq)
    except Exception:
        there = 0.0

    return best[0],(best[1]/len(seq)),there


def return_MAP_by_mode(seq,name):
    acc = {}
    for model in seq:
        if model.name not in acc:
            acc[model.name] = 1
        else:
            acc[model.name] += 1   
    best = sorted(acc.items(), key = lambda x : x[1], reverse=True)
    
    print(str(best[:4]))

    best = best[0]
    try:
        there = acc[actual(name).strip()]/len(seq)
    except Exception:
        there = 0.0

    return best[0],(best[1]/len(seq)),there,len(seq)

def bracket_number(name):
    if "True" in name:
        try:
            num = name.split("(")[1]
            num = int(num[0])
        except IndexError as I:
            print("Can't find",I)
            return np.NaN
    else:
        return np.NaN
    return num


def accumulate(grammar_file,new_model):
    name = new_model[0]
    grammar = read(Grammar_files+"/"+str(grammar_file))
    newcounts = CTL03.generate_transitions(name,grammar,allkernels)
    for k in newcounts.keys():
        grammar[k] = grammar[k] + newcounts[k]
    return grammar

def baseline():
    files = os.listdir(Greedy_seqs)
    print(files)
    bests = []
    for name in tqdm(files):
        if '.pickle' in name:
            best = read(Greedy_seqs+"/"+name)

            n = best.name
            m = best.mode
            n = n.split(" ")
            bests.append((name," ".join(n[:-1]),(n[-1]),m))
    
    return bests


MH = extract_MH(MH_seqs)
Trues, Falses = [],[]
print(MH.keys())

def print_results(emperical_mode):
    bests = {}
    for (name,data) in MH.items():
        if emperical_mode is True:
            global lens
            MAP, freq,lin_count,lens = return_MAP_by_mode(data,name)
        else:
            MAP, freq,lin_count = return_MAP(data,name)
        bests[name]=(MAP,freq,lin_count)

    N = np.array([0,0])
    U = 0
    print("\nFile".ljust(60),"ABCD Fit".ljust(40),"loss".ljust(15))
    
    try:
        for (name,model,loss,m) in baseline():
            print(name.ljust(60),model.ljust(40),loss.ljust(15))
            if str(model).strip() == actual(name):
                N[0] += 1
        print("(mode was {})".format(m))
    except Exception as E:
        print(E)

    
    if emperical_mode is True:
        print("\nFile".ljust(40),"Actual".ljust(20),"MH MAP (emperical)".ljust(40),"frequency".ljust(15),"True frequency".ljust(15))
    else:
        print("\nFile".ljust(40),"Actual".ljust(20),"MH MAP".ljust(40),"log(MAP)".ljust(15),"True = {} log(MAP)".ljust(15))
    
    for k,v in bests.items():
        print(k.ljust(40),actual(k).ljust(20),v[0].ljust(40),str(round(v[1],3)).ljust(15),str(round(v[2],3)).ljust(15))
        if emperical_mode is True:
            if "True" in k:
                Trues.append(float(v[2]))
            else:
                Falses.append(float(v[2]))

        if str(v[0]).strip() == actual(k):
            if "True" in k:
                N[1] += 1
            else:
                U += 1
    
    print("Sequence at: "+str(lens))
    print("Counts: ")
    anT = len(Trues)
    anF = len(Falses)
    
    print(N,U,int(anT+anF))
    try:
        print("\n\n +++ FINAL RESULTS: +++\nRound 0:   {}\nRound 1:   {}\nUninformed: {}\n".format(N[0]/anT,N[1]/anT,U/anF))
    except Exception as E:
        print(E)

    if emperical_mode is True:
        try:
            print("Informed mean correct: "+str(np.mean(Trues))+"+/-"+str(np.std(Trues)))
        except Exception:
            pass
        try:
            print("Uninformed mean correct: "+str(np.mean(Falses))+"+/-"+str(np.std(Falses)))
        except Exception:
            pass

print(Greedy_seqs, Grammar_files, MH_seqs)
print()
print_results(True)
print_results(False)


sys.stdout = open(MH_seqs+'/results.md', 'w')
print(Greedy_seqs, Grammar_files, MH_seqs)
print()
print_results(True)
print_results(False)

"""
input("Update Grammar?")
Gs = "Grammars"
files = os.listdir(Gs)

for f in files:
    n = bracket_number(f)
    for k in bests.keys():
        if bracket_number(k) is n:
            print("Matched Countsfile to Timeseries?",f.value(),k)
            new_grammar = accumulate(f.value(),bests[k])
            with open('Grammars/2_{}'.format(f), 'wb') as f:
                pickle.dump(new_grammar, f, pickle.HIGHEST_PROTOCOL)
"""
