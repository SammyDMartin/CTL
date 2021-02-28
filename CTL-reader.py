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

def read(the_name):
    #MOVE TO CTL03
    with open(the_name, 'rb') as f:
        data = pickle.load(f)

    if type(data) is list:
        return data

plt.figure(figsize=(20,20))

def plot_progress(name,sequence):
    """
    Updates the Summary Model pickle file (model.pickle), progress plot (progress.png) and tree txt file (tree.txt)
    """
    plt.figure(figsize=(30,15))
    plt.style.use('fivethirtyeight')

    acceptance = []
    for i,_ in enumerate(sequence):
        try:
            if sequence[i].name is sequence[i-1].name:
                acceptance.append(0)
            else:
                acceptance.append(1)
        except Exception:
            pass
    
    acceptance.insert(0,0)

    acceptance = np.cumsum(np.array(acceptance)[1:])/np.arange(len(acceptance))[1:]

    losses = np.array([-1*s.loss for s in sequence])
    poss = np.array([s.pos for s in sequence])


    plt.subplot(211)
    plt.plot(np.arange(len(acceptance)),acceptance)
    plt.xlim(0,lim)
    plt.title("Acceptance Average")

    plt.subplot(212)
    plt.title("Fitted Model Stats")
    plt.plot(np.arange(len(acceptance)),losses ,label = 'Cross_validation MLLs')
    plt.plot(np.arange(len(acceptance)),poss ,label = 'pos estimates')
    plt.xlim(0,lim)
    plt.legend()
    name = "progress_"+str(name)
    plt.savefig("Summary/{}.png".format(name))
    plt.close()


def plot_acc(name,check,figure):
    sequence = read(name)[:a]
    correct = []
    strict_correct = []
    print("\n{}\n".format(name))
    for visit in sequence:
        if verbose:
            print(visit.name.ljust(50),str(round(visit.pos,4)).ljust(15),str(round(visit.loss,4)).ljust(15))
        if check in visit.name:
            correct.append(1)
        else:
            correct.append(0)

        if visit.name == check:
            strict_correct.append(1)
        else:
            strict_correct.append(0)


    #correct.insert(0,0)
    acceptance = np.squeeze(np.cumsum(np.array(correct))/np.arange(a))
    acceptance_s = np.squeeze(np.cumsum(np.array(strict_correct))/np.arange(a))

    upper = acceptance + np.std(acceptance[1:])*2
    lower = acceptance - np.std(acceptance[1:])*2

    upper_s = acceptance_s + np.std(acceptance_s[1:])*2
    lower_s = acceptance_s - np.std(acceptance_s[1:])*2

    figure[0].fill_between(np.arange(a),lower,upper,alpha=0.3)
    figure[1].fill_between(np.arange(a),lower_s,upper_s,alpha=0.3)

    figure[0].plot(np.arange(a),acceptance, label = "{}".format(name))
    figure[1].plot(np.arange(a),acceptance_s, label = "{}".format(name))

def hist_save(name):
    acc = {}
    seq = read(MH_seqs+"/"+name)
    for model in seq[:a]:
        if model.name not in acc:
            acc[model.name] = 1
        else:
            acc[model.name] += 1
    plt.figure(figsize=(12,6))

    print("\n\n"+name)
    
    data = sorted(acc.items(), key = lambda x : x[1], reverse=True)

    for k,v in data:
        for s in seq:
            if s.name is k:
                pos = s.pos
        print(k.ljust(30),str(v).ljust(5),str(round(pos,4)))


    plt.bar(np.arange(len(data)), [d[1] for d in data], color='g')

    visits = np.sum([d[1] for d in data])
    best = data[:5]
    frac = [(b[0],str(round((b[1]/visits),3))) for b in best]
    
    stats = []
    for n in [b[0] for b in best]:
        for itm in seq:
            if itm.name is n:
                stats.append((itm.name,str(round(itm.pos,2)),str(round(itm.loss,2))))
    plt.rcParams.update({'font.size': 5})
    plt.title("{} visits = {}, True = {}\n{}\n{}\n{}".format(name,visits,TRUE,best,frac,stats))

    plt.ylim((0,lim))
    if allC:
        for c in [b[0] for b in best[:2]]:
            CORRECT.append(c)
    
    if want_hists:
        plt.savefig("Summary/hist_{}_run_{}.png".format(name,str(a)))





################################################
TRUE = "PERIODIC"
CORRECT = [TRUE]
MH_seqs = "Summary"
a = []
want_hists = False
allC = False
verbose = False

f = []
files = os.listdir(MH_seqs)
for idx,name in enumerate(files):
    if 'model' in name:
        if 'pickle' in name[-8:]:
            f.append(name)
print(f)

lim = []
for name in tqdm(f):
    acc = {}
    data = read(MH_seqs+"/"+name)
    a.append(len(data))

a = min(a)

for name in tqdm(f):
    acc = {}
    data = read(MH_seqs+"/"+name)
    for model in data[:a]:
        if model.name not in acc:
            acc[model.name] = 1
        else:
            acc[model.name] += 1   
    data = sorted(acc.items(), key = lambda x : x[1], reverse=True)
    lim.append(data[0][1])

print(lim)
lim = max(lim)
print(lim)

for name in tqdm(f):
    hist_save(name)

"""
for name in tqdm(f):
    seq = read("Summary/{}".format(name))
    plot_progress(name,seq)


plt.figure()

set(CORRECT)
print(CORRECT)
for c in tqdm(CORRECT):
    fg,fig_traces = plt.subplots(2, 1, sharey=True,sharex=True,clear=True)
    fg.figsize=(12,6)    
    for name in f:
        nm = "Summary/{}".format(name)
        plot_acc(nm,c,fig_traces)

    tt = "Structure contains " + c
    ti = "Structure is " + c
    fig_traces[0].set_title(tt)
    fig_traces[1].set_title(ti)
    fig_traces[1].set_xlabel("Accepted Models")
    fig_traces[0].set_ylabel("Fraction Matching")
    fig_traces[1].set_ylabel("Fraction Matching")
    fig_traces[0].set_xlim(0,a)
    fig_traces[1].set_xlim(0,a)
    fig_traces[0].set_ylim(0,1)
    fig_traces[1].set_ylim(0,1)
    fig_traces[0].legend()
    fig_traces[1].legend()
    fg.savefig("Summary/comparator_{}_{}.png".format(c,str(a)))

plt.close()
"""

####averaging

def sum_traces(condition,allfiles,check,figure):
    sequences = []
    for name in allfiles:
        nm = MH_seqs+"/"+name
        if condition in nm:
            sequences.append(read(nm)[:a])
    
    strict_correct = correct = np.zeros(a)
    for sequence in sequences:

        correct0 = []
        strict_correct0 = []

        for visit in sequence:
            if verbose:
                print(visit.name.ljust(50),str(round(visit.pos,4)).ljust(15),str(round(visit.loss,4)).ljust(15))
            if check in visit.name:
                correct0.append(1)
            else:
                correct0.append(0)

            if visit.name == check:
                strict_correct0.append(1)
            else:
                strict_correct0.append(0)
        correct = correct + np.array(correct0)
        strict_correct = strict_correct + np.array(strict_correct0)


    correct = correct/len(sequences)
    strict_correct = strict_correct/len(sequences)

    #correct.insert(0,0)
    acceptance = np.squeeze(np.cumsum(correct)/np.arange(a))
    acceptance_s = np.squeeze(np.cumsum(strict_correct)/np.arange(a))

    upper = acceptance + np.std(acceptance[1:])*2
    lower = acceptance - np.std(acceptance[1:])*2

    upper_s = acceptance_s + np.std(acceptance_s[1:])*2
    lower_s = acceptance_s - np.std(acceptance_s[1:])*2

    #figure[0].fill_between(np.arange(a),lower,upper,alpha=0.3)
    figure.fill_between(np.arange(a),lower_s,upper_s,alpha=0.3)

    if "6ATrueFalse" in condition:
        l = "c = 1e-6, RBF+Periodic, RBF"
    elif "9ATrueFalse" in condition:
        l = "c = 1e-9, RBF+Periodic, RBF"    
    elif "6TrueFalse" in condition:
        l = "c = 1e-6, RBF"
    elif "9TrueFalse" in condition:
        l = "c = 1e-9, RBF"
    elif "TrueFalse" in condition:
        l = "c = 1e-9"
    else:
        l = "c = infinity"

    #figure[0].plot(np.arange(a),acceptance, label = "{}".format(l))
    figure.plot(np.arange(a),acceptance_s, label = "{}".format(l))


plt.rcParams.update({'font.size': 7})

def unified_progress(allfiles,condition):
    sequences = []
    for name in allfiles:
        nm = MH_seqs+"/"+name
        if condition in nm:
            sequences.append(read(nm)[:a])

    plt.figure(figsize=(30,15))
    plt.style.use('fivethirtyeight')
    acceptance = losses = poss = np.zeros(a)
    for sequence in sequences:
        acceptance0 = []
        for i,_ in enumerate(sequence):
            try:
                if sequence[i].name is sequence[i-1].name:
                    acceptance0.append(0)
                else:
                    acceptance0.append(1)
            except Exception:
                pass
        
        acceptance0.insert(0,0)

        acceptance0 = np.cumsum(np.array(acceptance0)[1:])/np.arange(len(acceptance0))[1:]

        losses0 = np.array([-1*s.loss for s in sequence])
        poss0 = np.array([s.pos for s in sequence])

        try:
            acceptance += acceptance0
            losses += losses0
            poss += poss0
        except Exception:
            acceptance = acceptance0
            losses = losses0
            poss = poss0

    acceptance = acceptance/len(acceptance)
    losses = losses/len(losses)
    poss = poss/len(poss)

    plt.subplot(211)
    plt.plot(np.arange(len(acceptance)),acceptance)
    plt.xlim(0,lim)
    plt.title("Acceptance Average")

    plt.subplot(212)
    plt.title("Fitted Model Stats")
    plt.plot(np.arange(len(acceptance)),losses ,label = 'Cross_validation MLLs')
    plt.plot(np.arange(len(acceptance)),poss ,label = 'pos estimates')
    plt.xlim(0,lim)
    plt.legend()
    name = "progress_"+str(name)
    plt.savefig(MH_seqs+"/U_{}.png".format(condition))
    plt.close()


for c in tqdm(CORRECT):
    fg,fig_traces = plt.subplots(1, 1, sharey=True,sharex=True,clear=True)
    fg.figsize=(20,40)    
    #for condition in ["9ATrueFalse","6ATrueFalse","9TrueFalse","6TrueFalse","FalseFalse"]:
    for condition in ["TrueFalse","FalseFalse"]:
        sum_traces(condition,f,c,fig_traces)

    tt = "Structure contains " + c
    ti = "Structure is " + c
    fig_traces.set_title(tt)
    #fig_traces[1].set_title(ti)
    #fig_traces[1].set_xlabel("Accepted Models")
    fig_traces.set_ylabel("Fraction Matching")
    #fig_traces[1].set_ylabel("Fraction Matching")
    fig_traces.set_xlim(0,a)
    #fig_traces[1].set_xlim(0,a)
    fig_traces.set_ylim(0,1)
    #fig_traces[1].set_ylim(0,1)
    fig_traces.legend(loc=4)
    #fig_traces[1].legend()
    fg.savefig(MH_seqs+"/U_comp_{}_{}.png".format(c,str(a)))

#for condition in tqdm(["TrueFalse","TrueTrue","FalseFalse"]):
#    unified_progress(f,condition)

