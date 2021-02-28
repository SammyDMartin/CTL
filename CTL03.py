from CTL02 import ScoredKernelNode
#import CTL02
import os
from copy import deepcopy
import pickle
import numpy as np
from IPython.display import display
import sys
import re
from tqdm import tqdm

def skim(the_name):
    #MOVE TO CTL03
    "To extract whatever the current grammar and kernel usage list is"
    with open(the_name, 'rb') as f:
        data = pickle.load(f)
    data.update_at_end(noprint=True)
    return data.counts, data.allkernels

def read(the_name):
    """
    Reads the tree files and returns the counts for each tree file
    """
    print(the_name)
    with open(the_name, 'rb') as f:
        data = pickle.load(f)
    data.update_at_end(noprint=False)
    #dataset = ''.join(re.findall('[A-Za-z]\w*',the_name))
    dataset = the_name
    print("### Final Structure/loss:", data.name)
    if data.params is None:
        data.params_breakdown = None
        print("Cross-validation Loss, no parameters!")
    else:
        print("#### Raw Params:")
        display(data.params)
        #print("#### Hierarchical Params:")
        #print(data.params_breakdown.getvalue())
    allsearch.append(data)
    if data.mode != "cross_validation_loss":
        print("\nNot CV Loss!\n")
        #return None
    #already = [x.name if x.mode is "cross_validation_loss" else '' for x in allsearch]
    #if dataset in already:
    #    print("\nCannot reuse data counts!\n")
    #    return None
    
    already.append(dataset)
    print("\nCounts returned\n")
    return data.counts


def add(counts):
    """
    Adds the counts from each tree file to the overall grammar
    """
    if counts:
        for k in grammar.keys():
            grammar[k] = grammar[k] + counts[k]


def normalize():
    """
    Normalizes each of the grammmar rule probabilities to equal to 1 for each choice point, prints rule probabilities
    """

    for k in allkernels:
        beta = 0.0
        for key,value in grammar.items():
            if key.startswith(k):
                beta += value
        print()
        for key,value in grammar.items():
            if key.startswith(k):
                value = float(value)/beta
                grammar[key] = value
    for key,value in grammar.items():
        print()
        print(key.ljust(30),str(value))

def generate_transitions(structure_string, grammar,allkernels):
    """
    Used in CTL03 and CTL04 to generate the transition counts from a structure string, relying on the fact that the grammar
    has no ambiguity. Returns dictionary of transition counts
    """
    started,finished = False,False

    counts = deepcopy(grammar)
    for key in counts.keys():
        counts[key] = 0

    #find the finishing kernel and update its counts
    for idx,_ in enumerate(structure_string):
        partial = structure_string[-idx-1:]
        for start in allkernels:
            if start in partial:
                k = str(start)+'-'
                counts[k] = counts[k] + 1
                finished = True
        if finished == True:
            break

    #find the starting kernel and update its counts
    for idx,_ in enumerate(structure_string):
        partial = structure_string[:idx+1]
        for start in allkernels:
            if start in partial:
                k = '+'+str(start)
                counts[k] = counts[k] + 1
                started = True
        if started == True:
            break

    #count all the non-starting transitions til the terminal state
    for key in counts.keys():
        if key[1:] not in allkernels and key[:-1] not in allkernels:
            #non-starting
            counts[key] = counts[key] + sum(1 for i in range(len(structure_string)) if structure_string.startswith(key, i))
    
    return {k:v for k,v in counts.items() if v!= 0.0}

C = 1e-9

if __name__ == "__main__":
    Trees = "Trees"
    """
    Creates the Grammar binary file and the report markdown file. Check in Summary!
    """
    sys.stdout = open('Summary/out.md', 'w')
    #C = len(grammar.items())
    #C=1

    allsearch = []
    already = []
    afiles = os.listdir(Trees)
    files = []
    for name in afiles:
        if '.pickle' in name:
            files.append(name)

    
    grammar, allkernels = deepcopy(skim(Trees+"/"+files[0]))
    allkernels.append('+')
    for k in grammar.keys():
        grammar[k] = C/len(grammar.keys())

    #print("## Uninformed Probabilities - Normalized")
    #normalize()

    for name in tqdm(files):
        print()
        print('## Data set: '+name)
        add(read(Trees+"/"+name))

    print("Files Read:", already)
    
    print("Unnormalized")
    allcounts = 0
    for key,value in grammar.items():
        print()
        print(key.ljust(30),str(value- C/len(grammar.keys())))
        allcounts += value- C/len(grammar.keys())

    #NORMALIZE
    vals = [float(x) for x in grammar.values()]

    print()
    print("# GRAMMAR")
    print("## {} transitions after {} data sets".format(allcounts,len(allsearch)))
    print("## Probabilities - Normalized with C-factor (no leaveout) {}".format(C))
    normalize()

    for tree in allsearch:
        #print()
        #print("### "+ tree.name)
        #print()
        #print(tree.mode)
        #print()
        counts = generate_transitions(tree.name, grammar,allkernels)
        #for k,v in counts.items():
        #    print(k.ljust(30),v)
        prior = 1.0
        for k in counts.keys():
            prior = prior * (grammar[k] * counts[k])
        prior = round(prior,4)
        #print("prior = {}".format(str(prior)))
        loss = float((re.findall("\d+\.\d+", tree.name))[0])
        #print()
        #print("#### log(prior) + mll: {} + {} = {}".format(round(np.log(prior),4), -1*loss, round(np.log(prior)-1*loss,4)))

    with open('Summary/grammar.pickle', 'wb') as f:
        pickle.dump(grammar, f, pickle.HIGHEST_PROTOCOL)