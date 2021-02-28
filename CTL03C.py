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
    dataset = ''.join(re.findall('[A-Za-z]\w*',the_name))
    print("### Final Structure/loss:", data.name)
    if data.params is None:
        data.params_breakdown = None
        print("Cross-validation Loss, no parameters!")
    else:
        print("#### Raw Params:")
        display(data.params)
        print("#### Hierarchical Params:")
        print(data.params_breakdown)
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


def normalize(grammar,allkernels):
    """
    Normalizes each of the grammmar rule probabilities to equal to 1 for each choice point, prints rule probabilities
    """
    allkernels.append("+")
    for k in allkernels:
        beta = 0.0
        for key,value in grammar.items():
            if key.startswith(k):
                beta += value
        for key,value in grammar.items():
            if key.startswith(k):
                value = float(value)/beta
                grammar[key] = value
    return grammar

if __name__ == "__main__":

    """
    Creates the Grammar binary file and the report markdown file. Check in Summary!
    """
    from CTL03 import C
    #C = len(grammar.items())
    #C=1
    allfiles = []
    
    for name in os.listdir("Trees"):
        if '.pickle' in name:
            allfiles.append(name)
    
    for exclude in allfiles:
        files = []
        allsearch = []
        already = []
        for name in allfiles:
            if exclude is not name:
                files.append(name)

        print("Excluded:", exclude)
        print("Files:", files)
    
        grammar, allkernels = deepcopy(skim("Trees/"+files[0]))
        allkernels.append('+')
        for k in grammar.keys():
            grammar[k] = C/len(grammar.items()) #non optimization

        #print("## Uninformed Probabilities - Normalized")
        #normalize()

        for name in tqdm(files):
            print()
            print('## Data set: '+name)
            add(read("Trees/"+name))

        print("Files Read:", already)

        #NORMALIZE
        vals = [float(x) for x in grammar.values()]

        print()
        print("# GRAMMAR")
        print("## {} transitions after {} data sets".format(int((np.sum(vals) - len(vals))),len(allsearch)))
        print("## Counts")
        #normalize()
        

        print(grammar)
        
        with open('Grammars/{}_grammar.pickle'.format(exclude), 'wb') as f:
            pickle.dump(grammar, f, pickle.HIGHEST_PROTOCOL)