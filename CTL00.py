from CTL02 import create_inputs
from CTL04 import MarkovKernelNode
import time
import CTL02
CTL02.u_time = CTL02.the_time = time.strftime("%m %d %Y, %H %M %S")
import anytree
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
print("Imported")
"""
#Duvenaud 2013
("RBF(LIN + LIN( PERIODIC + RQ))",'01-airline-res'),
("(RBF + RBF)PERIODIC",'02-solar-res'),
("(LIN(RBF + RBF)(PERIODIC + RQ))","03-mauna-res"),

"""

Autostat = [("((PERIODIC + RQ)LIN + LIN)RBF",'01-airline-res'),
    ("(RBF + RBF)PERIODIC",'02-solar-res')
    #("(((RBF + RBF)LIN)PERIODIC) + (((RBF + RBF)LIN)RQ)","03-mauna-res"),
    ]
    #Above is validation loss new run. Valid but didn't get solar data.

#BIC Full Run 08 05 2019, 00 57 26
BIC_100 = [
    ("(RBF + PERIODIC)PERIODIC + RBF + PERIODIC",'01-airline-res'),
    ("RBF + RBF",'02-solar-res'),
    ("(RBF + PERIODIC + RBF)LIN","03-mauna-res")]
    #Above is 100% BIC score, so its training on test set. big no for not BIC

#(80%) Original Data Run 08 04 2019, 20 22 41
BIC_80 = [("(LIN + PERIODIC)RBF + LIN",'01-airline-res'),
    ("RBF + RBF",'02-solar-res'),
    ("RBF + PERIODIC + RBF + LIN","03-mauna-res")]
    #Above is 80% BIC score, so limits are good

#VAL Loss run w. Linear 08 05 2019, 23 59 17
Val = [("(PERIODIC + PERIODIC + LIN)RBF",'01-airline-res'),
    ("PERIODIC + PERIODIC",'02-solar-res'),
    ("RBF","03-mauna-res")]
    #Above is 80% BIC score, so limits are good

#new val run
Val = [("((RBF + RBF + PERIODIC)RBF)LIN",'01-airline-res'),
    ("PERIODIC + PERIODIC",'02-solar-res'),
    ("RBF","03-mauna-res")]

#Extremely Long CV Run 08 05 2019, 10 34 21
CV_Long = [("((RBF + RBF + PERIODIC)RBF)RBF",'01-airline-res'),
    ("RBF",'02-solar-res'),
    ("(PERIODIC + RBF + PERIODIC)RBF + PERIODIC","03-mauna-res")]


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
    CTL02.Inputs = create_inputs(filename,m=mode)
    kernel = CTL02.ScoredKernelNode(kern)
    kernel.counts = {}
    kernel.best = True
    kernel.update_at_end()
    print("\n ABOVE IS: {} \n\n\n\n".format(name))
    sys.stdout = sys.__stdout__
    if hasattr(kernel,"plot"):
        fig = kernel.plot[1]
        savestr = kernel.plot[0]
        fig.savefig(savestr)
    pbar.update(1)


CTL02.LOAD, CTL02.Params = False, False
CTL02.LIM = 15
CTL02.thresh = 0.0
CTL02.CTLGPy.REST = 10
CTL02.CTLGPy.PLOT = True
pbar = tqdm(total = 15,desc="Models")

mode = 'validation_loss'

#fit_model(pattern_from_desc("(LIN)RBF + PERIODIC"),"03-mauna-res", "Mauna")

#fit_model(pattern_from_desc("((RBF + RBF + PERIODIC)RBF)LIN + LIN"),"01-airline-res", "Val Loss 80%")
#fit_model(pattern_from_desc("(PERIODIC + LIN + RBF)LIN"),"02-solar-res", "Val Loss 80%")

"""
fit_model([['RBF', 'RBF'],['PERIODIC' , 'RQ'],'LIN'],"03-mauna-res","Autostat")
for (kern,filename) in Autostat:
    fit_model(pattern_from_desc(kern),filename,"Autostat (unfair match)")
for (kern,filename) in BIC_80:
    fit_model(pattern_from_desc(kern),filename, "BIC 80%")
for (kern,filename) in Val:
    fit_model(pattern_from_desc(kern),filename, "Val Loss 80%")

mode = 'BIC'

fit_model([['RBF', 'RBF'],['PERIODIC' , 'RQ'],'LIN'],"03-mauna-res","Autostat")
for (kern,filename) in Autostat:
    fit_model(pattern_from_desc(kern),filename, "Autostat")
for (kern,filename) in BIC_100:
    fit_model(pattern_from_desc(kern),filename, "BIC 100%")
for (kern,filename) in CV_Long:
    fit_model(pattern_from_desc(kern),filename, "CV Long")
"""