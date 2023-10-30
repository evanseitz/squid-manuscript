import os, sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logomaker
from numpy import linalg as LA
from six.moves import cPickle
from scipy import stats
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

np.random.seed(6)

color_add = '#4daf4a' #green

if 1:
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')

    matplotlib.rc('text', usetex=True)
    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{amssymb}",
        r"\usepackage{bold-extra}"])
    

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandParentDir)
import squid.utils as squid_utils

fig = plt.figure(figsize=[15,7])#, constrained_layout=True)
gs1 = GridSpec(1, 3, left=0.05, right=0.97, bottom=0.5, top=0.97, wspace=0.2, hspace=0.1) #0.05, 0.95
ax3 = fig.add_subplot(gs1[0, 2])

# plot sequenc-function space schematic:
x_vals = np.arange(3.25, 4.5, 0.02)
sparse = np.random.choice(np.shape(x_vals)[0],int(np.shape(x_vals)[0]/2.), replace=False)
sparse = np.sort(sparse)
y_vals = []
y_vals_noise = []
noise = np.random.normal(loc=0.0, scale=1.5, size=(x_vals[sparse].shape[0],))

a = .2
for i, x in enumerate(x_vals[sparse]):
    y = ((x-a)*(np.sin(np.pi*(x-a))+np.cos(2*np.pi*(x-a))))
    y_vals_noise.append(y + noise[i])

for i, x in enumerate(x_vals):
    y = ((x-a)*(np.sin(np.pi*(x-a))+np.cos(2*np.pi*(x-a))))
    y_vals.append(y)

ax3.plot(x_vals[sparse], y_vals_noise, c='k')
ax3.scatter(x_vals[sparse], y_vals_noise, c='k', s=10)
if 1:
    ax3.plot(x_vals, y_vals, c='gray', linestyle='--', zorder=-10)
if 0:
    ax3.plot(x_vals, y_vals, c=color_add, zorder=-10, linewidth=3)
if 0:
    ax3.plot(x_vals, y_vals, c='white', zorder=-10, linewidth=3)


ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.axes.xaxis.set_ticklabels([])
ax3.axes.yaxis.set_ticklabels([])

plt.tight_layout()
if 1:
    plt.savefig(os.path.join(pyDir,'Landscape.pdf'), facecolor='w', dpi=200)
plt.show()

