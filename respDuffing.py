#### Main execution file ####
#### provided in CBO-HBM-ex ####
#### Author: Quentin Ragueneau ####
#### url: http://github.com/XXX/CBO-HBM-ex.git ####
#### License: MIT ####

#### Information ####
#### This Python file provide frequency response of Duffing oscillator fof differents set of parameters ####

import fobjDuffing as fobj
import matplotlib.pyplot as plt
import numpy as np



knl_list = np.linspace(0.25,2,8)
f_list = []
Drms_list = []
Arms_list = []
for knl in knl_list:
    f,Drms,Arms = fobj.solveDuffing([0.15,knl])
    f_list.append(f)
    Drms_list.append(Drms)
    Arms_list.append(Arms)
    plt.plot(f,Drms)
