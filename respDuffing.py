#### Main execution file ####
#### provided in CBO-HBM-ex ####
#### Author: Quentin Ragueneau ####
#### url: http://github.com/XXX/CBO-HBM-ex.git ####
#### License: MIT ####

#### Information ####
#### This Python file provide frequency response of Duffing oscillator fof differents set of parameters ####

import fobjDuffing as fobj
import matplotlib.pyplot as plt
import numpy
import os

ResName = "ParamDuffing"
knl_list = numpy.linspace(0.25,2,8)
xi_list = numpy.array([0.15,0.3,0.5,1.0])

if not os.path.exists(ResName):
        os.mkdir(ResName)

w_list = []
Drms_list = []
Arms_list = []
prm_list = []

for xi in xi_list:
    plt.figure()
    for knl in knl_list:
        w,Drms,Arms = fobj.solveDuffing([xi,knl])
        w_list.append(w)
        Drms_list.append(Drms)
        Arms_list.append(Arms)
        prm_list.append([xi,knl])
        plt.plot(w,Arms,label=f'knl = {knl}')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'${\ddot q}_{\rm RMS}$')
    plt.xlim(0,2.5)
    plt.ylim(0,2.6)
    plt.legend(loc='upper left')
    plt.savefig(ResName+f'/Arms_xi{int(100*xi)}.pdf')
    plt.close()
    
    
