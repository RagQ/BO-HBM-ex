#### Main execution file ####
#### provided in BO-HBM-ex ####
#### Author: Quentin Ragueneau ####
#### url: http://github.com/RagQ/BO-HBM-ex.git ####
#### License: MIT ####

#### Information ####
#### This Python file provide frequency response of Duffing oscillator fof differents set of parameters ####

import fobjDuffing as fobj
import matplotlib.pyplot as plt
import numpy
import os

plt.rcParams["text.usetex"] = True

ResName = "ParamDuffing"
knl_list = numpy.linspace(0.25, 2, 8)
xi_list = numpy.array([0.15, 0.3, 0.5, 1.0])

if not os.path.exists(ResName):
    os.mkdir(ResName)

w_list = []
Drms_list = []
Arms_list = []
prm_list = []

for xi in xi_list:
    plt.figure(0)
    plt.figure(1)
    for knl in knl_list:
        print("=== xi = {} - knl = {} ===".format(xi, knl))
        w, Drms, Arms = fobj.solveDuffing([xi, knl])
        w_list.append(w)
        Drms_list.append(Drms)
        Arms_list.append(Arms)
        prm_list.append([xi, knl])
        plt.figure(0)
        plt.plot(w, Drms, label=f"${knl}$")
        plt.figure(1)
        plt.plot(w, Arms, label=f"${knl}$")
    plt.figure(0)
    plt.xlabel(r"$\omega\,[\mathrm{rad}\cdot\mathrm{s}^{-1}]$")
    plt.ylabel(r"$q_{\rm RMS}\,[\mathrm{m}]$")
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.6)
    plt.legend(loc="upper left",title=f"$k_{{nl}}\,[\mathrm{{N}}\cdot\mathrm{{m}}^{{-3}}]$")
    plt.savefig(os.path.join(ResName, "Drms_xi{}.pdf".format(int(100 * xi))))
    plt.close()
    plt.figure(1)
    plt.xlabel(r"$\omega\,[\mathrm{rad}\cdot\mathrm{s}^{-1}]$")
    plt.ylabel(r"${\ddot q}_{\rm RMS}\,[\mathrm{m}\cdot\mathrm{s}^{-2}]$")
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.6)
    plt.legend(loc="upper left",title=f"$k_{{nl}}\,[\mathrm{{N}}\cdot\mathrm{{m}}^{{-3}}]$")
    plt.savefig(os.path.join(ResName, "Arms_xi{}.pdf".format(int(100 * xi))))
    plt.close()
