import sys
sys.path.append("..")

import fobjDuffing as fobj
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
import os



plt.rcParams["text.usetex"] = True
if not os.path.exists("figs"):
    os.mkdir("figs")


def latex_float(f):
    float_str = "{:10.3e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


knl_list = np.linspace(0, 2, 100)
f_list = []
Drms_list = []
Arms_list = []
for knl in knl_list:
    print("knl={}".format(knl))
    f, Drms, Arms = fobj.solveDuffing([0.15, knl])
    f_list.append(f)
    Drms_list.append(Drms)
    Arms_list.append(Arms)


f_np = np.concatenate(f_list, axis=0)
Drms_np = np.concatenate(Drms_list, axis=0)
Arms_np = np.concatenate(Arms_list, axis=0)


fig = plt.figure()  # initialise la figure
(line,) = plt.plot([], [])
plt.xlim(f_np.min(), f_np.max())
plt.ylim(Drms_np.min(), Drms_np.max())
for i in range(len(knl_list)):
    plt.plot(f_list[i], Drms_list[i])

plt.ylabel(r"$q_{\rm RMS} [\mathrm{m}]$")
plt.xlabel("$\omega [\mathrm{rad}\cdot\mathrm{s}^{-1}]$")


fig = plt.figure()  # initialise la figure
(line,) = plt.plot([], [])
plt.xlim(f_np.min(), f_np.max())
plt.ylim(Arms_np.min(), Arms_np.max())
for i in range(len(knl_list)):
    plt.plot(f_list[i], Arms_list[i])

plt.ylabel(r"$\ddot{q}_{\rm RMS} [\mathrm{m}\cdot\mathrm{s}^{-2}]$")
plt.xlabel("$\omega [\mathrm{rad}\cdot\mathrm{s}^{-1}]$")


for i in range(len(knl_list)):
    fig = plt.figure()
    plt.xlim(f_np.min(), f_np.max())
    plt.ylim(Drms_np.min(), Drms_np.max())
    plt.plot(f_list[i], Drms_list[i])
    plt.ylabel(r"$q_{\rm RMS} [\mathrm{m}]$")
    plt.xlabel("$\omega [\mathrm{rad}\cdot\mathrm{s}^{-1}]$")
    plt.title(
        "$\displaystyle k_{{nl}}={} \mathrm{{N}}/\mathrm{{m}}^{{-3}}$".format(
            latex_float(knl_list[i])
        )
    )
    plt.savefig(
        os.path.join("figs", "Drms_knl{:03d}.pdf".format(i)), bbox_inches="tight"
    )
    plt.close()


for i in range(len(knl_list)):
    fig = plt.figure()
    plt.xlim(f_np.min(), f_np.max())
    plt.ylim(Arms_np.min(), Arms_np.max())
    plt.plot(f_list[i], Arms_list[i])
    plt.ylabel(r"$\ddot{q}_{\rm RMS} [\mathrm{m}\cdot\mathrm{s}^{-2}]$")
    plt.xlabel("$\omega\, [\mathrm{rad}\cdot\mathrm{s}^{-1}]$")
    plt.title(
        "$\displaystyle k_{{nl}}={} \mathrm{{N}}/\mathrm{{m}}^{{-3}}$".format(
            latex_float(knl_list[i])
        )
    )
    plt.savefig(
        os.path.join("figs", "Arms_knl{:03d}.pdf".format(i)), bbox_inches="tight"
    )
    plt.close()
