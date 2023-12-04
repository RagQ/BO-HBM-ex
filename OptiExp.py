#### Main execution file ####
#### provided in CBO-HBM-ex ####
#### Author: Quentin Ragueneau ####
#### url: http://github.com/XXX/CBO-HBM-ex.git ####
#### License: MIT ####

#### Information ####
#### Application of Constrained Bayesian Optimization on a Duffing Oscillator ####
#### - based on the use of BOTorch and GPyTorch for Constrained Bayesian Optimization ####
#### - based on HBM and continuation method for solving Duffing problem ####


import numpy
import matplotlib.pyplot as plt
import pandas
import torch
import os
import sys
import time

plt.rcParams["text.usetex"] = True


import bogp_lib as bogp
from botorch.utils.transforms import normalize, unnormalize
from fobjDuffing import fobjDuffing as fobj

# Design space
bounds = torch.tensor([[0.1, 0.1], [1, 2]])
# bounds defined for acquisition function (Expected Improvement) (normalized on the square [0,1])
bndaqf = torch.stack([0 * torch.ones(2), 1 * torch.ones(2)]).double()

# Reference grid considered for testing
xgrid = torch.linspace(bounds[0, 0], bounds[1, 0], 500).double()
ygrid = torch.linspace(bounds[0, 1], bounds[1, 1], 500).double()
Xg, Yg = torch.meshgrid(xgrid, ygrid, indexing="xy")
Xref = bogp.grid_to_array(Xg, Yg)

# Coarse reference grid considered for plotting
xplt = torch.linspace(bounds[0, 0], bounds[1, 0], 25).double()
yplt = torch.linspace(bounds[0, 1], bounds[1, 1], 25).double()
Xp, Yp = torch.meshgrid(xplt, yplt, indexing="xy")
Xplt = bogp.grid_to_array(Xp, Yp)

# number of experiments for each ns
nbExp = 30
# Maximum number of calls to the mechanical solver
budget = 100

# list of number of samples
ns_list = [10, 15, 20, 25]
ResName = "ExpOptimDuffing"


def ExpOpti(ns, nb=0):
    print("=== Start Optimization for ns={} ===".format(ns))
    if not os.path.exists(ResName):
        os.mkdir(ResName)
    subdirs = [name for name in os.listdir(ResName) if os.path.isdir(ResName)]
    numbers = [
        int(name[-2:])
        for name in subdirs
        if name.startswith("Ns" + f"{ns:02d}" + "_Exp")
    ]
    new_num = max(numbers) + 1 if numbers else 0

    Nname = "Ns" + f"{ns:02d}" + "_Exp" + f"{new_num:02d}"

    workdir = os.path.join(ResName, Nname)
    os.mkdir(workdir)
    print("=== Current working directory: {} ===".format(workdir))

    # sampling of the design space using LHS
    print("=== Sampling ", end="")
    tic = time.time()
    Xs = bogp.lhs_distrib(bounds, ns).double()
    print("Done - {}s ===".format(time.time() - tic))
    # associated responses
    Zs = fobj(Xs)
    Nit = ns  # nb of calls

    # Scaling
    Xtest_sc = normalize(Xref, bounds)  # Normalized test grid
    Xplt_sc = normalize(Xplt, bounds)  # Normalized plot grid
    Xs_sc = normalize(Xs, bounds)  # Normalized samples
    Zs_sc, Sm = bogp.stdize(Zs)  # Normalized responses

    # Initialization
    it = 0
    delta = 1


    print("=== Create initial GP: ", end="")
    tic = time.time()
    gp_sc = bogp.initfit_GP(Xs_sc, Zs_sc, Noptim=100)  # Create initial GP model
    print("Done - {}s ===".format(time.time() - tic))

    # Renormalizetion predictions
    Zpred_sc = bogp.get_pred(gp_sc, Xtest_sc).T
    Zpplt_sc = bogp.get_pred(gp_sc, Xplt_sc).T
    Zpred = bogp.unstdize(Zpred_sc, Sm)
    Zpplt = bogp.unstdize(Zpplt_sc, Sm)

    # Evaluation of the acquisition function (EI) on grids
    EI = bogp.get_EI(gp_sc, Xtest_sc)
    EIplt = bogp.get_EI(gp_sc, Xplt_sc)

    EImax = torch.max(EI)
    XEImax = Xref[torch.argmax(EI)]

    # Best minimum
    Zbest = torch.min(Zs)
    Xbest = Xs[torch.argmin(Zs)]

    delta = float(EImax / abs(Zbest))

    # Prepare data for visualization on grids
    Fg = Zpred[:, 0].reshape(Xg.shape)
    EIg = EI.reshape(Xg.shape)

    plt.rc("font", family="serif", size=16)
    

    # Save Pred
    df = pandas.DataFrame()
    df["Xref"] = Xplt[:, 0]
    df["Yref"] = Xplt[:, 0]
    df["Fpred"] = Zpplt[:, 0]
    df["EI"] = EIplt
    df.to_csv(os.path.join(workdir, "Surf_pred_" + f"{it:02}" + ".csv"), index=None)
    # Save Points
    df = pandas.DataFrame()
    df["Xs"] = Xs[:, 0]
    df["Ys"] = Xs[:, 1]
    df["Fs"] = Zs[:, 0]
    df1 = pandas.DataFrame()
    df1["Xbest"] = [float(Xbest[0])]
    df1["Ybest"] = [float(Xbest[1])]
    df1["Fbest"] = [float(Zbest)]
    df = pandas.concat([df, df1], axis=1)
    df1 = pandas.DataFrame()
    df1["XEImax"] = [float(XEImax[0])]
    df1["YEImax"] = [float(XEImax[1])]
    df1["EImax"] = [float(EImax)]
    df1["delta"] = [float(delta)]
    df = pandas.concat([df, df1], axis=1)
    df.to_csv(os.path.join(workdir, "Enrich_" + f"{it:02}" + ".csv"), index=None)
    
    #Plot
    plt.figure()
    ax1 = plt.subplot()
    CSF = ax1.contour(
        Xg, Yg, Fg, cmap=plt.cm.viridis, levels=numpy.linspace(0, 2, 21)
    )
    ax1.clabel(
        CSF,
        levels=numpy.linspace(0, 2, 11),
        inline=1,
        inline_spacing=0,
        fontsize=15,
    )
    ax1.set_xlim(0.1, 1)
    ax1.set_ylim(0.1, 2)
    ax1.set_xlabel(r'$\xi\, [\mathrm{kg}/\mathrm{s}]$')
    ax1.set_ylabel(r'$k_{nl}\, [\mathrm{N}/\mathrm{m}^{-3}]$')
    ax1.set_title(r'Objective Function')
    ax1.plot(  #Initial Samples
        Xs[:, 0],
        Xs[:, 1],
        "D",
        label="Initial Samples",
        color="tab:red",
        zorder=100,
        markersize=8,
    )
    ax1.scatter(
        Xbest[0],
        Xbest[1],
        marker="$\\bigodot$",
        color="black",
        linestyle="None",
        zorder=90,
        label="Minimum",
        s=384,
        alpha=1,
    )
    plt.savefig(os.path.join(workdir, "ContourObj_" + f"{it:02}" + ".pdf"))
    plt.close()

    plt.figure()
    ax2 = plt.subplot()
    CSCEI = ax2.contour(Xg, Yg, EIg, cmap=plt.cm.YlOrRd, levels=50)
    ax2.plot(
        XEImax[0],
        XEImax[1],
        'o',
        color='tab:green',
        zorder=100,
        markersize=8,)
    ax2.set_xlim(0.1, 1)
    ax2.set_ylim(0.1, 2)
    ax2.set_xlabel(r'$\xi\, [\mathrm{kg}/\mathrm{s}]$')
    ax2.set_ylabel(r'$k_{nl}\, [\mathrm{N}/\mathrm{m}^{-3}]$')
    ax2.set_title(r'Expected Improvement')
    plt.savefig(os.path.join(workdir, "ContourCEI_" + f"{it:02}" + ".pdf"))
    plt.close()
    
    plt.figure()
    ax1 = plt.subplot(projection='3d')
    ax1.plot_surface(
        Xg, Yg, Fg, cmap=plt.cm.viridis
    )
    ax1.set_xlim(0.1, 1)
    ax1.set_ylim(0.1, 2)
    ax1.set_zlim(0.,4.5)
    ax1.set_xlabel(r'$\xi\, [\mathrm{kg}/\mathrm{s}]$')
    ax1.set_ylabel(r'$k_{nl}\, [\mathrm{N}/\mathrm{m}^{-3}]$')
    ax1.set_zlabel(r'$F_{\mathrm{obj}}(\xi,k_{nl})$')
    ax1.set_title(r'Objective Function')
    ax1.plot(  #Initial Samples
        numpy.array(Xs[:, 0]),
        numpy.array(Xs[:, 1]),
        numpy.array(Zs[:, 0]),
        marker="D",
        linestyle="None",
        label="Initial Samples",
        color="tab:red",
        zorder=100,
        markersize=8,
    )
    ax1.scatter(
        Xbest[0],
        Xbest[1],
        Zbest,
        marker="$\\bigodot$",
        color="black",
        linestyle="None",
        zorder=90,
        label="Minimum",
        s=384,
        alpha=1,
    )
    plt.savefig(os.path.join(workdir, "SurfaceObj_" + f"{it:02}" + ".pdf"))
    plt.close()

    plt.figure()
    ax2 = plt.subplot(projection='3d')
    ax2.plot_surface(Xg, Yg, EIg, cmap=plt.cm.YlOrRd)
    ax2.plot(
        XEImax[0],
        XEImax[1],
        EImax,
        marker='o',
        linestyle="None",
        color='tab:green',
        zorder=100,
        markersize=8,)
    ax2.set_xlim(0.1, 1)
    ax2.set_ylim(0.1, 2)
    ax2.set_xlabel(r'$\xi\, [\mathrm{kg}/\mathrm{s}]$')
    ax2.set_ylabel(r'$k_{nl}\, [\mathrm{N}/\mathrm{m}^{-3}]$')
    ax2.set_zlabel(r'$EI(\xi,k_{nl})$')
    ax2.set_title(r'Expected Improvement')
    plt.savefig(os.path.join(workdir, "SurfaceCEI_" + f"{it:02}" + ".pdf"))
    plt.close()

    dfg = pandas.DataFrame(
        [
            [
                it,
                Nit,
                float(Xbest[0]),
                float(Xbest[1]),
                float(Zbest),
                float(XEImax[0]),
                float(XEImax[1]),
                float(EImax),
                0,
                delta,
            ]
        ],
        columns=[
            "Iterations",
            "Calls",
            "Xmin",
            "Ymin",
            "Zmin",
            "XEI",
            "YEI",
            "EImax",
            "Eimax",
            "delta",
        ],
    )

    # Iterations
    print("=== Start enrichment iterations ===")
    while Nit < budget:
        it = it + 1
        print("=== Iteration: {} (ns={}, nb={}) ===".format(it, ns, nb))
        # enrichment
        print("== Get new point ", end="")
        tic = time.time()
        gp_sc, Xs_sc, Zs_sc, Sm, Eimax = bogp.enrich2(
            gp_sc,
            fobj,
            bounds,
            Sm,
            bndaqf,
            Noptim=100,
            num_restarts=150,
            raw_samples=150,
        )
        print("Done - {}s ".format(time.time() - tic))

        Nit = Nit + 1

        # Get data for visualization
        Zpred_sc = bogp.get_pred(gp_sc, Xtest_sc).T  # Prediction
        Zpplt_sc = bogp.get_pred(gp_sc, Xplt_sc).T
        Zpred = bogp.unstdize(Zpred_sc, Sm)
        Zpplt = bogp.unstdize(Zpplt_sc, Sm)

        EI = bogp.get_EI(gp_sc, Xtest_sc)  # EI
        EIplt = bogp.get_EI(gp_sc, Xplt_sc)  # EI
        EImax = torch.max(EI)
        XEImax = Xref[torch.argmax(EI)]
        Xm = unnormalize(Xs_sc, bounds)
        Zm = bogp.unstdize(Zs_sc, Sm)
        Zbest = torch.min(Zm)  # Best min
        Xbest = Xm[torch.argmin(Zm)]

        delta = float(EImax / abs(Zbest))
        # Prepare data for visualization on grids
        Fg = Zpred[:, 0].reshape(Xg.shape)
        EIg = EI.reshape(Xg.shape)


        # Save Pred
        df = pandas.DataFrame()
        df["Xref"] = Xplt[:, 0]
        df["Yref"] = Xplt[:, 0]
        df["Fpred"] = Zpplt[:, 0]
        df["EI"] = EIplt
        df.to_csv(os.path.join(workdir, "Surf_pred_" + f"{it:02}" + ".csv"), index=None)
        # Save Points
        df = pandas.DataFrame()
        df["Xs"] = Xs[:, 0]
        df["Ys"] = Xs[:, 1]
        df["Fs"] = Zs[:, 0]
        df1 = pandas.DataFrame()  # added points
        df1["Xm"] = Xm[ns:-1, 0]
        df1["Ym"] = Xm[ns:-1, 1]
        df1["Fm"] = Zm[ns:-1, 0]
        df = pandas.concat([df, df1], axis=1)
        df1 = pandas.DataFrame()  # New point
        df1["Xnew"] = [float(Xm[-1, 0])]
        df1["Ynew"] = [float(Xm[-1, 1])]
        df1["Fnew"] = [float(Zm[-1, 0])]
        df = pandas.concat([df, df1], axis=1)
        df1 = pandas.DataFrame()
        df1["Xbest"] = [float(Xbest[0])]
        df1["Ybest"] = [float(Xbest[1])]
        df1["Fbest"] = [float(Zbest)]
        df = pandas.concat([df, df1], axis=1)
        df1 = pandas.DataFrame()
        df1["XEImax"] = [float(XEImax[0])]
        df1["YEImax"] = [float(XEImax[1])]
        df1["EImax"] = [float(EImax)]
        df1["delta"] = [float(delta)]
        df = pandas.concat([df, df1], axis=1)
        df.to_csv(os.path.join(workdir, "Enrich_" + f"{it:02}" + ".csv"), index=None)
        
        #Plot
        plt.figure()
        ax1 = plt.subplot()
        CSF = ax1.contour(
            Xg, Yg, Fg, cmap=plt.cm.viridis, levels=numpy.linspace(0, 2, 21)
        )
        ax1.clabel(
            CSF,
            levels=numpy.linspace(0, 2, 11),
            inline=1,
            inline_spacing=0,
            fontsize=15,
        )
        ax1.set_xlim(0.1, 1)
        ax1.set_ylim(0.1, 2)
        ax1.set_xlabel(r'$\xi\, [\mathrm{kg}/\mathrm{s}]$')
        ax1.set_ylabel(r'$k_{nl}\, [\mathrm{N}/\mathrm{m}^{-3}]$')
        ax1.set_title(r'Objective Function')
        ax1.plot(  #Initial Samples
            Xs[:, 0],
            Xs[:, 1],
            "D",
            label="Initial Samples",
            color="tab:red",
            zorder=100,
            markersize=8,
        )
        ax1.plot( #Added Samples
            Xm[ns:-1, 0],
            Xm[ns:-1, 1],
            "o",
            label="Added Samples",
            color="tab:red",
            zorder=100,
            markersize=8,
        )
        ax1.plot( #New point
            Xm[-1, 0],
            Xm[-1, 1],
            "s",
            label="New Sample",
            color="#ccff00",
            zorder=100,
            markersize=8,
        )
        ax1.scatter(
            Xbest[0],
            Xbest[1],
            marker="$\\bigodot$",
            color="black",
            linestyle="None",
            zorder=90,
            label="Minimum",
            s=384,
            alpha=1,
        )
        plt.savefig(os.path.join(workdir, "ContourObj_" + f"{it:02}" + ".pdf"))
        plt.close()

        plt.figure()
        ax2 = plt.subplot()
        CSCEI = ax2.contour(Xg, Yg, EIg, cmap=plt.cm.YlOrRd, levels=50)
        ax2.plot(
            XEImax[0],
            XEImax[1],
            'o',
            color='tab:green',
            zorder=100,
            markersize=8,)
        ax2.set_xlim(0.1, 1)
        ax2.set_ylim(0.1, 2)
        ax2.set_xlabel(r'$\xi\, [\mathrm{kg}/\mathrm{s}]$')
        ax2.set_ylabel(r'$k_{nl}\, [\mathrm{N}/\mathrm{m}^{-3}]$')
        ax2.set_title(r'Expected Improvement')
        plt.savefig(os.path.join(workdir, "ContourCEI_" + f"{it:02}" + ".pdf"))
        plt.close()
        
        
        plt.figure()
        ax1 = plt.subplot(projection='3d')
        ax1.plot_surface(
            Xg, Yg, Fg, cmap=plt.cm.viridis
        )
        ax1.set_xlim(0.1, 1)
        ax1.set_ylim(0.1, 2)
        ax1.set_zlim(0.,4.5)
        ax1.set_xlabel(r'$\xi\, [\mathrm{kg}/\mathrm{s}]$')
        ax1.set_ylabel(r'$k_{nl}\, [\mathrm{N}/\mathrm{m}^{-3}]$')
        ax1.set_zlabel(r'$F_{\mathrm{obj}}(\xi,k_{nl})$')
        ax1.set_title(r'Objective Function')
        ax1.plot(  #Initial Samples
            numpy.array(Xs[:, 0]),
            numpy.array(Xs[:, 1]),
            numpy.array(Zs[:, 0]),
            marker="D",
            linestyle="None",
            label="Initial Samples",
            color="tab:red",
            zorder=100,
            markersize=8,
        )
        ax1.plot( #Added Samples
            numpy.array(Xm[ns:-1, 0]),
            numpy.array(Xm[ns:-1, 1]),
            numpy.array(Zm[ns:-1, 0]),
            marker = "o",
            linestyle = 'None',
            label="Added Samples",
            color="tab:red",
            zorder=100,
            markersize=8,
        )
        ax1.plot( #New point
            Xm[-1, 0],
            Xm[-1, 1],
            Zm[-1,0],
            marker = "s",
            linestyle = 'None',
            label="New Sample",
            color="#ccff00",
            zorder=100,
            markersize=8,
        )
        ax1.scatter(
            Xbest[0],
            Xbest[1],
            Zbest,
            marker="$\\bigodot$",
            color="black",
            linestyle="None",
            zorder=90,
            label="Minimum",
            s=384,
            alpha=1,
        )
        plt.savefig(os.path.join(workdir, "SurfaceObj_" + f"{it:02}" + ".pdf"))
        plt.close()

        plt.figure()
        ax2 = plt.subplot(projection='3d')
        ax2.plot_surface(Xg, Yg, EIg, cmap=plt.cm.YlOrRd)
        ax2.plot(
            XEImax[0],
            XEImax[1],
            EImax,
            marker = 'o',
            linestyle = 'None',
            color='tab:green',
            zorder=100,
            markersize=8,)
        ax2.set_xlim(0.1, 1)
        ax2.set_ylim(0.1, 2)
        ax2.set_xlabel(r'$\xi\, [\mathrm{kg}/\mathrm{s}]$')
        ax2.set_ylabel(r'$k_{nl}\, [\mathrm{N}/\mathrm{m}^{-3}]$')
        ax2.set_zlabel(r'$EI(\xi,k_{nl})$')
        ax2.set_title(r'Expected Improvement')
        plt.savefig(os.path.join(workdir, "SurfaceCEI_" + f"{it:02}" + ".pdf"))
        plt.close()

        dfg2 = pandas.DataFrame(
            [
                [
                    it,
                    Nit,
                    float(Xbest[0]),
                    float(Xbest[1]),
                    float(Zbest),
                    float(XEImax[0]),
                    float(XEImax[1]),
                    float(EImax),
                    delta,
                ]
            ],
            columns=[
                "Iterations",
                "Calls",
                "Xmin",
                "Ymin",
                "Zmin",
                "XEI",
                "YEI",
                "EImax",
                "delta",
            ],
        )
        dfg = pandas.concat([dfg, dfg2])

    return dfg


dflist = []
dfStat = pandas.DataFrame()
for ns in ns_list:
    dfStat = pandas.DataFrame()
    for nb in range(1, nbExp+1):
        print("===", nb, "===")
        df = ExpOpti(ns, nb)

        Amin = df["Zmin"].to_numpy()
        AEI = df["EImax"]
        Adelta = df["delta"]

        if nb == 1:
            Anit = df["Calls"].to_numpy()
            DbEI = AEI
            Dbmin = Amin
            Dbdelta = Adelta
        else:
            DbEI = numpy.column_stack((DbEI, AEI))
            Dbmin = numpy.column_stack((Dbmin, Amin))
            Dbdelta = numpy.column_stack((Dbdelta, Adelta))

    dfEI = pandas.DataFrame(DbEI)
    dfEI.to_csv(os.path.join(ResName, "DBEImax_ns" + f"{ns:02}" + ".csv"), index=None)
    dfZmin = pandas.DataFrame(Dbmin)
    dfZmin.to_csv(os.path.join(ResName, "DBZmin_ns" + f"{ns:02}" + ".csv"), index=None)
    dfDelta = pandas.DataFrame(Dbdelta)
    dfDelta.to_csv(
        os.path.join(ResName, "DBdelta_ns" + f"{ns:02}" + ".csv"), index=None
    )

    dfStat["Nit"] = Anit
    dfStat["moyZmin"] = numpy.mean(Dbmin, axis=1)
    dfStat["varZmin"] = numpy.var(Dbmin, axis=1)
    dfStat["stdZmin"] = numpy.std(Dbmin, axis=1)
    dfStat["minZmin"] = numpy.min(Dbmin, axis=1)
    dfStat["maxZmin"] = numpy.max(Dbmin, axis=1)
    dfStat["medZmin"] = numpy.median(Dbmin, axis=1)
    dfStat["q1Zmin"] = numpy.quantile(Dbmin, 0.25, axis=1)
    dfStat["q3Zmin"] = numpy.quantile(Dbmin, 0.75, axis=1)
    dfStat["moyEImax"] = numpy.mean(DbEI, axis=1)
    dfStat["varEImax"] = numpy.var(DbEI, axis=1)
    dfStat["stdEImax"] = numpy.std(DbEI, axis=1)
    dfStat["minEImax"] = numpy.min(DbEI, axis=1)
    dfStat["maxEImax"] = numpy.max(DbEI, axis=1)
    dfStat["medEImax"] = numpy.median(DbEI, axis=1)
    dfStat["q1EImax"] = numpy.quantile(DbEI, 0.25, axis=1)
    dfStat["q3EImax"] = numpy.quantile(DbEI, 0.75, axis=1)
    dfStat["moydelta"] = numpy.mean(Dbdelta, axis=1)
    dfStat["vardelta"] = numpy.var(Dbdelta, axis=1)
    dfStat["stddelta"] = numpy.std(Dbdelta, axis=1)
    dfStat["mindelta"] = numpy.min(Dbdelta, axis=1)
    dfStat["maxdelta"] = numpy.max(Dbdelta, axis=1)
    dfStat["meddelta"] = numpy.median(Dbdelta, axis=1)
    dfStat["q1delta"] = numpy.quantile(Dbdelta, 0.25, axis=1)
    dfStat["q3delta"] = numpy.quantile(Dbdelta, 0.75, axis=1)

    dfStat.to_csv(os.path.join(ResName, "Stats_ns" + f"{ns:02}" + ".csv"), index=None)
