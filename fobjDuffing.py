#### Libraries ####
#### provided in BO-HBM-ex ####
#### Author: Quentin Ragueneau ####
#### url: http://github.com/RagQ/BO-HBM-ex.git ####
#### License: MIT ####

### Evaluation of the objective function for the Duffing oscillator example
### - use of the HBM method
### - use of the pseudo arc length continuation method
### - use of the Newton-Raphson method


import numpy
import torch
import scipy.sparse as sps


def solveDuffing(Xpara:numpy.array):
    """
    Compute the frequency response.

    Args:
        X (torch.Tensor): Array like - Size (ns x 2) - X = [xi,knl]

    Raises:
        StopIteration: in the case of no convergence

    Returns:
        w_list: List of angular frequencies.
        Drms: Root Mean Square displacement list
        Arms: Root Mean Square acceleration list
    """
    

    # Oscillator definition
    m = 1  # Mass (kg)
    k = 1  # Linear stiffness (N.m-1)
    f0 = 0.3  # Excitation amplitude (N)

    # HBM parameters
    nh = 20  # Harmonics number
    Nt = 60  # AFT time samples number
    dsmin = 1e-10  # Minimum arclength step
    dsmax = 0.05  # Maximum arclength step
    it_tar = 8  # Targeted number of Newton iterations
    it_max = 50  # Maximum number of iteration for first point
    eps_hbm = 1e-4  # Stopping criterion
    w0 = 0.01  # Starting angular frequency (rad)
    wf = 2.5  # Ending angular frequency (rad)


    xi = float(Xpara[0])  # Damping
    knl = float(Xpara[1])  # Nonlinear stiffness
    # Initialization --------------------------------------------------------
    w_list = []  # list of frequencies
    qh_list = []  # list of Fourrier coefficients
    fnlh_list = []  # list of nonlinear terms

    fexth = numpy.zeros(2 * nh + 1)  # Excitation force fourrier coefficients
    fexth[1] = f0

    F = numpy.zeros((2 * nh + 1, Nt))  # Direct Fourier transform matrix
    F[0] = 0.5 * numpy.ones(Nt)
    for i in range(1, nh):
        for j in range(Nt):
            F[2 * i - 1][j] = numpy.cos(i * 2 * numpy.pi * j / Nt)
            F[2 * i][j] = numpy.sin(i * 2 * numpy.pi * j / Nt)
    i = nh
    if nh == Nt / 2:  # if Nt even
        for j in range(Nt):
            F[2 * i - 1][j] = 0.5 * numpy.cos(i * 2 * numpy.pi * j / Nt)
            F[2 * i][j] = 0.5 * numpy.sin(i * 2 * numpy.pi * j / Nt)
    else:  # if Nt odd
        for j in range(Nt):
            F[2 * i - 1][j] = numpy.cos(i * 2 * numpy.pi * j / Nt)
            F[2 * i][j] = numpy.sin(i * 2 * numpy.pi * j / Nt)
    Fd = sps.csc_matrix(2.0 / Nt * F)

    F = numpy.zeros((Nt, 2 * nh + 1))  # Inverse Fourier transform matrix
    F[:, 0] = numpy.ones(Nt)
    for i in range(Nt):
        for j in range(1, nh + 1):
            F[i][2 * j - 1] = numpy.cos(j * 2 * numpy.pi * i / Nt)
            F[i][2 * j] = numpy.sin(j * 2 * numpy.pi * i / Nt)
    Fi = sps.csc_matrix(F)

    # -----------------------------------------------------------------------

    # First point -----------------------------------------------------------
    qh = numpy.zeros(2 * nh + 1)  # HBM unknowns
    w = w0
    Q = f0 / (-(w**2) * m + w * xi * 1j + k)  # Initial guess = linear solution
    qh[1] = abs(Q) * numpy.cos(numpy.angle(Q))  # Fourier coefficients
    qh[2] = -abs(Q) * numpy.sin(numpy.angle(Q))

    Z = Build_Z(w, m, xi, k, nh)

    Mag = 1
    it = 0
    while Mag > eps_hbm:  # HBM iterations
        it = it + 1
        fnlh, dfnlh = Buildfnlh(qh, nh, Nt, knl, Fd, Fi)  # Nonlinear terms
        rh = Z.dot(qh) + fnlh - fexth  # Residual
        Jrh = Z + dfnlh  # Jacobian /qh
        Dqh = sps.linalg.spsolve(Jrh, -rh)  # Increment
        qh = qh + Dqh
        Mag = numpy.linalg.norm(rh)
        if it == it_max:
            raise StopIteration("Initialization failed - No convergence")

    w_list.append(w)
    qh_list.append(qh)
    fnlh_list.append(fnlh)

    # -----------------------------------------------------------------------

    # Arc length continuation -----------------------------------------------
    ds = dsmax / 5
    dZ = Build_dZ(w, m, xi, nh)  # dZ/dw
    Jrw = dZ.dot(qh)[..., None]  # Jacobian /qh
    while w < wf:
        # Tangent Vector
        t = sps.linalg.spsolve(Jrh, -Jrw)
        sgn = numpy.linalg.slogdet(Jrh.toarray())[0]
        sigma = w / numpy.linalg.norm(qh)  # Scaling
        tw = (
            sgn * 1 / numpy.sqrt(1 + (sigma * t.T).dot(sigma * t))
        )  # Tangent vector
        tqh = tw * t.T

        ds, w, qh, fnlh, dfnlh, Jrh, Jrw, it = p_arc_HBM(
            ds,
            qh,
            w,
            tqh,
            tw,
            m,
            xi,
            k,
            knl,
            nh,
            Nt,
            dsmin,
            it_tar,
            eps_hbm,
            fexth,
            Fd,
            Fi,
        )

        w_list.append(w)
        qh_list.append(qh)
        fnlh_list.append(fnlh)

        # ds adaptation
        ds = ds * (2 ** ((it_tar - it) / it_tar))
        if ds > dsmax:
            ds = dsmax

    # -----------------------------------------------------------------------

    # Arms/Drms computation ------------------------------------------------------
    Arms = numpy.zeros(len(w_list)) # RMS accelerations
    Drms = numpy.zeros(len(w_list)) # RMS displacements
    W_blocks = [numpy.zeros((1, 1))]
    for i in range(1, nh + 1):
        W_blocks.append(numpy.array([[0, i], [-i, 0]]))
    W = sps.block_diag(W_blocks, "csc", "float")  # Differential operator
    for i in range(len(w_list)):
        ddqh = w_list[i] ** 2 * W.dot(
            W.dot(qh_list[i])
        )  # Fourier coefficient of the acceleration
        Arms[i] = numpy.sqrt(numpy.sum(0.5 * ddqh**2))
        Drms[i] = numpy.sqrt(numpy.sum(0.5 * qh_list[i]**2))
    # -----------------------------------------------------------------------

    return w_list,Drms,Arms

# Objective function
def fobjDuffing(X: torch.Tensor):
    """
    Compute the objective function.

    Args:
        X (torch.Tensor): Torch tensor of float64 - Size (ns x 2) - X = [xi,knl]

    Raises:
        StopIteration: in the case of no convergence

    Returns:
        y: maximum of the amplitude of the response along frequencies.
    """
    # Output
    ns = X.shape[0]    
    y = torch.zeros((ns, 1)).double()
    
    #along parameter's sets
    for ni in range(ns):        
        _,_,Arms = solveDuffing(numpy.array(X[ni, :]))
        y[ni, 0] = numpy.max(Arms)

    return y


def Build_Z(w, m, xi, k, nh):  # Build Z
    Z_blocks = [[k]]
    for i in range(1, nh + 1):
        Z_blocks.append(
            numpy.array(
                [
                    [k - (i * w) ** 2 * m, i * w * xi],
                    [-i * w * xi, k - (i * w) ** 2 * m],
                ]
            )
        )
    return sps.block_diag(Z_blocks, "csc", "float")


def Build_dZ(w, m, xi, nh):  # Build dZ/dw
    DZ_blocks = [[0]]
    for i in range(1, nh + 1):
        DZ_blocks.append(
            numpy.array(
                [[-2 * (i) ** 2 * w * m, i * xi], [-i * xi, -2 * (i) ** 2 * w * m]]
            )
        )
    return sps.block_diag(DZ_blocks, "csc", "float")


def trigo_to_expo(
    qh, nh, Nt
):  # Fourier coefficients from trigonometric to exponential form
    X_expo = numpy.zeros((nh + 1), complex)
    X_expo[0] = Nt * qh[0] + 0j
    for i in range(1, nh):
        X_expo[i] = Nt / 2 * (qh[(2 * i - 1) : 2 * i] - qh[2 * i : (2 * i + 1)] * 1j)
    i = nh
    if Nt % 2 == 0:
        X_expo[i] = Nt * (qh[(2 * i - 1) : 2 * i] - qh[2 * i : (2 * i + 1)] * 1j)
    else:
        X_expo[i] = Nt / 2 * (qh[(2 * i - 1) : 2 * i] - qh[2 * i : (2 * i + 1)] * 1j)
    return X_expo


def expo_to_trigo(
    qh, nh, Nt
):  # Fourier coefficients from exponential to trigonometric form
    X_trig = numpy.zeros(2 * nh + 1)
    X_trig[0] = qh[0].real / Nt
    for i in range(1, nh):
        X_trig[(2 * i - 1) : 2 * i] = 2 / Nt * qh[i].real
        X_trig[2 * i : (2 * i + 1)] = -2 / Nt * qh[i].imag
    i = nh
    if Nt % 2 == 0:
        X_trig[(2 * i - 1) : 2 * i] = 1.0 / Nt * qh[i].real
        X_trig[2 * i : (2 * i + 1)] = -1.0 / Nt * qh[i].imag
    else:
        X_trig[(2 * i - 1) : 2 * i] = 2 / Nt * qh[i].real
        X_trig[2 * i : (2 * i + 1)] = -2 / Nt * qh[i].imag
    return X_trig


def Buildfnlh(qh, nh, Nt, knl, Fd, Fi):  # Build fnlh and d fnlh/d qh with AFT method
    qhexp = trigo_to_expo(qh, nh, Nt)
    qt = numpy.fft.irfft(qhexp, n=Nt)
    fnlt = knl * qt**3
    dfnlt = sps.diags(3 * knl * qt**2, format="csc", shape=(Nt, Nt))
    fnlhexp = numpy.fft.rfft(fnlt)
    fnlh = expo_to_trigo(fnlhexp, nh, Nt)
    dfnlh = Fd.dot(dfnlt.dot(Fi))
    return fnlh, dfnlh


def p_arc_HBM(
    ds, qh, w, tqh, tw, m, xi, k, knl, nh, Nt, dsmin, it_tar, eps_hbm, fexth, Fd, Fi
):
    args0 = (
        qh,
        w,
        tqh,
        tw,
        m,
        xi,
        k,
        knl,
        nh,
        Nt,
        dsmin,
        it_tar,
        eps_hbm,
        fexth,
        Fd,
        Fi,
    )

    # Prediction
    qh = qh + ds * tqh
    w = w + ds * tw

    # Pseudo arc length corrections
    Mag = 1
    it = 0
    while Mag > eps_hbm:
        it = it + 1
        Z = Build_Z(w, m, xi, k, nh)
        dZ = Build_dZ(w, m, xi, nh)  # dZ/dw
        fnlh, dfnlh = Buildfnlh(qh, nh, Nt, knl, Fd, Fi)
        rh = Z.dot(qh) + fnlh - fexth
        r = numpy.hstack((rh, 0))  # Pseudo arclength residual
        Jrh = Z + dfnlh  # Jacobian /qh
        Jrw = dZ.dot(qh)[..., None]  # Jacobian /w
        Jr = numpy.block([[Jrh.toarray(), Jrw], [tqh, tw]])  # Pseudo arclength Jacobian
        Dq = numpy.linalg.solve(Jr, -r)  # Increment
        qh = qh + Dq[:-1]
        w = w + Dq[-1]
        Mag = numpy.linalg.norm(r)
        if it == it_tar:  # If no convergence after it_tar
            ds = ds / 5
            if ds < dsmin:
                raise StopIteration(f"Continuation failed at w = {w} - No convergence")
            else:
                return p_arc_HBM(ds, *args0)

    return ds, w, qh, fnlh, dfnlh, Jrh, Jrw, it
