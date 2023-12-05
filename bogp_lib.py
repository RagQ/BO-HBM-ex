#### Libraries ####
#### provided in BO-HBM-ex ####
#### Author: Quentin Ragueneau ####
#### url: http://github.com/RagQ/BO-HBM-ex.git ####
#### License: MIT ####

#### Information ####
#### User must refer to the following additional libraries and associated documentation
#### - pyTorch : https://pytorch.org/
#### - BOTorch : https://botorch.org/
#### - GPyTorch : https://gpytorch.ai/

import numpy
import torch
import gpytorch
import matplotlib.pyplot as plt
import pyDOE as doe

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.analytic import (
    ConstrainedExpectedImprovement,
    LogConstrainedExpectedImprovement,
    ExpectedImprovement,
)
from botorch.optim import optimize_acqf


def initfit_GP(
    train_X,
    train_Z,
    GPtype=SingleTaskGP,
    mllfun=ExactMarginalLogLikelihood,
    likelihoodfun=gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
    mean_module=gpytorch.means.ConstantMean(input_size=1),
    covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),
    noise_lvl=1e-4,
    optimfun=torch.optim.Adam,
    lr=0.1,
    Noptim=30,
):
    """Initialize and fit a GP model.

    Args:
        train_X : sample points.
        train_Z : associated responses.
        GPtype (optional): Gaussian Process model (see BOtorch's documentation). Defaults to SingleTaskGP.
        mllfun (optional): fitting method based on Maximum Likelihood (see GPyTorch's documentation). Defaults to ExactMarginalLogLikelihood.
        likelihoodfun (optional): likelihood function (see GPyTorch's documentation). Defaults to gpytorch.likelihoods.FixedNoiseGaussianLikelihood.
        mean_module (optional): type of mean model for GP (see GPyTorch's documentation). Defaults to gpytorch.means.ConstantMean(input_size=1).
        covar_module (optional): select kernel function (see GPyTorch's documentation)). Defaults to gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()).
        noise_lvl (optional): select noise level. Defaults to 1e-4.
        optimfun (optional): select type of optimizer for hyperperameters estimation (see pyTorch's documentation). Defaults to torch.optim.Adam.
        lr (float, optional): learning rate for Adam. Defaults to 0.1.
        Noptim (int, optional): number of iterations for optimization. Defaults to 30.

    Returns:
         - gp: trained GP model.
    """

    # initiatialize the GP model, structure and hyperparameters estimation method
    batch_shape = torch.Size([train_Z.shape[1]])
    likelihood = likelihoodfun(
        noise=torch.ones(train_X.shape[0]) * noise_lvl,
        num_tasks=1,
        rank=0,
        batch_shape=batch_shape,
    )
    mean_module = gpytorch.means.LinearMean(
        input_size=train_X.shape[1], batch_shape=batch_shape, bias=False
    )
    covar_module = gpytorch.kernels.MaternKernel(batch_shape=batch_shape)
    gp = GPtype(
        train_X,
        train_Z,
        likelihood=likelihood,
        covar_module=covar_module,
        mean_module=mean_module,
    )
    mll = mllfun(gp.likelihood, gp)
    mll = mll.to(train_X)

    optimizer = optimfun(gp.parameters(), lr=lr)

    # start training GP
    gp.train()
    for epoch in range(Noptim):
        # clear gradients
        optimizer.zero_grad()
        # forward pass through the model to obtain the output MultivariateNormal
        output = gp(gp.train_inputs[0])
        # Compute negative marginal log likelihood
        loss = -mll(output, gp.train_targets).sum()
        # back prop gradients
        loss.backward()
        optimizer.step()
    gp.eval()

    return gp


def get_pred(gp, Xtest):
    """Get prediction (mean value) from a GP model at points (Xtest).

    Args:
        gp : trained GP model.
        Xtest : array of sample points

    Returns:
        - array of evaluated response values by GP.
    """
    GpPred = gp(Xtest)
    return GpPred.mean.detach()


def get_var(gp, Xtest):
    """Get variance of GP model at points (Xtest).

    Args:
        gp : trained GP model.
        Xtest : array of sample points.

    Returns:
        - array of evaluated variance values by GP.
    """
    GpPred = gp(Xtest)
    return GpPred.variance.detach()


def get_best(Z, objective_index=0, constraints={1: (None, 0.0)}):
    """Get best values of responses based on constraints.

    Args:
        Z: array of actual response values.
        objective_index (int, optional): index of responses (if many functions are fitted). Defaults to 0.
        constraints (dict, optional): list of constraints. Defaults to {1: (None, 0.0)}.

    Returns:
        - best value of actual response values.
    """
    acc = torch.ones(Z.shape[0], dtype=bool)
    for i, (lower, upper) in constraints.items():
        lower = -float("inf") if lower is None else lower
        upper = float("inf") if upper is None else upper
        acc &= (Z[:, i] > lower) & (Z[:, i] < upper)
    if any(acc):
        Zbest = torch.min(Z[:, objective_index][acc])
    else:
        Zbest = torch.min(Z[:, objective_index])
    return Zbest


def get_EI_fun(gp):
    """Get Expected Improvement function from a GP model.

    Args:
        gp : trained GP model.

    Returns:
        expected improvement function for BO.
    """
    Zmin = torch.min(gp.train_targets)
    EI_bo = ExpectedImprovement(gp, best_f=Zmin, maximize=False)
    return EI_bo


def get_EI(gp, Xtest):
    """Get evaluation of Expected Improvement function from a GP model at points (Xtest).

    Args:
        gp : trained GP model.
        Xtest : array of sample points.

    Returns:
        - array of evaluated Expected Improvement values by GP.
    """
    Zmin = torch.min(gp.train_targets)
    EI_bo = ExpectedImprovement(gp, best_f=Zmin, maximize=False)
    Xt = Xtest.reshape(Xtest.shape[0], 1, Xtest.dim())
    return EI_bo(Xt).detach()


def get_EImax(gp, bounds, q=1, num_restarts=10, raw_samples=30):
    """Get maximum of Expected Improvement function from a GP model and associated points.

    Args:
        gp : trained GP model.
        bounds : bounds of the search space.
        q (int, optional): number of maximum points. Defaults to 1.
        num_restarts (int, optional): number of restarts of the optimizer. Defaults to 10.
        raw_samples (int, optional): number of sample points . Defaults to 30.

    Returns:
        - array of points where the maximum of Expected Improvement function is reached.
        - value(s) of the maximum of Expected Improvement function.
    """
    train_X = gp.train_inputs[0]
    train_Z = gp.train_targets.reshape(len(gp.train_targets), 1)

    Zmin = torch.min(gp.train_targets)

    EI = ExpectedImprovement(gp, best_f=Zmin, maximize=False)

    candidate, Eimax = optimize_acqf(
        EI, bounds=bounds, q=q, num_restarts=num_restarts, raw_samples=raw_samples
    )
    return candidate, Eimax


def grid_to_array(X, Y):
    """Convert grid points to array by horizontal concatenation.

    Args:
        X,Y: two grid (numpy ndarray or torch Tensor)

    Returns:
        - array of points (same type as X and Y).
    """
    if type(X) == torch.Tensor and type(Y) == torch.Tensor:
        n = X.nelement()
        Z = torch.cat((X.reshape(n, 1), Y.reshape(n, 1)), -1)
    elif type(X) == numpy.ndarray and type(Y) == numpy.ndarray:
        n = X.size
        Z = numpy.hstack((X.reshape(n, 1), Y.reshape(n, 1)))
    return Z


def lhs_distrib(bounds, n_sample):
    """Generate sampling points using Latin Hypercube Sampling fitted to the bounds.

    Args:
        bounds: bounds of the design space.
        n_sample: number of sampling points.

    Returns:
        - array of sample points.
    """
    ndim = bounds.size()[1]
    A = doe.lhs(ndim, samples=n_sample, criterion="maximin")
    Sample = torch.zeros((n_sample, ndim))
    for i in range(ndim):
        Sample[:, i] = (bounds[1][i] - bounds[0][i]) * A[:, i] + bounds[0][i]
    return Sample


def stdize(Z):
    """Normalize data and get mean and standard deviation.

    Args:
        Z: array of data

    Returns:
        - normalized data.
        - mean and standard deviation.
    """
    nout = Z.shape[1]
    Zsc = torch.zeros_like(Z)
    StdMoy = torch.zeros((2, nout)).double()
    for j in range(nout):
        std, moy = torch.std_mean(Z[:, j])
        Zsc[:, j] = (Z[:, j] - moy) / std
        StdMoy[:, j] = torch.tensor([std, moy])
    return Zsc, StdMoy


def stdize_obj(Z, std, moy):
    """Normalize data using mean and standard deviation.

    Args:
        Z : value to be normalized.
        std : standard deviation.
        moy : mean

    Returns:
        - normalize data
    """
    return (Z - moy) / std


def stdize_cons(constraints, StdMoy):
    """Normalized constraints using mean and standard deviation.

    Args:
        constraints : dictionary of constraints.
        StdMoy : data for normalization.

    Returns:
        - dictionary of normalized constraints.
    """
    stdcons = {}
    for i in constraints.keys():
        std = StdMoy[0, i]
        moy = StdMoy[1, i]
        a, b = constraints[i]
        low = (a - moy) / std if a is not None else None
        up = (b - moy) / std if b is not None else None
        stdcons[i] = (low, up)
    return stdcons


def unstdize(Zsc, StdMoy):
    """Denormalize data using mean and standard deviation.

    Args:
        Zsc : data to be denormalized.
        StdMoy : data for denormalization.

    Returns:
        - denoramlized data.
    """
    nout = Zsc.shape[1]
    Z = torch.zeros_like(Zsc)
    for j in range(nout):
        std, moy = StdMoy[:, j]
        Z[:, j] = std * Zsc[:, j] + moy
    return Z


def enrich2(
    gp,
    fct,
    bounds,
    StdMoy,
    boundsaqf,
    argf=(),
    q=1,
    num_restarts=10,
    raw_samples=30,
    Noptim=50,
):
    """Enrichment of the GP using Expected Improvement function without noramlization of data

    Args:
        gp : initial train GP model.
        fct : function to evaluate the actual responses.
        bounds : bounds of the design space.
        StdMoy: data for normalization.
        boundsaqf : bounds for design space for acquisition function.
        argf (tuple, optional): specific additional arguments required for fct. Defaults to ().
        q (int, optional): number of new points added. Defaults to 1.
        num_restarts (int, optional): number of restarts of the optimizer. Defaults to 10.
        raw_samples (int, optional): number of sample points . Defaults to 30.
        Noptim (int, optional): number of iterations for estimation hyperparameters. Defaults to 50.

    Returns:
        - new trained GP
        - full set of normalized sample points with new ones
        - full set of noralize responses with new ones
        - new normalization data
        - value of the maximum of Expected Improvement function.
    """
    train_X = gp.train_inputs[0]
    train_Z = gp.train_targets.reshape(len(gp.train_targets), 1)
    Z = unstdize(train_Z, StdMoy)

    Zmin = torch.min(gp.train_targets)

    EI = ExpectedImprovement(gp, best_f=Zmin, maximize=False)

    candidate, Eimax = optimize_acqf(
        EI, bounds=boundsaqf, q=q, num_restarts=num_restarts, raw_samples=raw_samples
    )

    new_X = candidate.detach()
    X = unnormalize(new_X, bounds)
    new_Z = fct(X, *argf)

    Xesc = torch.cat([train_X, new_X]).double()
    Ze = torch.cat([Z, new_Z]).double()
    Zesc, Sme = stdize(Ze)

    gp = initfit_GP(Xesc, Zesc, Noptim=Noptim)

    return gp, Xesc, Zesc, Sme, Eimax.detach()

