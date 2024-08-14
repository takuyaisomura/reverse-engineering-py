from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from reverse_engineering.utils.utils import check_double_layer_args


@dataclass
class NNVariables:
    """Neural network variables."""

    x: np.ndarray  # (T, Nx)
    W1: np.ndarray  # (Nsess, Nx, No)
    W0: np.ndarray  # (Nsess, Nx, No)
    phi1: np.ndarray  # (Nx,)
    phi0: np.ndarray  # (Nx,)
    h1: np.ndarray | None = None  # (Nsess, Nx); optional because h can be calculated from W and phi
    h0: np.ndarray | None = None  # (Nsess, Nx); optional because h can be calculated from W and phi
    y: np.ndarray | None = None  # (T, Ny)
    K1: np.ndarray | None = None  # (Nsess, Nx, Nx)
    K0: np.ndarray | None = None  # (Nsess, Nx, Nx)
    V1: np.ndarray | None = None  # (Nsess, Ny, Nx)
    V0: np.ndarray | None = None  # (Nsess, Ny, Nx)
    psi1: np.ndarray | None = None  # (Ny,)
    psi0: np.ndarray | None = None  # (Ny,)
    m1: np.ndarray | None = None  # (Nsess, Ny)
    m0: np.ndarray | None = None  # (Nsess, Ny)


@dataclass
class POMDPVariables:
    """POMDP (Bayesian) variables."""

    qs: np.ndarray  # (T, Ns * 2)
    qA: np.ndarray  # (Nsess, No * 2, Ns * 2)
    lnD: np.ndarray  # (Ns * 2,)
    qd: np.ndarray | None = None  # (T, Nd * 2)
    qBi: np.ndarray | None = None  # (Nsess, Ns * 2, Ns * 2)
    qCi: np.ndarray | None = None  # (Nsess, Ns * 2, Nd * 2)
    lnE: np.ndarray | None = None  # (Nd * 2,)


def pomdp2nn(pomdp: POMDPVariables) -> NNVariables:
    """Convert POMDP variables to neural network variables."""
    is_double_layer = check_double_layer_args(pomdp.qd, pomdp.qBi, pomdp.qCi, pomdp.lnE)

    No = pomdp.qA.shape[1] // 2
    Nx = pomdp.qs.shape[1] // 2  # Nx == Ns
    if is_double_layer:
        Ny = pomdp.qd.shape[1] // 2  # Ny == Nd

    x = pomdp.qs[:, :Nx]  # (T, Nx)
    W = logit(pomdp.qA.transpose(0, 2, 1))  # sig^-1(qA.T); (Nsess, Nx * 2, No * 2)
    W1 = W[:, :Nx, :No]  # (Nsess, Nx, No)
    W0 = W[:, Nx:, :No]  # (Nsess, Nx, No)
    phi1 = pomdp.lnD[:Nx]  # (Nx,)
    phi0 = pomdp.lnD[Nx:]  # (Nx,)

    if not is_double_layer:
        h1 = np.log(1 - expit(W1)) @ np.ones(No) + phi1  # (Nsess, Nx)
        h0 = np.log(1 - expit(W0)) @ np.ones(No) + phi0  # (Nsess, Nx)
        nn = NNVariables(x, W1, W0, phi1, phi0, h1, h0)
    else:
        y = pomdp.qd[:, :Ny]  # (T, Ny)
        K = logit(pomdp.qBi.transpose(0, 2, 1))  # sig^-1(qB.T); (Nsess, Nx * 2, Nx * 2)
        K1 = K[:, :Nx, :Nx]  # (Nsess, Nx, Nx)
        K0 = K[:, Nx:, :Nx]  # (Nx, Nx)
        V = logit(pomdp.qCi.T.transpose(0, 2, 1))  # sig^-1(qC.T); (Ny * 2, Nx * 2)
        V1 = V[:, :Ny, :Nx]  # (Nsess, Ny, Nx)
        V0 = V[:, Ny:, :Nx]  # (Nsess, Ny, Nx)
        psi1 = pomdp.lnE[:Ny]  # (Ny,)
        psi0 = pomdp.lnE[Ny:]  # (Ny,)
        h1 = np.log(1 - expit(W1)) @ np.ones(No) + np.log(1 - expit(K1)) @ np.ones(Nx) + phi1  # (Nsess, Nx)
        h0 = np.log(1 - expit(W0)) @ np.ones(No) + np.log(1 - expit(K0)) @ np.ones(Nx) + phi0  # (Nsess, Nx)
        m1 = np.log(1 - expit(V1)) @ np.ones(Nx) + psi1  # (Nsess, Ny)
        m0 = np.log(1 - expit(V0)) @ np.ones(Nx) + psi0  # (Nsess, Ny)
        nn = NNVariables(x, W1, W0, phi1, phi0, h1, h0, y, K1, K0, V1, V0, psi1, psi0, m1, m0)

    return nn


def nn2pomdp(nn: NNVariables) -> POMDPVariables:
    """Convert neural network variables to POMDP variables."""
    is_double_layer = check_double_layer_args(nn.y, nn.K1, nn.K0, nn.V1, nn.V0, nn.psi1, nn.psi0, nn.m1, nn.m0)

    qs = np.concatenate((nn.x, 1 - nn.x), axis=1)  # (T, Ns * 2,); Ns == Nx
    qA11 = expit(nn.W1).transpose(0, 2, 1)  # (Nsess, No, Ns)
    qA10 = expit(nn.W0).transpose(0, 2, 1)  # (Nsess, No, Ns)
    qA1 = np.concatenate((qA11, qA10), axis=2)  # (Nsess, No, Ns * 2)
    qA = np.concatenate((qA1, 1 - qA1), axis=1)  # (Nsess, No * 2, Ns * 2)
    lnD = np.concatenate((nn.phi1, nn.phi0))  # (Ns * 2,)

    if not is_double_layer:
        pomdp = POMDPVariables(qs, qA, lnD)
    else:
        qd = np.concatenate((nn.y, 1 - nn.y), axis=1)  # (T, Nd * 2,); Nd == Ny
        qBi11 = expit(nn.K1).transpose(0, 2, 1)  # (Nsess, Ns, Ns)
        qBi10 = expit(nn.K0).transpose(0, 2, 1)  # (Nsess, Ns, Ns)
        qBi1 = np.concatenate((qBi11, qBi10), axis=2)  # (Nsess, Ns, Ns * 2)
        qBi = np.concatenate((qBi1, 1 - qBi1), axis=1)  # (Nsess, Ns * 2, Ns * 2)
        qCi11 = expit(nn.V1).transpose(0, 2, 1)  # (Nsess, Ns, Nd)
        qCi10 = expit(nn.V0).transpose(0, 2, 1)  # (Nsess, Ns, Nd)
        qCi1 = np.concatenate((qCi11, qCi10), axis=2)  # (Nsess, Ns, Nd * 2)
        qCi = np.concatenate((qCi1, 1 - qCi1), axis=1)  # (Nsess, Ns * 2, Nd * 2)
        lnE = np.concatenate((nn.psi1, nn.psi0))  # (Nd * 2,)
        pomdp = POMDPVariables(qs, qA, lnD, qd, qBi, qCi, lnE)

    return pomdp
