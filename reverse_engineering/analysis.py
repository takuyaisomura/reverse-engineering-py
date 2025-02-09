import numpy as np
from scipy.special import expit, logit

from reverse_engineering.utils.analysis_utils import compute_conditional_expectations


def normalize_responses(
    r: np.ndarray,
    s: np.ndarray,
    baseline: float,
    Nsession: int,
    session_len: int,
):
    """Normalize the responses.

    Args:
        r (np.ndarray): Neuronal responses (Nr, Nsession * session_len); Nr is the number of neurons.
        s (np.ndarray): Hidden states (Ns, Nsession * session_len); Ns is the number of hidden states.
        baseline (float): Baseline excitability of the neurons.
        Nsession (int): Number of sessions.
        session_len (int): Number of time steps per session.

    Returns:
        np.ndarray: Normalized responses (Nx, Nsession * session_len); Nx is the number of neuronal ensembles.
    """
    # categorise into sources 1- and 2-preferring ensembles
    r_, r_s_11, r_s_10, r_s_01, r_s_00, _, _, _, _ = compute_conditional_expectations(r.T, s.T, Nsession, session_len)
    g1 = np.where(((r_.mean(axis=0) > 1) & (r_.min(axis=0) > 0.1) & ((r_s_10 - r_s_01).mean(axis=0) > 0.5)))[0]
    g2 = np.where(((r_.mean(axis=0) > 1) & (r_.min(axis=0) > 0.1) & ((r_s_10 - r_s_01).mean(axis=0) < -0.5)))[0]
    if len(g1) == 1:
        g1 = np.array([g1[0], g1[0]])
    if len(g2) == 1:
        g2 = np.array([g2[0], g2[0]])

    # compute ensemble responses (Nx, Nsession * session_len)
    x = np.vstack((r[g1, :].mean(axis=0), r[g2, :].mean(axis=0)))
    # compute mean response intensity in each session (Nx, Nsession)
    x_ = np.vstack((r_[:, g1].mean(axis=1), r_[:, g2].mean(axis=1)))
    # compute conditional ensemble responses given s
    x_s_11 = np.vstack((r_s_11[:, g1].mean(axis=1), r_s_11[:, g2].mean(axis=1)))  # s = (1,1)
    x_s_10 = np.vstack((r_s_10[:, g1].mean(axis=1), r_s_10[:, g2].mean(axis=1)))  # s = (1,0)
    x_s_01 = np.vstack((r_s_01[:, g1].mean(axis=1), r_s_01[:, g2].mean(axis=1)))  # s = (0,1)
    x_s_00 = np.vstack((r_s_00[:, g1].mean(axis=1), r_s_00[:, g2].mean(axis=1)))  # s = (0,0)

    # normalise ensemble responses
    # subtract initial values to remove the stimulus-specific components
    x[:, (s[0, :] == 1) & (s[1, :] == 1)] -= x_s_11[:, 0][:, np.newaxis]
    x[:, (s[0, :] == 1) & (s[1, :] == 0)] -= x_s_10[:, 0][:, np.newaxis]
    x[:, (s[0, :] == 0) & (s[1, :] == 1)] -= x_s_01[:, 0][:, np.newaxis]
    x[:, (s[0, :] == 0) & (s[1, :] == 0)] -= x_s_00[:, 0][:, np.newaxis]
    # subtract mean response (trend) in each session
    x -= np.kron(x_ - x_[:, 0][:, np.newaxis], np.ones(session_len))
    # normalise to zero mean and unit variance
    x = np.linalg.inv(np.diag(x.std(axis=1, ddof=1))) @ (x - x.mean(axis=1)[:, np.newaxis])
    # add baseline excitability for each sample (relative value)
    x = 0.5 + x / 2 + (x_[:, 0][:, np.newaxis] - baseline) / baseline / 4
    # normalise in the range between 0 and 1
    x = np.clip(x, 0, 1)

    return x


def estimate_firing_threshold_factor(x: np.ndarray):
    """Estimate the firing threshold factors.

    Args:
        x (np.ndarray): Normalized responses (Nx, Nsession * session_len); Nx is the number of neuronal ensembles.

    Returns:
        np.ndarray: Firing threshold factors (Nx, 1).
    """
    # estimate firing threshold factor based on initial empirical data
    phi1 = np.log(x.mean(axis=1, keepdims=True))  # firing threshold factor phi1 = log(<x>)
    phi0 = np.log(1 - x.mean(axis=1, keepdims=True))  # firing threshold factor phi0 = log(1-<x>)
    return phi1, phi0


def estimate_from_responses(
    x: np.ndarray,
    o: np.ndarray,
    phi1: np.ndarray,
    phi0: np.ndarray,
    Nsession: int,
    lambda_: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate synaptic weights and cost function from neuronal responses.

    Args:
        x (np.ndarray): Normalized responses (Nx, Nsession * session_len);
            Nx is the number of neuronal ensembles.
        o (np.ndarray): Observations (No, Nsession * session_len)
            No is the number of observations.
        phi1 (np.ndarray): Firing threshold factors of excitatory neurons (Nx, 1)
        phi0 (np.ndarray): Firing threshold factors of inhibitory neurons (Nx, 1)
        Nsession (int): Number of sessions.
        lambda_ (float): Inverse learning rate factor (prior strength).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        - Estimated excitatory synaptic weights per session (Nx, No, Nsession).
        - Estimated inhibitory synaptic weights per session (Nx, No, Nsession).
        - Cost function per session (Nsession,).
    """
    # estimation of effective synaptic connectivity
    Nx = x.shape[0]
    No = o.shape[0]
    session_len = x.shape[1] // Nsession

    W1 = np.zeros((Nx, No, Nsession))  # estimated synaptic weights
    W0 = np.zeros((Nx, No, Nsession))  # estimated synaptic weights
    L = np.zeros(Nsession)  # cost function (= variational free energy)
    Hebb1 = np.ones((Nx, No)) * lambda_ / 2  # matrix for Hebbian product
    Hebb0 = np.ones((Nx, No)) * lambda_ / 2  # matrix for Hebbian product
    Home1 = np.ones((Nx, No)) * lambda_  # matrix for homeostatic term
    Home0 = np.ones((Nx, No)) * lambda_  # matrix for homeostatic term
    for sess in range(Nsession):
        ts = slice(session_len * sess, session_len * (sess + 1))  # time steps in session
        W1[:, :, sess] = logit(Hebb1 / Home1)  # update synaptic weights
        W0[:, :, sess] = logit(Hebb0 / Home0)  # update synaptic weights
        h1 = np.log(1 - expit(W1[:, :, sess])) @ np.ones((No, 1)) + phi1  # compute firing threshold
        h0 = np.log(1 - expit(W0[:, :, sess])) @ np.ones((No, 1)) + phi0  # compute firing threshold
        Hebb1 += x[:, ts] @ o[:, ts].T  # compute synaptic plasticity
        Hebb0 += (1 - x[:, ts]) @ o[:, ts].T  # compute synaptic plasticity
        Home1 += x[:, ts] @ np.ones((No, session_len)).T  # compute synaptic plasticity
        Home0 += (1 - x[:, ts]) @ np.ones((No, session_len)).T  # compute synaptic plasticity
        L[sess] = np.sum(
            np.vstack((x[:, ts], 1 - x[:, ts]))
            * (
                np.log(np.vstack((x[:, ts], 1 - x[:, ts])) + 1e-6)
                - np.vstack((W1[:, :, sess], W0[:, :, sess])) @ o[:, ts]
                - np.vstack((h1, h0))
            )
        )  # compute cost function
    return W1, W0, L


def predict_from_initial_responses(
    x_init: np.ndarray,
    o: np.ndarray,
    W1_init: np.ndarray,
    W0_init: np.ndarray,
    phi1: np.ndarray,
    phi0: np.ndarray,
    session_len: int,
    lambda_: float,
    gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate synaptic weights and cost function from initial neuronal responses.

    Args:
        x_init (np.ndarray): Normalized responses in the initial sessions (Nx, Ninit * session_len);
            Nx is the number of neuronal ensembles.
            Ninit is the number of initial sessions.
        o (np.ndarray): Observations (No, Nsession * session_len)
            No is the number of observations.
        W1_init (np.ndarray): Excitatory synaptic weights in the initial sessions (Nx, No, Ninit)
        W0_init (np.ndarray): Inhibitory synaptic weights in the initial sessions (Nx, No, Ninit)
        phi1 (np.ndarray): Firing threshold factors of excitatory neurons (Nx, 1)
        phi0 (np.ndarray): Firing threshold factors of inhibitory neurons (Nx, 1)
        session_len (int): Number of time steps per session.
        lambda_ (float): Inverse learning rate factor (prior strength).
        gain (float): Relative strength of initial Hebb and Home.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        - Predicted neuronal responses (Nx, Nsession * session_len).
        - Predicted excitatory synaptic weights (Nx, No, Nsession).
        - Predicted inhibitory synaptic weights (Nx, No, Nsession).
    """
    # prediction of neuronal responses and effective synaptic connectivity
    Nx = x_init.shape[0]
    No = o.shape[0]
    Nsession = o.shape[1] // session_len
    Ninit = x_init.shape[1] // session_len

    xp = np.zeros((Nx, Nsession * session_len))  # predicted neuronal responses
    W1p = np.zeros((Nx, No, Nsession))  # predicted synaptic weights
    W0p = np.zeros((Nx, No, Nsession))  # predicted synaptic weights
    Hebb1 = np.ones((Nx, No)) * lambda_ * gain / 2  # matrix for Hebbian product
    Hebb0 = np.ones((Nx, No)) * lambda_ * gain / 2  # matrix for Hebbian product
    Home1 = np.ones((Nx, No)) * lambda_ * gain  # matrix for homeostatic term
    Home0 = np.ones((Nx, No)) * lambda_ * gain  # matrix for homeostatic term

    for sess in range(Ninit):
        ts = slice(session_len * sess, session_len * (sess + 1))  # select data in session t
        W1p[:, :, sess] = W1_init[:, :, sess]  # update synaptic weights
        W0p[:, :, sess] = W0_init[:, :, sess]  # update synaptic weights
        h1 = np.log(1 - expit(W1p[:, :, sess])) @ np.ones((No, 1)) + phi1  # compute firing threshold
        h0 = np.log(1 - expit(W0p[:, :, sess])) @ np.ones((No, 1)) + phi0  # compute firing threshold
        xp[:, ts] = expit(
            (W1p[:, :, sess] - W0p[:, :, sess]) @ o[:, ts] + h1 - h0
        )  # compute predicted neuronal responses
        Hebb1 += x_init[:, ts] @ o[:, ts].T * gain  # compute synaptic plasticity
        Hebb0 += (1 - x_init[:, ts]) @ o[:, ts].T * gain  # compute synaptic plasticity
        Home1 += x_init[:, ts] @ np.ones((No, session_len)).T * gain  # compute synaptic plasticity
        Home0 += (1 - x_init[:, ts]) @ np.ones((No, session_len)).T * gain  # compute synaptic plasticity

    for sess in range(Ninit, Nsession):
        ts = slice(session_len * sess, session_len * (sess + 1))  # select data in session t
        W1p[:, :, sess] = logit(Hebb1 / Home1)  # update synaptic weights
        W0p[:, :, sess] = logit(Hebb0 / Home0)  # update synaptic weights
        h1 = np.log(1 - expit(W1p[:, :, sess])) @ np.ones((No, 1)) + phi1  # compute firing threshold
        h0 = np.log(1 - expit(W0p[:, :, sess])) @ np.ones((No, 1)) + phi0  # compute firing threshold
        xp[:, ts] = expit(
            (W1p[:, :, sess] - W0p[:, :, sess]) @ o[:, ts] + h1 - h0
        )  # compute predicted neuronal responses
        Hebb1 += xp[:, ts] @ o[:, ts].T  # compute synaptic plasticity
        Hebb0 += (1 - xp[:, ts]) @ o[:, ts].T  # compute synaptic plasticity
        Home1 += xp[:, ts] @ np.ones((No, session_len)).T  # compute synaptic plasticity
        Home0 += (1 - xp[:, ts]) @ np.ones((No, session_len)).T  # compute synaptic plasticity

    return xp, W1p, W0p


def compute_prediction_errors(
    x: np.ndarray,
    xp: np.ndarray,
    W_hat: np.ndarray,
    Wp_hat: np.ndarray,
    qA11_id: np.ndarray,
    qA10_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute prediction errors.

    Args:
        x (np.ndarray): Observed neuronal responses (Nx, Nsession * session_len);
            Nx is the number of neuronal ensembles.
        xp (np.ndarray): Predicted neuronal responses (Nx, Nsession * session_len)
        W_hat (np.ndarray): Estimated synaptic weights with sigmoid applied (Nx, No, Nsession)
        Wp_hat (np.ndarray): Predicted synaptic weights from initial sessions with sigmoid applied (Nx, No, Nsession)
        qA11_id (np.ndarray): Ideal bayesian posterior belief about A matrix (s=1) (Nx, No, Nsession)
        qA10_id (np.ndarray): Ideal bayesian posterior belief about A matrix (s=0) (Nx, No, Nsession)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        - Prediction error between observed and predicted responses (Nsession,).
        - Prediction error between empirical and ideal posteriors (4, Nsession).
        - Prediction error between estimated and predicted synaptic weights (Nsession,).
    """
    Nsession = W_hat.shape[2]
    session_len = x.shape[1] // Nsession
    qA1_id = np.hstack((qA11_id, qA10_id))

    # compute prediction errors
    err_x_xp = np.zeros(Nsession)
    err_W_qA = np.zeros((4, Nsession))
    err_W_Wp = np.zeros(Nsession)
    for sess in range(Nsession):
        # select data in session t
        ts = slice(session_len * sess, session_len * (sess + 1))
        # error between observed and predicted responses
        err_x_xp[sess] = np.mean(np.square(x[:, ts] - xp[:, ts]))
        # error between empirical and ideal posteriors
        err_W_qA[:, sess] = np.square(W_hat[:, :, sess].T - qA1_id).mean(axis=0) / np.square(qA1_id).mean()
        # error between estimated and predicted synaptic weights
        err_W_Wp[sess] = np.sum(np.square(W_hat[:, :, sess] - Wp_hat[:, :, sess])) / np.sum(
            np.square(W_hat[:, :, sess])
        )

    return err_x_xp, err_W_qA, err_W_Wp
