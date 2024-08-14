import numpy as np


def compute_conditional_expectations(
    r: np.ndarray,
    s: np.ndarray,
    Nsession: int,
    session_len: int,
):
    """Compute conditional expectations of responses and sources.

    Args:
        r (np.ndarray): Neuronal responses (Nsession * session_len, Nr);
            Nr: Number of neuronal ensembles.
        s (np.ndarray): Hidden states (Nsession * session_len, Ns);
            Ns: Number of hidden states.
        Nsession (int): Number of sessions.
        session_len (int): Length of each session.

    Returns:
        tuple: Expected values of responses under different conditions.
    """
    assert r.shape[0] == Nsession * session_len
    assert s.shape[0] == Nsession * session_len
    Nr = r.shape[1]
    Ns = s.shape[1]

    r = r.reshape(session_len, Nsession, Nr, order="F")
    s = s.reshape(session_len, Nsession, Ns, order="F").astype(bool)

    r_ = r.mean(axis=0)
    r_s_11 = r[s[:, :, 0] & s[:, :, 1]].reshape(-1, Nsession, Nr).mean(axis=0)
    r_s_10 = r[s[:, :, 0] & ~s[:, :, 1]].reshape(-1, Nsession, Nr).mean(axis=0)
    r_s_01 = r[~s[:, :, 0] & s[:, :, 1]].reshape(-1, Nsession, Nr).mean(axis=0)
    r_s_00 = r[~s[:, :, 0] & ~s[:, :, 1]].reshape(-1, Nsession, Nr).mean(axis=0)
    r_s1_1 = r[s[:, :, 0]].reshape(-1, Nsession, Nr).mean(axis=0)
    r_s1_0 = r[~s[:, :, 0]].reshape(-1, Nsession, Nr).mean(axis=0)
    r_s2_1 = r[s[:, :, 1]].reshape(-1, Nsession, Nr).mean(axis=0)
    r_s2_0 = r[~s[:, :, 1]].reshape(-1, Nsession, Nr).mean(axis=0)

    return (
        r_,
        r_s_11,
        r_s_10,
        r_s_01,
        r_s_00,
        r_s1_1,
        r_s1_0,
        r_s2_1,
        r_s2_0,
    )
