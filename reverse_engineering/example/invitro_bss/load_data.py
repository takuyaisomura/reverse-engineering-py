from pathlib import Path

import numpy as np
import scipy.io

from reverse_engineering.utils.analysis_utils import compute_conditional_expectations

No = 32  # number of sensory stimuli
Nx = 2  # number of neuronal ensembles
Nsession = 100  # number of sessions
session_len = 256  # number of time steps per session
Ninit = 10  # number of initial sessions

# types of neuronal response data
datatypes = [
    "ctrl",  # control condition
    "bic",  # downregulated condition (Bicuculline)
    "dzp",  # upregulated condition (Diazepam)
    "mix0",  # 0% source mixed condition
    "mix50",  # 50% source mixed condition
]

data_dict = {}
baselines_dict = {}
plot_samp_idx_dict = {
    "ctrl": 18,
    "bic": 0,
    "dzp": 4,
    "mix0": 0,
    "mix50": 0,
}


def get_data_const():
    return datatypes, Nx, Nsession, session_len, Ninit


def load_data():
    # We need to load all datatypes to compute baseline
    # so we load all data here and keep them in global variables

    # load empirical neuronal response data
    global data_dict, baselines_dict
    data_dir = Path(__file__).parent / "data"
    data_dict = {
        datatype: scipy.io.loadmat(data_dir / f"response_data_{datatype}.mat", simplify_cells=True)[f"data_{datatype}"]
        for datatype in datatypes
    }

    # compute baseline excitability
    baselines_dict = compute_baseline_excitability(data_dict, Nsession, session_len)


def get_data(datatype: str):
    return data_dict[datatype], baselines_dict[datatype], plot_samp_idx_dict[datatype]


def compute_baseline_excitability(
    data_dict: dict,
    Nsession: int,
    session_len: int,
) -> dict:
    mean_resp1 = []
    mean_resp2 = []
    mean_resp3 = []
    mean_resp4 = []

    for datatype in datatypes:
        data = data_dict[datatype]

        for idx_samp, samp in enumerate(data):
            # read sources and responses
            s = samp["s"]  # hidden sources (T, Ns)
            r = samp["r"]  # neuronal responses (T, Nr)

            # compute ensemble averages
            # categorise into sources 1- and 2-preferring ensembles
            r_, _, r_s_10, r_s_01, _, _, _, _, _ = compute_conditional_expectations(r, s, Nsession, session_len)
            g1 = np.where((r_.mean(axis=0) > 1) & (r_.min(axis=0) > 0.1) & ((r_s_10 - r_s_01).mean(axis=0) > 0.5))[0]
            g2 = np.where((r_.mean(axis=0) > 1) & (r_.min(axis=0) > 0.1) & ((r_s_10 - r_s_01).mean(axis=0) < -0.5))[0]
            if len(g1) == 1:
                g1 = [g1[0], g1[0]]
            if len(g2) == 1:
                g2 = [g2[0], g2[0]]

            # We separated the control group into two groups:
            # one obtained in previous work (Isomura et al., 2015)
            # and one newly obtained for this work because of different noise and excitability levels in these groups.
            x_ = np.vstack((r_[:, g1].mean(axis=1), r_[:, g2].mean(axis=1)))
            mean_resp = x_[:, :Ninit].mean(axis=1)
            if datatype == "ctrl" and idx_samp <= 22:
                mean_resp1.append(mean_resp)
            elif datatype == "ctrl" and idx_samp >= 23:
                mean_resp2.append(mean_resp)
            elif datatype == "bic":
                mean_resp2.append(mean_resp)
            elif datatype == "dzp":
                mean_resp2.append(mean_resp)
            elif datatype == "mix0":
                mean_resp3.append(mean_resp)
            elif datatype == "mix50":
                mean_resp4.append(mean_resp)

    # compute average baseline excitability in each condition
    baseline_dict = {
        "ctrl": np.concatenate((np.ones(23) * np.mean(mean_resp1), np.ones(7) * np.mean(mean_resp2))),
        "bic": np.ones(len(data_dict["bic"])) * np.mean(mean_resp2),
        "dzp": np.ones(len(data_dict["dzp"])) * np.mean(mean_resp2),
        "mix0": np.ones(len(data_dict["mix0"])) * np.mean(mean_resp3),
        "mix50": np.ones(len(data_dict["mix50"])) * np.mean(mean_resp4),
    }

    return baseline_dict


def get_ideal_likelihood_mapping(datatype: str) -> tuple[np.ndarray, np.ndarray]:
    # ideal Bayesian posterior belief about A matrix
    if datatype == "mix0":
        qA11id = np.repeat([[1, 0.5], [0.5, 1]], No // 2, axis=0)
        qA10id = np.repeat([[0, 0.5], [0.5, 0]], No // 2, axis=0)
    elif datatype == "mix50":
        qA11id = np.repeat([[0.75, 0.75], [0.75, 0.75]], No // 2, axis=0)
        qA10id = np.repeat([[0.25, 0.25], [0.25, 0.25]], No // 2, axis=0)
    else:
        qA11id = np.repeat([[0.875, 0.625], [0.625, 0.875]], No // 2, axis=0)
        qA10id = np.repeat([[0.125, 0.375], [0.375, 0.125]], No // 2, axis=0)

    return qA11id, qA10id
