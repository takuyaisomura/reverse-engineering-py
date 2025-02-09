import numpy as np

from reverse_engineering.utils.analysis_utils import compute_conditional_expectations

from dataset import InVitroBSSDataset, Ninit, Nsession, session_len  # isort: skip


def compute_baselines(dataset_list: list[InVitroBSSDataset]) -> dict[str, np.ndarray]:
    """Compute baselines for all datasets

    Args:
        dataset_list: List of dataset

    Returns:
        Dictionary mapping dataset type to its baseline
    """
    # Initialize containers for mean responses
    mean_resp_old_lab = []
    mean_resp_new_lab = []
    mean_resp_mix0 = []
    mean_resp_mix50 = []
    data_lengths = {}

    # Compute baselines for each dataset
    for dataset in dataset_list:
        datatype = dataset.datatype
        data_lengths[datatype] = len(dataset.data)

        # Compute mean responses for each sample
        for idx_samp, samp in enumerate(dataset.data):
            # read sources and responses
            s = samp["s"]  # hidden sources (T, Ns)
            r = samp["r"]  # neuronal responses (T, Nr)

            # compute ensemble averages
            r_, _, r_s_10, r_s_01, _, _, _, _, _ = compute_conditional_expectations(r, s, Nsession, session_len)
            g1 = np.where((r_.mean(axis=0) > 1) & (r_.min(axis=0) > 0.1) & ((r_s_10 - r_s_01).mean(axis=0) > 0.5))[0]
            g2 = np.where((r_.mean(axis=0) > 1) & (r_.min(axis=0) > 0.1) & ((r_s_10 - r_s_01).mean(axis=0) < -0.5))[0]

            if len(g1) == 1:
                g1 = np.array([g1[0], g1[0]])
            if len(g2) == 1:
                g2 = np.array([g2[0], g2[0]])

            x_ = np.vstack((r_[:, g1].mean(axis=1), r_[:, g2].mean(axis=1)))
            mean_resp = x_[:, :Ninit].mean(axis=1)

            if datatype == "ctrl" and idx_samp <= 22:
                mean_resp_old_lab.append(mean_resp)
            elif datatype == "ctrl" and idx_samp >= 23:
                mean_resp_new_lab.append(mean_resp)
            elif datatype in ["bic", "dzp"]:
                mean_resp_new_lab.append(mean_resp)
            elif datatype == "mix0":
                mean_resp_mix0.append(mean_resp)
            elif datatype == "mix50":
                mean_resp_mix50.append(mean_resp)

    # Compute baselines
    return {
        "ctrl": np.concatenate((np.ones(23) * np.mean(mean_resp_old_lab), np.ones(7) * np.mean(mean_resp_new_lab))),
        "bic": np.ones(data_lengths["bic"]) * np.mean(mean_resp_new_lab),
        "dzp": np.ones(data_lengths["dzp"]) * np.mean(mean_resp_new_lab),
        "mix0": np.ones(data_lengths["mix0"]) * np.mean(mean_resp_mix0),
        "mix50": np.ones(data_lengths["mix50"]) * np.mean(mean_resp_mix50),
    }
