from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import binned_statistic

from reverse_engineering.analysis import (
    compute_prediction_errors,
    estimate_firing_threshold_factor,
    estimate_from_responses,
    normalize_responses,
    predict_from_initial_responses,
)

from dataset import get_datasets, InVitroBSSDataset, Ninit, Nx, Nsession, session_len  # isort: skip
from plot_summary_figure import plot_summary_figure  # isort: skip
from plot_weights_trajectory import plot_weights_trajectory  # isort: skip


Npost = 10  # number of "post" sessions for statistics
lambda_ = 3000  # prior strength (insensitivity to plasticity)
gain = 2  # relative strength of initial Hebb and Home for prediction
show_video = True  # show animations and save them as videos


@dataclass
class SampleResults:
    """Analysis results for a single sample"""

    o: np.ndarray  # sensory stimuli (No, T)
    x: np.ndarray  # normalized responses (Nx, T)
    xp: np.ndarray  # predicted responses (Nx, T)
    W: np.ndarray  # synaptic weights (Nx, No, Nsession)
    Wp: np.ndarray  # predicted synaptic weights (Nx, No, Nsession)
    W_hat: np.ndarray  # sig([W1' W0'])' (No * 2, Nx, Nsession)
    Wp_hat: np.ndarray  # sig([W1p' W0p'])' (No * 2, Nx, Nsession)
    qA1: np.ndarray  # empirical posterior belief about A1 (No, Nx, 4)
    phi1: float  # firing threshold factor
    phi0: float  # firing threshold factor
    L: np.ndarray  # cost function values (Nsession,)


@dataclass
class DatasetResults:
    """Analysis results for a dataset of a single datatype"""

    x1_list: np.ndarray  # response of source 1-preferring ensemble (n_samp, Nsession, Nx)
    x_post: np.ndarray  # observed responses in last 10 sessions (n_samp, Nx, session_len * Npost)
    xp_post: np.ndarray  # predicted responses in last 10 sessions (n_samp, Nx, session_len * Npost)
    err_x_xp_list: np.ndarray  # error between observed and predicted responses (n_samp, Nsession)
    err_W_Wp_list: np.ndarray  # error between estimated and predicted synaptic weights (n_samp, Nsession)
    err_W_qA_list: np.ndarray  # error between empirical and ideal posteriors (n_samp * 4, Nsession)
    L_list: np.ndarray  # values of cost function (n_samp, Nsession)


def main(out_dir: Path | str | None = None) -> None:
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create figure to store all plots
    fig = plt.figure(figsize=(25, 23))

    # Analyze and visualize each dataset condition
    for i, dataset in enumerate(get_datasets()):
        print(f"Analyzing {dataset.datatype} condition...")
        analysis_results, sample_results = analyze_dataset(dataset)

        fig_pos = i + 1  # 1-indexed
        visualize_dataset(analysis_results, sample_results, dataset.datatype, fig, fig_pos, out_dir)

    fig.savefig(out_dir / "fig.png")
    fig.show()


def analyze_dataset(dataset: InVitroBSSDataset) -> tuple[DatasetResults, SampleResults]:
    """Analyze a single dataset condition."""

    # Initialize results container
    n_samp = len(dataset.data)
    res = DatasetResults(
        x1_list=np.zeros((n_samp, Nsession, Nx)),  # response of source 1-preferring ensemble
        x_post=np.zeros((n_samp, Nx, session_len * Npost)),  # observed responses in last 10 sessions
        xp_post=np.zeros((n_samp, Nx, session_len * Npost)),  # predicted responses in last 10 sessions
        err_x_xp_list=np.zeros((n_samp, Nsession)),  # error between observed and predicted responses
        err_W_Wp_list=np.zeros((n_samp, Nsession)),  # error between estimated and predicted synaptic weights
        err_W_qA_list=np.zeros((n_samp * 4, Nsession)),  # save error between empirical and ideal posteriors
        L_list=np.zeros((n_samp, Nsession)),  # values of cost function
    )

    # analyze each sample
    selected_sample_res = None  # will store results for the selected sample for visualization
    for idx_samp, samp in enumerate(dataset.data):
        # read sources, stimuli, and responses
        s = samp["s"].T  # hidden sources (Ns, T)
        o = samp["o"].T  # sensory stimuli (No, T)
        r = samp["r"].T  # neuronal responses (Nr, T)
        sample_res = analyze_sample(s, o, r, dataset.baseline[idx_samp])

        # store results for the selected sample
        if idx_samp == dataset.sample_index:
            selected_sample_res = sample_res

        # response of source 1-preferring ensemble
        res.x1_list[idx_samp, :, 0] = sample_res.x[0, s[0] == 1].reshape(-1, Nsession, order="F").mean(axis=0)
        res.x1_list[idx_samp, :, 1] = sample_res.x[0, s[0] == 0].reshape(-1, Nsession, order="F").mean(axis=0)

        # observed/predicted responses in last 10 sessions
        res.x_post[idx_samp, :, : session_len * Npost] = sample_res.x[:, -session_len * Npost :]
        res.xp_post[idx_samp, :, : session_len * Npost] = sample_res.xp[:, -session_len * Npost :]

        res.L_list[idx_samp] = sample_res.L

        qA11id, qA10id = dataset.ideal_likelihood
        (
            res.err_x_xp_list[idx_samp],
            res.err_W_qA_list[idx_samp * 4 : (idx_samp + 1) * 4],
            res.err_W_Wp_list[idx_samp],
        ) = compute_prediction_errors(sample_res.x, sample_res.xp, sample_res.W_hat, sample_res.Wp_hat, qA11id, qA10id)

    if selected_sample_res is None:
        raise ValueError(f"Sample index {dataset.sample_index} not found in dataset")

    return res, selected_sample_res


def visualize_dataset(
    dataset_results: DatasetResults,
    sample_results: SampleResults,
    datatype: str,
    fig: plt.Figure,
    fig_pos: int,
    out_dir: Path,
):
    """Visualize dataset analysis results."""
    # compute relationship between observed (x_post) and predicted (xp_post) responses during last 10 sessions
    # divide into 25 bins according to the value
    x_post = dataset_results.x_post.ravel()
    xp_post = dataset_results.xp_post.ravel()
    x_xp_mean, *_ = binned_statistic(x_post, xp_post, "mean", bins=25, range=(0, 1))
    x_xp_std, *_ = binned_statistic(x_post, xp_post, "std", bins=25, range=(0, 1))

    plot_summary_figure(
        fig,
        fig_pos,
        datatype,
        dataset_results.x1_list,
        x_xp_mean,
        x_xp_std,
        dataset_results.err_x_xp_list,
        sample_results.W,
        sample_results.Wp,
        dataset_results.err_W_Wp_list,
        dataset_results.err_W_qA_list,
        dataset_results.L_list,
    )

    if show_video:
        line_width = 6 if datatype == "mix0" else 3
        plot_weights_trajectory(
            sample_results.W,
            sample_results.Wp,
            sample_results.o,
            sample_results.phi1,
            sample_results.phi0,
            out_dir,
            datatype,
            Ninit,
            line_width=line_width,
        )


def analyze_sample(
    s: np.ndarray,
    o: np.ndarray,
    r: np.ndarray,
    baseline: float,
) -> SampleResults:
    """Analyze a single sample and return the results."""
    x = normalize_responses(r, s, baseline, Nsession, session_len)  # (Nx, T)

    # estimate firing threshold factor from initial responses
    x_init = x[:, : Ninit * session_len]
    phi1, phi0 = estimate_firing_threshold_factor(x_init)

    # estimate synaptic weights and cost function from responses
    W1, W0, L = estimate_from_responses(x, o, phi1, phi0, Nsession, lambda_)

    # predict responses and synaptic weights from initial responses & weights
    W1_init = W1[:, :, :Ninit]
    W0_init = W0[:, :, :Ninit]
    xp, W1p, W0p = predict_from_initial_responses(x_init, o, W1_init, W0_init, phi1, phi0, session_len, lambda_, gain)

    # compute empirical posterior belief
    qA11 = expit(W1[:, :, -1]).T  # A_11 (No, Nx)
    qA10 = expit(W0[:, :, -1]).T  # A_10 (No, Nx)
    qA1 = np.stack(
        (
            qA11[:, 0] * qA11[:, 1],
            qA11[:, 0] * qA10[:, 1],
            qA10[:, 0] * qA11[:, 1],
            qA10[:, 0] * qA10[:, 1],
        ),
        axis=-1,
    )  # A_1; (No, Nx, 4)

    W = W1 - W0  # sum of excitatory (W1) and inhibitory (W0) synaptic strengths
    Wp = W1p - W0p  # sum of predicted excitatory (W1p) and inhibitory (W0p) synaptic strengths
    W_hat = expit(np.vstack((W1, W0)))  # W_hat = sig([W1' W0'])'; (No * 2, Nx, Nsession)
    Wp_hat = expit(np.vstack((W1p, W0p)))  # Wp_hat = sig([W1p' W0p'])'; (No * 2, Nx, Nsession)

    return SampleResults(o, x, xp, W, Wp, W_hat, Wp_hat, qA1, phi1, phi0, L)


if __name__ == "__main__":
    main()
