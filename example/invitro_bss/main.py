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

from .load_data import get_data, get_data_const, get_ideal_likelihood_mapping, load_data
from .plot_summary_figure import plot_summary_figure
from .plot_weights_trajectory import plot_weights_trajectory

Npost = 10  # number of "post" sessions for statistics
lambda_ = 3000  # prior strength (insensitivity to plasticity)
gain = 2  # relative strength of initial Hebb and Home for prediction
SHOWVIDEO = 1  # show animations and save them as videos

datatypes, Nx, Nsession, session_len, Ninit = get_data_const()


def main(out_dir: Path | str | None = None) -> None:
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    load_data()  # load neuronal responses data
    fig = plt.figure(figsize=(25, 23))  # share between datatypes

    for datatype in datatypes:
        print(f"{datatype} condition")

        # get data
        data, baselines, plot_samp_idx = get_data(datatype)
        qA11id, qA10id = get_ideal_likelihood_mapping(datatype)

        # result containers
        n_samp = len(data)
        x1_list = np.zeros((n_samp, Nsession, Nx))  # response of source 1-preferring ensemble
        x_post = np.zeros((n_samp, Nx, session_len * Npost))  # observed responses in last 10 sessions
        xp_post = np.zeros((n_samp, Nx, session_len * Npost))  # predicted responses in last 10 sessions
        err_x_xp_list = np.zeros((n_samp, Nsession))  # error between observed and predicted responses
        err_W_Wp_list = np.zeros((n_samp, Nsession))  # error between estimated and predicted synaptic weights
        err_W_qA_list = np.zeros((n_samp * 4, Nsession))  # save error between empirical and ideal posteriors
        L_list = np.zeros((n_samp, Nsession))  # values of cost function
        plot_vars = {}

        # analyze each sample
        for idx_samp, samp in enumerate(data):
            # read sources, stimuli, and responses
            s = samp["s"].T  # hidden sources (Ns, T)
            o = samp["o"].T  # sensory stimuli (No, T)
            r = samp["r"].T  # neuronal responses (Nr, T)
            x, xp, W, Wp, W_hat, Wp_hat, qA1, phi1, phi0, L = analyze_sample(s, o, r, baselines[idx_samp])

            if idx_samp == plot_samp_idx:
                plot_vars["W"] = W
                plot_vars["Wp"] = Wp
                plot_vars["qA1"] = qA1
                plot_vars["phi1"] = phi1
                plot_vars["phi0"] = phi0

            # response of source 1-preferring ensemble
            x1_list[idx_samp, :, 0] = x[0, s[0] == 1].reshape(-1, Nsession, order="F").mean(axis=0)
            x1_list[idx_samp, :, 1] = x[0, s[0] == 0].reshape(-1, Nsession, order="F").mean(axis=0)

            # observed/predicted responses in last 10 sessions
            x_post[idx_samp, :, : session_len * Npost] = x[:, -session_len * Npost :]
            xp_post[idx_samp, :, : session_len * Npost] = xp[:, -session_len * Npost :]

            L_list[idx_samp] = L

            (
                err_x_xp_list[idx_samp],
                err_W_qA_list[idx_samp * 4 : (idx_samp + 1) * 4],
                err_W_Wp_list[idx_samp],
            ) = compute_prediction_errors(x, xp, W_hat, Wp_hat, qA11id, qA10id)

        # visualize
        # compute relationship between observed (x_post) and predicted (xp_post) responses during last 10 sessions
        # divide into 25 bins according to the value
        x_xp_mean, *_ = binned_statistic(x_post.ravel(), xp_post.ravel(), "mean", bins=25, range=(0, 1))
        x_xp_std, *_ = binned_statistic(x_post.ravel(), xp_post.ravel(), "std", bins=25, range=(0, 1))

        fig_pos = datatypes.index(datatype) + 1
        plot_summary_figure(
            fig,
            datatype,
            fig_pos,
            x1_list,
            x_xp_mean,
            x_xp_std,
            err_x_xp_list,
            plot_vars["W"],
            plot_vars["Wp"],
            err_W_Wp_list,
            err_W_qA_list,
            L_list,
        )

        if SHOWVIDEO:
            plot_weights_trajectory(
                out_dir,
                Ninit,
                o,
                plot_vars["W"],
                plot_vars["Wp"],
                plot_vars["phi1"],
                plot_vars["phi0"],
                data_name=datatype,
                line_width=6 if datatype == "mix0" else 3,
            )

    fig.savefig(out_dir / "fig.png")
    fig.show()


def analyze_sample(
    s: np.ndarray,
    o: np.ndarray,
    r: np.ndarray,
    baseline: float,
):
    x = normalize_responses(r, s, baseline, Nsession, session_len)  # (Nx, T)

    x_init = x[:, : Ninit * session_len]
    phi1, phi0 = estimate_firing_threshold_factor(x_init)

    W1, W0, L = estimate_from_responses(
        x,
        o,
        phi1,
        phi0,
        Nsession,
        lambda_,
    )

    xp, W1p, W0p = predict_from_initial_responses(
        x_init,
        o,
        W1[:, :, :Ninit],
        W0[:, :, :Ninit],
        phi1,
        phi0,
        session_len,
        lambda_,
        gain,
    )

    qA11 = expit(W1[:, :, -1]).T  # compute empirical posterior belief about A11; (No, Nx)
    qA10 = expit(W0[:, :, -1]).T  # compute empirical posterior belief about A10; (No, Nx)
    qA1 = np.stack(
        (
            qA11[:, 0] * qA11[:, 1],
            qA11[:, 0] * qA10[:, 1],
            qA10[:, 0] * qA11[:, 1],
            qA10[:, 0] * qA10[:, 1],
        ),
        axis=-1,
    )  # compute empirical posterior belief about A1; (No, Nx, 4)

    W = W1 - W0  # sum of excitatory (W1) and inhibitory (W0) synaptic strengths
    Wp = W1p - W0p  # sum of predicted excitatory (W1p) and inhibitory (W0p) synaptic strengths
    W_hat = expit(np.vstack((W1, W0)))  # W_hat = sig([W1' W0'])'; (No * 2, Nx, Nsession)
    Wp_hat = expit(np.vstack((W1p, W0p)))  # Wp_hat = sig([W1p' W0p'])'; (No * 2, Nx, Nsession)

    return (
        x,
        xp,
        W,
        Wp,
        W_hat,
        Wp_hat,
        qA1,
        phi1,
        phi0,
        L,
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
