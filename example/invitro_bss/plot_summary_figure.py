import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
    }
)


def plot_summary_figure(
    fig: plt.Figure,
    fig_pos: int,
    datatype: str,
    x1_list: np.ndarray,  # (n_samp, Nsession, Nx)
    x_xp_mean: np.ndarray,  # (n_bins,)
    x_xp_std: np.ndarray,  # (n_bins,)
    err_x_xp_list: np.ndarray,  # (n_samp, Nsession)
    W: np.ndarray,  # (Nx, No, Nsession)
    Wp: np.ndarray,  # (Nx, No, Nsession)
    err_W_Wp_list: np.ndarray,  # (n_samp, Nsession)
    err_W_qA_list: np.ndarray,  # (n_samp * 4, Nsession)
    L_list: np.ndarray,  # (n_samp, Nsession)
):
    Nsession = x1_list.shape[1]
    sess = np.arange(Nsession) + 1
    sess_ticks = np.arange(0, Nsession + 1, Nsession // 2)
    resp_ticks = np.arange(0, 1.1, 1 / 2)

    # response of source 1-preferring ensemble
    ax = fig.add_subplot(6, 5, fig_pos + 0)
    # when s^(1) = 0
    mean = x1_list[:, :, 1].mean(axis=0)
    std = x1_list[:, :, 1].std(axis=0, ddof=1)
    ax.fill_between(
        sess,
        mean + std,
        mean - std,
        color="blue",
        alpha=0.2,
        edgecolor="none",
    )
    ax.plot(sess, mean, "b-", label="$s^{(1)} = 0$")
    # when s^(1) = 1
    mean = x1_list[:, :, 0].mean(axis=0)
    std = x1_list[:, :, 0].std(axis=0, ddof=1)
    ax.fill_between(
        sess,
        mean + std,
        mean - std,
        color="red",
        alpha=0.2,
        edgecolor="none",
    )
    ax.plot(sess, mean, "r-", label="$s^{(1)} = 1$")
    ax.set_title(f"Data: {datatype}\nSource-1 ensemble response")
    ax.set_xlabel("Session")
    ax.set_ylabel("Response")
    ax.axis((0, Nsession, 0, 1))
    ax.set_xticks(sess_ticks)
    ax.set_yticks(resp_ticks)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()

    # observed (x_post) and predicted (xp_post) responses during last 10 sessions
    ax = fig.add_subplot(6, 5, fig_pos + 5)
    n_bins = len(x_xp_mean)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.fill_between(
        bin_centers,
        x_xp_mean + x_xp_std,
        x_xp_mean - x_xp_std,
        color="blue",
        alpha=0.2,
        edgecolor="none",
    )
    ax.plot(bin_centers, x_xp_mean, "b-")
    ax.set_title("Response prediction (sess 91-100)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$x^P$")
    ax.axis((0, 1, 0, 1))
    ax.set_xticks(resp_ticks)
    ax.set_yticks(resp_ticks)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # error between observed and predicted responses
    ax = fig.add_subplot(6, 5, fig_pos + 10)
    mean = err_x_xp_list.mean(axis=0)
    std = err_x_xp_list.std(axis=0, ddof=1)
    ax.fill_between(
        sess,
        mean + std,
        mean - std,
        color="blue",
        alpha=0.2,
        edgecolor="none",
    )
    ax.plot(sess, mean, "b-")
    ax.set_title("Response prediction error")
    ax.set_xlabel("Session")
    ax.set_ylabel("Error")
    ax.axis((0, Nsession, 0, 0.5))
    ax.set_xticks(sess_ticks)
    ax.set_yticks([0, 0.2, 0.4])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # observed vs predicted weights
    ax = fig.add_subplot(6, 5, fig_pos + 15)
    ax.plot([-1, 4], [-1, 4], "k--")
    ax.plot(W[:, :, -1].ravel(), Wp[:, :, -1].ravel(), "b+", markersize=10)
    ax.set_title("Weight prediction (sess 100)")
    ax.set_xlabel("$W$")
    ax.set_ylabel("$W^P$")
    ax.axis((-1, 4, -1, 4))
    ax.set_xticks([0, 2, 4])
    ax.set_yticks([0, 2, 4])

    # observed and predicted weights
    ax = fig.add_subplot(6, 5, fig_pos + 20)
    mean = err_W_Wp_list.mean(axis=0)
    std = err_W_Wp_list.std(axis=0, ddof=1)
    ax.fill_between(
        sess,
        mean + std,
        mean - std,
        color="red",
        alpha=0.2,
        edgecolor="none",
    )
    ax.plot(sess, mean, "r-", label=r"$\mathrm{sig}(W)$ vs $\mathrm{sig}(W^P)$")

    mean = err_W_qA_list.mean(axis=0)
    std = err_W_qA_list.std(axis=0, ddof=1)
    ax.fill_between(
        sess,
        mean + std,
        mean - std,
        color="blue",
        alpha=0.2,
        edgecolor="none",
    )
    ax.plot(sess, mean, "b-", label=r"$\mathrm{sig}(W)$ vs $\mathrm{qA}^{Id}$")
    ax.set_title(r"Weight prediction error")
    ax.set_xlabel("Session")
    ax.set_ylabel("Error")
    ax.axis((0, Nsession, 0, 0.3))
    ax.set_xticks(sess_ticks)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()

    # free energy
    ax = fig.add_subplot(6, 5, fig_pos + 25)
    mean = L_list.mean(axis=0)
    std = L_list.std(axis=0, ddof=1)
    ax.fill_between(
        sess,
        mean + std,
        mean - std,
        color="blue",
        alpha=0.2,
        edgecolor="none",
    )
    ax.plot(sess, mean, "b-")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useMathText=True)
    ax.set_title("Free energy")
    ax.set_xlabel("Session")
    ax.set_ylabel("$F$")
    ax.axis((0, Nsession, 10000, 12000))
    ax.set_xticks(sess_ticks)
    ax.set_yticks([1e4, 1.1e4, 1.2e4])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.tight_layout()
    plt.draw()
