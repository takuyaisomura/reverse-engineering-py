import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.special import expit

grid_size = 100
log_interval = 10
fps = 15
dpi = 100


def _create_landscape(
    o,
    Nsession,
    phi1,
    phi0,
):
    No = o.shape[0]
    session_len = o.shape[1] // Nsession

    # compute cost for virtual weights
    L_W = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            base = np.array([[i, j], [j, i]]) * 2 / grid_size
            W1 = np.repeat(base, No // 2, axis=1)  # (2, No)
            W0 = np.repeat(-base, No // 2, axis=1)  # (2, No)
            h1 = np.log(1 - expit(W1)) @ np.ones((No, 1)) + phi1  # (2, 1)
            h0 = np.log(1 - expit(W0)) @ np.ones((No, 1)) + phi0  # (2, 1)
            x = expit((W1 - W0) @ o[:, :session_len] + (h1 - h0) @ np.ones((1, session_len)))
            x_ = np.vstack((x, 1 - x))  # (4, session_len)
            L_W[i, j] = np.sum(
                x_
                * (
                    np.log(x_ + 1e-6)
                    - np.vstack((W1, W0)) @ o[:, :session_len]
                    - np.vstack((h1, h0)) @ np.ones((1, session_len))
                )
            )
    # normalize
    L_W = (L_W - L_W.min()) / (L_W.max() - L_W.min())

    # set color
    img = np.zeros((grid_size, grid_size, 3))
    img[:, :, 0] = np.clip((L_W * 3 - 1) / 2, 0, 1)
    img[:, :, 1] = np.clip(L_W * 3, 0, 1)
    img[:, :, 2] = np.clip((L_W * 3 - 1) / 2, 0, 1)
    # set grid lines
    intv = grid_size // 5
    img[intv:grid_size:intv, :, :] = 0.5
    img[:, intv:grid_size:intv, :] = 0.5

    return L_W, img


def _create_figure(L_W, img):
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    ax.plot_surface(X, Y, L_W, rstride=1, cstride=1, facecolors=img, linewidth=0, antialiased=False, zorder=1)
    ax.view_init(elev=45, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_box_aspect((1, 1, 0.5))

    return fig, ax


def _plot_trajectory(xs, ys, L_W, lines):
    idx_w0 = np.ceil(xs - 1).astype(int) - 1
    idx_w1 = np.ceil(ys - 1).astype(int) - 1
    zs = L_W[idx_w0, idx_w1] + 0.01
    for line, x, y, z in zip(lines, xs, ys, zs):
        line.set_data(x, y)
        line.set_3d_properties(z)


def plot_weights_trajectory(
    out_dir,
    Ninit,
    o,
    W,
    Wp,
    phi1,
    phi0,
    data_name="",
    line_width=3,
):
    Nx, No, Nsession = W.shape
    assert Nx == 2, "Nx != 2 is not supported"

    desc = f"_{data_name}" if data_name else ""  # description to be added to file name

    L_W, img = _create_landscape(o, Nsession, phi1, phi0)

    # --- plot empirically estimated trajectory ---
    W_s = np.clip(W * 2, 0.001, 3.96) * grid_size / 4  # scale to grid size

    fig, ax = _create_figure(L_W, img)
    # placeholders for lines
    lines_s1 = [ax.plot([], [], [], "r-", linewidth=line_width)[0] for _ in range(No // 2)]
    lines_s2 = [ax.plot([], [], [], "b-", linewidth=line_width)[0] for _ in range(No // 2)]

    def update_plot_empirical(idx_sess):
        _plot_trajectory(  # trajectories of weights for source 1-biased observations
            W_s[0, : No // 2, : idx_sess + 1] + 1,
            W_s[1, : No // 2, : idx_sess + 1] + 1,
            L_W,
            lines_s1,
        )
        _plot_trajectory(  # trajectories of weights for source 2-biased observations
            W_s[0, No // 2 :, : idx_sess + 1] + 1,
            W_s[1, No // 2 :, : idx_sess + 1] + 1,
            L_W,
            lines_s2,
        )

        if (idx_sess + 1) % log_interval == 0:
            print(f"Frame: {idx_sess + 1}/{Nsession}")

        return lines_s1 + lines_s2  # updated things

    anim = FuncAnimation(
        fig,
        update_plot_empirical,
        frames=Nsession,
        interval=1,
        blit=True,
        repeat=False,
    )
    anim.save(
        out_dir / f"free_energy_gradient_W{desc}.mp4",
        writer="ffmpeg",
        fps=fps,
        dpi=dpi,
    )
    fig.savefig(out_dir / f"free_energy_gradient_W{desc}.png")

    # --- plot theoretically predicted trajectory ---
    W_s = np.clip(Wp * 2, 0.001, 3.96) * grid_size / 4

    fig, ax = _create_figure(L_W, img)
    # placeholders for lines
    lines_s1_init = [ax.plot([], [], [], "r-", linewidth=line_width)[0] for _ in range(No // 2)]
    lines_s2_init = [ax.plot([], [], [], "b-", linewidth=line_width)[0] for _ in range(No // 2)]
    lines_s1_pred = [ax.plot([], [], [], color=[1, 0.75, 0], linewidth=line_width)[0] for _ in range(No // 2)]
    lines_s2_pred = [ax.plot([], [], [], color=[0, 0.75, 1], linewidth=line_width)[0] for _ in range(No // 2)]

    def update_plot_predicted(idx_sess):
        if idx_sess < Ninit:
            _plot_trajectory(  # trajectories of weights for source 1-biased observations
                W_s[0, : No // 2, : idx_sess + 1] + 1,
                W_s[1, : No // 2, : idx_sess + 1] + 1,
                L_W,
                lines_s1_init,
            )
            _plot_trajectory(  # trajectories of weights for source 2-biased observations
                W_s[0, No // 2 :, : idx_sess + 1] + 1,
                W_s[1, No // 2 :, : idx_sess + 1] + 1,
                L_W,
                lines_s2_init,
            )
        else:
            _plot_trajectory(  # trajectories of weights for source 1-biased observations
                W_s[0, : No // 2, :Ninit] + 1,
                W_s[1, : No // 2, :Ninit] + 1,
                L_W,
                lines_s1_init,
            )
            _plot_trajectory(  # trajectories of weights for source 2-biased observations
                W_s[0, No // 2 :, :Ninit] + 1,
                W_s[1, No // 2 :, :Ninit] + 1,
                L_W,
                lines_s2_init,
            )
            _plot_trajectory(  # trajectories of weights for source 1-biased observations (predicted)
                W_s[0, : No // 2, Ninit : idx_sess + 1] + 1,
                W_s[1, : No // 2, Ninit : idx_sess + 1] + 1,
                L_W,
                lines_s1_pred,
            )
            _plot_trajectory(  # trajectories of weights for source 2-biased observations (predicted)
                W_s[0, No // 2 :, Ninit : idx_sess + 1] + 1,
                W_s[1, No // 2 :, Ninit : idx_sess + 1] + 1,
                L_W,
                lines_s2_pred,
            )

        if (idx_sess + 1) % log_interval == 0:
            print(f"Frame: {idx_sess + 1}/{Nsession}")

        return lines_s1_init + lines_s2_init + lines_s1_pred + lines_s2_pred  # updated things

    anim = FuncAnimation(
        fig,
        update_plot_predicted,
        frames=Nsession,
        interval=1,
        blit=True,
        repeat=False,
    )
    anim.save(
        out_dir / f"free_energy_gradient_Wp{desc}.mp4",
        writer="ffmpeg",
        fps=fps,
        dpi=dpi,
    )
    fig.savefig(out_dir / f"free_energy_gradient_Wp{desc}.png")
