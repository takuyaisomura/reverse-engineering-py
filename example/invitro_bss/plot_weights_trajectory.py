import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.special import expit

grid_size = 100
log_interval = 10
fps = 15
dpi = 100


def _compute_cost(grid_points, o, No, session_len, phi1, phi0):
    """Compute cost for given grid points in the landscape."""
    W1 = np.repeat(grid_points, No // 2, axis=1)  # (2, No)
    W0 = np.repeat(-grid_points, No // 2, axis=1)  # (2, No)
    h1 = np.log(1 - expit(W1)) @ np.ones((No, 1)) + phi1  # (2, 1)
    h0 = np.log(1 - expit(W0)) @ np.ones((No, 1)) + phi0  # (2, 1)
    x = expit((W1 - W0) @ o[:, :session_len] + (h1 - h0) @ np.ones((1, session_len)))
    x_ = np.vstack((x, 1 - x))  # (4, session_len)
    return np.sum(
        x_
        * (
            np.log(x_ + 1e-6)
            - np.vstack((W1, W0)) @ o[:, :session_len]
            - np.vstack((h1, h0)) @ np.ones((1, session_len))
        )
    )


def _create_landscape(
    o,
    phi1,
    phi0,
    Nsession,
):
    No = o.shape[0]
    session_len = o.shape[1] // Nsession

    # compute cost for virtual weights
    landscape = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            grid_points = np.array([[i, j], [j, i]]) * 2 / grid_size  # (i, j) and (j, i) on the grid
            landscape[i, j] = _compute_cost(grid_points, o, No, session_len, phi1, phi0)

    # normalize
    landscape = (landscape - landscape.min()) / (landscape.max() - landscape.min())

    return landscape


def _create_landscape_img(landscape):
    """Create color image from landscape values."""
    img = np.zeros((grid_size, grid_size, 3))
    img[:, :, 0] = np.clip((landscape * 3 - 1) / 2, 0, 1)
    img[:, :, 1] = np.clip(landscape * 3, 0, 1)
    img[:, :, 2] = np.clip((landscape * 3 - 1) / 2, 0, 1)

    # set grid lines
    intv = grid_size // 5
    img[intv:grid_size:intv, :, :] = 0.5
    img[:, intv:grid_size:intv, :] = 0.5
    return img


def _create_figure(landscape, img):
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    ax.plot_surface(X, Y, landscape, rstride=1, cstride=1, facecolors=img, linewidth=0, antialiased=False, zorder=1)
    ax.view_init(elev=45, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_box_aspect((1, 1, 0.5))

    return fig, ax


class WeightTrajectoryLines:
    """Class to manage a collection of weight trajectory lines in a 3D plot."""

    def __init__(self, W, landscape, ax, color, width, style="-"):
        num_lines = W.shape[1]
        self.lines = [ax.plot([], [], [], color=color, linewidth=width, linestyle=style)[0] for _ in range(num_lines)]
        self.W = W  # (2, num_lines, num_sessions)
        self.landscape = landscape  # (grid_size, grid_size)

    def update(self, idx):
        xs = self.W[0, :, : idx + 1] + 1
        ys = self.W[1, :, : idx + 1] + 1

        idx_w0 = np.ceil(xs - 1).astype(int) - 1
        idx_w1 = np.ceil(ys - 1).astype(int) - 1
        zs = self.landscape[idx_w0, idx_w1] + 0.01

        for line, x, y, z in zip(self.lines, xs, ys, zs):
            line.set_data(x, y)
            line.set_3d_properties(z)

    def get_lines(self):
        return self.lines


def plot_empirical_weights_trajectory(
    W,
    landscape,
    img,
    out_dir,
    condition_name,
    No,
    Nsession,
    line_width=3,
):
    """Plot empirically estimated weight trajectories."""
    fig, ax = _create_figure(landscape, img)
    # Create line collections using the new class with pre-sliced weights
    lines_source1 = WeightTrajectoryLines(W[:, : No // 2], landscape, ax, "r", line_width)
    lines_source2 = WeightTrajectoryLines(W[:, No // 2 :], landscape, ax, "b", line_width)

    def update_plot_empirical(idx):
        if (idx + 1) % log_interval == 0:
            print(f"Frame: {idx + 1}/{Nsession}")

        lines_source1.update(idx)
        lines_source2.update(idx)
        return lines_source1.get_lines() + lines_source2.get_lines()

    anim = FuncAnimation(fig, update_plot_empirical, frames=Nsession, interval=1, blit=True, repeat=False)
    anim.save(out_dir / f"free_energy_gradient_W_{condition_name}.mp4", writer="ffmpeg", fps=fps, dpi=dpi)
    fig.savefig(out_dir / f"free_energy_gradient_W_{condition_name}.png")


def plot_predicted_weights_trajectory(
    Wp,
    landscape,
    img,
    out_dir,
    condition_name,
    No,
    Nsession,
    Ninit,
    line_width=3,
):
    """Plot theoretically predicted weight trajectories."""
    fig, ax = _create_figure(landscape, img)
    # Create line collections for initial and predicted trajectories
    lines_source1_init = WeightTrajectoryLines(Wp[:, : No // 2, :Ninit], landscape, ax, "r", line_width)
    lines_source2_init = WeightTrajectoryLines(Wp[:, No // 2 :, :Ninit], landscape, ax, "b", line_width)
    lines_source1_pred = WeightTrajectoryLines(Wp[:, : No // 2, Ninit:], landscape, ax, "orange", line_width)
    lines_source2_pred = WeightTrajectoryLines(Wp[:, No // 2 :, Ninit:], landscape, ax, "cyan", line_width)

    def update_plot_predicted(idx):
        if (idx + 1) % log_interval == 0:
            print(f"Frame: {idx + 1}/{Nsession}")

        if idx < Ninit:
            # Update trajectories for initial weights
            lines_source1_init.update(idx)
            lines_source2_init.update(idx)
            return lines_source1_init.get_lines() + lines_source2_init.get_lines()
        else:
            # Update only predicted trajectories after Ninit-th session
            lines_source1_pred.update(idx - Ninit)
            lines_source2_pred.update(idx - Ninit)
            return lines_source1_pred.get_lines() + lines_source2_pred.get_lines()

    anim = FuncAnimation(
        fig,
        update_plot_predicted,
        frames=Nsession,
        interval=1,
        blit=True,
        repeat=False,
    )
    anim.save(
        out_dir / f"free_energy_gradient_Wp_{condition_name}.mp4",
        writer="ffmpeg",
        fps=fps,
        dpi=dpi,
    )
    fig.savefig(out_dir / f"free_energy_gradient_Wp_{condition_name}.png")


def plot_weights_trajectory(
    W,
    Wp,
    o,
    phi1,
    phi0,
    out_dir,
    condition_name,
    Ninit,
    line_width=3,
):
    """Plot both empirical and predicted weight trajectories."""
    # Common setup
    Nx, No, Nsession = W.shape
    assert Nx == 2, "Nx != 2 is not supported"
    assert W.shape == Wp.shape, "W and Wp must have the same shape"

    # Create landscape and scale weights
    landscape = _create_landscape(o, phi1, phi0, Nsession)
    img = _create_landscape_img(landscape)
    W_scaled = np.clip(W * 2, 0.001, 3.96) * grid_size / 4
    Wp_scaled = np.clip(Wp * 2, 0.001, 3.96) * grid_size / 4

    # Plot empirical weights
    print("Plotting empirical weights...")
    plot_empirical_weights_trajectory(W_scaled, landscape, img, out_dir, condition_name, No, Nsession, line_width)

    # Plot predicted weights
    print("Plotting predicted weights...")
    plot_predicted_weights_trajectory(
        Wp_scaled, landscape, img, out_dir, condition_name, No, Nsession, Ninit, line_width
    )
