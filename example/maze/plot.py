import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "savefig.bbox": "tight",
    }
)


def plot(
    fig: plt.Figure,
    maze: np.ndarray,  # (size_y, size_x)
    pos: np.ndarray,  # (2, t + 1)
    o_mat: np.ndarray,  # (size_view, size_view)
    qs_mat: np.ndarray,  # (size_view, size_view)
    qd_mat: np.ndarray,  # (size_view, size_view)
    next_decision_prob: np.ndarray,  # (n_decisions,)
    trial_time_history: np.ndarray,  # (idx_trial,)
    T: int,
    n_trial: int,
    n_intvl: int,
):
    _, size_x = maze.shape
    size_view = o_mat.shape[0]
    idx_trial = len(trial_time_history)
    t = pos.shape[1] - 1

    # maze
    img_maze = (1 - maze[:, :, np.newaxis]).repeat(3, axis=-1).astype(np.float32)  # (size_y, size_x, 3)
    # draw trajectory
    traj_len = min(n_intvl, t + 1)  # t: 0-indexed
    traj_time = np.arange(t - traj_len + 1, t)
    img_maze[pos[0, traj_time], pos[1, traj_time], :] = [0.8, 0.8, 1]  # light blue
    # agent moves two points at once, so draw mid point of trajectory
    mid_y = (pos[0, traj_time] + pos[0, traj_time + 1]) // 2
    mid_x = (pos[1, traj_time] + pos[1, traj_time + 1]) // 2
    img_maze[mid_y, mid_x, :] = [0.8, 0.8, 1]
    # draw current position
    img_maze[pos[0, t], pos[1, t], :] = [0, 0, 1]  # blue

    # agent observation
    img_o = (1 - o_mat[:, :, np.newaxis]).repeat(3, axis=-1).astype(np.float32)

    # agent state posterior
    img_qs = (1 - qs_mat[:, :, np.newaxis]).repeat(3, axis=-1).astype(np.float32)

    # agent decision posterior
    img_qd = (1 - qd_mat[:, :, np.newaxis]).repeat(3, axis=-1).astype(np.float32)
    img_qd[:, :, 2] = 1  # blue

    fig.clf()
    gs = fig.add_gridspec(2, 5)

    # show maze on the top row
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(img_maze)
    # show stats on title
    trials_success = trial_time_history < T
    if not any(trials_success):
        success_rate = 0
        mean_success_step = 20000
    else:
        success_rate = (trials_success.mean() * 100).round().astype(int)
        mean_success_step = trial_time_history[trials_success].mean().round().astype(int)
    ax.set_title(
        f"Trial {idx_trial + 1}: t = {t + 1}  Success rate: {success_rate}%  Mean success step: {mean_success_step}"
    )
    ax.axis("off")

    # bottom row
    view_extent = [-size_view / 2, size_view / 2, -size_view / 2, size_view / 2]
    view_ticks = [np.ceil(-size_view / 2).astype(int), 0, np.floor(size_view / 2).astype(int)]

    # observation
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(img_o, extent=view_extent)
    ax.set_xticks(view_ticks)
    ax.set_yticks(view_ticks)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("o")

    # state posterior
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(img_qs, extent=view_extent)
    ax.set_xticks(view_ticks)
    ax.set_yticks(view_ticks)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(r"$\mathbf{s}$")

    # decision posterior
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(img_qd, extent=view_extent)
    ax.set_xticks(view_ticks)
    ax.set_yticks(view_ticks)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(rf"$\mathbf{{\delta}}$ (4step) L:{next_decision_prob[2]:.2f}, R:{next_decision_prob[3]:.2f}")

    # position
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(np.arange(1, t + 1), pos[1, :t] + 1, "b-")  # 1-indexed
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useMathText=True)
    ax.axis([0, T, 0, size_x + 1])
    ax.set_xticks([0, T // 2, T])
    ax.set_yticks([0, (size_x + 1) // 2, size_x + 1])
    ax.set_xlabel("Time")
    ax.set_ylabel("X")
    ax.set_title("Position")

    ax = fig.add_subplot(gs[1, 4])
    times = np.concatenate([trial_time_history, np.full(n_trial - idx_trial, np.nan)])
    ax.plot(np.arange(1, n_trial + 1), times, "bo", fillstyle="none")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useMathText=True)
    ax.axis([1, n_trial, 0, T])
    ax.set_xticks([0, n_trial // 2, n_trial])
    ax.set_yticks([0, T // 2, T])
    ax.set_xlabel("Trial")
    ax.set_ylabel("Time")
    ax.set_title("Duration")

    fig.tight_layout()

    return img_maze
