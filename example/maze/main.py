from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reverse_engineering.pomdp import POMDP

from plot import plot, qd_matrix  # isort: skip
from env import MazeEnv, process_risks  # isort: skip


size_x = 99  # length of maze (should be odd)
size_y = 19  # width of maze (should be odd)
size_view = 11  # size of agent-centered view (should be odd)
size_view_half = (size_view - 1) // 2

directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # (dy, dx) for up, down, left, right
n_decisions = len(directions)  # number of possible decisions per step
n_decision_steps = 4  # How many steps ahead to make decisions

T = 20000  # maximum duration
No = size_view * size_view  # observation dimensionality
Ns = No  # state dimensionality
Nd = n_decisions**n_decision_steps  # decision dimensionality; possible decision sequences for n_decision_steps
sim_type = 2

risk_chunk_size = 200
delta_thresholds = [-10.0, 0.0, 10.0]
risk_values = [0.55, 0.55, 0.45, 0]

E_right = 0.25  # prior of selecting rightward motion
# In the paper 'Canonical neural networks perform active inference',
# E_right = 0.15 corresponds to the E_right = 0.0023 condition (black)
# E_right = 0.25 corresponds to the E_right = 0.0039 condition (blue)
# E_right = 0.35 corresponds to the E_right = 0.0055 condition (cyan)
# Please refer to Figs. 4 and 5 in the paper.


def create_init_pomdp(buf_size: int = T) -> POMDP:
    # Model initialization constants
    fixed_param_scale = np.float32(10000000)  # High value to reduce plasticity in A and B models
    fixed_param_diag = np.float32(990000000)  # Extra high value for diag elements in A for 1-to-1 obs-state mapping
    learning_param_scale = np.float32(3000)  # Low value to allow learning in C model

    qa0 = np.full((No * 2, Ns * 2), fixed_param_scale)  # likelihood mapping
    qbi0 = np.full((Ns * 2, Ns * 2), fixed_param_scale)  # inverse state transition mapping
    qci0 = np.full((Ns * 2, Nd * 2), learning_param_scale)  # inverse policy mapping
    qa0[:No, :Ns] += np.eye(No, Ns, dtype=np.float32) * fixed_param_diag
    qa0[No:, Ns:] += np.eye(No, Ns, dtype=np.float32) * fixed_param_diag
    D = np.full((Ns * 2,), 0.5 / Nd, dtype=np.float32)  # state prior
    E = np.full((Nd * 2,), 0.5 / Nd, dtype=np.float32)  # decision prior

    # Set directional preferences (action prior probabilities)
    E_up = 0.25  # Probability of moving up
    E_down = 0.25  # Probability of moving down
    E_left = 0.5 - E_right  # Probability of moving left (complement of right)
    # Repeat probabilities for each decision step and normalize
    E[:Nd] = np.repeat([E_up, E_down, E_left, E_right], Nd // n_decisions) / (Nd // n_decisions)
    E[Nd:] = 1 - E[:Nd]  # Complement probabilities for one-hot encoding

    pomdp = POMDP(qa0, D, qbi0, qci0, E, sim_type, buf_size=buf_size)

    return pomdp


def main(
    out_dir: Path | str | None = None,
    n_sample: int = 30,
    n_trial: int = 100,
    n_intvl: int = 2000,
) -> None:
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    E_est_list = np.zeros((n_sample, n_trial, Nd), dtype=np.float32)
    fig = plt.figure(figsize=(17, 6))

    for idx_samp in range(n_sample):
        env = MazeEnv(size_x, size_y, size_view)
        pomdp = create_init_pomdp(buf_size=T)
        time_history = np.zeros(n_trial, dtype=np.int32)

        for idx_trial in range(n_trial):

            def _plot(pos, o_mat, qs_mat, qd_mat, next_decision_prob):
                qd_mat = np.minimum(qd_mat * 32, 1)
                times = time_history[:idx_trial]
                plot(fig, env.maze, pos, o_mat, qs_mat, qd_mat, next_decision_prob, times, T, n_trial, n_intvl)
                plt.draw()
                plt.pause(0.001)

            pomdp, trial_time, F, E_est, pos_history = run_trial(
                env,
                pomdp,
                T,
                n_intvl=n_intvl,
                plot_fn=_plot,
            )
            time_history[idx_trial] = trial_time + 1
            E_est_list[idx_samp, idx_trial] = E_est

            if idx_samp == 0 and (idx_trial == 0 or idx_trial == n_trial - 1):
                dpi = fig.dpi
                fig.savefig(
                    out_dir / f"maze_sample{idx_samp + 1}_trial{idx_trial + 1}_E{E_right}.png",
                    format="png",
                    dpi=120,
                )
                fig.dpi = dpi

    np.save(out_dir / f"Eest_E{E_right}.npy", E_est_list)


def run_trial(
    env: MazeEnv,
    pomdp: POMDP,
    T: int = 10000,
    n_intvl: int = 2000,
    plot_fn=None,
):
    """Run a single trial of maze navigation."""
    pos, o_mat = env.reset()
    pomdp.reset_buffer()
    step_risks = []

    pos_history = np.zeros((2, T), dtype=np.int32)
    pos_history[:, 0] = pos

    for t in range(1, T):
        # Get observation and select action
        s = o_mat.ravel(order="F")  # (Ns,)
        # observations (No * 2,)
        # o[:, :No] contains bits representing whether o^(0), ..., o^(No-1) is 1.
        # o[:, No:] contains bits representing whether o^(0), ..., o^(No-1) is 0.
        o = np.concatenate([s, 1 - s])  # (No * 2,)

        # inference
        qs, qd_prob, d = pomdp.infer(o)
        action = np.argmax(d) // (Nd // n_decisions)  # in {0, 1, 2, 3}

        # Environment step
        pos, o_mat, risk, done = env.step(action)
        pos_history[:, t] = pos
        step_risks.append(risk)

        # Visualization
        if plot_fn is not None and ((t + 1) % n_intvl == 0 or done):
            qs_mat = np.reshape(qs[:Ns], (size_view, size_view), order="F")
            qd_mat = qd_matrix(qd_prob, directions, n_decision_steps, size_view)
            qd_prob_next = qd_prob.reshape((n_decisions, Nd // n_decisions)).sum(axis=1)
            plot_fn(pos_history[:, : t + 1], o_mat, qs_mat, qd_mat, qd_prob_next)
            plt.pause(0.001)

        if done:
            break

    # Process results and update model
    chunk_risks = process_risks(np.array(step_risks), risk_chunk_size)
    F = pomdp.compute_vfe(chunk_risks, risk_chunk_size)
    E_est = pomdp.d_buf[:, 2 : pomdp.buf_idx].mean(axis=-1)
    pomdp.learn(chunk_risks, risk_chunk_size)

    return pomdp, t, F, E_est, pos_history


if __name__ == "__main__":
    np.random.seed(1000)
    main(n_sample=1)
