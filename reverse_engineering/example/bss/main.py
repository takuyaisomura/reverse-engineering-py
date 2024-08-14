from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from reverse_engineering.pomdp import POMDP

Ns = 2  # number of hidden states
No = 32  # number of observations
Nsample = 100
Nsession = 100
session_len = 256
T = Nsession * session_len
qa_eps = 0.01  # bias for prior of parameters
qa_amp_min = 200  # amplitude for prior of parameters
qa_amp_max = 400  # amplitude for prior of parameters
sim_type = 1  # 1:MDP, 2:neural network


def main(out_dir: Path | str | None = None) -> None:
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    s_history = np.empty((Nsample, T, Ns * 2))
    qs_history = np.empty((Nsample, T, Ns * 2))

    for idx_samp in tqdm(range(Nsample)):
        s, o = generate(Ns, No, T)

        qa_init = create_qa_init(Ns, No, qa_eps, qa_amp_min, qa_amp_max)
        D = np.array([0.5, 0.5, 0.5, 0.5])  # (Ns * 2,)
        pomdp = POMDP(qa_init, D, sim_type=sim_type, buf_size=T)

        qs = simulate(pomdp, o)  # (T, Ns * 2)

        s_history[idx_samp] = s
        qs_history[idx_samp] = qs

    plot_state(s_history, qs_history, out_dir)


def simulate(
    pomdp: POMDP,
    o: np.ndarray,
) -> np.ndarray:
    qs_history = []
    for o_t in o:
        qs = pomdp.infer(o_t)
        qs_history.append(qs)
        pomdp.learn(o_t, qs)
        pomdp.reset_buffer()
    return np.array(qs_history)  # (T, Ns * 2)


def generate(
    Ns: int,
    No: int,
    T: int,
) -> tuple[np.ndarray, np.ndarray]:
    # define generative process
    # A: likelihood mapping that maps hidden states to observations
    # A[i, k, m, n] = P(o^(i)=1-k | s^(0)=1-m, s^(1)=1-n)
    A = np.zeros((No, 2, 2, 2))
    # for o^(0) - o^(No/2-1): o=s^(0) with prob 3/4, o=s^(1) with prob 1/4
    for i in range(No // 2):
        A[i, 0, :, :] = [[1, 3 / 4], [1 / 4, 0]]
    # for o^(No/2+1) - o^(No): o=s^(0) with prob 1/4, o=s^(1) with prob 3/4
    for i in range(No // 2, No):
        A[i, 0, :, :] = [[1, 1 / 4], [3 / 4, 0]]
    # P(o^(i)=0) = 1 - P(o^(i)=1)
    A[:, 1, :, :] = 1 - A[:, 0, :, :]

    # s: hidden states
    # s[:, :Ns] contains bits representing whether s^(0), ..., s^(Ns-1) is 1.
    # s[:, Ns:] contains bits representing whether s^(0), ..., s^(Ns-1) is 0.
    s = np.zeros((T, Ns * 2), dtype=int)
    # o: observations
    # o[:, :No] contains bits representing whether o^(0), ..., o^(No-1) is 1.
    # o[:, No:] contains bits representing whether o^(0), ..., o^(No-1) is 0.
    o = np.zeros((T, No * 2), dtype=int)
    for t in range(T):
        s[t, :Ns] = np.random.randint(0, 2, size=Ns)
        # o^(i)=1 with prob A[i, 0, 1-s^(0), 1-s^(1)]
        o[t, :No] = (np.random.rand(No) < A[:, 0, 1 - s[t, 0], 1 - s[t, 1]]).astype(int)
    s[:, Ns:] = 1 - s[:, :Ns]
    o[:, No:] = 1 - o[:, :No]

    return s, o


def create_qa_init(
    Ns: int,
    No: int,
    qa_eps: float,
    qa_amp_min: float,
    qa_amp_max: float,
) -> np.ndarray:
    # Initial connection strengths (prior beliefs about parameters)
    qa_init = np.zeros((No * 2, Ns * 2))  # parameter prior or initial synaptic strengths
    for i in range(No // 2):
        qa_init[i, :] = [
            0.5 + qa_eps * 2,
            0.5 + qa_eps,
            0.5 - qa_eps * 2,
            0.5 - qa_eps,
        ]
    for i in range(No // 2, No):
        qa_init[i, :] = [
            0.5 + qa_eps,
            0.5 + qa_eps * 2,
            0.5 - qa_eps,
            0.5 - qa_eps * 2,
        ]
    qa_init[No:, :] = 1 - qa_init[:No, :]

    for i in range(No):
        for j in range(Ns):
            amp = qa_amp_min + (qa_amp_max - qa_amp_min) * np.random.rand()
            qa_init[i, j] *= amp
            qa_init[i, j + Ns] *= amp
            qa_init[i + No, j] *= amp
            qa_init[i + No, j + Ns] *= amp

    return qa_init


def plot_state(
    s_history: np.ndarray,
    qs_history: np.ndarray,
    out_dir: Path,
):
    # s_history: (Nsample, T, Ns * 2)
    # qs_history: (Nsample, T, Ns * 2)

    # compute conditional mean of qs for each session
    qs_sess_mean = np.empty((Nsample, Nsession, Ns, 2))

    s_reshaped = s_history.reshape(Nsample, Nsession, session_len, Ns * 2)
    qs_reshaped = qs_history.reshape(Nsample, Nsession, session_len, Ns * 2)
    for i in range(2):  # s_i = 1 (i = 0), 0 (i = 1)
        # mask that indicates whether s^(k) == s_i
        mask = s_reshaped[:, :, :, i * Ns : (i + 1) * Ns]  # (Nsample, Nsession, session_len, Ns)
        # extract qs^(k) at time s^(k) == s_i for each k
        qs_masked = np.where(mask, qs_reshaped[:, :, :, :Ns], np.nan)  # (Nsample, Nsession, session_len, Ns)
        # session means of qs^(k) at time s^(k) == s_i for each k
        qs_sess_mean[:, :, :, i] = np.nanmean(qs_masked, axis=2)  # (Nsample, Nsession, Ns)

    # plot qs^(0)
    # select qs^(0)
    qs0_sess_mean = qs_sess_mean[:, :, 0, :]  # (Nsample, Nsession, 2)
    # subtract the first session
    qs0_sess_mean -= qs0_sess_mean[:, 0:1, :]
    # subtract state mean
    qs0_sess_mean -= qs0_sess_mean.mean(axis=2, keepdims=True)
    # sample mean and std
    qs0_mean = qs0_sess_mean.mean(axis=0)  # (Nsession, 2)
    qs0_std = qs0_sess_mean.std(axis=0, ddof=1)

    sess = np.arange(Nsession)
    for i, color in enumerate(["r", "b"]):  # s^(0) = s_i
        plt.plot(sess, qs0_mean[:, i], color=color, label=rf"$s^{{(1)}} = {i}$")  # 1-indexed
        plt.fill_between(
            sess,
            qs0_mean[:, i] - qs0_std[:, i],
            qs0_mean[:, i] + qs0_std[:, i],
            alpha=0.2,
            color=color,
        )
        plt.axis([0, Nsession, -0.2, 0.2])
        plt.xlabel("Session #")
        plt.ylabel(r"Change in posterior $\mathbf{s}^{(1)}$")  # 1-indexed
        plt.xticks([0, Nsession // 2, Nsession])
        plt.yticks([-0.2, 0, 0.2])
        plt.legend()

    plt.tight_layout()

    plt.savefig(out_dir / "bss.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    main()
