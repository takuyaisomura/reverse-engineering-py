import numpy as np
from scipy.special import psi

from reverse_engineering.utils.utils import check_double_layer_args


def compute_dirichlet_expectation(
    qa: np.ndarray,
    sim_type: int,
    qA_array: np.ndarray | None = None,
    qlnA_array: np.ndarray | None = None,
):
    """Compute the posterior expectation of the Dirichlet distribution.

    Args:
        qa (np.ndarray): Dirichlet concentration parameter (No * 2, Ns * 2).
        sim_type (int): Type of simulation. 1 for psi, 2 for log.
        qA_array (np.ndarray | None, optional): Output placeholder for the posterior expectation (No * 2, Ns * 2).
        qlnA_array (np.ndarray | None, optional): Output placeholder for the posterior expectation of log(A)
            (No * 2, Ns * 2).

    Returns:
        qA (np.ndarray): Posterior expectation (No * 2, Ns * 2).
        qlnA (np.ndarray): Posterior expectation of log(A) (No * 2, Ns * 2).
    """
    qA = qA_array if qA_array is not None else np.empty_like(qa)
    qlnA = qlnA_array if qlnA_array is not None else np.empty_like(qa)
    No = qa.shape[0] // 2

    qa_sum = qa[:No] + qa[No:]  # (No, Ns * 2)

    qA[:No] = qa[:No] / qa_sum
    qA[No:] = 1 - qA[:No]

    if sim_type == 1:
        psi_sum = psi(np.maximum(1e-6, qa_sum))
        qlnA[:] = psi(np.maximum(1e-6, qa)) - np.tile(psi_sum, (2, 1))
    elif sim_type == 2:
        log_sum = np.log(np.maximum(1e-6, qa_sum))
        qlnA[:] = np.log(np.maximum(1e-6, qa)) - np.tile(log_sum, (2, 1))
    else:
        raise ValueError(f"Invalid sim_type: {sim_type}")

    return qA, qlnA


class POMDP:
    """Bayesian POMDP model."""

    def __init__(
        self,
        qa_init: np.ndarray,
        D: np.ndarray,
        qbi_init: np.ndarray | None = None,
        qci_init: np.ndarray | None = None,
        E: np.ndarray | None = None,
        sim_type: int = 2,
        buf_size: int = 1000,
    ) -> None:
        """
        Args:
            qa_init (np.ndarray): Initial Dirichlet concentration parameter (No * 2, Ns * 2).
            D (np.ndarray): State prior (Ns * 2,).
            qbi_init (np.ndarray | None, optional): Initial Dirichlet concentration parameter for the recurrent layer
            (Ns * 2, Ns * 2).
            qci_init (np.ndarray | None, optional): Initial Dirichlet concentration parameter for the output layer
            (Ns * 2, Nd * 2).
            E (np.ndarray | None, optional): Decision prior (Nd * 2,).
            sim_type (int, optional): Type of simulation. 1 for psi, 2 for log. Defaults to 2.
            buf_size (int, optional): Buffer size. This would need to be longer than the interval between learn() calls.
            Defaults to 1000.
        """
        self.is_double_layer = check_double_layer_args(E, qbi_init, qci_init)

        self.No = len(qa_init) // 2
        self.Ns = len(D) // 2
        self.sim_type = sim_type
        self.buf_size = buf_size

        self.qa = qa_init
        self.qA = np.empty_like(self.qa)
        self.qlnA = np.empty_like(self.qa)
        compute_dirichlet_expectation(self.qa, self.sim_type, self.qA, self.qlnA)

        self.D = D
        self.lnD = np.log(np.maximum(1e-6, D))

        if self.is_double_layer:
            self.Nd = len(E) // 2
            self.qbi = qbi_init
            self.qBi = np.empty_like(self.qbi)
            self.qlnBi = np.empty_like(self.qbi)
            compute_dirichlet_expectation(self.qbi, self.sim_type, self.qBi, self.qlnBi)

            self.qci = qci_init
            self.qCi = np.empty_like(self.qci)
            self.qlnCi = np.empty_like(self.qci)
            compute_dirichlet_expectation(self.qci, self.sim_type, self.qCi, self.qlnCi)

            self.E = E
            self.lnE = np.log(np.maximum(1e-6, E))

        self.reset_buffer()

    def reset_buffer(self) -> None:
        """Reset the buffer."""
        self.o_buf = np.empty((self.No * 2, self.buf_size), dtype=np.int32)
        self.qs_buf = np.empty((self.Ns * 2, self.buf_size))  # float64
        self.vs_buf = np.empty((self.Ns * 2, self.buf_size))  # float64
        if self.is_double_layer:
            self.d_buf = np.empty((self.Nd, self.buf_size), dtype=np.int64)
            self.qd_buf = np.empty((self.Nd * 2, self.buf_size))  # float64

        self.buf_idx = 0  # buffer index of the next data to be written
        self.buffer_data(
            o=np.zeros((self.No * 2,), dtype=np.int32),
            qs=self.D,
            vs=self.D,
            d=np.zeros(self.Nd, dtype=np.int64) if self.is_double_layer else None,  # multinomial outputs int64
            qd=self.E if self.is_double_layer else None,
        )

    def buffer_data(
        self,
        o: np.ndarray,
        qs: np.ndarray,
        vs: np.ndarray,
        d: np.ndarray | None = None,
        qd: np.ndarray | None = None,
    ) -> None:
        """Buffer data.

        Args:
            o (np.ndarray): Observation (No * 2,).
            qs (np.ndarray): State posterior (Ns * 2,).
            vs (np.ndarray): pre-sigmoid state posterior (Ns * 2,).
            d (np.ndarray | None, optional): Decision (Nd,).
            qd (np.ndarray | None, optional): Decision posterior (Nd * 2,).
        """
        if self.buf_idx >= self.buf_size:
            raise ValueError("Buffer is full. Consider increasing the buffer size.")

        self.o_buf[:, self.buf_idx] = o
        self.qs_buf[:, self.buf_idx] = qs
        self.vs_buf[:, self.buf_idx] = vs
        if self.is_double_layer:
            self.d_buf[:, self.buf_idx] = d
            self.qd_buf[:, self.buf_idx] = qd

        self.buf_idx += 1

    def infer(
        self,
        o: np.ndarray,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Infer the state (and decision if double layer).

        Args:
            o (np.ndarray): Observation (No * 2,).

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
            - single layer: State posterior (Ns * 2,).
            - double layer: State posterior (Ns * 2,), decision (Nd,), decision posterior (Nd * 2,).
        """
        if not self.is_double_layer:
            return self._infer_single_layer(o)
        else:
            qs_prev = self.qs_buf[:, self.buf_idx - 1]  # (Ns * 2,)
            return self._infer_double_layer(o, qs_prev)

    def _infer_single_layer(
        self,
        o: np.ndarray,
    ) -> np.ndarray:
        """Infer the state in single layer model.

        Args:
            o (np.ndarray): Observation (No * 2,).

        Returns:
            np.ndarray: State posterior (Ns * 2,).
        """
        vs = self.lnD + self.qlnA.T @ o  # faster than einsum (for small data only?)
        qs = np.empty_like(self.lnD)  # (Ns * 2,)
        qs[: self.Ns] = 1 / (1 + np.exp(-(vs[: self.Ns] - vs[self.Ns :])))
        qs[self.Ns :] = 1 - qs[: self.Ns]

        self.buffer_data(o, qs, vs)

        return qs

    def _infer_double_layer(
        self,
        o: np.ndarray,
        qs_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Infer the state and decision in double layer model.

        Args:
            o (np.ndarray): Observation (No * 2,).
            qs_prev (np.ndarray): Previous state posterior (Ns * 2,).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
            State posterior (Ns * 2,), decision (Nd,), decision posterior (Nd * 2,).
        """
        vs = self.lnD + np.einsum("ij,i->j", self.qlnA, o) + np.einsum("ij,i->j", self.qlnBi, qs_prev)  # faster than @
        qs = np.empty((self.Ns * 2))
        qs[: self.Ns] = 1 / (1 + np.exp(-(vs[: self.Ns] - vs[self.Ns :])))
        qs[self.Ns :] = 1 - qs[: self.Ns]

        vd = self.lnE + np.einsum("ij,i->j", self.qlnCi, qs_prev)
        qd = np.empty((self.Nd * 2))
        qd[: self.Nd] = 1 / (1 + np.exp(-(vd[: self.Nd] - vd[self.Nd :])))
        qd[self.Nd :] = 1 - qd[: self.Nd]

        # decision
        qd_prob = qd[: self.Nd] / (qd[self.Nd :] + 1e-6)  # (Nd,)  # ratio
        qd_prob /= qd_prob.sum()
        d = np.random.multinomial(1, qd_prob)

        # update trajectory
        self.buffer_data(o, qs, vs, d, qd)

        return qs, qd_prob, d

    def learn(
        self,
        gammas: np.ndarray | None = None,
        risk_chunk_size: int | None = None,
    ):
        """Model learning.

        Args:
            gammas (np.ndarray | None, optional): Chunk-wise risk values with shape (num_chunk,).
            Used for decision layer learning.
            risk_chunk_size (int | None, optional): Chunk size for risk calculation. Used for decision layer learning.
        """
        if self.is_double_layer:
            assert gammas is not None and risk_chunk_size is not None

        # gamma: (num_chunk,)
        o_traj = self.o_buf[:, : self.buf_idx]  # (No * 2, buf_idx)
        qs_traj = self.qs_buf[:, : self.buf_idx]  # (Ns * 2, buf_idx)

        # input-middle connection
        self.qa += o_traj[:, 1:] @ qs_traj[:, 1:].T
        compute_dirichlet_expectation(self.qa, self.sim_type, self.qA, self.qlnA)

        # recurrent connection and middle-output connection
        if self.is_double_layer:
            self.qbi += qs_traj[:, :-1] @ qs_traj[:, 1:].T
            compute_dirichlet_expectation(self.qbi, self.sim_type, self.qBi, self.qlnBi)

            qs_traj_ = qs_traj[:, :-1]  # (Ns * 2, buf_idx - 1)
            d1 = self.d_buf[:, 1 : self.buf_idx]  # (Nd, buf_idx - 1)
            d_traj_ = np.concatenate([d1, 1 - d1], axis=0)
            for i, gamma in enumerate(gammas):
                ts = slice(i * risk_chunk_size, (i + 1) * risk_chunk_size)
                self.qci += qs_traj_[:, ts] @ d_traj_[:, ts].T * (1 - 2 * gamma)
                self.qci = np.maximum(1, self.qci)
            compute_dirichlet_expectation(self.qci, self.sim_type, self.qCi, self.qlnCi)

        return self

    def compute_vfe(
        self,
        gammas: np.ndarray | None = None,
        risk_chunk_size: int | None = None,
    ):
        """Compute the variational free energy.

        Args:
            gammas (np.ndarray | None, optional): Chunk-wise risk values with shape (num_chunk,).
            Used in double layer model.
            risk_chunk_size (int | None, optional): Chunk size. Used in double layer model.

        Returns:
            float: Variational free energy.
        """
        if self.is_double_layer:
            assert gammas is not None and risk_chunk_size is not None

        # gamma: (num_chunk,)
        qs_traj = self.qs_buf[:, : self.buf_idx]  # (Ns * 2, buf_idx)
        vs_traj = self.vs_buf[:, : self.buf_idx]  # (Ns * 2, buf_idx)
        ln_qs_traj = np.log(np.maximum(1e-6, qs_traj))

        # input-middle connection
        ts = slice(1, self.buf_idx - 1)
        F = np.sum(qs_traj[:, ts] * (ln_qs_traj[:, ts] - vs_traj[:, ts])).astype(np.float32)

        if not self.is_double_layer:
            return F

        # recurrent connection and middle-output connection
        qs_traj = qs_traj.astype(np.float32)
        qd_traj = self.qd_buf[:, : self.buf_idx].astype(np.float32)  # (Nd * 2, buf_idx)
        ln_qd_traj = np.log(np.maximum(1e-6, qd_traj))
        for i, gamma in enumerate(gammas):
            ts = np.arange(
                max(1, i * risk_chunk_size),
                min(self.buf_idx - 1, (i + 1) * risk_chunk_size),
            )
            vd = (1 - 2 * gamma) * self.qlnCi.T @ qs_traj[:, ts - 1] + self.lnE[:, np.newaxis]  # faster than einsum
            F += np.sum(qd_traj[:, ts] * (ln_qd_traj[:, ts] - vd))

        return F
