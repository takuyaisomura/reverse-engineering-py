from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from example.bss.main import create_qa_init, generate, simulate
from reverse_engineering.pomdp import POMDP


def test_bss():
    N = 2
    M = 32
    T = 100 * 256
    qa_eps = 0.01
    qa_amp_min = 200
    qa_amp_max = 400
    sim_type = 1
    np.random.seed(0)

    _, o = generate(N, M, T)
    qa_init = create_qa_init(N, M, qa_eps, qa_amp_min, qa_amp_max)
    D = np.array([0.5, 0.5, 0.5, 0.5])  # (Ns * 2,)

    pomdp = POMDP(qa_init, D, sim_type=sim_type, buf_size=T)
    qs = simulate(pomdp, o)

    rtol = 1e-5
    atol = 1e-8
    data = np.load(Path(__file__).parent / "correct_results" / "bss.npz")
    assert_allclose(qs, data["qs"], rtol=rtol, atol=atol)
    data.close()


if __name__ == "__main__":
    test_bss()
    print("All tests passed!")
