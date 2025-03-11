import numpy as np
from numpy.testing import assert_allclose

from example.bss.main import create_qa_init, generate, simulate
from reverse_engineering.pomdp import POMDP
from reverse_engineering.utils.variable_conversion import POMDPVariables, nn2pomdp, pomdp2nn


def test_variable_conversion():
    N = 2
    M = 32
    T = 100 * 256
    qa_eps = 0.01
    qa_amp_min = 200
    qa_amp_max = 400
    sim_type = 1
    np.random.seed(0)

    s, o = generate(N, M, T)

    qa_init = create_qa_init(N, M, qa_eps, qa_amp_min, qa_amp_max)
    D = np.array([0.5, 0.5, 0.5, 0.5])  # (Ns * 2,)

    pomdp = POMDP(qa_init.copy(), D, sim_type=sim_type, buf_size=T)
    qs = simulate(pomdp, o)

    pomdp_vars = POMDPVariables(
        qs=qs,
        qA=pomdp.qA[np.newaxis, :, :],  # add time axis
        lnD=pomdp.lnD,
    )
    nn_vars = pomdp2nn(pomdp_vars)
    pomdp_vars_recovered = nn2pomdp(nn_vars)

    rtol = 1e-5
    atol = 1e-8
    assert_allclose(pomdp_vars.qs, pomdp_vars_recovered.qs, rtol=rtol, atol=atol)
    assert_allclose(pomdp_vars.qA, pomdp_vars_recovered.qA, rtol=rtol, atol=atol)
    assert_allclose(pomdp_vars.lnD, pomdp_vars_recovered.lnD, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_variable_conversion()
    print("All tests passed!")
    test_variable_conversion()
    print("All tests passed!")
