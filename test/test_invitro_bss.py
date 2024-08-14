from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from reverse_engineering.example.invitro_bss.load_data import get_data, load_data
from reverse_engineering.example.invitro_bss.main import analyze_sample


def test_ctrl():
    _test_datatype("ctrl")


def test_bic():
    _test_datatype("bic")


def test_dzp():
    _test_datatype("dzp")


def test_mix0():
    _test_datatype("mix0")


def test_mix50():
    _test_datatype("mix50")


def _test_datatype(datatype):
    load_data()
    data, baselines, plot_samp_idx = get_data(datatype)
    x, xp, W, Wp, W_hat, Wp_hat, qA1, phi1, phi0, L = analyze_sample(
        data[plot_samp_idx]["s"].T,
        data[plot_samp_idx]["o"].T,
        data[plot_samp_idx]["r"].T,
        baselines[plot_samp_idx],
    )

    rtol = 1e-5
    atol = 1e-8
    data = np.load(Path(__file__).parent / "correct_results" / f"invitro_bss_{datatype}.npz")
    assert_allclose(x, data["x"], rtol=rtol, atol=atol)
    assert_allclose(xp, data["xp"], rtol=rtol, atol=atol)
    assert_allclose(W, data["W"], rtol=rtol, atol=atol)
    assert_allclose(Wp, data["Wp"], rtol=rtol, atol=atol)
    assert_allclose(qA1, data["qA1"], rtol=rtol, atol=atol)
    assert_allclose(phi1, data["phi1"], rtol=rtol, atol=atol)
    assert_allclose(phi0, data["phi0"], rtol=rtol, atol=atol)
    assert_allclose(L, data["L_list"], rtol=rtol, atol=atol)
    data.close()


if __name__ == "__main__":
    test_ctrl()
    # test_bic()
    # test_dzp()
    # test_mix0()
    # test_mix50()
    print("All tests passed!")
