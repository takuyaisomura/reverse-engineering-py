from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from example.invitro_bss.dataset import get_datasets
from example.invitro_bss.main import analyze_sample


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
    datasets = get_datasets()
    dataset = next(d for d in datasets if d.datatype == datatype)
    sample_res = analyze_sample(
        dataset.data[dataset.sample_index]["s"].T,
        dataset.data[dataset.sample_index]["o"].T,
        dataset.data[dataset.sample_index]["r"].T,
        dataset.baseline,
    )

    rtol = 1e-5
    atol = 1e-8
    data = np.load(Path(__file__).parent / "correct_results" / f"invitro_bss_{datatype}.npz")
    assert_allclose(sample_res.x, data["x"], rtol=rtol, atol=atol)
    assert_allclose(sample_res.xp, data["xp"], rtol=rtol, atol=atol)
    assert_allclose(sample_res.W, data["W"], rtol=rtol, atol=atol)
    assert_allclose(sample_res.Wp, data["Wp"], rtol=rtol, atol=atol)
    assert_allclose(sample_res.qA1, data["qA1"], rtol=rtol, atol=atol)
    assert_allclose(sample_res.phi1, data["phi1"], rtol=rtol, atol=atol)
    assert_allclose(sample_res.phi0, data["phi0"], rtol=rtol, atol=atol)
    assert_allclose(sample_res.L, data["L_list"], rtol=rtol, atol=atol)
    data.close()


if __name__ == "__main__":
    test_ctrl()
    # test_bic()
    # test_dzp()
    # test_mix0()
    # test_mix50()
    print("All tests passed!")
