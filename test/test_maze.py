from pathlib import Path

import numpy as np

from example.maze.main import create_init_pomdp, main, run_trial
from example.maze.utils import create_maze


def test_maze_trial():
    nx = 99
    ny = 19
    T = 20000

    np.random.seed(0)
    maze = create_maze(nx, ny)

    np.random.seed(0)
    Q, time, F, e_est, pos = run_trial(
        create_init_pomdp(buf_size=T),
        maze,
        T,
    )

    rtol = 1e-5
    atol = 1e-8
    data = np.load(Path(__file__).parent / "correct_results" / "maze_trial.npz")
    np.testing.assert_allclose(time, data["time"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(Q.qd_buf[..., : time + 1], data["qd"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(Q.d_buf[..., : time + 1], data["d"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(pos[:, : time + 1], data["pos"], rtol=rtol, atol=atol)  # 1-indexed
    np.testing.assert_allclose(e_est, data["e_est"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(Q.qa, data["qa"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(Q.qbi, data["qbi"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(Q.qci, data["qci"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(F, data["F"], rtol=rtol, atol=atol)
    data.close()


def profile():
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    # profile trial
    # nx = 99
    # ny = 19
    # T = 20000
    # out_dir = Path(__file__).parent / "output/maze/profile_trial"
    # out_dir.mkdir(exist_ok=True, parents=True)
    # run_trial(
    #     create_init_Q(buf_size=T),
    #     create_maze(nx, ny),
    #     T,
    # )

    # profile main
    out_dir = Path(__file__).parent / "output/maze/profile_main"
    out_dir.mkdir(exist_ok=True, parents=True)
    main(
        out_dir=out_dir,
        n_sample=1,
        n_trial=1,
    )

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()

    stats.dump_stats(out_dir / "profile.prof")


if __name__ == "__main__":
    test_maze_trial()
    # profile()

    print("All tests passed!")
