import numpy as np


def create_maze(
    size_x: int,
    size_y: int,
) -> np.ndarray:
    maze = np.zeros((size_y, size_x), dtype=np.int32)

    # set borders
    maze[[0, -1], :] = 1
    maze[:, [0, -1]] = 1

    # set goal
    maze[(size_y + 1) // 2 - 1, -1] = 0

    # set obstacles
    # create a grid of fixed obstacles
    maze[2:-2:2, 2:-2:2] = 1
    # add random walls around fixed obstacles
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    for y in range(2, size_y - 2, 2):
        for x in range(2, size_x - 2, 2):
            while True:
                dy, dx = directions[np.random.randint(4)]
                y_new, x_new = y + dy, x + dx
                if maze[y_new, x_new] == 0:
                    maze[y_new, x_new] = 1
                    break

    return maze


def qd_matrix(
    qd,
    directions,
    n_decision_steps,
    size_view,
):
    """Visualize decision probabilities in matrix form.
    qd (Nd,): decision probabilities
    directions (np.ndarray): (n_decisions, 2)
    n_decision_steps (int): how many steps ahead to make decisions
    size_view (int): size of the view
    """
    n_decisions = len(directions)
    Nd = n_decisions**n_decision_steps
    assert len(qd) == Nd

    qd_mat = np.zeros((size_view, size_view))

    # Convert each linear index of qd to n_decision_steps-dimensional index
    # Then, combine them to (n_decision_steps, Nd)
    indices = np.array(np.unravel_index(np.arange(Nd), (n_decisions,) * n_decision_steps))  # (n_decision_steps, Nd)
    # sum the moves at all decision steps
    total_moves = directions[indices].sum(axis=0)  # (n_decision_steps, Nd, 2) -> (Nd, 2)
    # convert to the center-based coordinates
    pos = total_moves + (size_view + 1) // 2 - 1

    # return the maximum probability at each position
    np.maximum.at(qd_mat, (pos[:, 0], pos[:, 1]), qd)

    return qd_mat
