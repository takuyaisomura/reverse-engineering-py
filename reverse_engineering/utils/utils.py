def check_double_layer_args(*args) -> bool:
    """Check arguments for the double layer model.

    Args:
        args (Any): Arguments to check.

    Raises:
        AssertionError: If some of the arguments are None and some are not None.

    Returns:
        bool: True if all arguments are not None, False if all arguments are None.
    """
    is_not_none_set = set(var is not None for var in args)
    assert len(is_not_none_set) == 1, "All variables for double layer model must be None or not None"
    return is_not_none_set.pop()
