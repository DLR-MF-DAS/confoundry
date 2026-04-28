import numpy as np

def child_parent_dict_to_prior_knowledge(relations, variables=None, dtype=int):
    """
    Convert a dict of the form {(child, parent): -1|0|1} into a
    DirectLiNGAM prior knowledge matrix.

    Parameters
    ----------
    relations : dict
        Keys are (child, parent).
        Values are:
            1  -> parent is known to be an ancestor of child
            0  -> parent is known NOT to be an ancestor of child
           -1  -> unknown
    variables : list, optional
        Variable ordering to use in the matrix.
        If None, order is inferred from first appearance in `relations`.
    dtype : type
        Matrix dtype, default int.

    Returns
    -------
    pk : np.ndarray of shape (n_variables, n_variables)
        Prior knowledge matrix for lingam.DirectLiNGAM.
    name_to_idx : dict
        Mapping from variable name to matrix index.
    """
    valid_values = {-1, 0, 1}

    # Infer variable order if not provided
    if variables is None:
        seen = set()
        variables = []
        for (child, parent) in relations:
            if parent not in seen:
                variables.append(parent)
                seen.add(parent)
            if child not in seen:
                variables.append(child)
                seen.add(child)

    name_to_idx = {name: i for i, name in enumerate(variables)}
    n = len(variables)

    # Default: unknown everywhere
    pk = np.full((n, n), -1, dtype=dtype)

    for key, value in relations.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f"Invalid key {key!r}; expected (child, parent).")
        if value not in valid_values:
            raise ValueError(
                f"Invalid value {value!r} for {key!r}; expected one of -1, 0, 1."
            )

        child, parent = key

        if child not in name_to_idx or parent not in name_to_idx:
            raise ValueError(f"Unknown variable in key {key!r}.")

        i = name_to_idx[parent]  # source / ancestor
        j = name_to_idx[child]   # destination / descendant
        pk[i, j] = value

    # Diagonal should stay unknown
    np.fill_diagonal(pk, -1)

    return pk, name_to_idx
