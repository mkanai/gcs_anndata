"""Utility functions for GCS AnnData."""

from typing import Dict, Any


def infer_sparse_matrix_format(X_group) -> str:
    """
    Infer the sparse matrix format from the HDF5 group.

    Parameters
    ----------
    X_group : h5py.Group
        HDF5 group containing the sparse matrix data

    Returns
    -------
    str
        'csc' or 'csr'

    Raises
    ------
    ValueError
        If the format cannot be inferred
    """
    # Check if format is explicitly stored
    if "format" in X_group.attrs:
        format_str = X_group.attrs["format"].lower()
        if "csc" in format_str:
            return "csc"
        elif "csr" in format_str:
            return "csr"

    # Infer from shape and indptr length
    if "shape" in X_group.attrs and "indptr" in X_group:
        shape = X_group.attrs["shape"]
        indptr_len = X_group["indptr"].shape[0]

        if indptr_len - 1 == shape[1]:
            return "csc"
        elif indptr_len - 1 == shape[0]:
            return "csr"

    raise ValueError("Unable to infer sparse matrix format")
