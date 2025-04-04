"""Utility functions for GCS AnnData."""

import h5py


def infer_sparse_matrix_format(X_group: h5py.Group) -> str:
    """
    Infer the sparse matrix format from the HDF5 group.

    This function attempts to determine whether a sparse matrix stored in an HDF5 group
    is in CSC (Compressed Sparse Column) or CSR (Compressed Sparse Row) format.
    It first checks for explicit format information in the group attributes,
    and if not found, tries to infer the format from the shape and indptr length.

    Parameters
    ----------
    X_group : h5py.Group
        HDF5 group containing the sparse matrix data. This should typically be
        the 'X' group in an h5ad file.

    Returns
    -------
    str
        The inferred sparse matrix format: 'csc' or 'csr'

    Raises
    ------
    ValueError
        If the format cannot be inferred from the available information
    """
    # Check if format is explicitly stored in attributes
    if "format" in X_group.attrs:
        format_str = X_group.attrs["format"].lower()
        if "csc" in format_str:
            return "csc"
        elif "csr" in format_str:
            return "csr"

    # If not explicitly stored, infer from shape and indptr length
    if "shape" in X_group.attrs and "indptr" in X_group.keys():
        shape = tuple(X_group.attrs["shape"])
        indptr_len = X_group["indptr"].shape[0]

        # In CSC format, indptr length = num_columns + 1
        if indptr_len - 1 == shape[1]:
            return "csc"
        # In CSR format, indptr length = num_rows + 1
        elif indptr_len - 1 == shape[0]:
            return "csr"

    # If we can't determine the format, raise an error
    raise ValueError("Unable to infer sparse matrix format")
