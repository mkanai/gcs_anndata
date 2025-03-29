"""Core functionality for GCS AnnData."""

import h5py
import gcsfs
import numpy as np
import warnings
from scipy.sparse import csc_matrix, csr_matrix
from typing import List, Union, Tuple, Optional, Sequence

from .exceptions import InvalidFormatError
from .utils import infer_sparse_matrix_format


class GCSAnnData:
    """
    Class for partially reading AnnData objects stored in h5ad format on Google Cloud Storage.

    Parameters
    ----------
    gcs_path : str
        Path to the h5ad file on GCS (e.g., 'gs://bucket/path/to/file.h5ad')

    Attributes
    ----------
    shape : tuple
        Shape of the data matrix (n_obs, n_vars)
    sparse_format : str
        Format of the sparse matrix ('csc' or 'csr')
    obs_names : np.ndarray
        Names of observations (cell barcodes)
    var_names : np.ndarray
        Names of variables (genes)
    """

    def __init__(self, gcs_path: str):
        """Initialize the GCSAnnData object."""
        self.gcs_path = gcs_path
        self.fs = gcsfs.GCSFileSystem()
        self.obs_names = None
        self.var_names = None
        self._initialize()

    def _initialize(self):
        """Initialize by reading metadata from the h5ad file."""
        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                # Get shape from attributes or infer it
                if "X" in h5f and "shape" in h5f["X"].attrs:
                    self.shape = tuple(h5f["X"].attrs["shape"])
                else:
                    # Try to infer shape from indptr length
                    self.shape = self._infer_shape(h5f)

                # Determine sparse format
                self.sparse_format = infer_sparse_matrix_format(h5f["X"])

                # Load observation names (cell barcodes)
                if "obs" in h5f:
                    if "index" in h5f["obs"]:
                        self.obs_names = self._decode_string_array(h5f["obs"]["index"][:])
                    elif "_index" in h5f["obs"]:
                        self.obs_names = self._decode_string_array(h5f["obs"]["_index"][:])

                # Load variable names (genes)
                if "var" in h5f:
                    if "index" in h5f["var"]:
                        self.var_names = self._decode_string_array(h5f["var"]["index"][:])
                    elif "_index" in h5f["var"]:
                        self.var_names = self._decode_string_array(h5f["var"]["_index"][:])

    def _decode_string_array(self, arr):
        """Decode byte strings to unicode if necessary."""
        if arr.dtype.kind == "S":  # byte string
            return np.array([s.decode("utf-8") for s in arr])
        return arr

    def _infer_shape(self, h5f) -> Tuple[int, int]:
        """Infer the shape of the data matrix if not explicitly stored."""
        X_group = h5f["X"]
        if "indptr" in X_group and "indices" in X_group:
            indptr = X_group["indptr"][:]
            if len(X_group["indices"]) > 0:
                indices_max = X_group["indices"][:].max()
            else:
                indices_max = 0

            # Determine if it's CSC or CSR based on other indicators
            if "format" in X_group.attrs:
                format_str = X_group.attrs["format"].lower()
                if "csc" in format_str:
                    return (indices_max + 1, len(indptr) - 1)
                elif "csr" in format_str:
                    return (len(indptr) - 1, indices_max + 1)

            # Make a best guess
            return (indices_max + 1, len(indptr) - 1)

        raise ValueError("Cannot infer shape of the data matrix")

    def get_columns(self, columns: Union[List[int], List[str], int, str]) -> csc_matrix:
        """
        Get specific columns from the data matrix.

        Parameters
        ----------
        columns : list, int, or str
            Column indices, variable names, or a single index/name

        Returns
        -------
        scipy.sparse.csc_matrix
            A CSC matrix containing only the requested columns

        Examples
        --------
        >>> adata = GCSAnnData('gs://bucket/file.h5ad')
        >>> # Get columns by index
        >>> cols = adata.get_columns([0, 5, 10])
        >>> # Get columns by gene name
        >>> cols = adata.get_columns(['GAPDH', 'CD3D', 'CD8A'])
        >>> # Get a single column
        >>> col = adata.get_columns('GAPDH')
        """
        # Convert single item to list
        if isinstance(columns, (int, str)):
            columns = [columns]

        # Convert variable names to indices if needed
        if self.var_names is not None and isinstance(columns[0], str):
            try:
                # Create a mapping of var_names to indices for efficient lookup
                var_to_idx = {name: idx for idx, name in enumerate(self.var_names)}
                column_indices = [var_to_idx[col] for col in columns]
            except KeyError as e:
                raise KeyError(f"Variable name not found: {e}")
        else:
            column_indices = columns

        # Validate indices
        if not all(isinstance(idx, int) for idx in column_indices):
            raise TypeError("Column indices must be integers")

        if max(column_indices) >= self.shape[1] or min(column_indices) < 0:
            raise IndexError(f"Column index out of bounds. Shape: {self.shape}")

        if self.sparse_format == "csc":
            return self._get_csc_columns(column_indices)
        elif self.sparse_format == "csr":
            warnings.warn(
                "Extracting columns from a CSR matrix is inefficient. "
                "Consider converting your data to CSC format for better performance when accessing columns.",
                UserWarning,
            )
            return self._get_csr_columns(column_indices)
        else:
            raise InvalidFormatError(f"Unsupported sparse format: {self.sparse_format}")

    def get_rows(self, rows: Union[List[int], List[str], int, str]) -> csr_matrix:
        """
        Get specific rows from the data matrix.

        Parameters
        ----------
        rows : list, int, or str
            Row indices, observation names, or a single index/name

        Returns
        -------
        scipy.sparse.csr_matrix
            A CSR matrix containing only the requested rows

        Examples
        --------
        >>> adata = GCSAnnData('gs://bucket/file.h5ad')
        >>> # Get rows by index
        >>> rows = adata.get_rows([0, 5, 10])
        >>> # Get rows by cell barcode
        >>> rows = adata.get_rows(['AAACCCAAGCGCCCAT-1', 'AAACCCATCAGCCCAG-1'])
        >>> # Get a single row
        >>> row = adata.get_rows('AAACCCAAGCGCCCAT-1')
        """
        # Convert single item to list
        if isinstance(rows, (int, str)):
            rows = [rows]

        # Convert observation names to indices if needed
        if self.obs_names is not None and isinstance(rows[0], str):
            try:
                # Create a mapping of obs_names to indices for efficient lookup
                obs_to_idx = {name: idx for idx, name in enumerate(self.obs_names)}
                row_indices = [obs_to_idx[row] for row in rows]
            except KeyError as e:
                raise KeyError(f"Observation name not found: {e}")
        else:
            row_indices = rows

        # Validate indices
        if not all(isinstance(idx, int) for idx in row_indices):
            raise TypeError("Row indices must be integers")

        if max(row_indices) >= self.shape[0] or min(row_indices) < 0:
            raise IndexError(f"Row index out of bounds. Shape: {self.shape}")

        if self.sparse_format == "csc":
            warnings.warn(
                "Extracting rows from a CSC matrix is inefficient. "
                "Consider converting your data to CSR format for better performance when accessing rows.",
                UserWarning,
            )
            return self._get_csc_rows(row_indices)
        elif self.sparse_format == "csr":
            return self._get_csr_rows(row_indices)
        else:
            raise InvalidFormatError(f"Unsupported sparse format: {self.sparse_format}")

    def _get_csc_columns(self, column_indices: List[int]) -> csc_matrix:
        """Get columns from a CSC matrix."""
        column_indices = sorted(set(column_indices))

        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                X_group = h5f["X"]

                # Read indptr for all requested columns
                col_indptr = X_group["indptr"][column_indices + [max(column_indices) + 1]]

                # Calculate start and end positions for each column
                starts = col_indptr[:-1]
                ends = col_indptr[1:]

                # Initialize lists to store data
                all_data = []
                all_indices = []
                all_indptr = [0]

                # Read data and indices for all needed ranges
                for start, end in zip(starts, ends):
                    if end > start:  # Only read if there's data
                        data_slice = X_group["data"][start:end]
                        indices_slice = X_group["indices"][start:end]

                        all_data.append(data_slice)
                        all_indices.append(indices_slice)

                    # Update indptr for the new matrix
                    all_indptr.append(all_indptr[-1] + (end - start))

                # Concatenate all data and indices
                if all_data:
                    data = np.concatenate(all_data)
                    indices = np.concatenate(all_indices)
                else:
                    data = np.array([], dtype=np.float32)
                    indices = np.array([], dtype=np.int32)

                indptr = np.array(all_indptr, dtype=np.int32)

                # Create the CSC matrix with only the requested columns
                result_shape = (self.shape[0], len(column_indices))
                result = csc_matrix((data, indices, indptr), shape=result_shape)

                return result

    def _get_csr_rows(self, row_indices: List[int]) -> csr_matrix:
        """Get rows from a CSR matrix."""
        row_indices = sorted(set(row_indices))

        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                X_group = h5f["X"]

                # Read indptr for all requested rows
                row_indptr = X_group["indptr"][row_indices + [max(row_indices) + 1]]

                # Calculate start and end positions for each row
                starts = row_indptr[:-1]
                ends = row_indptr[1:]

                # Initialize lists to store data
                all_data = []
                all_indices = []
                all_indptr = [0]

                # Read data and indices for all needed ranges
                for start, end in zip(starts, ends):
                    if end > start:  # Only read if there's data
                        data_slice = X_group["data"][start:end]
                        indices_slice = X_group["indices"][start:end]

                        all_data.append(data_slice)
                        all_indices.append(indices_slice)

                    # Update indptr for the new matrix
                    all_indptr.append(all_indptr[-1] + (end - start))

                # Concatenate all data and indices
                if all_data:
                    data = np.concatenate(all_data)
                    indices = np.concatenate(all_indices)
                else:
                    data = np.array([], dtype=np.float32)
                    indices = np.array([], dtype=np.int32)

                indptr = np.array(all_indptr, dtype=np.int32)

                # Create the CSR matrix with only the requested rows
                result_shape = (len(row_indices), self.shape[1])
                result = csr_matrix((data, indices, indptr), shape=result_shape)

                return result

    def _get_csc_rows(self, row_indices: List[int]) -> csr_matrix:
        """Get rows from a CSC matrix (less efficient)."""
        # This is a fallback method that gets all columns and then selects rows
        # For better performance with large matrices, implement a direct method
        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                X_group = h5f["X"]

                # Read all data, indices, and indptr
                data = X_group["data"][:]
                indices = X_group["indices"][:]
                indptr = X_group["indptr"][:]

                # Create the full CSC matrix
                full_matrix = csc_matrix((data, indices, indptr), shape=self.shape)

                # Extract the requested rows
                return full_matrix[row_indices, :].tocsr()

    def _get_csr_columns(self, column_indices: List[int]) -> csc_matrix:
        """Get columns from a CSR matrix (less efficient)."""
        # This is a fallback method that gets all rows and then selects columns
        # For better performance with large matrices, implement a direct method
        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                X_group = h5f["X"]

                # Read all data, indices, and indptr
                data = X_group["data"][:]
                indices = X_group["indices"][:]
                indptr = X_group["indptr"][:]

                # Create the full CSR matrix
                full_matrix = csr_matrix((data, indices, indptr), shape=self.shape)

                # Extract the requested columns
                return full_matrix[:, column_indices].tocsc()
