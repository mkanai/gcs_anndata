"""Core functionality for GCS AnnData."""

import h5py
import gcsfs
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from typing import List, Union, Tuple, Optional

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
    """

    def __init__(self, gcs_path: str):
        """Initialize the GCSAnnData object."""
        self.gcs_path = gcs_path
        self.fs = gcsfs.GCSFileSystem()
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

                # Store variable and observation names if available
                if "var" in h5f and "index" in h5f["var"]:
                    self.var_names = h5f["var"]["index"][:]
                if "obs" in h5f and "index" in h5f["obs"]:
                    self.obs_names = h5f["obs"]["index"][:]

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

    def get_columns(self, column_indices: Union[List[int], int]) -> csc_matrix:
        """
        Get specific columns from the data matrix.

        Parameters
        ----------
        column_indices : list or int
            Column index or list of column indices to extract

        Returns
        -------
        scipy.sparse.csc_matrix
            A CSC matrix containing only the requested columns
        """
        if isinstance(column_indices, int):
            column_indices = [column_indices]

        if self.sparse_format == "csc":
            return self._get_csc_columns(column_indices)
        elif self.sparse_format == "csr":
            return self._get_csr_columns(column_indices)
        else:
            raise InvalidFormatError(f"Unsupported sparse format: {self.sparse_format}")

    def get_rows(self, row_indices: Union[List[int], int]) -> csr_matrix:
        """
        Get specific rows from the data matrix.

        Parameters
        ----------
        row_indices : list or int
            Row index or list of row indices to extract

        Returns
        -------
        scipy.sparse.csr_matrix
            A CSR matrix containing only the requested rows
        """
        if isinstance(row_indices, int):
            row_indices = [row_indices]

        if self.sparse_format == "csc":
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
