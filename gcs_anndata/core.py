"""Core functionality for GCS AnnData."""

import gcsfs
import h5py
import numpy as np
import pandas as pd
import warnings
from functools import cached_property
from scipy.sparse import csc_matrix, csr_matrix
from typing import List, Union, Tuple, Optional

from .exceptions import InvalidFormatError
from .utils import infer_sparse_matrix_format


class GCSAnnData:
    """
    Class for partially reading AnnData objects stored in h5ad format on Google Cloud Storage.

    This class provides efficient access to specific parts of an AnnData object stored
    in h5ad format on Google Cloud Storage without loading the entire dataset into memory.
    It supports reading specific rows or columns from sparse matrices (CSR or CSC format).

    Parameters
    ----------
    gcs_path : str
        Path to the h5ad file on GCS (e.g., 'gs://bucket/path/to/file.h5ad')

    Attributes
    ----------
    gcs_path : str
        Path to the h5ad file on GCS
    fs : gcsfs.GCSFileSystem
        GCS filesystem object
    shape : Tuple[int, int]
        Shape of the data matrix (n_obs, n_vars)
    sparse_format : str
        Format of the sparse matrix ('csc' or 'csr')
    obs_names : np.ndarray
        Names of observations (cell barcodes)
    var_names : np.ndarray
        Names of variables (genes)
    obs_to_idx : Dict[str, int]
        Mapping from observation names to indices
    var_to_idx : Dict[str, int]
        Mapping from variable names to indices

    Raises
    ------
    InvalidFormatError
        If the h5ad file does not contain a sparse matrix in CSR or CSC format
    ValueError
        If the h5ad file does not contain an X matrix or if the shape cannot be inferred
    """

    def __init__(self, gcs_path: str) -> None:
        """
        Initialize the GCSAnnData object.

        Parameters
        ----------
        gcs_path : str
            Path to the h5ad file on GCS (e.g., 'gs://bucket/path/to/file.h5ad')
        """
        self.gcs_path = gcs_path
        self.fs = gcsfs.GCSFileSystem()
        self.shape: Optional[Tuple[int, int]] = None
        self.sparse_format: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Initialize by reading basic metadata from the h5ad file.

        Reads the sparse format and shape of the data matrix from the h5ad file.
        Validates that the file contains a sparse matrix in CSR or CSC format.

        Raises
        ------
        InvalidFormatError
            If the h5ad file does not contain a sparse matrix in CSR or CSC format
        ValueError
            If the h5ad file does not contain an X matrix or if the shape cannot be inferred
        """
        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                # Check if X exists
                if "X" not in h5f.keys():
                    raise ValueError("No X matrix found in the h5ad file")

                # Check if the matrix is in sparse format
                X_group = h5f["X"]

                # Determine sparse format
                try:
                    self.sparse_format = infer_sparse_matrix_format(X_group)
                    if self.sparse_format not in ["csr", "csc"]:
                        raise InvalidFormatError(f"Unsupported sparse format: {self.sparse_format}")
                except ValueError as e:
                    raise InvalidFormatError(f"Only sparse matrices (CSR or CSC) are supported: {str(e)}")

                # Get shape from attributes or infer it
                if "shape" in X_group.attrs:
                    self.shape = tuple(X_group.attrs["shape"])
                else:
                    # Try to infer shape from indptr length
                    self.shape = self._infer_shape(h5f)

    @cached_property
    def obs_names(self):
        """Get observation names (lazy loaded)."""
        return self._get_index_names("obs")

    @cached_property
    def var_names(self):
        """Get variable names (lazy loaded)."""
        return self._get_index_names("var")

    @cached_property
    def obs_to_idx(self):
        """Get observation name to index mapping (lazy loaded)."""
        if self.obs_names is not None:
            return {name: idx for idx, name in enumerate(self.obs_names)}
        return None

    @cached_property
    def var_to_idx(self):
        """Get variable name to index mapping (lazy loaded)."""
        if self.var_names is not None:
            return {name: idx for idx, name in enumerate(self.var_names)}
        return None

    @cached_property
    def obs(self):
        """Get observation annotations as pandas DataFrame (lazy loaded)."""
        return self._get_dataframe("obs")

    @cached_property
    def var(self):
        """Get variable annotations as pandas DataFrame (lazy loaded)."""
        return self._get_dataframe("var")

    @cached_property
    def n_obs(self):
        """Get number of observations (cells)."""
        return self.shape[0]

    @cached_property
    def n_var(self):
        """Get number of variables (genes)."""
        return self.shape[1]

    def _decode_string_array(self, arr):
        """Decode byte strings to unicode if necessary."""
        if arr.dtype.kind == "S" or isinstance(arr[0], bytes):  # byte string
            return np.array([s.decode("utf-8") for s in arr])
        return arr

    def _get_index_names(self, group_name):
        """Get index names from a group in the h5ad file."""
        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                if group_name in h5f.keys():
                    if "_index" in h5f[group_name].attrs:
                        return self._decode_string_array(h5f[group_name][h5f[group_name].attrs["_index"]][:])
                return None

    def _get_dataframe(self, group_name):
        """
        Get a dataframe from a group in the h5ad file.

        Parameters
        ----------
        group_name : str
            Name of the group to read ('obs' or 'var')

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the data from the specified group
        """
        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                if group_name not in h5f.keys():
                    return pd.DataFrame()

                group = h5f[group_name]
                data = {}

                # Get all datasets in the group
                for key in group.keys():
                    if key == group.attrs.get("_index", None):
                        continue  # Skip the index column as it will be used for the DataFrame index

                    try:
                        # Check if it's a dataset before trying to read it
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            # Read the dataset
                            dataset = item[()]  # Use [()] instead of [:] to read the entire dataset

                            # Decode if it's a string array
                            if dataset.dtype.kind == "S" or (len(dataset) > 0 and isinstance(dataset[0], bytes)):
                                dataset = self._decode_string_array(dataset)

                            data[key] = dataset
                    except Exception as e:
                        warnings.warn(f"Error reading dataset {key}: {str(e)}")

                # Create DataFrame
                df = pd.DataFrame(data)

                # Set index using cached properties
                if group_name == "obs":
                    df.index = self.obs_names if self.obs_names is not None else pd.RangeIndex(len(df))
                elif group_name == "var":
                    df.index = self.var_names if self.var_names is not None else pd.RangeIndex(len(df))

                return df

    def _infer_shape(self, h5f) -> Tuple[int, int]:
        """Infer the shape of the data matrix if not explicitly stored."""
        X_group = h5f["X"]
        if "indptr" in X_group.keys() and "indices" in X_group.keys():
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

    def _get_indices_from_names(self, names, is_column=True):
        """
        Convert names to indices.

        Parameters
        ----------
        names : list of str or int, or single str or int
            Names or indices to convert
        is_column : bool, default=True
            If True, convert variable names, otherwise convert observation names

        Returns
        -------
        list of int
            Converted indices
        """
        # Convert single item to list
        if isinstance(names, (int, str)):
            names = [names]

        # Convert names to indices if needed
        if isinstance(names[0], str):
            name_to_idx = self.var_to_idx if is_column else self.obs_to_idx
            idx_type = "Variable" if is_column else "Observation"

            if name_to_idx is None:
                raise ValueError(f"{idx_type} names not available in this h5ad file")
            try:
                indices = [name_to_idx[name] for name in names]
            except KeyError as e:
                raise KeyError(f"{idx_type} name not found: {e}")
        else:
            indices = names

        # Validate indices
        if not all(isinstance(idx, int) for idx in indices):
            raise TypeError(f"{'Column' if is_column else 'Row'} indices must be integers")

        max_dim = self.shape[1] if is_column else self.shape[0]
        if max(indices) >= max_dim or min(indices) < 0:
            raise IndexError(f"{'Column' if is_column else 'Row'} index out of bounds. Shape: {self.shape}")

        return indices

    def _get_matrix_slice(self, indices, is_column=True):
        """
        Get a slice of the matrix (rows or columns).

        Parameters
        ----------
        indices : list of int
            Indices to extract
        is_column : bool, default=True
            If True, extract columns, otherwise extract rows

        Returns
        -------
        scipy.sparse.spmatrix
            Extracted slice as a sparse matrix
        """
        # Determine the optimal and fallback methods based on sparse format and slice type
        if is_column:  # Getting columns
            if self.sparse_format == "csc":
                result = self._get_direct_slice(indices, is_column=True)
            elif self.sparse_format == "csr":
                warnings.warn(
                    "Extracting columns from a CSR matrix is inefficient. "
                    "Consider converting your data to CSC format for better performance when accessing columns.",
                    UserWarning,
                )
                result = self._get_indirect_slice(indices, is_column=True)
            else:
                raise InvalidFormatError(f"Unsupported sparse format: {self.sparse_format}")
        else:  # Getting rows
            if self.sparse_format == "csr":
                result = self._get_direct_slice(indices, is_column=False)
            elif self.sparse_format == "csc":
                warnings.warn(
                    "Extracting rows from a CSC matrix is inefficient. "
                    "Consider converting your data to CSR format for better performance when accessing rows.",
                    UserWarning,
                )
                result = self._get_indirect_slice(indices, is_column=False)
            else:
                raise InvalidFormatError(f"Unsupported sparse format: {self.sparse_format}")

        return result

    def _extract_sparse_data(self, X_group, indices, indptr_values, indptr_map):
        """Extract data, indices and indptr arrays for the sparse matrix slice."""
        all_data = []
        all_indices = []
        all_indptr = [0]

        for idx in indices:
            # Get positions in the indptr_values array
            start_pos = indptr_map[idx]
            end_pos = indptr_map[idx + 1]

            # Get start and end values
            start = indptr_values[start_pos]
            end = indptr_values[end_pos]

            if end > start:  # Only read if there's data
                all_data.append(X_group["data"][start:end])
                all_indices.append(X_group["indices"][start:end])

            # Update indptr for the new matrix
            all_indptr.append(all_indptr[-1] + (end - start))

        # Concatenate all data and indices
        if all_data:
            data = np.concatenate(all_data)
            indices_array = np.concatenate(all_indices)
        else:
            data = np.array([], dtype=np.float32)
            indices_array = np.array([], dtype=np.int32)

        indptr = np.array(all_indptr, dtype=np.int32)

        return data, indices_array, indptr

    def _get_direct_slice(self, indices, is_column=True):
        """
        Get a slice directly from the sparse matrix in its native format.

        This is the efficient method when the slice type matches the matrix format
        (columns from CSC or rows from CSR).

        Parameters
        ----------
        indices : list of int
            Indices to extract
        is_column : bool, default=True
            If True, extract columns from CSC, otherwise extract rows from CSR

        Returns
        -------
        scipy.sparse.spmatrix
            Extracted slice as a sparse matrix
        """

        # Get unique indices while preserving order
        indices_array = np.array(indices)
        unique_indices, inverse_indices = np.unique(indices_array, return_inverse=True)
        # Check if indices are already unique and sorted
        is_unique_sorted = len(unique_indices) == len(indices_array) and np.array_equal(unique_indices, indices_array)

        # Calculate result shape based on slice type
        if is_column:
            result_shape = (self.shape[0], len(indices))
            unique_matrix_shape = (self.shape[0], len(unique_indices))
            matrix_class = csc_matrix
        else:
            result_shape = (len(indices), self.shape[1])
            unique_matrix_shape = (len(unique_indices), self.shape[1])
            matrix_class = csr_matrix

        # Handle empty indices case
        if not unique_indices.size:
            return matrix_class((0, 0), shape=result_shape)

        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                X_group = h5f["X"]

                # Create indptr indices needed for reading (each index and index+1)
                indptr_indices = sorted({i for idx in unique_indices for i in (idx, idx + 1)})

                # Read only the necessary indptr values
                indptr_values = X_group["indptr"][indptr_indices]

                # Create a mapping from original indices to positions in indptr_values
                indptr_map = {idx: pos for pos, idx in enumerate(indptr_indices)}

                # Extract and process the data
                all_data, all_indices, all_indptr = self._extract_sparse_data(
                    X_group, unique_indices, indptr_values, indptr_map
                )

                # Create the sparse matrix with unique slices
                unique_matrix = matrix_class((all_data, all_indices, all_indptr), shape=unique_matrix_shape)

                # If indices are already unique and sorted, return directly
                if is_unique_sorted:
                    return unique_matrix

                # Otherwise, rearrange columns/rows to match original indices
                if is_column:
                    return unique_matrix[:, inverse_indices]
                else:
                    return unique_matrix[inverse_indices, :]

    def _get_indirect_slice(self, indices, is_column=True):
        """
        Get a slice indirectly by loading the full matrix and then slicing.

        This is the less efficient fallback method when the slice type doesn't match
        the matrix format (rows from CSC or columns from CSR).

        Parameters
        ----------
        indices : list of int
            Indices to extract
        is_column : bool, default=True
            If True, extract columns, otherwise extract rows

        Returns
        -------
        scipy.sparse.spmatrix
            Extracted slice as a sparse matrix
        """
        with self.fs.open(self.gcs_path, "rb") as f:
            with h5py.File(f, "r") as h5f:
                X_group = h5f["X"]

                # Read all data, indices, and indptr
                data = X_group["data"][:]
                indices_array = X_group["indices"][:]
                indptr = X_group["indptr"][:]

                # Create the full sparse matrix in its native format
                if self.sparse_format == "csc":
                    full_matrix = csc_matrix((data, indices_array, indptr), shape=self.shape)
                else:  # CSR
                    full_matrix = csr_matrix((data, indices_array, indptr), shape=self.shape)

                # Extract the requested slice
                if is_column:  # Get columns
                    return full_matrix[:, indices].tocsc()
                else:  # Get rows
                    return full_matrix[indices, :].tocsr()

    def get_columns(
        self, columns: Union[List[int], List[str], int, str], as_df: bool = False
    ) -> Union[csc_matrix, pd.DataFrame]:
        """
        Get specific columns from the data matrix.

        Parameters
        ----------
        columns : list, int, or str
            Column indices, variable names, or a single index/name
        as_df : bool, default=False
            If True, return result as pandas DataFrame with appropriate indices

        Returns
        -------
        scipy.sparse.csc_matrix or pandas.DataFrame
            A CSC matrix containing only the requested columns, or a DataFrame if as_df=True

        Examples
        --------
        >>> adata = GCSAnnData('gs://bucket/file.h5ad')
        >>> # Get columns by index
        >>> cols = adata.get_columns([0, 5, 10])
        >>> # Get columns by gene name
        >>> cols = adata.get_columns(['GAPDH', 'CD3D', 'CD8A'])
        >>> # Get a single column
        >>> col = adata.get_columns('GAPDH')
        >>> # Get columns as DataFrame
        >>> df = adata.get_columns(['GAPDH', 'CD3D'], as_df=True)
        """
        # Convert names/indices and validate
        column_indices = self._get_indices_from_names(columns, is_column=True)

        # Get the matrix slice
        result = self._get_matrix_slice(column_indices, is_column=True)

        if as_df:
            # Get the original column names for the selected indices
            if isinstance(columns, (list, tuple)) and isinstance(columns[0], str):
                col_names = columns
            else:
                col_names = [self.var_names[idx] for idx in column_indices]

            # Convert to dense matrix for DataFrame creation
            dense_matrix = result.toarray()

            # Create DataFrame with appropriate row and column indices
            df = pd.DataFrame(dense_matrix, index=self.obs_names, columns=col_names)
            return df

        return result

    def get_rows(
        self, rows: Union[List[int], List[str], int, str], as_df: bool = False
    ) -> Union[csr_matrix, pd.DataFrame]:
        """
        Get specific rows from the data matrix.

        Parameters
        ----------
        rows : list, int, or str
            Row indices, observation names, or a single index/name
        as_df : bool, default=False
            If True, return result as pandas DataFrame with appropriate indices

        Returns
        -------
        scipy.sparse.csr_matrix or pandas.DataFrame
            A CSR matrix containing only the requested rows, or a DataFrame if as_df=True

        Examples
        --------
        >>> adata = GCSAnnData('gs://bucket/file.h5ad')
        >>> # Get rows by index
        >>> rows = adata.get_rows([0, 5, 10])
        >>> # Get rows by cell barcode
        >>> rows = adata.get_rows(['AAACCCAAGCGCCCAT-1', 'AAACCCATCAGCCCAG-1'])
        >>> # Get a single row
        >>> row = adata.get_rows('AAACCCAAGCGCCCAT-1')
        >>> # Get rows as DataFrame
        >>> df = adata.get_rows(['AAACCCAAGCGCCCAT-1', 'AAACCCATCAGCCCAG-1'], as_df=True)
        """
        # Convert names/indices and validate
        row_indices = self._get_indices_from_names(rows, is_column=False)

        # Get the matrix slice
        result = self._get_matrix_slice(row_indices, is_column=False)

        if as_df:
            # Get the original row names for the selected indices
            if isinstance(rows, (list, tuple)) and isinstance(rows[0], str):
                row_names = rows
            else:
                row_names = [self.obs_names[idx] for idx in row_indices]

            # Convert to dense matrix for DataFrame creation
            dense_matrix = result.toarray()

            # Create DataFrame with appropriate row and column indices
            df = pd.DataFrame(dense_matrix, index=row_names, columns=self.var_names)
            return df

        return result
