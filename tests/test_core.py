"""Tests for the core functionality of GCS AnnData."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tempfile
from scipy import sparse
from gcs_anndata import GCSAnnData
from gcs_anndata.exceptions import InvalidFormatError


def get_gcs_path(test_bucket, filename):
    return f"gs://{test_bucket}/{filename}"


class TestGCSAnnData:
    def test_init(self, test_bucket, test_files):
        """Test initialization of GCSAnnData object."""
        for matrix_type, filename in test_files.items():
            gcs_path = get_gcs_path(test_bucket, filename)

            if matrix_type == "standard":
                # Standard format should raise an InvalidFormatError
                with pytest.raises(InvalidFormatError, match="Only sparse matrices"):
                    GCSAnnData(gcs_path)
            else:
                # CSR and CSC formats should initialize successfully
                adata = GCSAnnData(gcs_path)
                assert isinstance(adata, GCSAnnData)
                assert adata.gcs_path == gcs_path
                assert adata.shape is not None
                assert adata.sparse_format in ["csc", "csr"]

    def test_metadata(self, test_bucket, test_files):
        """Test reading metadata from GCS."""
        for matrix_type, filename in test_files.items():
            # Skip standard format as it will raise an error
            if matrix_type == "standard":
                continue

            gcs_path = get_gcs_path(test_bucket, filename)
            adata = GCSAnnData(gcs_path)

            # Check basic properties
            assert adata.shape[0] > 0
            assert adata.shape[1] > 0

            # Check index names
            assert adata.obs_names is not None
            assert len(adata.obs_names) == adata.shape[0]
            assert adata.var_names is not None
            assert len(adata.var_names) == adata.shape[1]

            # Check index mappings
            assert adata.obs_to_idx is not None
            assert len(adata.obs_to_idx) == adata.shape[0]
            assert adata.var_to_idx is not None
            assert len(adata.var_to_idx) == adata.shape[1]

            # Check sparse format
            if matrix_type == "csr":
                assert adata.sparse_format == "csr"
            elif matrix_type == "csc":
                assert adata.sparse_format == "csc"

    def _get_data(self, test_bucket, test_files, gcs_fs, dimension="columns"):
        """Generic test for getting rows or columns from the data matrix."""
        is_column_test = dimension == "columns"
        get_method = "get_columns" if is_column_test else "get_rows"
        expected_format = "csc" if is_column_test else "csr"

        for matrix_type, filename in test_files.items():
            # Skip standard format as it will raise an error
            if matrix_type == "standard":
                continue

            gcs_path = get_gcs_path(test_bucket, filename)
            adata_gcs = GCSAnnData(gcs_path)

            # Download the file to a temporary location for comparison
            with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp:
                local_path = tmp.name
                gcs_fs.get(gcs_path, local_path)
                adata_local = ad.read_h5ad(local_path)

                # Test getting data by index
                indices = [0, 3, 7, 3, 25]

                # Handle expected warnings about inefficient operations
                if (is_column_test and adata_gcs.sparse_format == "csr") or (
                    not is_column_test and adata_gcs.sparse_format == "csc"
                ):
                    # We expect a warning about inefficient operations
                    with pytest.warns(UserWarning, match="inefficient"):
                        data_gcs = getattr(adata_gcs, get_method)(indices)
                else:
                    # No warning expected
                    data_gcs = getattr(adata_gcs, get_method)(indices)

                # Verify format and shape
                assert getattr(sparse, f"isspmatrix_{expected_format}")(data_gcs)
                expected_shape = (
                    (len(indices), adata_gcs.shape[1]) if not is_column_test else (adata_gcs.shape[0], len(indices))
                )
                assert data_gcs.shape == expected_shape

                # Compare with anndata
                if sparse.issparse(adata_local.X):
                    if is_column_test:
                        data_local = adata_local.X[:, indices].tocsc()
                    else:
                        data_local = adata_local.X[indices, :].tocsr()
                else:
                    matrix_type = sparse.csc_matrix if is_column_test else sparse.csr_matrix
                    if is_column_test:
                        data_local = matrix_type(adata_local.X[:, indices])
                    else:
                        data_local = matrix_type(adata_local.X[indices, :])

                # Check that the data is the same
                assert (data_gcs.data == data_local.data).all()
                assert (data_gcs.indices == data_local.indices).all()
                assert (data_gcs.indptr == data_local.indptr).all()

                # Test getting a single element
                single_idx = 1
                single_data_gcs = getattr(adata_gcs, get_method)(single_idx)
                expected_single_shape = (1, adata_gcs.shape[1]) if not is_column_test else (adata_gcs.shape[0], 1)
                assert single_data_gcs.shape == expected_single_shape

                # Test getting data as DataFrame
                data_df = getattr(adata_gcs, get_method)(indices, as_df=True)
                assert isinstance(data_df, pd.DataFrame)
                assert data_df.shape == expected_shape

                # Test getting data by name
                names_attr = "obs_names" if not is_column_test else "var_names"
                if getattr(adata_gcs, names_attr) is not None:
                    names = [getattr(adata_gcs, names_attr)[i] for i in indices[:2]]
                    data_by_name_gcs = getattr(adata_gcs, get_method)(names)
                    expected_name_shape = (
                        (len(names), adata_gcs.shape[1]) if not is_column_test else (adata_gcs.shape[0], len(names))
                    )
                    assert data_by_name_gcs.shape == expected_name_shape

    def test_get_columns(self, test_bucket, test_files, gcs_fs):
        """Test getting columns from the data matrix."""
        self._get_data(test_bucket, test_files, gcs_fs, dimension="columns")

    def test_get_rows(self, test_bucket, test_files, gcs_fs):
        """Test getting rows from the data matrix."""
        self._get_data(test_bucket, test_files, gcs_fs, dimension="rows")

    def test_obs_var_dataframes(self, test_bucket, test_files, gcs_fs):
        """Test getting obs and var DataFrames."""
        for matrix_type, filename in test_files.items():
            # Skip standard format as it will raise an error
            if matrix_type == "standard":
                continue

            gcs_path = get_gcs_path(test_bucket, filename)
            adata_gcs = GCSAnnData(gcs_path)

            # Download the file to a temporary location for comparison
            with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp:
                local_path = tmp.name
                gcs_fs.get(gcs_path, local_path)
                adata_local = ad.read_h5ad(local_path)

                # Test obs DataFrame
                obs_df = adata_gcs.obs
                assert isinstance(obs_df, pd.DataFrame)
                assert obs_df.shape[0] == adata_gcs.shape[0]  # Number of rows should match n_obs

                # Check that index matches obs_names
                assert obs_df.index.equals(pd.Index(adata_gcs.obs_names))

                # Compare with anndata's obs DataFrame
                # Note: We only check columns that exist in both DataFrames
                for col in set(obs_df.columns).intersection(set(adata_local.obs.columns)):
                    # Convert to same dtype for comparison if needed
                    gcs_col = obs_df[col]
                    local_col = adata_local.obs[col]

                    if isinstance(local_col.dtype, pd.CategoricalDtype):
                        # For categorical data, compare the values
                        assert np.array_equal(gcs_col.values, local_col.values)
                    elif pd.api.types.is_numeric_dtype(local_col) and pd.api.types.is_numeric_dtype(gcs_col):
                        # For numeric data, allow for small differences
                        assert np.allclose(gcs_col.values, local_col.values, equal_nan=True)
                    else:
                        # For other types, compare directly
                        assert np.array_equal(gcs_col.values, local_col.values)

                # Test var DataFrame
                var_df = adata_gcs.var
                assert isinstance(var_df, pd.DataFrame)
                assert var_df.shape[0] == adata_gcs.shape[1]  # Number of rows should match n_vars

                # Check that index matches var_names
                assert var_df.index.equals(pd.Index(adata_gcs.var_names))

                # Compare with anndata's var DataFrame
                # Note: We only check columns that exist in both DataFrames
                for col in set(var_df.columns).intersection(set(adata_local.var.columns)):
                    # Convert to same dtype for comparison if needed
                    gcs_col = var_df[col]
                    local_col = adata_local.var[col]

                    if isinstance(local_col.dtype, pd.CategoricalDtype):
                        # For categorical data, compare the values
                        assert np.array_equal(gcs_col.values, local_col.values)
                    elif pd.api.types.is_numeric_dtype(local_col) and pd.api.types.is_numeric_dtype(gcs_col):
                        # For numeric data, allow for small differences
                        assert np.allclose(gcs_col.values, local_col.values, equal_nan=True)
                    else:
                        # For other types, compare directly
                        assert np.array_equal(gcs_col.values, local_col.values)

    def test_error_handling(self, test_bucket, test_files):
        """Test error handling for invalid inputs."""
        for matrix_type, filename in test_files.items():
            # Skip standard format as it will raise an error
            if matrix_type == "standard":
                continue

            gcs_path = get_gcs_path(test_bucket, filename)
            adata = GCSAnnData(gcs_path)

            # Test invalid column index
            with pytest.raises(IndexError):
                adata.get_columns(adata.shape[1] + 10)

            # Test invalid row index
            with pytest.raises(IndexError):
                adata.get_rows(adata.shape[0] + 5)

            # Test invalid column name
            if adata.var_names is not None:
                with pytest.raises(KeyError):
                    adata.get_columns("non_existent_column_name")

            # Test invalid row name
            if adata.obs_names is not None:
                with pytest.raises(KeyError):
                    adata.get_rows("non_existent_row_name")
