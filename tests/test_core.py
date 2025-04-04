"""Tests for the core functionality of GCS AnnData."""

import os
import pytest
import numpy as np
import pandas as pd
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

    def test_get_columns(self, test_bucket, test_files):
        """Test getting columns from the data matrix."""
        for matrix_type, filename in test_files.items():
            # Skip standard format as it will raise an error
            if matrix_type == "standard":
                continue

            gcs_path = get_gcs_path(test_bucket, filename)
            adata = GCSAnnData(gcs_path)

            # Test getting columns by index
            col_indices = [0, 2, 5]
            cols = adata.get_columns(col_indices)
            assert sparse.isspmatrix_csc(cols)
            assert cols.shape == (adata.shape[0], len(col_indices))

            # Test getting a single column
            single_col = adata.get_columns(1)
            assert sparse.isspmatrix_csc(single_col)
            assert single_col.shape == (adata.shape[0], 1)

            # Test getting columns as DataFrame
            cols_df = adata.get_columns(col_indices, as_df=True)
            assert isinstance(cols_df, pd.DataFrame)
            assert cols_df.shape == (adata.shape[0], len(col_indices))
            assert list(cols_df.index) == list(adata.obs_names)

            # Test getting columns by name if var_names is available
            if adata.var_names is not None:
                var_names = [adata.var_names[i] for i in col_indices[:2]]
                cols_by_name = adata.get_columns(var_names)
                assert sparse.isspmatrix_csc(cols_by_name)
                assert cols_by_name.shape == (adata.shape[0], len(var_names))

                # Test getting columns by name as DataFrame
                cols_by_name_df = adata.get_columns(var_names, as_df=True)
                assert isinstance(cols_by_name_df, pd.DataFrame)
                assert cols_by_name_df.shape == (adata.shape[0], len(var_names))
                assert list(cols_by_name_df.columns) == var_names

    def test_get_rows(self, test_bucket, test_files):
        """Test getting rows from the data matrix."""
        for matrix_type, filename in test_files.items():
            # Skip standard format as it will raise an error
            if matrix_type == "standard":
                continue

            gcs_path = get_gcs_path(test_bucket, filename)
            adata = GCSAnnData(gcs_path)

            # Test getting rows by index
            row_indices = [0, 3, 7]
            rows = adata.get_rows(row_indices)
            assert sparse.isspmatrix_csr(rows)
            assert rows.shape == (len(row_indices), adata.shape[1])

            # Test getting a single row
            single_row = adata.get_rows(1)
            assert sparse.isspmatrix_csr(single_row)
            assert single_row.shape == (1, adata.shape[1])

            # Test getting rows as DataFrame
            rows_df = adata.get_rows(row_indices, as_df=True)
            assert isinstance(rows_df, pd.DataFrame)
            assert rows_df.shape == (len(row_indices), adata.shape[1])
            assert list(rows_df.columns) == list(adata.var_names)

            # Test getting rows by name if obs_names is available
            if adata.obs_names is not None:
                obs_names = [adata.obs_names[i] for i in row_indices[:2]]
                rows_by_name = adata.get_rows(obs_names)
                assert sparse.isspmatrix_csr(rows_by_name)
                assert rows_by_name.shape == (len(obs_names), adata.shape[1])

                # Test getting rows by name as DataFrame
                rows_by_name_df = adata.get_rows(obs_names, as_df=True)
                assert isinstance(rows_by_name_df, pd.DataFrame)
                assert rows_by_name_df.shape == (len(obs_names), adata.shape[1])
                assert list(rows_by_name_df.index) == obs_names

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
