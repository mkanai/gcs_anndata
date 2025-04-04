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

    def test_get_columns(self, test_bucket, test_files, gcs_fs):
        """Test getting columns from the data matrix."""

        for matrix_type, filename in test_files.items():
            # Skip standard format as it will raise an error
            if matrix_type == "standard":
                continue

            gcs_path = get_gcs_path(test_bucket, filename)
            adata_gcs = GCSAnnData(gcs_path)

            # Download the file to a temporary location
            with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp:
                local_path = tmp.name
                gcs_fs.get(gcs_path, local_path)

                # Read with anndata for comparison
                adata_local = ad.read_h5ad(local_path)

                # Test getting columns by index
                col_indices = [0, 2, 5]
                cols_gcs = adata_gcs.get_columns(col_indices)
                assert sparse.isspmatrix_csc(cols_gcs)
                assert cols_gcs.shape == (adata_gcs.shape[0], len(col_indices))

                # Compare with anndata
                if sparse.issparse(adata_local.X):
                    cols_local = adata_local.X[:, col_indices].tocsc()
                else:
                    cols_local = sparse.csc_matrix(adata_local.X[:, col_indices])

                # Check that the data is the same
                assert (cols_gcs.data == cols_local.data).all()
                assert (cols_gcs.indices == cols_local.indices).all()
                assert (cols_gcs.indptr == cols_local.indptr).all()

                # Test getting a single column
                single_col_gcs = adata_gcs.get_columns(1)
                assert sparse.isspmatrix_csc(single_col_gcs)
                assert single_col_gcs.shape == (adata_gcs.shape[0], 1)

                # Compare with anndata
                if sparse.issparse(adata_local.X):
                    single_col_local = adata_local.X[:, 1].tocsc()
                else:
                    single_col_local = sparse.csc_matrix(adata_local.X[:, 1])

                # Check that the data is the same
                assert (single_col_gcs.data == single_col_local.data).all()
                assert (single_col_gcs.indices == single_col_local.indices).all()
                assert (single_col_gcs.indptr == single_col_local.indptr).all()

                # Test getting columns as DataFrame
                cols_df = adata_gcs.get_columns(col_indices, as_df=True)
                assert isinstance(cols_df, pd.DataFrame)
                assert cols_df.shape == (adata_gcs.shape[0], len(col_indices))
                assert list(cols_df.index) == list(adata_gcs.obs_names)

                # Compare with anndata DataFrame
                cols_df_local = pd.DataFrame(
                    adata_local.X[:, col_indices].toarray(),
                    index=adata_local.obs_names,
                    columns=[adata_local.var_names[i] for i in col_indices],
                )
                pd.testing.assert_frame_equal(cols_df, cols_df_local, check_dtype=False)

                # Test getting columns by name if var_names is available
                if adata_gcs.var_names is not None:
                    var_names = [adata_gcs.var_names[i] for i in col_indices[:2]]
                    cols_by_name_gcs = adata_gcs.get_columns(var_names)
                    assert sparse.isspmatrix_csc(cols_by_name_gcs)
                    assert cols_by_name_gcs.shape == (adata_gcs.shape[0], len(var_names))

                    # Compare with anndata
                    local_indices = [list(adata_local.var_names).index(name) for name in var_names]
                    if sparse.issparse(adata_local.X):
                        cols_by_name_local = adata_local.X[:, local_indices].tocsc()
                    else:
                        cols_by_name_local = sparse.csc_matrix(adata_local.X[:, local_indices])

                    # Check that the data is the same
                    assert (cols_by_name_gcs.data == cols_by_name_local.data).all()
                    assert (cols_by_name_gcs.indices == cols_by_name_local.indices).all()
                    assert (cols_by_name_gcs.indptr == cols_by_name_local.indptr).all()

                    # Test getting columns by name as DataFrame
                    cols_by_name_df = adata_gcs.get_columns(var_names, as_df=True)
                    assert isinstance(cols_by_name_df, pd.DataFrame)
                    assert cols_by_name_df.shape == (adata_gcs.shape[0], len(var_names))
                    assert list(cols_by_name_df.columns) == var_names

                    # Compare with anndata DataFrame
                    cols_by_name_df_local = pd.DataFrame(
                        adata_local.X[:, local_indices].toarray(), index=adata_local.obs_names, columns=var_names
                    )
                    pd.testing.assert_frame_equal(cols_by_name_df, cols_by_name_df_local, check_dtype=False)

    def test_get_rows(self, test_bucket, test_files, gcs_fs):
        """Test getting rows from the data matrix."""

        for matrix_type, filename in test_files.items():
            # Skip standard format as it will raise an error
            if matrix_type == "standard":
                continue

            gcs_path = get_gcs_path(test_bucket, filename)
            adata_gcs = GCSAnnData(gcs_path)

            # Download the file to a temporary location
            with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp:
                local_path = tmp.name
                gcs_fs.get(gcs_path, local_path)

                # Read with anndata for comparison
                adata_local = ad.read_h5ad(local_path)

                # Test getting rows by index
                row_indices = [0, 3, 7]
                rows_gcs = adata_gcs.get_rows(row_indices)

                assert sparse.isspmatrix_csr(rows_gcs)
                assert rows_gcs.shape == (len(row_indices), adata_gcs.shape[1])

                # Compare with anndata
                if sparse.issparse(adata_local.X):
                    rows_local = adata_local.X[row_indices, :].tocsr()
                else:
                    rows_local = sparse.csr_matrix(adata_local.X[row_indices, :])

                print(gcs_path)
                print(adata_local)
                print(GCSAnnData(gcs_path).get_rows(row_indices, as_df=True))
                # print(rows_local)

                # Check that the data is the same
                assert (rows_gcs.data == rows_local.data).all()
                assert (rows_gcs.indices == rows_local.indices).all()
                assert (rows_gcs.indptr == rows_local.indptr).all()

                # Test getting a single row
                single_row_gcs = adata_gcs.get_rows(1)
                assert sparse.isspmatrix_csr(single_row_gcs)
                assert single_row_gcs.shape == (1, adata_gcs.shape[1])

                # Compare with anndata
                if sparse.issparse(adata_local.X):
                    single_row_local = adata_local.X[1, :].tocsr()
                else:
                    single_row_local = sparse.csr_matrix(adata_local.X[1, :])

                # Check that the data is the same
                assert (single_row_gcs.data == single_row_local.data).all()
                assert (single_row_gcs.indices == single_row_local.indices).all()
                assert (single_row_gcs.indptr == single_row_local.indptr).all()

                # Test getting rows as DataFrame
                rows_df = adata_gcs.get_rows(row_indices, as_df=True)
                assert isinstance(rows_df, pd.DataFrame)
                assert rows_df.shape == (len(row_indices), adata_gcs.shape[1])
                assert list(rows_df.columns) == list(adata_gcs.var_names)

                # Compare with anndata DataFrame
                rows_df_local = pd.DataFrame(
                    adata_local.X[row_indices, :].toarray(),
                    index=[adata_local.obs_names[i] for i in row_indices],
                    columns=adata_local.var_names,
                )
                pd.testing.assert_frame_equal(rows_df, rows_df_local, check_dtype=False)

                # Test getting rows by name if obs_names is available
                if adata_gcs.obs_names is not None:
                    obs_names = [adata_gcs.obs_names[i] for i in row_indices[:2]]
                    rows_by_name_gcs = adata_gcs.get_rows(obs_names)
                    assert sparse.isspmatrix_csr(rows_by_name_gcs)
                    assert rows_by_name_gcs.shape == (len(obs_names), adata_gcs.shape[1])

                    # Compare with anndata
                    local_indices = [list(adata_local.obs_names).index(name) for name in obs_names]
                    if sparse.issparse(adata_local.X):
                        rows_by_name_local = adata_local.X[local_indices, :].tocsr()
                    else:
                        rows_by_name_local = sparse.csr_matrix(adata_local.X[local_indices, :])

                    # Check that the data is the same
                    assert (rows_by_name_gcs.data == rows_by_name_local.data).all()
                    assert (rows_by_name_gcs.indices == rows_by_name_local.indices).all()
                    assert (rows_by_name_gcs.indptr == rows_by_name_local.indptr).all()

                    # Test getting rows by name as DataFrame
                    rows_by_name_df = adata_gcs.get_rows(obs_names, as_df=True)
                    assert isinstance(rows_by_name_df, pd.DataFrame)
                    assert rows_by_name_df.shape == (len(obs_names), adata_gcs.shape[1])
                    assert list(rows_by_name_df.index) == obs_names

                    # Compare with anndata DataFrame
                    rows_by_name_df_local = pd.DataFrame(
                        adata_local.X[local_indices, :].toarray(), index=obs_names, columns=adata_local.var_names
                    )
                    pd.testing.assert_frame_equal(rows_by_name_df, rows_by_name_df_local, check_dtype=False)

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
