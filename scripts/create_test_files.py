# scripts/create_test_files.py
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from gcsfs import GCSFileSystem

# Get test bucket from environment variable or use default
DEFAULT_TEST_BUCKET = "gcs-anndata-test"
TEST_BUCKET = os.environ.get("GCS_ANNDATA_TEST_BUCKET", DEFAULT_TEST_BUCKET)


def create_test_anndata(n_obs=500, n_vars=200, matrix_type="standard", seed=42):
    """
    Create a test AnnData object with given dimensions and matrix type.

    Parameters
    ----------
    n_obs : int
        Number of observations (rows)
    n_vars : int
        Number of variables (columns)
    matrix_type : str
        Type of matrix to use: "standard" (numpy array), "csr" (sparse CSR), or "csc" (sparse CSC)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    AnnData object
    """
    np.random.seed(seed)

    # Generate random sparse data
    X = np.random.poisson(0.5, size=n_obs * n_vars).reshape(n_obs, n_vars).astype(np.uint32)

    # Convert to specified matrix type
    if matrix_type == "csr":
        X = sparse.csr_matrix(X)
    elif matrix_type == "csc":
        X = sparse.csc_matrix(X)
    # For "standard", keep as numpy array

    # Create observation and variable annotations
    obs = pd.DataFrame(
        {
            "group": np.random.choice(["A", "B", "C"], size=n_obs),
            "value1": np.random.rand(n_obs),
            "value2": np.random.rand(n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    var = pd.DataFrame(
        {"type": np.random.choice(["gene", "protein"], size=n_vars), "score": np.random.rand(n_vars)},
        index=[f"feature_{i}" for i in range(n_vars)],
    )

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var, dtype=X.dtype)

    # Add some additional data
    adata.uns["metadata"] = {"experiment": "test", "date": "2023-01-01", "matrix_type": matrix_type}

    return adata


def create_and_upload_test_files():
    """Create test files and upload them to GCS."""
    # Create test files with different matrix types
    test_files = {
        "test_standard.h5ad": create_test_anndata(matrix_type="standard"),
        "test_csr.h5ad": create_test_anndata(matrix_type="csr"),
        "test_csc.h5ad": create_test_anndata(matrix_type="csc"),
    }

    # Initialize GCS filesystem
    fs = GCSFileSystem()

    # Create bucket if it doesn't exist
    if not fs.exists(TEST_BUCKET):
        fs.mkdir(TEST_BUCKET)

    # Save files locally and upload to GCS
    for filename, adata in test_files.items():
        # Save locally first
        local_path = filename
        adata.write_h5ad(local_path)

        # Upload to GCS
        gcs_path = f"{TEST_BUCKET}/{filename}"
        fs.put(local_path, gcs_path)

        # Remove local file
        os.remove(local_path)

        print(f"Uploaded {filename} to gs://{gcs_path}")
        print(f"Matrix type: {adata.uns['metadata']['matrix_type']}, Shape: {adata.shape}")


if __name__ == "__main__":
    print(f"Using test bucket: {TEST_BUCKET}")
    create_and_upload_test_files()
