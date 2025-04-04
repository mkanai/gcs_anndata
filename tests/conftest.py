import os
import pytest
from gcsfs import GCSFileSystem

# Get test bucket from environment variable or use default
DEFAULT_TEST_BUCKET = "gcs-anndata-test"
TEST_BUCKET = os.environ.get("GCS_ANNDATA_TEST_BUCKET", DEFAULT_TEST_BUCKET)

# Test files
TEST_FILES = {"csr": "test_csr.h5ad", "csc": "test_csc.h5ad", "standard": "test_standard.h5ad"}


@pytest.fixture(scope="session")
def gcs_fs():
    """Provide a GCSFileSystem instance for tests."""
    return GCSFileSystem()


@pytest.fixture(scope="session")
def test_bucket():
    """Provide the test bucket name."""
    return TEST_BUCKET


@pytest.fixture(scope="session")
def test_files():
    """Provide the test file names."""
    return TEST_FILES


def pytest_configure(config):
    """Configure pytest."""
    # Skip tests if credentials are not available
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and not os.environ.get("GCSFS_GOOGLE_TOKEN"):
        print("No GCS credentials found, skipping GCS tests")
