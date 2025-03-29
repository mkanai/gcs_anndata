"""Tests for the core functionality of GCS AnnData."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from gcs_anndata.core import GCSAnnData


class TestGCSAnnData(unittest.TestCase):
    """Test cases for GCSAnnData class."""

    @patch("gcs_anndata.core.gcsfs.GCSFileSystem")
    @patch("gcs_anndata.core.h5py.File")
    def test_initialization(self, mock_h5py_file, mock_gcsfs):
        """Test initialization of GCSAnnData."""
        # Mock the h5py File and GCSFileSystem
        mock_fs = MagicMock()
        mock_gcsfs.return_value = mock_fs

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Mock the X group with shape attribute
        mock_x_group = MagicMock()
        mock_x_group.attrs = {"shape": (100, 200), "format": "csc"}
        mock_file.__getitem__.return_value = mock_x_group

        # Initialize GCSAnnData
        adata = GCSAnnData("gs://bucket/file.h5ad")

        # Check that shape was correctly read
        self.assertEqual(adata.shape, (100, 200))
        self.assertEqual(adata.sparse_format, "csc")

    # Add more tests for get_columns, get_rows, etc.


if __name__ == "__main__":
    unittest.main()
