"""
GCS AnnData - Partial reading of h5ad files from Google Cloud Storage.

This package provides functionality for efficiently reading and working with
AnnData objects stored in h5ad format on Google Cloud Storage without loading
the entire dataset into memory. It supports reading specific rows or columns
from sparse matrices (CSR or CSC format).

Classes
-------
GCSAnnData
    Main class for partially reading AnnData objects from GCS

Examples
--------
>>> from gcs_anndata import GCSAnnData
>>> adata = GCSAnnData('gs://bucket/file.h5ad')
>>> # Get specific columns by index
>>> cols = adata.get_columns([0, 5, 10])
>>> # Get specific rows by name
>>> rows = adata.get_rows(['AAACCCAAGCGCCCAT-1', 'AAACCCATCAGCCCAG-1'])
"""

from .core import GCSAnnData

__version__ = "0.1.0"
__all__ = ["GCSAnnData"]
