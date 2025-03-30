"""GCS AnnData - Partial reading of h5ad files from Google Cloud Storage."""

from .core import GCSAnnData
from .io import write_indexed_h5ad

__version__ = "0.1.0"
__all__ = ["GCSAnnData", "write_indexed_h5ad"]
