"""I/O utilities for GCS AnnData."""

import anndata
import numpy as np
from typing import Optional, Union, Dict, Any


def write_indexed_h5ad(
    adata: anndata.AnnData,
    filename: str,
    compression: Optional[str] = None,
    compression_opts: Optional[Union[int, Dict[str, Any]]] = None,
    max_string_length: int = 100,
) -> None:
    """
    Prepare an h5ad file for fast indexing by precomputing and storing name-to-index mappings.

    This function adds precomputed mappings from observation/variable names to their indices,
    which allows for faster lookups when accessing data by name using GCSAnnData.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to prepare.
    filename : str
        The filename or GCS path to write to.
    compression : str, optional
        Compression method. Options are 'gzip', 'lzf', or None.
    compression_opts : int or dict, optional
        Compression options. See h5py documentation for details.
    max_string_length : int, default=100
        Maximum length for string names in the precomputed indices.
        Increase this if you have very long observation or variable names.

    Examples
    --------
    >>> import anndata
    >>> from gcs_anndata.io import write_indexed_h5ad
    >>>
    >>> # Create or load your AnnData object
    >>> adata = anndata.read_h5ad('input_file.h5ad')
    >>>
    >>> # Prepare it for fast indexing and save to GCS
    >>> write_indexed_h5ad(
    ...     adata,
    ...     'gs://your-bucket/optimized_data.h5ad',
    ...     compression='gzip'
    ... )
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()

    # Precompute var_to_idx and obs_to_idx
    var_to_idx = {name: idx for idx, name in enumerate(adata.var_names)}
    obs_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}

    # Create structured arrays for the indices
    var_dtype = [("name", f"S{max_string_length}"), ("idx", "int32")]
    obs_dtype = [("name", f"S{max_string_length}"), ("idx", "int32")]

    var_index_array = np.array([(k.encode("utf-8"), v) for k, v in var_to_idx.items()], dtype=var_dtype)
    obs_index_array = np.array([(k.encode("utf-8"), v) for k, v in obs_to_idx.items()], dtype=obs_dtype)

    # Store precomputed indices in uns
    adata.uns["_var_to_idx"] = var_index_array
    adata.uns["_obs_to_idx"] = obs_index_array

    adata.write_h5ad(filename, compression=compression, compression_opts=compression_opts)
