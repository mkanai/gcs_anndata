"""I/O utilities for GCS AnnData."""

import anndata
import numpy as np
import gcsfs
from typing import Optional, Union, Dict, Any


def write_indexed_h5ad(
    adata: anndata.AnnData,
    filename: str,
    compression: Optional[str] = None,
    compression_opts: Optional[Union[int, Dict[str, Any]]] = None,
) -> None:
    """
    Write an h5ad file with precomputed name-to-index mappings for fast indexing.

    This function adds precomputed mappings from observation/variable names to their indices,
    which allows for faster lookups when accessing data by name using GCSAnnData.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to write.
    filename : str
        The filename or GCS path to write to.
    compression : str, optional
        Compression method. Options are 'gzip', 'lzf', or None.
    compression_opts : int or dict, optional
        Compression options. See h5py documentation for details.

    Examples
    --------
    >>> import anndata
    >>> from gcs_anndata.io import write_indexed_h5ad
    >>>
    >>> # Create or load your AnnData object
    >>> adata = anndata.read_h5ad('input_file.h5ad')
    >>>
    >>> # Write it with precomputed indices for fast indexing
    >>> write_indexed_h5ad(
    ...     adata,
    ...     'indexed_data.h5ad',
    ...     compression='gzip'
    ... )
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()

    # Precompute var_to_idx and obs_to_idx
    var_to_idx = {name: idx for idx, name in enumerate(adata.var_names)}
    obs_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}

    # Compute max string length based on actual names
    var_max_length = len(adata.var_names[np.argmax(adata.var_names.str.len())].encode("utf-8"))
    obs_max_length = len(adata.obs_names[np.argmax(adata.obs_names.str.len())].encode("utf-8"))

    # Create structured arrays for the indices
    var_dtype = [("name", f"S{var_max_length}"), ("idx", "int32")]
    obs_dtype = [("name", f"S{obs_max_length}"), ("idx", "int32")]

    var_index_array = np.array([(k.encode("utf-8"), v) for k, v in var_to_idx.items()], dtype=var_dtype)
    obs_index_array = np.array([(k.encode("utf-8"), v) for k, v in obs_to_idx.items()], dtype=obs_dtype)

    # Store precomputed indices in uns
    adata.uns["_var_to_idx"] = var_index_array
    adata.uns["_obs_to_idx"] = obs_index_array

    adata.write_h5ad(filename, compression=compression, compression_opts=compression_opts)
