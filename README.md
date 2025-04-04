# GCS AnnData (gcs_anndata)

GCS AnnData is a Python package that enables efficient partial reading of AnnData objects stored in h5ad format on Google Cloud Storage (GCS). It provides a convenient interface to access specific rows (observations) or columns (variables) of large-scale single-cell sequencing data without loading the entire dataset into memory.

## Features

- Efficient partial reading of h5ad files stored on Google Cloud Storage
- Support for sparse matrices in CSC (Compressed Sparse Column) and CSR (Compressed Sparse Row) formats
- Access to specific rows (cells) or columns (genes) by index or name
- Return data as sparse matrices or pandas DataFrames
- Automatic handling of observation and variable names
- Warnings for potentially inefficient operations
- Robust error handling for unsupported matrix formats

## Installation

You can install gcs_anndata using pip:

```bash
# TODO: pip install gcs-anndata
git clone https://github.com/mkanai/gcs_anndata
pip install ./gcs_anndata
```

## Usage

Here's a basic example of how to use GCS AnnData:

```python
from gcs_anndata import GCSAnnData

# Initialize the GCSAnnData object with the path to your h5ad file on GCS
# Note: Only h5ad files with sparse matrices (CSR or CSC) are supported
adata = GCSAnnData('gs://your-bucket/your-file.h5ad')

# Get specific columns (genes) by name as a sparse matrix
gene_data = adata.get_columns(['GAPDH', 'CD3D', 'CD8A'])

# Get specific columns as a pandas DataFrame with appropriate row and column labels
gene_df = adata.get_columns(['GAPDH', 'CD3D', 'CD8A'], as_df=True)

# Get specific rows (cells) by barcode
cell_data = adata.get_rows(['AAACCCAAGCGCCCAT-1', 'AAACCCATCAGCCCAG-1'])

# Get specific rows as a pandas DataFrame
cell_df = adata.get_rows(['AAACCCAAGCGCCCAT-1', 'AAACCCATCAGCCCAG-1'], as_df=True)

# Get columns by index
cols = adata.get_columns([0, 10, 20])

# Get rows by index
rows = adata.get_rows([100, 200])

# Get a single column
col = adata.get_columns('ACTB')

# Get a single row
row = adata.get_rows('AAACCCAAGCGCCCAT-1')

# Print the shape of the full matrix
print(adata.shape)

# Print the names of all variables (genes)
print(adata.var_names)

# Print the names of all observations (cells)
print(adata.obs_names)
```

### Efficient Access Patterns

GCS AnnData will warn you if you're using an inefficient access pattern (e.g., accessing rows from a CSC matrix or columns from a CSR matrix). To get the best performance:

- Use `get_columns()` when your data is in CSC format
- Use `get_rows()` when your data is in CSR format

If you find yourself frequently using the inefficient access pattern, consider converting your data to the other format before saving it as an h5ad file.

### Matrix Format Requirements

GCS AnnData only supports h5ad files containing sparse matrices in CSR or CSC format. Attempting to open an h5ad file with a dense matrix or other sparse formats will raise an `InvalidFormatError`.

## API Reference

### `GCSAnnData`

#### `__init__(gcs_path: str)`

Initialize the GCSAnnData object.

- `gcs_path`: Path to the h5ad file on GCS (e.g., 'gs://bucket/path/to/file.h5ad')
- Raises: `InvalidFormatError` if the h5ad file does not contain a sparse matrix in CSR or CSC format

#### `get_columns(columns: Union[List[int], List[str], int, str], as_df: bool = False) -> Union[csc_matrix, pd.DataFrame]`

Get specific columns from the data matrix.

- `columns`: Column indices, variable names, or a single index/name
- `as_df`: If True, return result as pandas DataFrame with appropriate indices
- Returns: A CSC matrix containing only the requested columns, or a DataFrame if as_df=True
- Raises: `IndexError` if column indices are out of bounds, `KeyError` if column names are not found

#### `get_rows(rows: Union[List[int], List[str], int, str], as_df: bool = False) -> Union[csr_matrix, pd.DataFrame]`

Get specific rows from the data matrix.

- `rows`: Row indices, observation names, or a single index/name
- `as_df`: If True, return result as pandas DataFrame with appropriate indices
- Returns: A CSR matrix containing only the requested rows, or a DataFrame if as_df=True
- Raises: `IndexError` if row indices are out of bounds, `KeyError` if row names are not found

#### Attributes

- `gcs_path`: Path to the h5ad file on GCS
- `shape`: Tuple representing the shape of the full data matrix (n_obs, n_vars)
- `sparse_format`: String indicating the sparse matrix format ('csc' or 'csr')
- `obs_names`: Array of observation names (cell barcodes)
- `var_names`: Array of variable names (genes)
- `obs_to_idx`: Dictionary mapping observation names to indices
- `var_to_idx`: Dictionary mapping variable names to indices
