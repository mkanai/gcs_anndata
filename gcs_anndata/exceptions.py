"""Custom exceptions for GCS AnnData."""

from typing import Optional


class InvalidFormatError(Exception):
    """
    Exception raised when the sparse matrix format is invalid or unsupported.

    This exception is raised when attempting to work with a matrix that is not
    in a supported sparse format (CSR or CSC) or when the format cannot be
    determined from the available information.

    Parameters
    ----------
    message : str, optional
        The error message explaining why the format is invalid

    Examples
    --------
    >>> raise InvalidFormatError("Only CSR and CSC formats are supported")
    >>> raise InvalidFormatError(f"Unsupported sparse format: {format_name}")
    """

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialize the InvalidFormatError with an optional message.

        Parameters
        ----------
        message : str, optional
            The error message explaining why the format is invalid
        """
        self.message = message if message is not None else "Invalid or unsupported sparse matrix format"
        super().__init__(self.message)
