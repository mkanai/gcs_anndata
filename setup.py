"""Setup script for gcs_anndata package."""

from setuptools import setup, find_packages

setup(
    name="gcs_anndata",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "h5py>=3.1.0",
        "gcsfs>=2021.4.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    author="Masahiro Kanai",
    author_email="mkanai@broadinstitute.org",
    description="Partial reading of h5ad files from Google Cloud Storage",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mkanai/gcs_anndata",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.6",
)
