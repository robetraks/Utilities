"""Setup configuration for utilities package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="utilities-robetraks",
    version="0.1.0",
    author="Akshay Jain",
    author_email="jain.akshay@icloud.com",
    description="A collection of utility functions for data analysis and plotting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robetraks/Utilities",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "matplotlib>=3.0.0",
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "Pillow>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
    },
)
