# setup.py at the root
from setuptools import setup, find_packages

setup(
    name="up-pce",
    version="0.1",
    package_dir={"": "uq_pce"},
    packages=find_packages(where="uq_pce"),
)
