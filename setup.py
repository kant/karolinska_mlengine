"""Setup script that takes care of packaging up the repo for use on the cloud."""
from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
]

setup(
    name='Carvana',
    version='1.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[])
