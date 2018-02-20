from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy', 'scipy', 'pillow']
FOUND_PACKAGES = find_packages()
IGNORE_PACKAGES = ['tests']
KEEP_PACKAGES = [i_pack for i_pack in FOUND_PACKAGES if i_pack not in IGNORE_PACKAGES]

setup(
    name='Carvana',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=KEEP_PACKAGES,
    include_package_data=True,
    requires=[]
)
