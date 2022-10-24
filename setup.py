from setuptools import setup, find_packages

setup(name='rddlgym',
      version='1.0.0',
      install_requires=['ply', 'PIL', 'matplotlib', 'numpy', 'gym', 'pygame'],
      packages=find_packages(),
      include_package_data=True,
)
