# This file is part of pyRDDLGym.

# pyRDDLGym is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation.

# pyRDDLGym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.

# You should have received a copy of the MIT License
# along with pyRDDLGym. If not, see <https://opensource.org/licenses/MIT>.

from setuptools import setup, find_packages

setup(
      name='pyRDDLGym',
      version='1.0.0',
      author="Ayal Taitler, Scott Sanner, Michael Gimelfarb, Sriram Gopalakrishnan, sgopal28@asu.edu, Martin Mladenov,jack liu",
      author_email="ataitler@gmail.com, ssanner@mie.utoronto.ca, mike.gimelfarb@mail.utoronto.ca, mmladenov@google.com, jack.liu.to@gmail.com",
      description="pyRDDLGym: RDDL automatic generation tool for OpenAI Gym",
      license="MIT License",
      url="https://github.com/ataitler/pyRDDLGym",
      packages=find_packages(),
      install_requires=['ply', 'PIL', 'matplotlib', 'numpy', 'gym', 'pygame'],
      python_requires=">=3.6",
      include_package_data=True,
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
