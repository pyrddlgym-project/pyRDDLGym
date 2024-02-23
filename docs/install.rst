Installation Guide
==================

Requirements
------------
We require Python 3.8+.

* ply
* pillow>=9.2.0
* numpy>=1.22
* matplotlib>=3.5.0
* gymnasium
* pygame
* termcolor

Installing via pip
-----------------
.. code-block:: shell

    pip install pyRDDLGym

To run the basic examples, you will also need ``rddlrepository``

.. code-block:: shell

    pip install rddlrepository

We recommend installing under a conda virtual environment:

.. code-block:: shell

    conda create -n rddl python=3.11
    conda activate rddl
    pip install pyrddlgym rddlrepository

Installing the pre-release version via git
---------
.. code-block:: shell

    pip install git+https://github.com/ataitler/pyRDDLGym.git

