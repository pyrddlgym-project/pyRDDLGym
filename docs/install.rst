Installation Guide
==================

Requirements
------------
We require Python 3.8+.

* ply
* pillow>=9.2.0
* numpy>=1.22
* matplotlib>=3.5.0
* gym>=0.24.0
* pygame

Installing via pip
-----------------
.. code-block:: shell

    pip install pyRDDLGym

We recommend installing under a conda virtual environment:

.. code-block:: shell

    conda create -n rddl python=3.8
    conda activate rddl
    pip install pyrddlgym

Installing the pre-release version via git
---------
.. code-block:: shell

    pip install git+https://github.com/ataitler/pyRDDLGym.git

Installing requirements for JAX planner
---------
To also run the JAX planner that ships with pyRDDLGym:

* tqdm
* bayesian-optimization
* jax>=0.3.25
* optax>=0.1.4
* dm-haiku>=0.0.9
* tensorflow>=2.11.0
* tensorflow-probability>=0.19.0

This can be installed as follows:

.. code-block:: shell

    pip install -r requirements_jax.txt

