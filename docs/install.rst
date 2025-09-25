Installation Guide
==================

Requirements
------------
We require Python 3.8+ and the packages `listed here <https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/requirements.txt>`_.

Installation
-----------------

Official Version from PyPI
^^^^^^^^^^^^^^^^^

We recommend installing pyRDDLGym and rddlrepository together in a shared conda virtual environment:

.. code-block:: shell

    conda create -n rddl python=3.11
    conda activate rddl
    pip install pyrddlgym rddlrepository

Pre-Release Version via git
^^^^^^^^^^^^^^^^^

If you wish to install the latest pre-release version from Github:

.. code-block:: shell

    pip install git+https://github.com/pyrddlgym-project/pyRDDLGym.git
    pip install git+https://github.com/pyrddlgym-project/rddlrepository.git

