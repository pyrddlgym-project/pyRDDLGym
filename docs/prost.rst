pyRDDLGym-prost: The PROST Planner
===============

The PROST planner, introduced `in this paper 
<https://ai.dmi.unibas.ch/papers/keller-eyerich-icaps2012.pdf>`_, works well for 
stochastic problems with finite action spaces. It uses a Monte-Carlo UCT search 
that is heavily informed by the RDDL problem structure in order to improve the search.

Installation
-------------------

This package is not designed to be installed like an ordinary Python package, so there is no pip installer.
Instead, you will need to download the necessary Docker files and run scripts packaged with this repository.

.. code-block:: shell
    git clone https://github.com/pyrddlgym-project/pyRDDLGym-prost /path/to/dockerfiles
    cd /path/to/dockerfiles/prost

Here, it is assumed that ``/path/to/dockerfiles`` is a valid path on the file system 
where the project files will be cloned into. 

When you run this command, you will find:
* a ``Dockerfile`` that instructs Docker how to build the image with all the dependencies
* ``prost.sh`` file that calls PROST from the command line
* ``rddlsim.py`` file that runs ``prost.sh`` from Python
* ``runprost.sh`` file that you can use to automate the build and run process (as described below).


Building the Image
-------------------

To build the Docker image, you will need to install Docker. Then, with Docker daemon running, build the image as follows:

.. code-block:: shell

    docker build -t <docker name> .
    
where ``<docker name>`` is the name of the image.


Running the Container
-------------------

To run a container from the built image:

.. code-block:: shell

    docker run --name <docker name> --mount type=bind,source=<rddl dir>,target=/RDDL prost <rounds> "<prost args>"

where:
* ``<docker name>`` is the name specified above during build
* ``<rddl dir>`` is the path of the directory containing the RDDL domain.rddl and instance.rddl files you wish to run
* ``<rounds>`` is the number of runs/episodes/trials of optimization
* ``<prost args>`` are the arguments to pass to PROST, whose syntax is described `here <https://github.com/prost-planner/prost/blob/master/src/search/main.cc>`_. 

For example, to run the IPC 2014 version of PROST with default parameters, set ``<prost args>`` to ``[PROST -se [IPC2014]]``. 
Additional notes about PROST command line arguments from Thomas Keller can be found 
`here <https://github.com/ataitler/pyRDDLGym/tree/main/pyRDDLGym/Docker/PROST_Command_Line_Option_Notes_Thomas_Keller.txt>`_.

After the container runs, you can then copy the files from the container to a 
directory ``<output dir>`` in your local file system for further analysis:

.. code-block:: shell

    docker cp <docker name>:/OUTPUTS/ <output dir>


Using the Convenience Script
-------------------

You do not need to run the commands described above, as we provide a script ``runprost.sh`` to automate the process:

.. code-block:: shell

    bash runprost.sh <docker name> <rddl dir> <rounds> <prost args> <output dir>
 
where the arguments are as described above.


Reporting PROST Bugs
-------------------

Any PROST-specific issues should be directed to Thomas Keller by filing a bug report
`here <https://github.com/prost-planner/prost>`_.

