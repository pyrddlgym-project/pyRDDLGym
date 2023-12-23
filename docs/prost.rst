Baselines: The PROST Planner
===============

The PROST planner, introduced `in this paper 
<https://ai.dmi.unibas.ch/papers/keller-eyerich-icaps2012.pdf>`_, works well for 
stochastic problems with finite action spaces. It uses a Monte-Carlo UCT search 
that is heavily informed by the problem structure in order to improve the search.

The RDDLSimAgent Class
-------------------

PROST requires a TCP connection with a server that provides the environment interaction, 
which was originally provided by the Java version of RDDL.
pyRDDLGym provides the replacement ``RDDLSimAgent`` class, which establishes an identical server
that listens for any messages (i.e. actions) passed by client planners connected to that port. 

The following code establishes this connection by reading the domain and instance RDDL files at the specified path,
and exposing the resulting Gym environment through a TCP connection:

.. code-block:: python
	
    from pyRDDLGym.Core.Policies.RDDLSimAgent import RDDLSimAgent
	
    server = RDDLSimAgent(<domain-path>, <instance-path>, <numrounds>, <time>, port=2323)
    server.run()	
	
The ``numrounds`` specifies the number of epsiodes/rounds of simulation to perform,
and ``time`` specifies the time the server connection should remain open. The optional ``port``
parameter allows multiple connections to be established in parallel at different ports, 
which is useful for parallel processing applications. Finally, the ``run()`` command starts the server
to listen in on the specified port.

The Docker Image
-------------------

While PROST and other planners can be used to establish connections with the ``RDDLSimAgent`` instance,
the setup of the PROST planner in most environments (especially, i.e. Windows machines) is quite difficult.
To automate the process of setting up the PROST planner in a standard environment, 
pyRDDLGym provides a generic Docker image that can run PROST with minimal effort on the user side and on most OS.

The first step is to download the Docker files from `here <https://github.com/ataitler/pyRDDLGym/tree/main/pyRDDLGym/Docker>`_
(if you have already installed a recent version of pyRDDLGym through pip or git commands, 
simply locate the installation folder and navigate to the Docker sub-folder).

Navigate to the Docker folder, and build the Docker image from the provided ``DOCKERFILE``:

.. code-block:: shell
	
    docker build -t <docker name> .

This may take a while. Once built, a container can be created for a specified domain and instance RDDL file. 
To do this, mount a local directory containing a RDDL domain and instance file into the /RDDL directory of the container
when calling ``docker run``:

.. code-block:: shell
	
    docker run --name <docker name> --mount type=bind,source=<rddl dir>,target=/RDDL prost <rounds> "<prost args>"

where ``<rddl dir>`` is a directory on the local machine containing a valid 
``domain.rddl`` and ``instance.rddl`` file, 
``<rounds>`` is the number of rounds/episodes to evaluate on the server, and
``<prost args>`` is the arguments to pass to PROST, 
which is typically of the form ``[PROST -se [...] ...]`` (see below for details).

To save the logs and simulation results to a local directory, 
copy the /OUTPUTS directory in the container using ``docker cp``:

.. code-block:: shell
	 
    docker cp <docker name>:/OUTPUTS/ <output dir>

where ``<output dir>`` is a valid directory on the local machine.

pyRDDLGym provides a convenient bash script ``runprost.sh`` in the Docker folder that automates the two previous commands:

.. code-block:: shell
	
    bash runprost.sh <docker name> <rddl dir> <rounds> <prost args> <output dir>

A complete list of ``<prost args>`` arguments can be found 
`here <https://github.com/prost-planner/prost/blob/master/src/search/main.cc>`_.
For example, to run the IPC 2014 version of PROST with default parameters, set this argument to ``[PROST -se [IPC2014]]``. 
Additional notes about PROST command line arguments from Thomas Keller can be found 
`here <https://github.com/ataitler/pyRDDLGym/tree/main/pyRDDLGym/Docker/PROST_Command_Line_Option_Notes_Thomas_Keller.txt>`_.

Any PROST-specific issues should be directed to Thomas Keller by filing a bug report
`here <https://github.com/prost-planner/prost>`_.
