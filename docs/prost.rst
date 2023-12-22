Baselines: The PROST Planner
===============

The PROST planner is an alternative to the Gurobi nonlinear solver that works 
for stochastic problems with finite action spaces. As discussed `in this paper 
<https://ai.dmi.unibas.ch/papers/keller-eyerich-icaps2012.pdf>`_, PROST uses
a Monte-Carlo tree search with the UCT heuristic, as well as other problem-tailored
heuristics to improve the efficacy of the search.

The RDDLSimAgent Class
-------------------

PROST requires a TCP connection with a server that provides the environment interaction. 
Originally, the Java RDDL simulator was used to establish such a connection.
However pyRDDLGym provides the replacement ``RDDLSimAgent`` class, which establishes an identical server-side port 
that listens for any messages (i.e. actions) passed by client planners connected to that port. The following code
establishes this connection by reading the domain and instance RDDL files at the specified path,
creating the OpenAI gym environment, and exposing this environment through TCP connections.

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
we provide a generic Docker image that can be used to run PROST easily and reproducibly on most machines and OS.

The first step is to download the Docker files from `here <https://github.com/ataitler/pyRDDLGym/tree/main/pyRDDLGym/Docker>`_
(if you have already installed a recent version of pyRDDLGym through pip or git commands, 
simply locate the installation folder and navigate to the Docker sub-folder).

Once inside the Docker folder, build the Docker image (this will take a while) with the necessary prerequisite packages

.. code-block:: shell
	
    docker build -t prost .

Once the Docker image is built, a container can be created for a specified domain and instance RDDL file and run. 
Fortunately, pyRDDLGym provides a convenient bash script ``runprost.sh`` 
that automates the process of instantiating a ``RDDLSimAgent`` server and a PROST client:

.. code-block:: shell
	
    bash runprost.sh prost <rddl dir> <rounds> <prost args> <output dir>
	
where ``<rddl dir>`` is a directory on the local machine containing a valid 
``domain.rddl`` and ``instance.rddl`` file,
``<rounds>`` is the number of rounds/episodes to evaluate on the server, 
``<prost args>`` is a set of arguments passed to PROST, such as time limit, 
search parameters, etc., and ``<output dir>`` is a valid directory on the local 
machine into which all logs and simulation results from PROST are to be copied 
once the experiment finishes.

A complete list of ``<prost args>`` arguments can be found 
`here <https://github.com/prost-planner/prost/blob/master/src/search/main.cc>`_.
For example, to run the IPC 2014 version of PROST, simply set this argument to ``[IPC2014]``. 
Additional notes about PROST command line arguments from Thomas Keller can be found 
`here <https://github.com/ataitler/pyRDDLGym/tree/main/pyRDDLGym/Docker/PROST_Command_Line_Option_Notes_Thomas_Keller.txt>`_.

Any PROST-specific issues should be directed to Thomas Keller by filing a bug report
`here <https://github.com/prost-planner/prost>`_.
