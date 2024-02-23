Managing Problems using rddlrepository
===============

The rddlrepository is a growing collection of RDDL domain and instance description files. 
It:

- hosts a diverse collection of domain and instance RDDL files, covering problems from a wide range of disciplines
- contains a growing collection of problems, including those previously used in the probabilistic and learning track of the International Planning Competitions
- contains custom visualizers for a subset of domains, to be used with the pyRDDLGym package
- provides out-of-the-box compatibility with pyRDDLGym.

The Repository Manager
---------

The core object for extracting problems and instances is the ``RDDLRepoManager`` object:

.. code-block:: python

    from rddlrepository.core.manager import RDDLRepoManager
    manager = RDDLRepoManager(rebuild=True)
    
.. note::
   ``rebuild`` instructs the manager to rebuild the manifest, which is an index 
   containing the locations of all domains and instances for fast access. 
   While you do not need this option in normal operation, in case you add your 
   own domains or the manifest becomes corrupt, you can force it to be recreated.

To list all domains currently available in rddlrepository:

.. code-block:: python

    print(manager.list_problems())

Problems are organized by context (e.g. year of the competition, standalone):

.. code-block:: python

    print(manager.list_context())
    print(manager.list_problems_by_context("standalone"))   # list all standalone problems
    print(manager.list_problems_by_context("ippc2018"))     # list all problems from IPPC 2018


Retrieving Specific Problems
---------

The information for a specific problem or domain is a ``ProblemInfo`` instance:

.. code-block:: python

    problem_info = manager.get_problem("EarthObservation_ippc2018")

will load the EarthObservation domain information from the ippc2018 context.

To list all the instances of a domain:

.. code-block:: python

    print(problem_info.list_instances())

To return the paths of the domain and an instance (1):

.. code-block:: python

    print(problem_info.get_domain())
    print(problem_info.get_instance("1"))
 
To return the pyRDDLGym visualizer class:

.. code-block:: python

    viz_class = problem_info.get_visualizer()

 
Loading Environments in pyRDDLGym
---------

In the introduction to pyRDDLGym, we already presented the standard way to load an environment:

.. code-block:: python

    import pyRDDLGym
    env = pyRDDLGym.make("EarthObservation_ippc2018", "1")

This can also be done directly using rddlrepository:

.. code-block:: python
    
    problem_info = manager.get_problem("EarthObservation_ippc2018")
    env = pyRDDLGym.make(domain=problem_info.get_domain(), instance=problem_info.get_instance("1"))
    env.set_visualizer(problem_info.get_visualizer())


Registering your Own Problems and Instances
---------

To register a new context in rddlrepository for later access:

.. code-block:: python

    manager.register_context("MyContext")

To register a new problem in a given context for later access:

.. code-block:: python

    domain_content = """
        domain ... {
            ...
        }
    """
    manager.register_domain("MyDomain", "MyContext", domain_content,
                            desc="a description of this domain", viz="ModuleName.ClassName") 

Here, ``"ModuleName.ClassName"`` refers to the Module name and the Class name of the visualizer (optional).

To register an instance for an existing domain for later access:

.. code-block:: python

    instance_content = """
        instance ... {
            ...
        }
    """
    problem_info.register_instance("MyInstance", instance_content)
 