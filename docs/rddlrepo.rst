Managing Problems using rddlrepository
===============

The rddlrepository:

- hosts a diverse collection of domain and instance RDDL files, covering problems from a wide range of disciplines (`full listing here <https://github.com/pyrddlgym-project/rddlrepository/blob/main/domains.pdf>`_)
- includes all domains used in the probabilistic and learning track of the International Planning Competitions (2011, 2014, 2018, 2023)
- is updated and expanded frequently with the help of the community
- contains custom visualizers for a subset of domains, to be used with the pyRDDLGym package
- provides out-of-the-box compatibility with pyRDDLGym.


Installing
---------

To install with pip:

.. code-block:: shell

    pip install rddlrepository


Retrieving Information about Domains and Instances
---------

The core object for extracting domains and instances is the ``RDDLRepoManager`` object:

.. code-block:: python

    from rddlrepository.core.manager import RDDLRepoManager
    manager = RDDLRepoManager(rebuild=True)
    
.. note::
   ``rebuild`` instructs the manager to rebuild the manifest, which is an index 
   containing the locations of all domains and instances for fast access. 
   While you do not need this option in normal operation, in case you add your 
   own domains or the manifest becomes corrupt, you can force it to be recreated.


Listing Available Domains
^^^^^^^^^^^^^^

To list all domains currently available in rddlrepository:

.. code-block:: python

    print(manager.list_problems())

Domains are organized by context (e.g. competition year, benchmark set) 
with names usually following the syntax ``<domain name>_<context>`` 
(except for standalone domains where the context is excluded from the name)

.. code-block:: python

    print(manager.list_context())
    print(manager.list_problems_by_context("standalone"))   # list all standalone problems
    print(manager.list_problems_by_context("ippc2018"))     # list all problems from IPPC 2018

.. raw:: html 

   <a href="notebooks/loading_problems_in_rddlrepository.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Loading a problem from the rddlrepository.
   </a>
   
   
Retrieving Information about a Domain or Instance
---------

The information for a specific domain is stored in a ``ProblemInfo`` instance:

.. code-block:: python

    problem_info = manager.get_problem("EarthObservation_ippc2018")

will load the EarthObservation domain from the ippc2018 context.

To list all the instances of a domain:

.. code-block:: python

    print(problem_info.list_instances())

To return the paths of the domain and instance:

.. code-block:: python

    print(problem_info.get_domain())
    print(problem_info.get_instance("1"))
 
To return the pyRDDLGym visualizer class:

.. code-block:: python

    viz_class = problem_info.get_visualizer()


.. raw:: html 

   <a href="notebooks/loading_problems_in_rddlrepository.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Loading a problem from the rddlrepository.
   </a>
   
   
 
Loading an Environment in pyRDDLGym
---------

In the introduction to pyRDDLGym, we presented the recommended way to load an environment:

.. code-block:: python

    import pyRDDLGym
    env = pyRDDLGym.make("EarthObservation_ippc2018", "1")

This can also be done explicitly using rddlrepository:

.. code-block:: python
    
    problem_info = manager.get_problem("EarthObservation_ippc2018")
    env = pyRDDLGym.make(domain=problem_info.get_domain(), instance=problem_info.get_instance("1"))
    env.set_visualizer(problem_info.get_visualizer())


.. raw:: html 

   <a href="notebooks/loading_problems_in_rddlrepository.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Loading a problem from the rddlrepository.
   </a>
   
   

Registering a New Domain or Instance
---------

To register a new context in rddlrepository for later access:

.. code-block:: python

    manager.register_context("MyContext")

To register a new domain in a given context for later access:

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
 

.. raw:: html 

   <a href="notebooks/adding_domains_to_rddlrepository.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Adding domains to the rddlrepository.
   </a>
   