.. MLPro Documentations documentation master file, created by
   sphinx-quickstart on Wed Sep 15 12:06:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLPro-Int-Gymnasium - Integration between Gymnasium and MLPro
==============================================================

Welcome to MLPro-Int-Gymnasium, an extension to MLPro to integrate the Gymnasium package.
MLPro is a middleware framework for standardized machine learning in Python. It is 
developed by the South Westphalia University of Applied Sciences, Germany, and provides 
standards, templates, and processes for hybrid machine learning applications. Gymnasium, in 
turn, provides a diverse suite of reinforcement learning environments.

MLPro-Int-Gymnasium offers wrapper classes that allow the reuse of environments from Gymnasium in MLPro,
or the other way around.


**Preparation**
   
Before running the examples, please install the latest versions of MLPro, Gymnasium, and MLPro-Int-Gymnasium as follows:

.. code-block:: bash

   pip install mlpro-int-gymnasium[full] --upgrade


**See also**
   - `MLPro - Machine Learning Professional <https://mlpro.readthedocs.io>`_ 
   - `MLPro-RL - Sub-framework for reinforcement learning <https://mlpro.readthedocs.io/en/latest/content/03_machine_learning/mlpro_rl/main.html>`_
   - `Gymnasium - An API standard for reinforcement learning with a diverse collection of reference environments <https://gymnasium.farama.org/index.html>`_ 
   - `Further MLPro extensions <https://mlpro.readthedocs.io/en/latest/content/04_extensions/main.html>`_
   - `MLPro-Int-Gymnasium on GitHub <https://github.com/fhswf/MLPro-Int-Gymnasium>`_


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Home

   self


.. toctree::
   :maxdepth: 2
   :caption: Example Pool
   :glob:

   content/01_example_pool/*


.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :glob:

   content/02_api/*


.. toctree::
   :maxdepth: 2
   :caption: About
   :glob:

   content/03_about/*
