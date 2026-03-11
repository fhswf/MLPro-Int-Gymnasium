.. _howto_rl_agent_010:

Howto RL-AGENT-010: Train a SB3 PPO Agent on the Rubik's Cube 2x2x2 Environment
=================================================================================

This howto demonstrates how to train a `Stable Baselines3 (SB3) <https://stable-baselines3.readthedocs.io>`_
PPO agent on the 2x2x2 Rubik's Cube environment using the MLPro RL training infrastructure.

.. seealso::

    :ref:`env_rubiks_cube_2x2x2` — full environment description, observation/action spaces,
    and reward shaping strategy.


**Prerequisites**

.. code-block:: bash

    pip install mlpro==2.0.0 mlpro-int-gymnasium mlpro-int-sb3==1.0.7 stable-baselines3 rubiks-cube-gym


**You will learn**

    - How to wrap the Gymnasium environment ``rubiks-cube-222-v0`` as an MLPro RL environment
    - How to activate the built-in shaped reward curriculum via ``p_shaped_reward=True``
    - How to wrap an SB3 PPO policy with ``WrPolicySB32MLPro``
    - How to configure an ``RLScenario`` and run an ``RLTraining``


**Executable Code**

.. literalinclude:: ../../../../../test/howtos/rl/howto_rl_agent_010_train_agent_ppo_on_rubiks_cube_2x2x2_environment.py.off
    :language: python


**Results**

The training log shows milestone progress printed to the console every 500 steps::

    2026-02-28 14:10:56  I  Training "RL": Instantiated
    2026-02-28 14:10:56  I  RL-Scenario "RubiksCube222_Scenario": Instantiated
    2026-02-28 14:10:58  I  Environment "RubiksCube222": Instantiated
    2026-02-28 14:10:58  I  Wrapper Gym2MLPro "(rubiks-cube-222-v0)": Instantiated
    2026-02-28 14:10:58  I  Wrapper SB3 -> MLPro "Policy PPO": Instantiated
    2026-02-28 14:10:58  S  Agent "PPO_Agent": Adaptivity switched on
    2026-02-28 14:10:58  W  Training "RL": Training run 0 started...

    [Step 500]  Score: 23.0 | TopLimit: 20 | Scramble: 1
    [Step 1000] Score: 46.0 | TopLimit: 20 | Scramble: 1
    ...
    ==================================================
    BREAK: FIRST LAYER COMPLETE (Start Dynamic Patience)
    ==================================================
    ...
    >>> LEVEL UP! Scramble length -> 2
    >>> SUCCESS! Cube Solved.


The short demo below shows the visualisation window during a training run:

.. image:: images/rubiks_cube_2x2x2_demo.gif
   :width: 500


**Cross Reference**

    - :ref:`env_rubiks_cube_2x2x2`
    - :ref:`API Reference <api_basics>`
    - `Project repository <https://github.com/fhswf/eet-fat-is2025-g32>`_