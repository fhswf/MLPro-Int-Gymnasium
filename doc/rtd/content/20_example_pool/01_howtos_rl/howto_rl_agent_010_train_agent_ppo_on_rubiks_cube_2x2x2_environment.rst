.. _Howto RL AGENT 010:

Howto RL-AGENT-010: Train a SB3 PPO Agent on the Rubik's Cube 2x2 Environment
===============================================================================

This howto demonstrates how to train a `Stable Baselines3 (SB3) <https://stable-baselines3.readthedocs.io>`_ 
PPO agent on the 2x2x2 Rubik's Cube environment using the MLPro RL training infrastructure.

**Prerequisites**

.. code-block:: bash

    pip install mlpro==2.0.0 mlpro-int-gymnasium mlpro-int-sb3==1.0.7 stable-baselines3 rubiks-cube-gym
 

**You will learn**

    - How to wrap the Gymnasium environment ``rubiks-cube-222-v0`` as an MLPro RL environment
    - How to apply custom reward shaping via a ``gym.Wrapper`` (``ShapedRewardCubeWrapper``)
    - How to use a curriculum learning strategy (progressive scramble difficulty)
    - How to wrap an SB3 PPO policy with ``WrPolicySB32MLPro``
    - How to configure an ``RLScenario`` and run an ``RLTraining``


**Reward Shaping Strategy**

The ``ShapedRewardCubeWrapper`` replaces the sparse default reward with a dense,
milestone-based curriculum signal across 9 checkpoints:

+-------------+-----------------------------------+--------+
| Milestone   | Description                       | Score  |
+=============+===================================+========+
| 0           | Yellow sticker on bottom face     | 2.0    |
+-------------+-----------------------------------+--------+
| 1           | 1st bottom corner solved          | 9.0    |
+-------------+-----------------------------------+--------+
| 2           | 2nd bottom corner solved          | 23.0   |
+-------------+-----------------------------------+--------+
| 3           | 3rd bottom corner solved          | 46.0   |
+-------------+-----------------------------------+--------+
| 4           | Bottom layer complete             | 69.0   |
+-------------+-----------------------------------+--------+
| 5           | Top layer permutation complete    | 75.0   |
+-------------+-----------------------------------+--------+
| 6-8         | Top layer orientation (1-3 white) | 78-84  |
+-------------+-----------------------------------+--------+
| 9           | Cube fully solved                 | 87.0   |
+-------------+-----------------------------------+--------+

Additional features:

    - **Checkpoint save/restore**: the cube reverts to the last milestone if the agent stalls
    - **Dynamic patience**: the step budget for the top layer grows up to 50 if the agent keeps getting stuck
    - **Curriculum**: scramble length starts at 1 and increases by 1 every 3 successful solves (up to 15)

**Executable Code**

.. literalinclude:: ../../../../../test/howtos/rl/howto_rl_agent_010_train_agent_ppo_on_rubiks_cube_2x2x2_environment.py
    :language: python


**Results**

The training log will show milestone progress printed to the console every 500 steps::

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

.. video::
    /images/rubiks_demo.mp4
    :width: 700
    :autoplay:
    :loop:


**Cross Reference**

    - :ref:`API Reference <api_basics>`
    - `Project repository <https://github.com/fhswf/eet-fat-is2025-g32>`_
