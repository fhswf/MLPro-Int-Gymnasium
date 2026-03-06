.. _env_rubiks_cube_2x2x2:

Rubik's Cube 2x2x2
==================

.. image:: images/rubiks_cube_2x2x2_demo.gif
   :width: 500
   :align: center
   :alt: 2x2x2 Rubik's Cube being solved by a PPO agent trained with MLPro


The 2x2x2 Pocket Rubik's Cube is a combinatorial puzzle with **3,674,160 states** under the FRU move system (Front, Right, Upper face rotations). The goal is
to reach the single solved state from a randomly scrambled configuration using the
minimum number of moves.

This environment wraps the third-party Gymnasium environment ``rubiks-cube-222-v0``
from the `rubiks-cube-gym <https://github.com/DoubleGremlin181/RubiksCubeGym>`_ package
via :class:`WrEnvGYM2MLPro`, exposing it as a fully MLPro-compatible single-agent RL
environment.


**Observation Space**

The observation is a single integer index in **[0, 3,674,159]** that uniquely identifies
the current cube configuration in a precomputed state dictionary. This compact
representation avoids raw pixel or tile-array observations and allows the agent to
distinguish all reachable states unambiguously.


**Action Space**

The agent has three discrete actions corresponding to clockwise quarter-turns of the
three available faces:

+--------+-------+---------------------------+
| Action | Move  | Description               |
+========+=======+===========================+
| 0      | F     | Front face clockwise      |
+--------+-------+---------------------------+
| 1      | R     | Right face clockwise      |
+--------+-------+---------------------------+
| 2      | U     | Upper face clockwise      |
+--------+-------+---------------------------+


**Reward**

Two reward modes are available via the ``p_shaped_reward`` parameter of
:class:`RubiksCube222`.


*Default sparse reward* (``p_shaped_reward=False``)

The agent receives **+100** when the cube is fully solved and **-1** on every other
step. The -1 step penalty discourages unnecessary moves and encourages the agent to
solve the cube as efficiently as possible. However, because a randomly acting agent
almost never reaches the solved state by chance, the agent receives almost no positive
learning signal in early training, making convergence extremely slow or infeasible for
longer scrambles.


*Shaped reward with curriculum* (``p_shaped_reward=True``)

Activates the built-in :class:`ShapedRewardCubeWrapper` which replaces the sparse
signal with a dense, milestone-based reward. The reward at each step equals the
**highest milestone score reached so far in the current episode**, minus a small decay
penalty to discourage wandering (0.05 during the bottom layer phase, 0.001 during the
top layer phase).

This design means the agent always receives a meaningful gradient signal from the very
first episode, even before it has ever seen the solved state.


**Reward Shaping Strategy**

The ``ShapedRewardCubeWrapper`` decomposes the solve into two phases — bottom layer
first, then top layer — mirroring the layer-by-layer method used by human beginners.
Progress is tracked across 9 milestones:

+-------------+-------------------------------------------+--------+
| Milestone   | Description                               | Score  |
+=============+===========================================+========+
| 0           | Yellow sticker visible on bottom face     | 2.0    |
+-------------+-------------------------------------------+--------+
| 1           | 1st bottom corner fully solved            | 9.0    |
+-------------+-------------------------------------------+--------+
| 2           | 2nd bottom corner fully solved            | 23.0   |
+-------------+-------------------------------------------+--------+
| 3           | 3rd bottom corner solved                  | 46.0   |
|             | (58.0 if 4th corner is in slot)           |        |
+-------------+-------------------------------------------+--------+
| 4           | Bottom layer complete                     | 69.0   |
+-------------+-------------------------------------------+--------+
| 5           | Top layer permutation complete            | 75.0   |
+-------------+-------------------------------------------+--------+
| 6           | 1st top corner oriented (white facing up) | 78.0   |
+-------------+-------------------------------------------+--------+
| 7           | 2nd top corner oriented                   | 81.0   |
+-------------+-------------------------------------------+--------+
| 8           | 3rd top corner oriented                   | 84.0   |
+-------------+-------------------------------------------+--------+
| 9           | Cube fully solved                         | 87.0   |
+-------------+-------------------------------------------+--------+

Three additional mechanisms accelerate learning:

- **Checkpoint save/restore** — when the agent reaches milestone 3 or higher and then
  fails to make progress within the allowed step budget, the cube is automatically
  reverted to that milestone state. This prevents the agent from accidentally undoing
  its own progress and forces it to focus only on the unsolved part of the cube.

- **Dynamic patience** — the step budget for the top layer starts at 20 and grows by 5
  each time the agent hits the limit, up to a maximum of 50. This adapts the allowed
  exploration window to the actual difficulty the agent is experiencing at each training
  stage rather than using a fixed hardcoded limit.

- **Curriculum learning** — the scramble length starts at 1 (one move away from solved)
  and increases by 1 after every 3 successful solves, up to a maximum of 15. This
  ensures the agent is never overwhelmed by a problem far beyond its current capability
  and always has a realistic chance of receiving a positive reward signal.


**Parameters**

+--------------------+----------+----------+-------------------------------------------+
| Parameter          | Type     | Default  | Description                               |
+====================+==========+==========+===========================================+
| p_shaped_reward    | bool     | False    | Activates ShapedRewardCubeWrapper         |
+--------------------+----------+----------+-------------------------------------------+
| p_visualize        | bool     | True     | Opens a live rendering window             |
+--------------------+----------+----------+-------------------------------------------+
| p_logging          | int      | LOG_ALL  | MLPro log level                           |
+--------------------+----------+----------+-------------------------------------------+


**Usage Example**

.. code-block:: python

    from mlpro_int_gymnasium.envs import RubiksCube222

    # Default sparse reward
    env = RubiksCube222(p_shaped_reward=False, p_visualize=True)

    # With shaped reward and curriculum learning
    env = RubiksCube222(p_shaped_reward=True, p_visualize=True)


**Cross Reference**

    - :ref:`Howto RL AGENT 010 <howto_rl_agent_010>` — Train a SB3 PPO agent on this environment
    - :ref:`API Reference <api_basics>`
    - `Project repository <https://github.com/fhswf/eet-fat-is2025-g32>`_