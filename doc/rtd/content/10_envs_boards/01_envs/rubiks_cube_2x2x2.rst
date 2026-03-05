.. _env_rubiks_cube_2x2x2:

Rubik's Cube 2x2x2
==================


.. video::
    images/rubiks_demo.mp4
    :width: 700
    :autoplay:
    :loop:



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



**Cross Reference**

    - :ref:`API Reference <api_basics>`
    - `Project repository <https://github.com/fhswf/eet-fat-is2025-g32>`_
