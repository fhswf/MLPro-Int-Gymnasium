## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_gymnasium.envs
## -- Module  : rubikscube2x2x2.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2026-02-26  0.0.0     DA       Initial preparation
## -- 2026-02-27  1.0.0     MTA      Define evnvironment class
## -- 2026-02-28  1.0.1     MTA      Released first version
## -- 2026-03-05  1.0.2     SFAB     Added ShapedRewardCubeWrapper
## -- 2026-03-10  1.0.3     DA       Code review/stabilization of RubiksCube222._init_env()
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2026-03-10)

This module provides a standardized MLPro environment for a 2x2x2 Rubik's Cube based on the
rubiks-cube-gym package (https://github.com/DoubleGremlin181/RubiksCubeGym).

The environment wraps the registered Gymnasium environment 'rubiks-cube-222-v0' using
WrEnvGYM2MLPro and exposes it as a single-agent MLPro RL environment.

Observation space : Discrete(3674160) -- index into the full state dictionary
Action space      : Discrete(3)       -- moves F=0, R=1, U=2
Reward            : Defined by reward strategy. Default: -1 per step, +100 on solve.
"""


from datetime import timedelta
import random

import numpy as np
import gymnasium as gym

from mlpro.rl.models import *
from mlpro.bf import *
from mlpro.bf.systems import *
from mlpro.bf.various import Persistent
from mlpro.bf.plot import *
from mlpro.bf.math import *
#import rubiks_cube_gym

from mlpro_int_gymnasium.wrappers import WrEnvGYM2MLPro



# Export list for public API
__all__ = ['RubiksCube222']



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ShapedRewardCubeWrapper (gym.Wrapper):
    """
    Gymnasium wrapper implementing a "Reverse Learning" curriculum for the 2x2x2 Rubik's Cube.

    Replaces the sparse default reward with a dense milestone-based signal that guides
    the PPO agent layer by layer from bottom to top, enabling significantly faster convergence
    compared to the default +100/-1 sparse reward.

    Key design decisions
    --------------------
    - Scramble length starts at 1 so the agent masters the endgame before harder scrambles.
    - Milestone scoring (0.0 - 87.0) tracks progress across 9 checkpoints:
        0-3  : bottom layer corners (scores: 0, 2, 9, 23, 46/58)
        4    : bottom layer complete (score: 69)
        5    : top-layer permutation complete (score: 75)
        6-9  : top-layer orientation, one white tile per step (scores: 78, 81, 84, 87)
    - Checkpoint save/restore: if the agent stalls at a milestone, the cube reverts
      to that checkpoint so it retries from a known-good state.
    - Dynamic patience: the step budget for the top layer grows by 5 each time
      the agent hits the limit, up to MAX_POSSIBLE_LIMIT (50).
    - Curriculum: scramble length increases by 1 every 3 successful solves, up to 15.

    Parameters
    ----------
    env : gym.Env
        The raw 'rubiks-cube-222-v0' gymnasium environment to wrap.
    """

    # --- Class-level state shared across all episodes ---
    overall_best_milestone = 0
    scramble_length        = 1
    success_tracker        = 0

    MAX_POSSIBLE_LIMIT       = 50
    BOTTOM_LAYER_FIXED_LIMIT = 15


## -------------------------------------------------------------------------------------------------
    def __init__(self, env):
        super().__init__(env)

        # Tile indices for each bottom-layer corner {target_color: tile_index}
        self.C1_INDICES = {'Y': 20, 'G': 14, 'O': 13}
        self.C2_INDICES = {'Y': 21, 'R': 16, 'G': 15}
        self.C3_INDICES = {'Y': 23, 'B': 18, 'R': 17}
        self.C4_INDICES = {'Y': 22, 'O': 12, 'B': 19}
        self.C4_SLOT_INDICES  = [22, 12, 19]
        self.C4_TARGET_COLORS = {'Y', 'O', 'B'}

        # Top-layer slot definitions {slot_name: [tile_indices]}
        self.TOP_SLOTS = {
            'FL': [2, 6, 5],  'FR': [3, 7, 8],
            'TR': [1, 10, 9], 'TL': [0, 11, 4]
        }
        self.REQUIRED_COLORS = {
            'FL': {'W', 'G', 'O'}, 'FR': {'W', 'R', 'G'},
            'TR': {'W', 'B', 'R'}, 'TL': {'W', 'O', 'B'}
        }

        self.highest_checkpoint_reached = 0.0
        self.step_counter               = 0
        self.backup_cube_state          = None
        self.moves_since_checkpoint     = 0
        self.current_milestone_id       = 0
        self.current_top_layer_limit    = 20


## -------------------------------------------------------------------------------------------------
    def _get_color(self, index, cube_mapped):
        return cube_mapped[index]


## -------------------------------------------------------------------------------------------------
    def _is_corner_solved(self, indices, cube_mapped):
        return all(self._get_color(idx, cube_mapped) == color for color, idx in indices.items())


## -------------------------------------------------------------------------------------------------
    def _is_corner_positioned(self, indices_list, target_colors, cube_mapped):
        return {self._get_color(i, cube_mapped) for i in indices_list} == target_colors


## -------------------------------------------------------------------------------------------------
    def _get_score_and_milestone(self, cube_mapped):
        """
        Evaluates the current cube state and returns (score, milestone_id).
        Score is a float in [0.0, 87.0]. Milestone is an integer in [0, 9].
        """
        c1 = self._is_corner_solved(self.C1_INDICES, cube_mapped)
        c2 = self._is_corner_solved(self.C2_INDICES, cube_mapped)
        c3 = self._is_corner_solved(self.C3_INDICES, cube_mapped)
        c4 = self._is_corner_solved(self.C4_INDICES, cube_mapped)
        bottom_count = sum([c1, c2, c3, c4])

        # --- Phase 1: Bottom layer ---
        if bottom_count < 4:
            if bottom_count == 3:
                if self._is_corner_positioned(self.C4_SLOT_INDICES, self.C4_TARGET_COLORS, cube_mapped):
                    return 58.0, 3       # 4th corner is in slot but not oriented
                return 46.0, 3           # 4th corner not yet in slot
            if bottom_count == 2: return 23.0, 2
            if c1:                return 9.0,  1
            if self._get_color(20, cube_mapped) == 'Y': return 2.0, 0
            return 0.0, 0

        # --- Phase 2: Bottom layer complete ---
        score, milestone = 69.0, 4

        # --- Phase 3: Top layer permutation ---
        perm_count = sum(
            self._is_corner_positioned(idxs, self.REQUIRED_COLORS[pos], cube_mapped)
            for pos, idxs in self.TOP_SLOTS.items()
        )
        if perm_count == 4:
            score, milestone = 75.0, 5

            # --- Phase 4: Top layer orientation ---
            whites = sum(1 for i in range(4) if self._get_color(i, cube_mapped) == 'W')
            if   whites == 1: score, milestone = 78.0, 6
            elif whites == 2: score, milestone = 81.0, 7
            elif whites == 3: score, milestone = 84.0, 8
            elif whites == 4: score, milestone = 87.0, 9

        return score, milestone


## -------------------------------------------------------------------------------------------------
    def generate_custom_scramble(self, length):
        """Generates a random scramble of the given length using only F, R, U moves."""
        moves, types = ['F', 'R', 'U'], ['', "'", '2']
        scramble, last = '', ''
        for _ in range(length):
            m = random.choice(moves)
            while m == last:
                m = random.choice(moves)
            scramble += m + random.choice(types) + ' '
            last = m
        return scramble.strip()


## -------------------------------------------------------------------------------------------------
    def save_checkpoint(self):
        """Saves the current cube tile array as a checkpoint for later restoration."""
        if hasattr(self.env, 'cube'):
            self.backup_cube_state = self.env.cube.copy()


## -------------------------------------------------------------------------------------------------
    def restore_checkpoint(self):
        """Restores the cube to the last saved checkpoint state."""
        if self.backup_cube_state is not None:
            self.env.cube = self.backup_cube_state.copy()
            self.env.update_cube_reduced()
            self.env.update_cube_state()


## -------------------------------------------------------------------------------------------------
    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        cube_mapped = info.get('cube_reduced')

        current_score, current_milestone = self._get_score_and_milestone(cube_mapped)

        # Latch the highest score reached this episode
        if current_score > self.highest_checkpoint_reached:
            self.highest_checkpoint_reached = current_score

        # Save checkpoint when a new milestone is reached
        if current_milestone > self.current_milestone_id and current_milestone >= 3:
            self.current_milestone_id   = current_milestone
            self.moves_since_checkpoint = 0
            if current_milestone >= 4:
                self.current_top_layer_limit = 20
            self.save_checkpoint()

        # Restore checkpoint if the agent stalls at the current milestone
        limit_hit = False
        if self.current_milestone_id == 3 and current_milestone == 3:
            self.moves_since_checkpoint += 1
            if self.moves_since_checkpoint >= self.BOTTOM_LAYER_FIXED_LIMIT:
                limit_hit = True
        elif self.current_milestone_id >= 4 and current_milestone == self.current_milestone_id:
            self.moves_since_checkpoint += 1
            if self.moves_since_checkpoint >= self.current_top_layer_limit:
                limit_hit = True
                if self.current_top_layer_limit < self.MAX_POSSIBLE_LIMIT:
                    self.current_top_layer_limit += 5

        if limit_hit:
            self.restore_checkpoint()
            self.moves_since_checkpoint = 0
            observation = self.env.cube_state
            info['cube_reduced'] = self.env.cube_reduced
            return observation, self.highest_checkpoint_reached - 0.1, terminated, truncated, info

        # Log new global milestone records
        if current_milestone > ShapedRewardCubeWrapper.overall_best_milestone:
            ShapedRewardCubeWrapper.overall_best_milestone = current_milestone
            labels = {
                2: 'BREAK: 2ND BOTTOM CORNER SOLVED',
                3: 'BREAK: 3 BOTTOM CORNERS SOLVED (Start 15-move Safety Net)',
                4: 'BREAK: FIRST LAYER COMPLETE (Start Dynamic Patience)',
                5: 'BREAK: TOP PERMUTATION COMPLETE',
                6: 'BREAK: 1ST TOP CORNER ORIENTED',
                7: 'BREAK: 2ND TOP CORNER ORIENTED',
                8: 'BREAK: 3RD TOP CORNER ORIENTED',
                9: 'BREAK: CUBE FULLY SOLVED'
            }
            if current_milestone in labels:
                print(f"\n{'='*50}\n{labels[current_milestone]}\n{'='*50}")

        # Compute final shaped reward
        decay        = 0.05 if self.highest_checkpoint_reached < 69.0 else 0.001
        final_reward = self.highest_checkpoint_reached - decay

        # Curriculum: terminate on solve and increase scramble difficulty every 3 wins
        if final_reward >= 86.9:
            terminated = True
            ShapedRewardCubeWrapper.success_tracker += 1
            if ShapedRewardCubeWrapper.success_tracker % 3 == 0:
                ShapedRewardCubeWrapper.scramble_length = min(
                    ShapedRewardCubeWrapper.scramble_length + 1, 15
                )
                print(f'>>> LEVEL UP! Scramble length -> {ShapedRewardCubeWrapper.scramble_length}')
            print('\n' + '='*60)
            print('>>> SUCCESS! Cube Solved.')
            print('='*60 + '\n')

        self.step_counter += 1
        if self.step_counter % 500 == 0:
            print(f'[Step {self.step_counter}] Score: {current_score:.1f} | '
                  f'TopLimit: {self.current_top_layer_limit} | '
                  f'Scramble: {ShapedRewardCubeWrapper.scramble_length}')

        return observation, final_reward, terminated, truncated, info


## -------------------------------------------------------------------------------------------------
    def reset(self, **kwargs):
        self.highest_checkpoint_reached = 0.0
        self.step_counter               = 0
        self.moves_since_checkpoint     = 0
        self.current_milestone_id       = 0
        self.backup_cube_state          = None
        self.current_top_layer_limit    = 20

        scramble_str = self.generate_custom_scramble(ShapedRewardCubeWrapper.scramble_length)
        obs, info    = self.env.reset(options={'scramble': scramble_str})

        current_score, current_milestone = self._get_score_and_milestone(info.get('cube_reduced'))
        if current_milestone >= 3:
            self.current_milestone_id = current_milestone
            self.save_checkpoint()

        return obs, info
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RubiksCube222 (Environment):
    """
    MLPro RL environment for the 2x2x2 Rubik's Cube.

    Wraps the Gymnasium environment 'rubiks-cube-222-v0' from the rubiks-cube-gym package
    using WrEnvGYM2MLPro.
    
    """

    C_NAME        = 'RubiksCube222'
    C_LATENCY     = timedelta(0, 1, 0)
    C_PLOT_ACTIVE = True


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_shaped_reward: bool = False,
                 p_visualize: bool = True,
                 p_logging=Log.C_LOG_ALL):

        # IMPORTANT: _gym_env_ext and _env must be assigned BEFORE super().__init__()
        # because MLPro's base class may trigger _init_env() internally during setup.
        # However, _state_space and _action_space must be assigned AFTER super().__init__()
        # because MLPro's base class overwrites them with None during its own __init__.
        self._env         = None
        self._shaped_reward = p_shaped_reward

        super().__init__(p_mode=Mode.C_MODE_SIM,
                         p_visualize=p_visualize,
                         p_logging=p_logging)

        self._state_space, self._action_space = self._setup_spaces()
        self._init_env()


## -------------------------------------------------------------------------------------------------
    def _init_env(self):
        """
        Instantiates the gym env, optionally wraps it with ShapedRewardCubeWrapper,
        then wraps the result with WrEnvGYM2MLPro.
        Called once during __init__ and again after loading from file (see load()).
        Guarded against premature calls before spaces are initialised.
        """
        # Guard: skip if spaces are not yet ready (premature call from base __init__)
        if not hasattr(self, '_state_space') or self._state_space is None:
            return

        if self.get_visualization():
            try:
                gym_env = gym.make('rubiks-cube-222-v0', render_mode='human')
            except:
                gym_env = gym.make('rubiks-cube-222-v0')
        else:
            gym_env = gym.make('rubiks-cube-222-v0')

        # Optionally apply the shaped reward wrapper before passing to MLPro
        if self._shaped_reward:
            gym_env = ShapedRewardCubeWrapper(gym_env)

        self._env = WrEnvGYM2MLPro(
            p_gym_env=gym_env,
            p_state_space=self._state_space,
            p_action_space=self._action_space,
            p_visualize=self.get_visualization(),
            p_logging=self.get_log_level()
        )

        self.reset()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def load(p_path, p_filename):
        rubiks = Persistent.load(p_path, p_filename)
        rubiks._init_env()
        return rubiks


## -------------------------------------------------------------------------------------------------
    def _save(self, p_path, p_filename) -> bool:
        # Clear unpicklable gym env before serialisation
        self._env = None

        import pickle as pkl
        import os
        pkl.dump(obj=self,
                 file=open(p_path + os.sep + p_filename, 'wb'),
                 protocol=pkl.HIGHEST_PROTOCOL)

        self._init_env()
        return True


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        if self._env is not None:
            self._env.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        return None, None


## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):
        """
        Defines the MLPro state and action spaces:
          - State  : one integer dimension, index in [0, 3674159]
          - Action : one integer dimension, index in [0, 2]  (F=0, R=1, U=2)
        """
        state_space = ESpace()
        state_space.add_dim(
            Dimension(p_name_short='S',
                      p_name_long='Cube State Index',
                      p_boundaries=[0, 3674159])
        )

        action_space = ESpace()
        action_space.add_dim(
            Dimension(p_name_short='A',
                      p_name_long='Move (F=0, R=1, U=2)',
                      p_boundaries=[0, 2])
        )

        return state_space, action_space


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        self._env.reset(p_seed)
        self._set_state(self._env.get_state())


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """
        Clips the continuous action value produced by SB3 to a valid discrete
        index {0, 1, 2} before passing it to the wrapped gym environment.
        """
        agent_id    = p_action.get_agent_ids()[0]
        action_elem = p_action.get_elem(agent_id)
        action_dim  = action_elem.get_dim_ids()[0]
        raw_value   = float(action_elem.get_value(action_dim))
        clipped     = int(np.clip(round(raw_value), 0, 2))
        action_elem.set_value(action_dim, clipped)

        new_state = self._env.simulate_reaction(p_state, p_action)
        self._set_state(new_state)
        return new_state


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        """
        Default reward: +100 on solve, -1 on every other step.
        When p_shaped_reward=True, ShapedRewardCubeWrapper overrides the gym env's
        reward signal before it reaches this method.
        """
        solved = self.compute_success(p_state_new)
        r      = 100.0 if solved else -1.0

        reward = Reward(Reward.C_TYPE_OVERALL)
        reward.set_overall_reward(r)
        return reward


## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        return self._env.get_state().get_success()


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        return self._env.get_state().get_broken()


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None, p_plot_settings=..., p_plot_depth: int = 0,
                  p_detail_level: int = 0, p_step_rate: int = 0, **p_kwargs):
        if self._env is not None:
            self._env.init_plot(p_figure=None)


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        if self._env is not None:
            self._env.update_plot(**p_kwargs)