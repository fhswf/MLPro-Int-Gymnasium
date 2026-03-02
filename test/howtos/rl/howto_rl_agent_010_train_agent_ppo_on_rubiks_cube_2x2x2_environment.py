## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_gymnasium
## -- Module  : howto_rl_agent_010_train_agent_ppo_on_rubiks_cube_2x2x2.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2026-02-26  0.0.0     DA       Initial preparation
## -- 2026-02-27  1.0.0     MTA      Initial SB3 PPO implementation
## -- 2026-02-28  1.0.1     SFAB     Added ShapedRewardCubeWrapper
## -- 2026-02-28  1.0.2     MTA      Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2026-02-28)

This howto demonstrates how to train a Stable Baselines3 (SB3) PPO agent on the 2x2x2 Rubik's
Cube environment using the MLPro RL training infrastructure.

You will learn:
  1. How to instantiate an MLPro-wrapped Gymnasium environment (RubiksCube222)
  2. How to define a custom reward strategy to guide faster convergence
  3. How to wrap an SB3 policy (PPO) with WrPolicySB32MLPro
  4. How to configure an RLScenario and run an RLTraining
"""

from pathlib import Path
import random

from mlpro.bf.ml import Model
from mlpro.bf.various import Log
from mlpro.rl import RLScenario, RLTraining
from mlpro.rl.models import *
from mlpro.bf.systems import *

from mlpro_int_sb3.wrappers import WrPolicySB32MLPro
from rubikscube2x2x2 import RubiksCube222

from stable_baselines3 import PPO
import gymnasium as gym
import rubiks_cube_gym                           


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1  Custom reward strategy
#
#    You can define your own reward shaping function and pass it to RubiksCube222 via
#    p_reward_strategy. The function receives:
#      - state_old : MLPro State before the action
#      - state_new : MLPro State after the action
#      - solved    : bool, True if the cube is solved
#    and must return a single float reward value.
#
#    The example below adds a small bonus whenever the new state index is lower than the
#    old one, rewarding progress toward a solved configuration.

class ShapedRewardCubeWrapper (gym.Wrapper):
    """
    Gymnasium wrapper implementing a "Reverse Learning" curriculum for the 2x2x2 Rubik's Cube.

    Key design decisions
    --------------------
    - Scramble length starts at 1 so the agent masters the endgame before harder scrambles.
    - Milestone scoring (0-87) tracks progress across 9 checkpoints:
        0-3  : bottom layer corners
        4    : bottom layer complete
        5    : top-layer permutation complete
        6-9  : top-layer orientation (one white per step)
    - Checkpoint save/restore: if the agent stalls at a milestone, the cube is
      reverted to that checkpoint so it can retry from a known-good state.
    - Dynamic patience: the step budget for the top layer grows by 5 each time
      the agent hits the limit, up to MAX_POSSIBLE_LIMIT.
    - Curriculum: scramble length increases by 1 every 3 successful solves.
    """

    # --- Class-level state shared across episodes ---
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
        self.C4_SLOT_INDICES    = [22, 12, 19]
        self.C4_TARGET_COLORS   = {'Y', 'O', 'B'}

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
        c1 = self._is_corner_solved(self.C1_INDICES, cube_mapped)
        c2 = self._is_corner_solved(self.C2_INDICES, cube_mapped)
        c3 = self._is_corner_solved(self.C3_INDICES, cube_mapped)
        c4 = self._is_corner_solved(self.C4_INDICES, cube_mapped)
        bottom_count = sum([c1, c2, c3, c4])

        # --- Bottom layer ---
        if bottom_count < 4:
            if bottom_count == 3:
                if self._is_corner_positioned(self.C4_SLOT_INDICES, self.C4_TARGET_COLORS, cube_mapped):
                    return 58.0, 3
                return 46.0, 3
            if bottom_count == 2: return 23.0, 2
            if c1:                return 9.0,  1
            if self._get_color(20, cube_mapped) == 'Y': return 2.0, 0
            return 0.0, 0

        # --- Bottom layer complete (milestone 4) ---
        score, milestone = 69.0, 4

        # --- Top layer permutation ---
        perm_count = sum(
            self._is_corner_positioned(idxs, self.REQUIRED_COLORS[pos], cube_mapped)
            for pos, idxs in self.TOP_SLOTS.items()
        )
        if perm_count == 4:
            score, milestone = 75.0, 5

            # --- Top layer orientation ---
            whites = sum(1 for i in range(4) if self._get_color(i, cube_mapped) == 'W')
            if   whites == 1: score, milestone = 78.0, 6
            elif whites == 2: score, milestone = 81.0, 7
            elif whites == 3: score, milestone = 84.0, 8
            elif whites == 4: score, milestone = 87.0, 9

        return score, milestone


## -------------------------------------------------------------------------------------------------
    def generate_custom_scramble(self, length):
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
        if hasattr(self.env, 'cube'):
            self.backup_cube_state = self.env.cube.copy()


## -------------------------------------------------------------------------------------------------
    def restore_checkpoint(self):
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
            self.current_milestone_id    = current_milestone
            self.moves_since_checkpoint  = 0
            if current_milestone >= 4:
                self.current_top_layer_limit = 20
            self.save_checkpoint()

        # Restore checkpoint if the agent stalls
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

        # Progress break messages
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
        decay = 0.05 if self.highest_checkpoint_reached < 69.0 else 0.001
        final_reward = self.highest_checkpoint_reached - decay

        # Curriculum: terminate on solve and increase difficulty every 3 wins
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
# 2  RL scenario
class RubiksPPOScenario (RLScenario):

    C_NAME = 'RubiksCube222_Scenario'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:

        # 2.1 Instantiate the MLPro environment.
        #     Pass p_reward_strategy=None to use the built-in default reward (+100/-1),
        #     or pass my_reward_strategy (or any other callable) for custom shaping.
        raw_gym_env   = gym.make('rubiks-cube-222-v0')
        shaped_gym_env = ShapedRewardCubeWrapper(raw_gym_env)
        self._env      = RubiksCube222(
            p_gym_env=shaped_gym_env,
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        # 2.2 Create the SB3 PPO policy.
        #     A plain gym env is passed so that SB3 can infer observation and action
        #     spaces internally. WrPolicySB32MLPro handles all actual env interaction.
        policy_sb3 = PPO(
            policy='MlpPolicy',
            env=None,
            _init_setup_model=False,
            n_steps=2048,
            learning_rate=1e-4,
            gamma=0.99,
            batch_size=64,
            ent_coef=0.2,
            device='cpu',
            seed=1
        )

        # 2.3 Wrap the SB3 policy for MLPro
        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_logging=p_logging
        )

        # 2.4 Create and return the agent
        return Agent(
            p_policy=policy_wrapped,
            p_id=1,
            p_name='PPO_Agent',
            p_ada=p_ada,
            p_logging=p_logging
        )


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 3  Parameterisation and training
if __name__ == '__main__':
    # 3.1 Parameters for interactive demo
    cycle_limit = 100
    logging     = Log.C_LOG_ALL
    visualize   = True
    path        = str(Path.home())

else:
    # 3.2 Parameters for internal unit test
    cycle_limit = 10
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None


# 3.3 Instantiate the training object and run
training = RLTraining(
    p_scenario_cls=RubiksPPOScenario,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=cycle_limit,
    p_ada=True,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging
)

training.run()