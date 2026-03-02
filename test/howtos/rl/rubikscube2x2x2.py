## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_gymnasium.envs
## -- Module  : rubikscube2x2x2.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2026-02-26  0.0.0     DA       Initial preparation
## -- 2026-02-27  1.0.0     MTA      Define evnvironment class
## -- 2026-02-28  1.0.1     SFAB     Added state, action space, and reward functions
## -- 2026-02-28  1.0.2     MTA      Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2026-02-26)

This module provides a standardized MLPro environment for a 2x2x2 Rubik's Cube based on the
rubiks-cube-gym package (https://github.com/DoubleGremlin181/RubiksCubeGym).

The environment wraps the registered Gymnasium environment 'rubiks-cube-222-v0' using
WrEnvGYM2MLPro and exposes it as a single-agent MLPro RL environment.

Observation space : Discrete(3674160) -- index into the full state dictionary
Action space      : Discrete(3)       -- moves F=0, R=1, U=2
Reward            : Defined by reward strategy. Default: -1 per step, +100 on solve.
"""

from mlpro.rl.models import *
from mlpro.bf import *
from mlpro_int_gymnasium.wrappers import WrEnvGYM2MLPro
from mlpro.bf.systems import *
from mlpro.bf.various import Persistent
from mlpro.bf.plot import *
from mlpro.bf.math import *
import gymnasium as gym
from datetime import timedelta
import rubiks_cube_gym
import numpy as np


# Export list for public API
__all__ = ['RubiksCube222']


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RubiksCube222 (Environment):
    """
    MLPro RL environment for the 2x2x2 Rubik's Cube.

    Wraps the Gymnasium environment 'rubiks-cube-222-v0' from the rubiks-cube-gym package
    using WrEnvGYM2MLPro.

    Custom reward shaping is supported by passing a pre-wrapped gym env via p_gym_env.
    For example, apply a gym.Wrapper subclass to the raw gym env first, then pass it here.
    If p_gym_env is None, the raw 'rubiks-cube-222-v0' env is created internally.

    Parameters
    ----------
    p_gym_env : gym.Env, optional
        A (optionally pre-wrapped) Gymnasium environment. If None, 'rubiks-cube-222-v0'
        is instantiated internally.
    p_visualize : bool
        If True, renders the environment visually. Default: True.
    p_logging
        MLPro log level. Default: Log.C_LOG_ALL.
    """

    C_NAME        = 'RubiksCube222'
    C_LATENCY     = timedelta(0, 1, 0)
    C_PLOT_ACTIVE = True


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_gym_env=None,
                 p_visualize: bool = True,
                 p_logging=Log.C_LOG_ALL):

        # IMPORTANT: _gym_env_ext and _env must be assigned BEFORE super().__init__()
        # because MLPro's base class may trigger _init_env() internally during setup.
        # However, _state_space and _action_space must be assigned AFTER super().__init__()
        # because MLPro's base class overwrites them with None during its own __init__.
        self._env         = None
        self._gym_env_ext = p_gym_env           # externally provided (possibly wrapped) gym env

        super().__init__(p_mode=Mode.C_MODE_SIM,
                         p_visualize=p_visualize,
                         p_logging=p_logging)

        self._state_space, self._action_space = self._setup_spaces()
        self._init_env()


## -------------------------------------------------------------------------------------------------
    def _init_env(self):
        """
        Wraps the Gymnasium environment with WrEnvGYM2MLPro.
        Uses the externally provided gym env if given, otherwise creates a fresh one.
        Called once during __init__ and again after loading from file (see load()).
        Guarded against premature calls before spaces are initialised.
        """
        # Guard: if spaces are not yet set up (premature call from base __init__), skip
        if not hasattr(self, '_state_space') or self._state_space is None:
            return

        if self._gym_env_ext is not None:
            gym_env = self._gym_env_ext
        elif self.get_visualization():
            gym_env = gym.make('rubiks-cube-222-v0', render_mode='human')
        else:
            gym_env = gym.make('rubiks-cube-222-v0')

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
        self._env         = None
        self._gym_env_ext = None

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
        When a ShapedRewardCubeWrapper is used as p_gym_env, the shaped reward
        returned by the wrapper's step() is forwarded here via the MLPro state.
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
