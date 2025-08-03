## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_gymnasium
## -- Module  : howto_rl_wp_001_mlpro_environment_to_gymnasium_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-30  0.0.0     SY       Creation
## -- 2021-09-30  1.0.0     SY       Released first version
## -- 2021-10-04  1.0.1     DA       Minor fixes
## -- 2021-12-22  1.0.2     DA       Cleaned up a bit
## -- 2022-03-21  1.0.3     MRD      Use Gym Env Checker
## -- 2022-05-30  1.0.4     DA       Little refactoring
## -- 2022-07-28  1.0.5     SY       Update due to the latest introduction of Gym 0.25
## -- 2022-10-14  1.0.6     SY       Refactoring 
## -- 2022-11-02  1.0.7     SY       Unable logging in unit test model
## -- 2023-03-02  1.0.8     LSB      Refactoring
## -- 2023-04-19  1.0.9     MRD      Refactor module import gym to gymnasium
## -- 2024-02-16  1.1.0     SY       Relocation from MLPro to MLPro-Int-Gymnasium
## -- 2024-10-10  1.1.1     SY       Temporary bypass the env checking due to GridWorld's issue
## -- 2024-12-03  1.1.2     SY       Re-enable the env checking
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.2 (2024-12-03)

This module shows how to wrap a native MLPro environment class to OpenAI Gym environment.

You will learn:

1. How to setup an MLPro environment.

2. How to wrap MLPro's native Environment class to the Gym environment object.
"""


from mlpro.bf import *
from mlpro.bf.math import *
from mlpro.bf.systems import *
from mlpro.bf.plot import *
import numpy as np
from mlpro_int_gymnasium.wrappers import WrEnvMLPro2GYM
from mlpro.rl.pool.envs.gridworld import GridWorld
from gymnasium.utils.env_checker import check_env


if __name__ == "__main__":
    logging = Log.C_LOG_ALL
else:
    logging = Log.C_LOG_NOTHING
    
# 1. Set up MLPro native environment
mlpro_env = GridWorld(p_logging=logging)

# 2. Wrap the MLPro environment to gym compatible environment
env = WrEnvMLPro2GYM(mlpro_env,
                     p_state_space=None,
                     p_action_space=None,
                     p_logging=logging)

# 3. Check whether the environment is valid
check_env(env)
