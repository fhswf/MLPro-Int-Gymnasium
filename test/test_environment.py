## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_gymnasium
## -- Module  : test_environment
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-02-16  1.0.0     SY       Creation
## -- 2024-02-16  1.0.0     SY       Release First Version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-02-16)

Unit test classes for environment.
"""


import pytest
import random
import numpy as np
from mlpro.rl.models import *
from mlpro_int_gymnasium.envs.multicartpole import MultiCartPole


## -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env_cls", [MultiCartPole])
def test_environment(env_cls):
    env = env_cls(p_visualize=False)
    assert isinstance(env, Environment)
    
    assert isinstance(env.get_state_space(), ESpace)
    assert env.get_state_space().get_num_dim() != 0
    
    assert isinstance(env.get_action_space(), ESpace)
    assert env.get_action_space().get_num_dim() != 0
    
    state = env.get_state()
    
    assert isinstance(state, State)
        
    my_action_values = np.zeros(env.get_action_space().get_num_dim())
    for d in range(env.get_action_space().get_num_dim()):
        my_action_values[d] = random.random() 

    my_action_values = Action(0, env.get_action_space(), my_action_values)

    env.process_action(my_action_values)

    reward = env.compute_reward()
    
    assert isinstance(reward, Reward)

    env.reset()
