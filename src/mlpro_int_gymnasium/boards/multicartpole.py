## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_gymnasium.boards
## -- Module  : multicartpole.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-06  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Released first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-12-03  1.0.1     DA       Refactoring
## -- 2022-11-07  1.1.0     DA       Refactoring
## -- 2023-04-12  1.1.1     SY       Refactoring 
## -- 2023-05-11  1.1.2     SY       Refactoring
## -- 2023-06-27  1.1.3     SY       Refactoring module name
## -- 2024-02-16  1.3.5     SY       Relocation from MLPro to MLPro-Int-Gymnasium
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.5 (2024-02-16)

This module provides game board classes based on the Multi-CartPole environment
of the reinforcement learning pool.
"""

from mlpro.rl.models import Reward
from mlpro_int_gymnasium.envs.multicartpole import MultiCartPole
from mlpro.gt import *
from mlpro.gt.dynamicgames.potential import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiCartPoleGT(MultiCartPole, GameBoard):

    """
    Game theoretical pendant for the reinforcement learning environment class MultiCartPole.
    """

    C_NAME          = 'MultiCartPole(GT)'

    def __init__(self, p_num_envs=2, p_visualize:bool=True, p_logging=Log.C_LOG_ALL):
        MultiCartPole.__init__( self, 
                                p_num_envs=p_num_envs, 
                                p_reward_type=Reward.C_TYPE_EVERY_AGENT, 
                                p_visualize=p_visualize, 
                                p_logging=p_logging )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiCartPolePGT(MultiCartPole, PGameBoard):
    """
    Potential Game theoretical pendant for the reinforcement learning environment class MultiCartPole.
    """

    C_NAME          = 'MultiCartPole(PGT)'

    def __init__(self, p_num_envs=2, p_visualize:bool=True, p_logging=Log.C_LOG_ALL):
        MultiCartPole.__init__( self, 
                                p_num_envs=p_num_envs, 
                                p_reward_type=Reward.C_TYPE_EVERY_AGENT, 
                                p_visualize=p_visualize,
                                p_logging=p_logging )
