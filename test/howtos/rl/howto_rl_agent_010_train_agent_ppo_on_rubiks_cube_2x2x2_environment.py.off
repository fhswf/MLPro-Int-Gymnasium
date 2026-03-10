## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_gymnasium
## -- Module  : howto_rl_agent_010_train_agent_ppo_on_rubiks_cube_2x2x2.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2026-02-26  0.0.0     DA       Initial preparation
## -- 2026-02-27  1.0.0     MTA      SB3 PPO implementation
## -- 2026-02-28  1.0.1     MTA      Released first version
## -- 2026-03-10  1.0.2     MTA      Added final evaluation section
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2026-03-10)

This howto demonstrates how to train a Stable Baselines3 (SB3) PPO agent on the 2x2x2 Rubik's
Cube environment using the MLPro RL training infrastructure.

You will learn:
  1. How to instantiate an MLPro-wrapped Gymnasium environment (RubiksCube222)
  2. How to define a custom reward strategy to guide faster convergence
  3. How to wrap an SB3 policy (PPO) with WrPolicySB32MLPro
  4. How to configure an RLScenario and run an RLTraining
"""

from pathlib import Path

from mlpro.bf.ml import Model
from mlpro.bf.various import Log
from mlpro.rl import RLScenario, RLTraining
from mlpro.rl.models import *

from mlpro_int_sb3.wrappers import WrPolicySB32MLPro
from mlpro_int_gymnasium.envs.rubikscube2x2x2 import RubiksCube222

from stable_baselines3 import PPO
import rubiks_cube_gym      
import os                     



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1  RL scenario
class RubiksPPOScenario (RLScenario):

    C_NAME = 'RubiksCube222_Scenario'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:

        # 1.1 Instantiate the MLPro environment.
        #     p_shaped_reward=True  activates the built-in ShapedRewardCubeWrapper
        #                           (dense milestone reward + curriculum learning).
        #     p_shaped_reward=False uses the default sparse reward (+100/-1).
        self._env = RubiksCube222(
            p_shaped_reward=True,
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        # 1.2 Create the SB3 PPO policy.
        #     A plain gym env is passed so that SB3 can infer observation and action
        #     spaces internally. WrPolicySB32MLPro handles all actual env interaction.
        policy_sb3 = PPO(
            "MlpPolicy", 
            n_steps=2048,
            learning_rate= 0.0001,
            gamma=0.99,
            batch_size=64,
            ent_coef=0.2, 
            env=None,  
            _init_setup_model=False,    
            device="cpu"    
        )
        # 1.3 Wrap the SB3 policy for MLPro
        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy    = policy_sb3,
            p_cycle_limit   = self._cycle_limit,
            p_observation_space = self._env.get_state_space(),
            p_action_space      = self._env.get_action_space(),
            p_ada           = p_ada,
            p_visualize     = p_visualize,
            p_logging       = p_logging
        )
        # 1.4 Create and return the agent
        return Agent( p_policy=policy_wrapped,     
                      p_envmodel=None,
                      p_name='PPO_Agent',
                      p_ada=p_ada,
                      p_visualize=p_visualize,
                      p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 2  Parameterisation and training
if __name__ == '__main__':
    # 2.1 Parameters for interactive demo
    cycle_limit = 100
    cycle_limit2 = 50
    logging     = Log.C_LOG_ALL
    visualize   = True
    path        = str(Path.home())

else:
    # 2.2 Parameters for internal unit test
    cycle_limit = 10
    cycle_limit2 = 5
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None


# 2.3 Instantiate the training object and run
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

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 3 Load the trained scenario for evaluation

## -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    input('\n\nTraining finished. Press ENTER to apply the trained agent...\n')

    ## -------------------------------------------------------------------------------------------------
    # 3.1 Load the trained scenario
    filename_scenario = training.get_scenario().get_filename()
    scenario = RubiksPPOScenario.load( p_path = training.get_training_path() + os.sep + 'scenario', 
                                       p_filename = filename_scenario )

    ## -------------------------------------------------------------------------------------------------
    # 3.2 Reset Scenario
    scenario.reset()  
    scenario.get_model().switch_adaptivity(False)

    ## -------------------------------------------------------------------------------------------------
    # 3.3 Run Scenario (using cycles for re-run/evaluation)
    print("\n--- Running Final Evaluation ---")
    scenario.set_cycle_limit(cycle_limit2)
    scenario.run()


if __name__ != '__main__':
    from shutil import rmtree
    rmtree(training.get_training_path())
else:
    input( '\nPress ENTER to finish...')
