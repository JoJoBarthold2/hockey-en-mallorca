from importlib import reload
from gymnasium import spaces

import logging
from Agents.Prio_n_step.Prio_DQN_Agent import Prio_DQN_Agent
from Agents.utils.actions import MORE_ACTIONS
from Agents.Combined_Agent_Double.Dueling_DDQN_Agent import (
    Dueling_DDQN_Agent as Combined_Agent,
)
import hockey.hockey_env as h_env
from training import train_agent_self_play

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
seed = SEED_TRAIN_1

USE_MORE_ACTIONS = False

reload(h_env)
    

env = h_env.HockeyEnv()
logging.basicConfig(level=logging.INFO)


state_space = env.observation_space

if USE_MORE_ACTIONS:
        action_space = spaces.Discrete(len(MORE_ACTIONS))
else:
        action_space = env.discrete_action_space

agent = Prio_DQN_Agent(
        state_space,
        action_space,
        seed=seed,
        eps=0.01,
        learning_rate=0.0001,
        hidden_sizes=[256, 256],
        n_steps=4,
        use_prio = True,
        #alpha = 0.2,
        #beta=0.4
    )
  
env_name = "prio_agent_24_02_25_n_step_4_normal_actions"

train_agent_self_play(agent = agent, use_more_actions = USE_MORE_ACTIONS, seed = SEED_TRAIN_1, env=env, env_name = env_name)