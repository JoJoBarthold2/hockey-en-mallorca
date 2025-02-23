import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import pickle
import imageio
import os
import random
import copy

#from Combined_Agent_Double.Dueling_DDQN_Agent import Dueling_DDQN_Agent
#from Combined_Agent_Double.utils.random_agent import RandomAgent
#from Combined_Agent_Double.utils.actions import MORE_ACTIONS
#import Combined_Agent_Double.utils.stats_functions as sf

from Agents.Prio_n_step.Prio_DQN_Agent import Prio_DQN_Agent
from Agents.Pablo.Adaptative_Dueling_Double_DQN.Agent import Adaptative_Dueling_Double_DQN
from Agents.Random.random_agent import RandomAgent
from Agents.utils.actions import MORE_ACTIONS
import Agents.utils.stats_functions as sf
import hockey.hockey_env as h_env

from Agents.Pablo.Agent import Dueling_DDQN_Agent
from Agents.Tapas_en_Mallorca.old.Agent import Combined_Agent
from Agents.Tapas_en_Mallorca.Adaptative_Dueling_Double_DQN.Agent import Adaptative_Dueling_Double_DQN_better_mem

from importlib import reload

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
SEEDS_TEST = [291 + i for i in range(10)]

seed = SEED_TRAIN_1

USE_MORE_ACTIONS = True

reload(h_env)
env_name = "../remake/Noisy_Dueling_Double_DQN_n_step_4_only_basic_remake_1/"
env = h_env.HockeyEnv()
h_env.HockeyEnv().seed(seed)
#env = gym.make("CartPole-v1", render_mode = "rgb_array")

state_space = env.observation_space

if(USE_MORE_ACTIONS):
    action_space = spaces.Discrete(len(MORE_ACTIONS))
else: 
    action_space = env.discrete_action_space

#agent = Adaptative_Dueling_Double_DQN_better_mem(state_space, action_space, seed = seed, eps = 0.01, learning_rate = 0.0001, hidden_sizes = [256, 256], n_steps = 5, env = env, use_more_actions = USE_MORE_ACTIONS)
agent = Adaptative_Dueling_Double_DQN(state_space, action_space, seed = seed,use_dueling = True, use_double = True, use_noisy = True ,eps = 0.01, learning_rate = 0.0001, hidden_sizes = [256, 256], env = env, use_more_actions = USE_MORE_ACTIONS)
#agent = Prio_DQN_Agent(state_space, action_space, seed = seed, eps = 0.01, learning_rate = 0.0001, hidden_sizes = [256, 256], n_steps = 5, env = env, use_more_actions = USE_MORE_ACTIONS)
agent.Q.load(env_name, name = "most_recent")
agent.Q_target.load(env_name, name = "most_recent")

max_episodes = 100000
opponents = [h_env.BasicOpponent(), h_env.BasicOpponent(weak=False)]

for episode in range(max_episodes):
    
    opponent = random.choice(opponents) 

    for game in range(50):

        state, _ = env.reset(seed = seed)
        env.action_space.seed(seed)
        obs_agent2 = env.obs_agent_two()
        total_reward = 0

        t = 0
        while True:
            
            done = False

            a1 = agent.act(state, eps = 0, validation = True)
            if(agent.use_more_actions):
                a1_cont = MORE_ACTIONS[a1]
            else: 
                a1_cont = env.discrete_to_continous_action(a1)

            a2 = opponent.act(obs_agent2)
            
            full_action = np.hstack([a1_cont, a2])

            next_state, reward, done, truncated, info = env.step(full_action)
            
            total_reward += reward

            one_step_transition = (state, a1, reward, next_state, done)
            if one_step_transition != ():
                agent.buffer.store(one_step_transition)

            state = next_state
            obs_agent2 = env.obs_agent_two()

            if done or truncated: break

        t += 1

    print(f"{agent.buffer.size}/{agent.buffer.max_size}")
    if agent.buffer.size >= agent.buffer.max_size:
        print("buffer lleno")
        with open(f"{env_name}/weights/replay_buffer.pkl", "wb") as f:
            pickle.dump(agent.buffer, f)
        print("Buffer guardado correctamente.")
        break

    #logging.debug(f" time per frame: {(time.time()-time_start)/frame_idx}")