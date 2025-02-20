from importlib import reload
import numpy as np

from Dueling_DQN_Agent.agentic_dqn import DQN_AGENT_priotized_buffer
import Dueling_DQN_Agent.utils.stats_functions as sf
import hockey.hockey_env as h_env

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
seed = SEED_TRAIN_1

reload(h_env)
env_name = "Hockey_100000_30000_(256_256_128)_2_vs_nonweak_agent"
env = h_env.HockeyEnv()

# Initialize the agent with the correct state/action space
state_space = env.observation_space
action_space = env.discrete_action_space

agent = DQN_AGENT_priotized_buffer(state_space, action_space, seed = seed, use_eps_decay = True, hidden_sizes = [256, 256, 128])
#agent.Q.load("Hockey_100000_30000_(256_256_128)_2") 
opponent = h_env.BasicOpponent(weak = False)

stats = []
losses = []


train_iterations = 32  # Number of training steps per episode

max_steps = 30000

ac_space = env.action_space
o_space = env.observation_space
print(ac_space)
print(o_space)
print(list(zip(env.observation_space.low, env.observation_space.high)))
alpha = 0.2
beta = 0.4

config = {
            "eps": 0.05,  # Epsilon in epsilon greedy policies
            "discount": 0.99,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0001,
            "update_target_every": 128,
            "use_target_net": True,
            "prioritized_replay_eps": 1e-6,
        }

use_target = True
target_update = 20
q_agent = agent(o_space, ac_space, discount=0.95, eps=0.2, 
                   use_target_net=use_target, update_target_every= target_update, alpha =  alpha, beta = beta, config = config, n_steps=8)

agent.Q.save(env_name, name = "training_finished")
sf.save_stats(env_name, stats, losses)
