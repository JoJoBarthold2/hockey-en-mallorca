import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random
import argparse
import logging

from Agents.Pablo.Adaptative_Dueling_Double_DQN.Agent import Adaptative_Dueling_Double_DQN
import Agents.utils.help_classes as hc
import Agents.utils.stats_functions as sf
from Agents.utils.actions import MORE_ACTIONS

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description = "Train Adaptative Dueling DDQN Agent.")
parser.add_argument("--use_dueling", type = str, default = "True", help = "Use Dueling Network")
parser.add_argument("--use_double", type = str, default = "True", help = "Use Double DQN")
parser.add_argument("--use_noisy", type = str, default = "False", help = "Use Noisy Linear layers")
parser.add_argument("--use_eps_decay", type = str, default = "False", help = "Use Epsilon Decay")
parser.add_argument("--env_description", type = str, default = "", help = "Additional description for env_name")
args = parser.parse_args()

use_dueling = True if args.use_dueling == "True" else False
use_double = True if args.use_double == "True" else False
use_noisy = True if args.use_noisy == "True" else False
use_eps_decay = True if args.use_eps_decay == "True" else False

name_parts = []
if use_noisy:
    name_parts.append("Noisy")
if use_dueling:
    name_parts.append("Dueling")
if use_double:
    name_parts.append("Double")
name_parts.append("DQN")
name = "_".join(name_parts)

env_name = f"../remake/{name}_{args.env_description}" if args.env_description != "" else f"../remake/{name}"
logging.info(env_name)

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
SEEDS_TEST = [291, 292, 293, 294, 295]

seed = SEED_TRAIN_1

USE_MORE_ACTIONS = True
USING_HOCKEY = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Ensure deterministic behavior in CUDA if available
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

#env_name = "CartPole-v1"
env = gym.make("CartPole-v1", render_mode = "rgb_array")
if isinstance(env.action_space, spaces.Box):
    env = hc.DiscreteActionWrapper(env,5)

state_space = env.observation_space
action_space = env.action_space
agent = Adaptative_Dueling_Double_DQN(
    state_space,
    action_space,
    seed = seed,
    update_target_every = 10,
    batch_size = 64,
    use_eps_decay = use_eps_decay,
    use_dueling = use_dueling,
    use_double = use_double,
    use_noisy = use_noisy,
    learning_rate = 0.0002
)

stats = []
losses = []

max_episodes = 2000
max_steps = 3000
train_iterations = 32  # Number of training steps per episode

for episode in range(max_episodes):

    state = env.reset(seed = seed)
    env.action_space.seed(seed)
    state = state[0] if isinstance(state, tuple) else state  # Handle Gymnasium compatibility
    total_reward = 0

    for t in range(max_steps):

        done = truncated = False

        if USING_HOCKEY:
            a1 = agent.act(state)
            if USE_MORE_ACTIONS:
                a1_cont = MORE_ACTIONS[a1]
            else:
                a1_cont = env.discrete_to_continous_action(a1)
        else:
            a1 = a1_cont = agent.act(state)

        next_state, reward, done, truncated, _ = env.step(a1_cont)
        
        total_reward += reward

        agent.buffer.store((state, a1, reward, next_state, done))

        state = next_state
    
        if done or truncated:
            break

    losses.extend(agent.train(train_iterations))
    stats.append([episode, total_reward, t + 1])
    
    if agent._config["use_eps_decay"] and episode > int(0.5 * max_episodes):
        agent._perform_epsilon_decay()

    logging.info(f"Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}")
        
    if ((episode) % int(max_episodes/10) == 0) and episode > 0:
        agent.Q.save(env_name, name = f"episode_{episode}")

agent.Q.save(env_name, name = "training_finished")
sf.save_stats(env_name, stats, losses)
