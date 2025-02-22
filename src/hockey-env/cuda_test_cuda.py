import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random
import time

from Agents.Pablo.DQN_CUDA.Agent import DQN
import Agents.utils.help_classes as hc
import Agents.utils.stats_functions as sf
from Agents.utils.actions import MORE_ACTIONS

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
    torch.backends.cudnn.benchmark = True
    torch.cuda.synchronize()
    torch.set_num_threads(1)

#env_name = "CartPole-v1"
env_name = "../remake/DQN_CartPole"
env = gym.make("CartPole-v1", render_mode = "rgb_array")
if isinstance(env.action_space, spaces.Box):
    env = hc.DiscreteActionWrapper(env,5)

state_space = env.observation_space
action_space = env.action_space
agent = DQN(state_space, action_space, seed = seed, update_target_every = 5, batch_size = 64)

stats = []
losses = []

max_episodes = 250
max_steps = 1000
train_iterations = 32  # Number of training steps per episode

start_time = time.time()
for episode in range(max_episodes):

    state = torch.tensor(env.reset(seed = seed)[0], dtype = torch.float32, device = agent.device)

    #print("MAIN LOOP - Initial State device:", state.device)  # Should be cuda:0

    env.action_space.seed(seed)
    state = state[0] if isinstance(state, tuple) else state  # Handle Gymnasium compatibility
    total_reward = 0
    step = 0
    
    for t in range(max_steps):

        step += 1
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

        next_state = torch.tensor(next_state, dtype=torch.float32, device=agent.device)

        total_reward += reward

        #print("MAIN LOOP - Next State device before storing:", type(next_state))

        agent.buffer.store((state, a1, torch.tensor(reward, dtype = torch.float32, device = agent.device), next_state, done))

        #print("MAIN LOOP - Next State device after conversion:", next_state.device)

        state = next_state
    
        if done or truncated:
            break

    losses.extend(agent.train(train_iterations))
    stats.append([episode, total_reward, t + 1])
    
    if agent._config["use_eps_decay"] and episode > int(0.5 * max_episodes):
        agent._perform_epsilon_decay()

    #print(f"Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}")
        
    if ((episode) % int(max_episodes/10) == 0) and episode > 0:
        agent.Q.save(env_name, name = f"episode_{episode}")

agent.Q.save(env_name, name = "training_finished")
sf.save_stats(env_name, stats, losses)


end_time = time.time()
print(f"Training Time for DQN with CUDA: {end_time - start_time:.2f} seconds")
