import os
import numpy as np
import time
import pickle
import random
import logging
import imageio
import argparse
from gymnasium import spaces
import gymnasium as gym
import Agents.utils.help_classes as hc
import Agents.utils.stats_functions as sf
from Agents.utils.actions import MORE_ACTIONS
from Agents.Tapas_en_Mallorca.old.Agent import Combined_Agent
from Agents.Prio_n_step.Prio_DQN_Agent import Prio_DQN_Agent
from Agents.Combined_Agent_Double.Dueling_DDQN_Agent import Dueling_DDQN_Agent as Previous_Combined_Agent
from Agents.Pablo.Adaptative_Dueling_Double_DQN.Agent import Adaptative_Dueling_Double_DQN
from Agents.Tapas_en_Mallorca.Adaptative_Dueling_Double_DQN.Agent import Adaptative_Dueling_Double_DQN_better_mem as Adaptive_Combined_Agent

def test(super_episode):
    print("Starting test:")

    test_rewards = []
    SEEDS_TEST = [291 + 563*i for i in range(100)]

    for episode in range(len(SEEDS_TEST)):

        state = env.reset(seed = SEEDS_TEST[episode])
        env.action_space.seed(SEEDS_TEST[episode])
        state = state[0] if isinstance(state, tuple) else state  # Handle Gymnasium compatibility
        total_reward = 0
        step = 0

        frames = []
        
        for t in range(max_steps):

            done = truncated = False

            """frame = env.render()
            if frame is not None:
                frames.append(frame)"""

            a1 = a1_cont = agent.act(state)
            
            next_state, reward, done, truncated, _ = env.step(a1_cont)
            
            total_reward += reward

            state = next_state
        
            if done or truncated:
                break

        test_rewards.append(total_reward)
        print(f"Test Episode {episode+1}, Total Reward: {total_reward}")
        
        if frames:
            os.makedirs(f"{env_name}/test_gifs", exist_ok=True)
            imageio.mimsave(f"{env_name}/test_gifs/test_episode_{episode}.gif", frames, fps=30)

    env.close()
    results_name = f"test_results_episode_{super_episode+1}"
    sf.save_test_results(env_name, test_rewards, name = results_name)
    print(f"Tests reults: {np.average(test_rewards)} avg.")

initalization_time = time.time()        # Debugging
parser = argparse.ArgumentParser(description = "Train Dueling DDQN Agent.")
parser.add_argument("--use_dueling", action="store_true", help = "Use Dueling Network")
parser.add_argument("--use_double", action="store_true", help = "Use Double DQN")
parser.add_argument("--use_eps_decay", action="store_true", help = "Use Epsilon Decay")
parser.add_argument("--use_noisy_net", action="store_true", help = "Use Noisy Net")
parser.add_argument("--use_prio",action="store_true", help = "Use Prioritized Buffuring Replay")
parser.add_argument("--n_step", type = int, default = 4, help = "Number of steps to look ahead")
parser.add_argument("--env_description", type = str, default = "", help = "Additional description for env_name")
parser.add_argument("--seed", type = int, default = 7489, help = "Seed for the training")
parser.add_argument("--use_more_actions", type = str, default = "False", help = "Use more actions")
parser.add_argument("--max_episodes", type = int, default = 4000, help = "Max number of episodes")
parser.add_argument("--train_iterations", type = int, default = 32, help = "Number of training iterations")
parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
parser.add_argument("--agent", type = str, default = "Adaptive_Combined", help = "Agent to use", choices = ["Combined", "Adaptive", "Previous_Combined_Agent", "Prio_DQN", "Adaptive_Combined", "adaptive", "Adaptative"])
args = parser.parse_args()

use_dueling = args.use_dueling
use_double = args.use_double
use_eps_decay = args.use_eps_decay
use_prio = args.use_prio
use_noisy = args.use_noisy_net

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
seed = SEED_TRAIN_1

USE_MORE_ACTIONS = False

random.seed(seed)
if args.verbose == "True":
    logging.basicConfig(level=logging.DEBUG)
else:    
    logging.basicConfig(level=logging.INFO)

name_parts = []
if use_noisy:
    name_parts.append("Noisy")
if use_dueling:
    name_parts.append("Dueling")
if use_double:
    name_parts.append("Double")
name_parts.append("DQN")
if use_eps_decay:
    name_parts.append("eps_decay")
name = "_".join(name_parts)

env_name = f"../last_chance/Cartpole/{name}"
env = gym.make("CartPole-v1", render_mode = "rgb_array")
if isinstance(env.action_space, spaces.Box):
    env = hc.DiscreteActionWrapper(env,5)
logging.info(env_name)

state_space = env.observation_space
action_space = env.action_space

if args.agent == "Combined":
    agent = Combined_Agent(
    state_space,
    action_space,
    env = env,
    seed = seed,
    use_eps_decay = use_eps_decay,
    use_dueling = use_dueling,
    use_double = use_double,
    use_noisy = use_noisy,
    use_prio = use_prio,
    n_step = args.n_step,
    hidden_sizes = [256, 256]
)

if args.agent == "Adaptive" or args.agent == "adaptive" or args.agent == "Adaptative":
    agent = Adaptative_Dueling_Double_DQN(
        state_space,
        action_space,
        env = env,
        seed = seed,
        use_eps_decay = use_eps_decay,
        use_dueling = use_dueling,
        use_double = use_double,
        use_noisy = use_noisy,
        hidden_sizes = [128, 128],
        use_more_actions = False,
        update_target_every = 20,
        learning_rate = 0.001
    )

if args.agent == "Adaptive_Combined":
    agent = Adaptive_Combined_Agent(
        state_space,
        action_space,
        env = env,
        seed = seed,
        use_eps_decay = use_eps_decay,
        use_dueling = use_dueling,
        use_double = use_double,
        use_noisy = use_noisy,
        use_prio = use_prio,
        n_step = args.n_step,
        hidden_sizes = [256, 256]
    )

if args.agent == "Previous_Combined_Agent":
    agent = Previous_Combined_Agent(
        state_space,
        action_space,
        env = env,
        seed = seed,
        use_eps_decay = use_eps_decay,
        use_dueling = use_dueling,
        use_double = use_double,
        use_noisy = use_noisy,
        use_prio = use_prio,
        n_step = args.n_step,
        hidden_sizes = [256, 256]
    )
if args.agent == "Prio_DQN":
    agent = Prio_DQN_Agent(
        state_space,
        action_space,
        seed = seed,
        eps = 0.01,
        learning_rate = 0.0001,
        hidden_sizes = [256, 256],
        n_steps = 4,
        env = env,
        use_more_actions = USE_MORE_ACTIONS,
    )

stats = []
losses = []
epsilons = []

frame_idx = 0

max_episodes = args.max_episodes # 10000 default
max_steps = 1000
train_iterations = args.train_iterations # 32 default

logging.info(f"Initialization time: {time.time()-initalization_time}")        # Debugging

time_start = time.time()        # Debugging
last_save_time = time.time()

for episode in range(max_episodes):

    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state  # Handle Gymnasium compatibility
    total_reward = 0
    step = 0
    
    for t in range(max_steps):

        step += 1
        done = truncated = False

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

    print(f"Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}")
        
    if ((episode) % int(max_episodes/10) == 0) and episode > 0:
        agent.Q.save(env_name, name = f"episode_{episode}")
        sf.plot_losses(losses, env_name)
        sf.plot_returns(stats, env_name)
        test(episode)

agent.Q.save(env_name, name = "training_finished")
sf.save_stats(env_name, stats, losses)
sf.plot_losses(losses, env_name)
sf.plot_returns(stats, env_name)

test(episode)