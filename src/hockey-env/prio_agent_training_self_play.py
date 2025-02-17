from importlib import reload
from gymnasium import spaces
import numpy as np
import logging
import random
import time
import copy
import os
from Prio_n_step_Agent.utils.actions import MORE_ACTIONS
from Prio_n_step_Agent.Prio_DQN_Agent import Prio_DQN_Agent
from Prio_n_step_Agent.utils.random_agent import RandomAgent
import Prio_n_step_Agent.utils.stats_functions as sf
from Combined_Agent_Double.Dueling_DDQN_Agent import (
    Dueling_DDQN_Agent as Combined_Agent,
)
import hockey.hockey_env as h_env

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
seed = SEED_TRAIN_1

USE_MORE_ACTIONS = True

random.seed(seed)

reload(h_env)
env_name = "pure_prio_training_2_2_25"

env = h_env.HockeyEnv()
logging.basicConfig(level=logging.INFO)

logging.info("Running Urban Planning")
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
)
# agent.Q.load("more_actions_test_1_150k_30k_(256_256)")
agent_copy = copy.deepcopy(agent)

opponent0 = RandomAgent(seed=seed)
opponent1 = h_env.BasicOpponent()
opponent2 = h_env.BasicOpponent(weak=False)
opponent3 = Prio_DQN_Agent(
    state_space,
    action_space,
    seed=seed,
    eps=0.01,
    learning_rate=0.0001,
    hidden_sizes=[256, 256],
    n_steps=4,
    env=env,
    use_more_actions=USE_MORE_ACTIONS,
)

opponent3.Q.load("pure_prio_training_2_2_25", name="episode_5000")

opponent4 = Prio_DQN_Agent(
    state_space,
    action_space,
    seed=seed,
    eps=0.01,
    learning_rate=0.0001,
    hidden_sizes=[256, 256],
    n_steps=4,
    env=env,
    use_more_actions=USE_MORE_ACTIONS,
)

opponent4.Q.load("pure_prio_training_2_2_25", name="episode_7500")
opponent5 = Combined_Agent(
    state_space,
    action_space,
    seed=seed,
    eps=0.01,
    learning_rate=0.0001,
    hidden_sizes=[256, 256],
    n_steps=4,
    env=env,
    use_more_actions=USE_MORE_ACTIONS,
)
opponent5.Q.load("combined_training_6_2_25", name="episode_5000")
opponent6 = Combined_Agent(
    state_space,
    action_space,
    seed=seed,
    eps=0.01,
    learning_rate=0.0001,
    hidden_sizes=[256, 256],
    n_steps=4,
    env=env,
    use_more_actions=USE_MORE_ACTIONS,
)
opponent6.Q.load("combined_training_6_2_25", name="episode_7500")

opponents = [
    opponent0,
    opponent1,
    opponent2,
    opponent3,
    opponent5,
    opponent6,
    agent_copy,
]
opponents_names = [
    "Random",
    "Weak",
    "NonWeak",
    "Prio Agent_5000",
    "Prio_Agent_7500",
    "Combined Agent_5000",
    "Combined Agent_7500",
    "self_play",
]

match_history = [[] for _ in opponents]

stats = []
losses = []
betas = []
epsilons = []

frame_idx = 0

max_episodes = 50000
games_to_play = 50
max_steps = 30000

train_iterations = 32  # Number of training steps per episode

beta_start = agent.beta  # Beta annealing parameters
beta_frames = max_episodes * 1000  # Increase this to make annealing slower

time_start = time.time()  # Debugging

for episode in range(max_episodes):
    saved_weights = []
    if len(saved_weights) > 0:
        selected = random.randint(0, len(opponents) - 2)

    else:
        selected = random.randint(0, len(opponents) - 1)

    opponent = opponents[selected]

    if opponents_names[selected] == "self_play":
        weights = random.choice(saved_weights)
        opponent.Q.load(env_name, name=weights)
    for game in range(games_to_play):

        state, _ = env.reset(seed=seed)
        obs_agent2 = env.obs_agent_two()
        total_reward = 0

        for t in range(max_steps):

            frame_idx += 1

            agent.beta = min(
                1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames
            )  # Smoother beta annealing

            done = False

            a1 = agent.get_step(state)

            if opponents_names in ["Random" "Weak", "NonWeak"]:
                a2 = opponent.act(obs_agent2)
            else:
                a2 = opponent.get_step(obs_agent2)

            full_action = np.hstack([a1, a2])

            start_time = time.time()  # Debbuging

            next_state, reward, done, truncated, info = env.step(full_action)

            logging.debug(f" Env time: {time.time()- start_time}")  # Debugging

            total_reward += reward

            if agent.use_n_step:
                one_step_transition = agent.n_buffer.add_transition(
                    (state, a1, reward, next_state, done)
                )
            else:
                one_step_transition = (state, a1, reward, next_state, done)

            if one_step_transition != ():
                agent.buffer.store(one_step_transition)

            state = next_state
            obs_agent2 = env.obs_agent_two()

            if done or truncated:
                break

        loss = agent.train(train_iterations)
        match_history[selected].append(info["winner"])
        logging.debug(info["winner"])

        if game % int(games_to_play / 2) == 0:
            losses.extend(loss)
            stats.append([episode, total_reward, t + 1])
            betas.append(agent.beta)
            epsilons.append(agent._eps)
            logging.info(
                f"Episode {episode+1}/{max_episodes}, Game {game+1}/{games_to_play} - Total Reward: {total_reward}, Beta: {agent.beta}"
            )

    if agent._config["use_eps_decay"] and episode > int(0.8 * max_episodes):
        agent._perform_epsilon_decay()

    if ((episode) % int(max_episodes / 20) == 0) and episode > 0:
        agent.Q.save(env_name, name=f"episode_{episode}")
        saved_weights.append(f"episode_{episode}")
        sf.save_betas(env_name, betas)
        sf.save_epsilons(env_name, epsilons)
        sf.save_stats(env_name, stats, losses)
        sf.save_match_history(env_name, match_history)
        sf.plot_returns(stats, env_name)
        sf.plot_losses(losses, env_name)
        sf.plot_beta_evolution(env_name, betas)
        sf.plot_epsilon_evolution(env_name, epsilons)
        sf.plot_match_evolution_by_chunks(
            env_name, match_history, opponents_names, games_to_play
        )

    logging.debug(f" time per frame: {(time.time()-time_start)/frame_idx}")
    logging.debug(f" mean sample time: {np.mean(agent.sample_times)}")

agent.Q.save(env_name, name="training_finished")
sf.save_betas(env_name, betas)
sf.save_epsilons(env_name, epsilons)
sf.save_stats(env_name, stats, losses)
sf.save_match_history(env_name, match_history)
sf.plot_returns(stats, env_name)
sf.plot_losses(losses, env_name)
sf.plot_beta_evolution(env_name, betas)
sf.plot_epsilon_evolution(env_name, epsilons)
sf.plot_match_evolution_by_chunks(
    env_name, match_history, opponents_names, games_to_play
)
