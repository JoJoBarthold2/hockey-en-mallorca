from importlib import reload
import numpy as np
import logging
import time

from Combined_Agent.Dueling_DQN_Agent import Dueling_DQN_Agent
from Combined_Agent.utils.random_agent import RandomAgent
import Combined_Agent.utils.stats_functions as sf
import hockey.hockey_env as h_env

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
seed = SEED_TRAIN_1

reload(h_env)
env_name = "Combined_test_4_DuelingDDQN_50k_30k_(128,128)"
env = h_env.HockeyEnv()

state_space = env.observation_space
action_space = env.discrete_action_space

agent = Dueling_DQN_Agent(state_space, action_space, seed = seed, use_eps_decay = True, hidden_sizes = [128, 128])

opponent0 = RandomAgent(seed = seed)
opponent1 = h_env.BasicOpponent()
opponent2 = h_env.BasicOpponent(weak = False)
opponent3 = agent

opponents = [opponent0, opponent1, opponent2, opponent3]

match_history = np.full((len(opponents), 20), -1)

iterations_to_train_against_random = 10000

def add_match_result(agent_index, result):
    global match_history
    match_history[agent_index] = np.roll(match_history[agent_index], -1)
    match_history[agent_index, -1] = result

stats = []
losses = []
betas = []
epsilons = []
winrates = np.empty((0, 4))

frame_idx = 0

max_episodes = 50000
iterations_to_train_against_random = max_episodes/5     # Better if we just try with 1000 probably
max_steps = 30000

train_iterations = 32  # Number of training steps per episode

beta_start = agent.beta     # Beta annealing parameters
beta_frames = max_episodes * 10000       # Increase this to make annealing slower

time_start = time.time()        # Debugging

for episode in range(max_episodes):

    state, _ = env.reset(seed = seed)

    obs_agent2 = env.obs_agent_two()
    #obs_agent2 = obs_agent2[0] if isinstance(obs_agent2, tuple) else obs_agent2

    total_reward = 0

    if episode < iterations_to_train_against_random:
        opponent = opponent0
    else:
        lowest_winrate_opponent = np.argmin(np.sum(match_history, axis=1)) 
        opponent = opponents[lowest_winrate_opponent]

    for t in range(max_steps):

        frame_idx += 1
        
        agent.beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)        # Smoother beta annealing

        done = False

        a1 = agent.perform_greedy_action(state)
        a2 = opponent.act(obs_agent2)
        full_action = np.hstack([env.discrete_to_continous_action(a1), a2])

        start_time = time.time()        # Debbuging

        next_state, reward, done, truncated, info = env.step(full_action)
        
        logging.debug(f" Env time: {time.time()- start_time}")      # Debugging

        total_reward += reward

        if agent.use_n_step:
            one_step_transition = agent.n_buffer.add_transition((state, a1, reward, next_state, done))
        else:
            one_step_transition = (state, a1, reward, next_state, done)

        if one_step_transition != ():
            agent.buffer.store(one_step_transition)

        state = next_state
        obs_agent2 = env.obs_agent_two()

        if done or truncated: break

    if episode >= iterations_to_train_against_random:
        add_match_result(lowest_winrate_opponent, info["winner"])

    loss = agent.train(train_iterations)

    if episode % 10 == 0:    
        losses.extend(loss)
        stats.append([episode, total_reward, t + 1])
        betas.append(agent.beta)
        epsilons.append(agent._eps)
        winrates = np.vstack([winrates, np.sum(match_history, axis=1)])
        print(f"Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}, Beta: {agent.beta}")
    
    if agent._config["use_eps_decay"] and episode > int(0.5 * max_episodes):
        agent._perform_epsilon_decay()  
        
    if ((episode) % int(max_episodes/50) == 0) and episode > 0:
        agent.Q.save(env_name, name = f"episode_{episode}")
        sf.save_betas(env_name, betas)
        sf.save_epsilons(env_name, epsilons)
        sf.save_stats(env_name, stats, losses)
        sf.save_winrates(env_name, winrates)
        sf.plot_returns(stats, env_name)
        sf.plot_losses(losses, env_name)

    logging.debug(f" time per frame: {(time.time()-time_start)/frame_idx}")
    logging.debug(f" mean sample time: {np.mean(agent.sample_times)}")

agent.Q.save(env_name, name = "training_finished")
sf.save_betas(env_name, betas)
sf.save_epsilons(env_name, epsilons)
sf.save_stats(env_name, stats, losses)
sf.save_winrates(env_name, winrates)
