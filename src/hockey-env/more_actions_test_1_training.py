from importlib import reload
from gymnasium import spaces
import numpy as np
import logging
import time

from Combined_Agent_Double.utils.actions import MORE_ACTIONS
from Combined_Agent_Double.Dueling_DDQN_Agent import Dueling_DDQN_Agent
from Combined_Agent_Double.utils.random_agent import RandomAgent
import Combined_Agent_Double.utils.stats_functions as sf
import hockey.hockey_env as h_env

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
seed = SEED_TRAIN_1

USE_MORE_ACTIONS = True

reload(h_env)
env_name = "more_actions_test_1_150k_30k_(256_256)"
env = h_env.HockeyEnv()

state_space = env.observation_space

if(USE_MORE_ACTIONS):
    action_space = spaces.Discrete(len(MORE_ACTIONS))
else: 
    action_space = env.discrete_action_space

agent = Dueling_DDQN_Agent(state_space, action_space, seed = seed, use_eps_decay = True, eps_decay = 0.9995, hidden_sizes = [256, 256], n_steps = 1)

opponent0 = RandomAgent(seed = seed)
opponent1 = h_env.BasicOpponent()
opponent2 = h_env.BasicOpponent(weak = False)
opponent3 = agent

opponents = [opponent0, opponent1, opponent2, opponent3]

stats = []
losses = []
betas = []
epsilons = []

frame_idx = 0

max_episodes = 150000
max_steps = 30000

train_iterations = 32  # Number of training steps per episode

beta_start = agent.beta     # Beta annealing parameters
beta_frames = max_episodes * 10000       # Increase this to make annealing slower

time_start = time.time()        # Debugging

opponent = opponent1

for episode in range(max_episodes):

    state, _ = env.reset(seed = seed)
    obs_agent2 = env.obs_agent_two()
    total_reward = 0

    for t in range(max_steps):

        frame_idx += 1
        
        agent.beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)        # Smoother beta annealing

        done = False

        a1 = agent.perform_greedy_action(state)
        a2 = opponent.act(obs_agent2)
        
        if(USE_MORE_ACTIONS):
            a1_cont = MORE_ACTIONS[a1]
        else: 
            a1_cont = env.discrete_to_continous_action(a1)

        full_action = np.hstack([a1_cont, a2])

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

    loss = agent.train(train_iterations)

    if episode % 10 == 0:    
        losses.extend(loss)
        stats.append([episode, total_reward, t + 1])
        betas.append(agent.beta)
        epsilons.append(agent._eps)
        print(f"Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}, Beta: {agent.beta}")
    
    if agent._config["use_eps_decay"] and episode > int(0.8 * max_episodes):
        agent._perform_epsilon_decay()  
        
    if ((episode) % int(max_episodes/20) == 0) and episode > 0:
        agent.Q.save(env_name, name = f"episode_{episode}")
        sf.save_betas(env_name, betas)
        sf.save_epsilons(env_name, epsilons)
        sf.save_stats(env_name, stats, losses)
        sf.plot_returns(stats, env_name)
        sf.plot_losses(losses, env_name)
        sf.plot_beta_evolution(env_name, betas)
        sf.plot_epsilon_evolution(env_name, epsilons)

    logging.debug(f" time per frame: {(time.time()-time_start)/frame_idx}")
    logging.debug(f" mean sample time: {np.mean(agent.sample_times)}")

agent.Q.save(env_name, name = "training_finished")
sf.save_betas(env_name, betas)
sf.save_epsilons(env_name, epsilons)
sf.save_stats(env_name, stats, losses)
sf.plot_returns(stats, env_name)
sf.plot_losses(losses, env_name)
sf.plot_beta_evolution(env_name, betas)
sf.plot_epsilon_evolution(env_name, epsilons)
