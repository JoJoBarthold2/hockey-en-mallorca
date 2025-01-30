from importlib import reload
import numpy as np
import logging
import time

from Combined_Agent.Dueling_DQN_Agent import Dueling_DQN_Agent
import Combined_Agent.utils.stats_functions as sf
import hockey.hockey_env as h_env

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
seed = SEED_TRAIN_1

reload(h_env)
env_name = "Combined_test_1_50k_30k_(128,128)"
env = h_env.HockeyEnv()

# Initialize the agent with the correct state/action space
state_space = env.observation_space
action_space = env.discrete_action_space

agent = Dueling_DQN_Agent(state_space, action_space, seed = seed, use_eps_decay = True, hidden_sizes = [128, 128]) 
opponent = h_env.BasicOpponent()

stats = []
losses = []

frame_idx = 0

max_episodes = 50000
max_steps = 30000

train_iterations = 32  # Number of training steps per episode

beta_start = agent.beta     # Beta annealing parameters
beta_frames = max_episodes * 700        # Increase this to make annealing slower

time_start = time.time()        # Debugging

for episode in range(max_episodes):

    state, _ = env.reset(seed = seed)
    #state = state[0] if isinstance(state, tuple) else state  # Handle Gymnasium compatibility

    obs_agent2 = env.obs_agent_two()
    obs_agent2 = obs_agent2[0] if isinstance(obs_agent2, tuple) else obs_agent2

    total_reward = 0
    
    for t in range(max_steps):

        frame_idx += 1
        
        agent.beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)        # Smoother beta annealing

        done = False

        a1 = agent.perform_greedy_action(state)
        a2 = opponent.act(obs_agent2)
        full_action = np.hstack([env.discrete_to_continous_action(a1), a2])

        start_time = time.time()        # Debbuging

        next_state, reward, done, truncated, _ = env.step(full_action)
        
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

    if episode % 20 == 0:    
        losses.extend(loss)
        stats.append([episode, total_reward, t + 1])
        print(f"Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}, Beta: {agent.beta}")
    
    if agent._config["use_eps_decay"] and episode > int(0.5 * max_episodes):
        agent._perform_epsilon_decay()  
        
    if ((episode) % int(max_episodes/10) == 0) and episode > 0:
        agent.Q.save(env_name, name = f"episode_{episode}")

    logging.debug(f" time per frame: {(time.time()-time_start)/frame_idx}")
    logging.debug(f" mean sample time: {np.mean(agent.sample_times)}")

agent.Q.save(env_name, name = "training_finished")
sf.save_stats(env_name, stats, losses)
