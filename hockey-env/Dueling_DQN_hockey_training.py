from importlib import reload
import numpy as np

from Dueling_DQN_Agent.Dueling_DQN_Agent import Dueling_DQN_Agent
import Dueling_DQN_Agent.utils.stats_functions as sf
import hockey.hockey_env as h_env

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
seed = SEED_TRAIN_1

reload(h_env)
env_name = "Hockey_200000_30000_(128_128)"
env = h_env.HockeyEnv()

# Initialize the agent with the correct state/action space
state_space = env.observation_space
action_space = env.discrete_action_space

agent = Dueling_DQN_Agent(state_space, action_space, seed = seed, use_eps_decay = True, hidden_sizes = [128, 128]) 
opponent = h_env.BasicOpponent()

stats = []
losses = []

max_episodes = 200000
train_iterations = 32  # Number of training steps per episode

max_steps = 30000

for episode in range(max_episodes):

    state, _ = env.reset(seed = seed)
    state = state[0] if isinstance(state, tuple) else state  # Handle Gymnasium compatibility

    obs_agent2 = env.obs_agent_two()
    obs_agent2 = obs_agent2[0] if isinstance(obs_agent2, tuple) else obs_agent2

    total_reward = 0
    
    for t in range(max_steps):

        done = False

        a1 = agent.perform_greedy_action(state)
        a2 = opponent.act(obs_agent2)
        full_action = np.hstack([env.discrete_to_continous_action(a1), a2])

        next_state, reward, done, truncated, _ = env.step(full_action)
        
        total_reward += reward

        agent.buffer.add_transition((state, a1, reward, next_state, done))      # Store transition in the agent"s memory and then train

        state = next_state
        obs_agent2 = env.obs_agent_two()

        if done or truncated: break

    loss = agent.train(train_iterations)

    if episode % 100 == 0:    
        losses.extend(loss)
        stats.append([episode, total_reward, t + 1])
        print(f"Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}")
    
    if agent._config["use_eps_decay"] and episode > int(0.5 * max_episodes):
        agent._perform_epsilon_decay()  
        
    if ((episode) % int(max_episodes/10) == 0) and episode > 0:
        agent.Q.save(env_name, name = f"episode_{episode}")

agent.Q.save(env_name, name = "training_finished")
sf.save_stats(env_name, stats, losses)
