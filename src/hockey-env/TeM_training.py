import time
import copy
import random
import logging
import argparse
import numpy as np
from importlib import reload
from gymnasium import spaces
import hockey.hockey_env as h_env
import Agents.utils.stats_functions as sf
from Agents.utils.actions import MORE_ACTIONS
from Agents.Random.random_agent import RandomAgent
from Agents.Tapas_en_Mallorca.Agent import Combined_Agent
from Agents.Prio_n_step.Prio_DQN_Agent import Prio_DQN_Agent
from Agents.Combined_Agent_Double.Dueling_DDQN_Agent import Dueling_DDQN_Agent as Previous_Combined_Agent

initalization_time = time.time()        # Debugging
parser = argparse.ArgumentParser(description = "Train Dueling DDQN Agent.")
parser.add_argument("--use_dueling", type = str, default = "True", help = "Use Dueling Network")
parser.add_argument("--use_double", type = str, default = "True", help = "Use Double DQN")
parser.add_argument("--use_eps_decay", type = str, default = "False", help = "Use Epsilon Decay")
parser.add_argument("--use_noisy_net", type = str, default = "True", help = "Use Noisy Net")
parser.add_argument("--use_prio", type = str, default = "True", help = "Use Prioritized Buffuring Replay")
parser.add_argument("--n_step", type = int, default = 5, help = "Number of steps to look ahead")
parser.add_argument("--env_description", type = str, default = "", help = "Additional description for env_name")
parser.add_argument("--seed", type = int, default = 7489, help = "Seed for the training")
parser.add_argument("--use_more_actions", type = str, default = "True", help = "Use more actions")
parser.add_argument("--max_episodes", type = int, default = 10000, help = "Max number of episodes")
parser.add_argument("--games_to_play", type = int, default = 50, help = "Number of games to play")
parser.add_argument("--train_iterations", type = int, default = 32, help = "Number of training iterations")
parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

args = parser.parse_args()

use_dueling = True if args.use_dueling == "True" else False
use_double = True if args.use_double == "True" else False
use_eps_decay = True if args.use_eps_decay == "True" else False
use_prio = True if args.use_prio == "True" else False
use_noisy = True if args.use_noisy_net == "True" else False

SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
seed = SEED_TRAIN_1

USE_MORE_ACTIONS = True

random.seed(seed)
if args.verbose == "True":
    logging.basicConfig(level=logging.DEBUG)
else:    
    logging.basicConfig(level=logging.INFO)

reload(h_env)
env = h_env.HockeyEnv()

name_parts = []
if use_dueling:
    name_parts.append("Dueling")
if use_double:
    name_parts.append("Double")
name_parts.append("DQN")
if use_prio:
    name_parts.append("Prio")
name_parts.append(f"n_step_{args.n_step}")
name = "_".join(name_parts)

env_name = f"{name}_{args.env_description}"
logging.info(env_name)

state_space = env.observation_space

if(USE_MORE_ACTIONS):
    action_space = spaces.Discrete(len(MORE_ACTIONS))
else: 
    action_space = env.discrete_action_space

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
agent.Q.load("Dueling_Double_DQN_Prio_n_step_5_20_2_25", name = "episode_1000")

opponent0 = RandomAgent(seed = seed)
opponent1 = h_env.BasicOpponent()
opponent2 = h_env.BasicOpponent(weak = False)
opponent3 = Prio_DQN_Agent(
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
opponent3.Q.load("pure_prio_training_2_2_25", name = "episode_5000")

opponent4 = Prio_DQN_Agent(
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
opponent4.Q.load("pure_prio_training_2_2_25", name = "episode_7500")

opponent5 = Previous_Combined_Agent(
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
opponent5.Q.load("combined_training_6_2_25", name = "episode_5000")

opponent6 = Previous_Combined_Agent(
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
opponent6.Q.load("combined_training_6_2_25", name = "episode_7500")

agent_copy = copy.deepcopy(agent)

opponents = [
    opponent0,
    opponent1,
    opponent2,
    opponent3,
    opponent4,
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

betas = []
stats = []
losses = []
epsilons = []
saved_weights = ["episode_500, episode_1000"]

frame_idx = 0

max_episodes = args.max_episodes # 10000 default
games_to_play = args.games_to_play # 50 default

train_iterations = args.train_iterations # 32 default

logging.info(f"Initialization time: {time.time()-initalization_time}")        # Debugging

time_start = time.time()        # Debugging
last_save_time = time.time()

for episode in range(max_episodes):

    if saved_weights != []:
        selected = random.randint(0, len(opponents) - 1)
    else:
        selected = random.randint(0, len(opponents) - 2)
    
    opponent = opponents[selected]
    logging.info(opponents_names[selected])

    if opponents_names[selected] == "Self_play":
        self_play_time = time.time()
        weights = random.choice(saved_weights)
        opponent.Q.load(env_name, name = weights)
        logging.debug(f" Self play time: {time.time()-self_play_time}")

    for game in range(games_to_play):

        state, _ = env.reset(seed = seed)
        obs_agent2 = env.obs_agent_two()
        total_reward = 0

        t = 0
        while True:
            
            frame_idx += 1
            done = False

            a1 = agent.act(state)
            if(USE_MORE_ACTIONS):
                a1_cont = MORE_ACTIONS[a1]
            else: 
                a1_cont = env.discrete_to_continous_action(a1)

            if opponents_names[selected] not in ["Random", "Weak", "NonWeak"]:
                a2 = opponent.act(obs_agent2, eps = 0, validation = True)
                if USE_MORE_ACTIONS:
                    a2 = MORE_ACTIONS[a2]
                else:
                    a2 = env.discrete_to_continous_action(a2)
            else:
                a2 = opponent.act(obs_agent2)

            full_action = np.hstack([a1_cont, a2])

            start_time = time.time()        # Debbuging

            next_state, reward, done, truncated, info = env.step(full_action)
            
            logging.debug(f" Env time: {time.time()- start_time}")      # Debugging

            total_reward += reward

            if agent.use_n_step:
                one_step_transition = agent.n_buffer.add_transition(
                    (state, a1, reward, next_state, done)
                )
                if one_step_transition != ():
                    agent.buffer.store(one_step_transition)
            else:
                one_step_transition = (state, a1, reward, next_state, done)
                if one_step_transition != ():
                    agent.buffer.add_transition(one_step_transition)        ## Store for vales

            state = next_state
            obs_agent2 = env.obs_agent_two()

            if done or truncated: break

        training_time = time.time()        # Debugging
        loss = agent.train(train_iterations)
        logging.info(f" Training time: {time.time()-training_time}")      # Debug
        match_history[selected].append(info["winner"])
        logging.debug(info["winner"])

        if game % int(games_to_play/2) == 0:    
            losses.extend(loss)
            stats.append([episode, total_reward, t + 1])
            betas.append(agent.beta)
            epsilons.append(agent._eps)
            logging.info(f"Episode {episode+1}/{max_episodes}, Game {game+1}/{games_to_play} - Total Reward: {total_reward}")
        
        t += 1

    if agent._config["use_eps_decay"] and episode > int(0.8 * max_episodes):
        epsilon_time = time.time()
        agent._perform_epsilon_decay()  
        logging.debug(f" Epsilon decay time: {time.time()-epsilon_time}")

    if ((episode) % int(max_episodes/20) == 0) and episode > 0:  
        agent.Q.save(env_name, name = f"episode_{episode}")
        saved_weights.append(f"episode_{episode}")
        sf.save_epsilons(env_name, epsilons)
        sf.save_betas(env_name, betas)
        sf.save_stats(env_name, stats, losses)
        sf.save_match_history(env_name, match_history)
        sf.plot_returns(stats, env_name)
        sf.plot_losses(losses, env_name)
        sf.plot_epsilon_evolution(env_name, epsilons)

    if (episode % 20 == 0) and episode > 0:  
        sf.plot_match_evolution_by_chunks(env_name, match_history, opponents_names, games_to_play)

    if time.time() - last_save_time >= 600:  # 600 segundos = 10 minutos
        agent.Q.save(env_name, name="more_recent")
        last_save_time = time.time()

    #logging.debug(f" time per frame: {(time.time()-time_start)/frame_idx}")

agent.Q.save(env_name, name = "training_finished")
sf.save_epsilons(env_name, epsilons)
sf.save_betas(env_name, betas)
sf.save_stats(env_name, stats, losses)
sf.save_match_history(env_name, match_history)
sf.plot_returns(stats, env_name)
sf.plot_losses(losses, env_name)
sf.plot_epsilon_evolution(env_name, epsilons)
sf.plot_match_evolution_by_chunks(env_name, match_history, opponents_names, games_to_play)
