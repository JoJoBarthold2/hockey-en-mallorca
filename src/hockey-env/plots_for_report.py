

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import imageio
import os
import random
import copy

from Agents.Prio_n_step.Prio_DQN_Agent import Prio_DQN_Agent
from Agents.Pablo.Adaptative_Dueling_Double_DQN.Agent import Adaptative_Dueling_Double_DQN
from Agents.Random.random_agent import RandomAgent
from Agents.utils.actions import MORE_ACTIONS
import hockey.hockey_env as h_env
import re

from Agents.Tapas_en_Mallorca.Adaptative_Dueling_Double_DQN.Agent import Adaptative_Dueling_Double_DQN_better_mem
import matplotlib.pyplot as plt
from importlib import reload
from Agents.utils.stats_functions import running_mean, load_stats, load_match_history

tournament_agent_path = "../weights/prio_agent_self_play_17_2_25"
n_step_6_path = "../weights/prio_agent_self_play_19_2_25_n_step_6"
n_step_1_path = "../weights/prio_agent_self_play_19_2_25"

prio_alpha_04 ="../weights/prio_agent_23_02_25_n_step_4_alpha_0.4"
prio_alpha_02_beta_04 = "../weights/prio_agent_23_02_25_n_step_4_alpha_0.2_beta_0.4"

no_prio_n_4 = "../weights/prio_agent_self_play_22_02_25_no_prio"
classic_dqn = "../weights/no_prio_agent_23_02_25_n_step_1"

Combined_n_4 = "../weights/Noisy_Dueling_Double_DQN_Prio_n_step_4_adaptive_training_fixed"
Combined_n_4_ten_games = "../weights/Noisy_Dueling_Double_DQN_Prio_n_step_4_10_games_only"



#### Experimentals

bigger_lr = "../weights/Noisy_Dueling_Double_DQN_Prio_n_step_4_lr_0.0005"
bigger_nn = "../weights/Noisy_Dueling_Double_DQN_Prio_n_step_4_bigger_nn"
plots_dir = "../../plots"
def plot_rewards(agents_names, episodes = 5000, name = "rewards"):
    plt.figure(figsize=(4, 3))
    for path,agent_name in agents_names:
        stats,_ = load_stats(path)
        #np_rewards =np.array([s[1] for s in stats])  
        np_rewards = np.array([s[1] for i, s in enumerate(stats) if i % 2 == 1])  # Only middle-episode rewards
      
        
        
     
        plotted_rewards = np_rewards[:int(episodes)]
        """print("================================")
        print(agent_name)
        print(np_rewards.shape)
        print(plotted_rewards.shape)
        
        print("================================")"""
        plt.plot(running_mean(plotted_rewards, 100), label=agent_name)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title("")
    save_path = os.path.join(plots_dir, f"{name}.png")
    plt.legend()
   
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")
    



def plot_winrate(agents, name, opponent_name, opponent_idx, chunk_size=200, games = 30000):	
    plt.figure(figsize=(4, 3))
    for path,agent_name in agents:
        match_history = load_match_history(path)
        match_history = match_history[opponent_idx][:games]
       
        running_means = running_mean(match_history, chunk_size)
       
        plt.plot(running_means, label=agent_name)
    plt.xlabel(f'Games played against {opponent_name}')
    plt.ylabel('Win rate')
    plt.title("")
    plt.legend()
    save_path = os.path.join(plots_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")
        
        



plot_rewards([(tournament_agent_path, "4 Steps"),(n_step_6_path, "6 Steps"),(n_step_1_path, "1 Step"), ], episodes = int(5000), name = "rewards_n_steps")
plot_rewards([(tournament_agent_path, "Alpha:0.2 Beta:0.6"),(prio_alpha_04, "Alpha:0.4 Beta:0.6"), (classic_dqn, "DQN"),(prio_alpha_02_beta_04, "Alpha:0.2 Beta:0.4") ], episodes = int(1500), name = "rewards_alpha_beta")
plot_rewards([(tournament_agent_path, "4 N steps with PER "),(no_prio_n_4, "4 N steps without PER"), (classic_dqn, "DQN")], episodes = int(2000), name = "rewards_no_prio")

plot_rewards([(tournament_agent_path, "PER + 4-Step"),(Combined_n_4, "Combined"), (classic_dqn, "DQN"), (bigger_nn, "Combined rectified")], episodes = int(2000), name = "rewards_vs_weak")
plot_rewards([(Combined_n_4,"50 games"),(Combined_n_4_ten_games, "10 games")], episodes = int(2000), name = "rewards_combined_vs_ten_games")

plot_rewards([(bigger_lr, "bigger_lr"),(bigger_nn, "bigger_nn"), (Combined_n_4, "Combined_n_4")], episodes = int(2000), name = "rewards_bigger_lr_nn")

plot_winrate([(tournament_agent_path, "4 Steps"),(n_step_6_path, "6 Steps"), (classic_dqn, "DQN"),(n_step_1_path, "1 Step") ], opponent_name=  "Weak", games = 10000, opponent_idx = 1, name = "win_rate_n_steps")
plot_winrate([(tournament_agent_path, "Alpha:0.2 Beta:0.6"),(prio_alpha_04,  "Alpha:0.4 Beta:0.6"), (classic_dqn, "DQN"),(prio_alpha_02_beta_04,  "Alpha:0.2 Beta:0.4") ], games = 10000, opponent_name=  "Weak", opponent_idx = 1, name = "win_rate_alpha_beta")
plot_winrate([(tournament_agent_path, "PER + 4-Step"),(Combined_n_4, "Combined"), (classic_dqn, "DQN"),(bigger_nn, "Combined rectified")], opponent_name=  "Weak", opponent_idx = 1, name = "win_rate_vs_weak", games = 10000)
plot_winrate([(Combined_n_4,"50 games"),(Combined_n_4_ten_games, "10 games")], opponent_name=  "Weak", opponent_idx = 1, name = "win_rate_combined_vs_ten_games", games=8000)
plot_winrate([(bigger_lr, "bigger_lr"),(bigger_nn, "bigger_nn"), (Combined_n_4, "Combined_n_4")], opponent_name=  "Weak", opponent_idx = 1, name = "win_rate_bigger_lr_nn")


plot_winrate([(tournament_agent_path, "PER + 4-Step"),(Combined_n_4, "Combined"), (classic_dqn, "DQN"), (bigger_nn, "Combined rectified")], opponent_name=  "Strong", opponent_idx = 2, name = "win_rate_vs_strong", games = 12000)

plot_winrate([(tournament_agent_path, "PER + 4-Step"),(Combined_n_4, "Combined"), (classic_dqn, "DQN"), (bigger_nn,"Combined rectified")], opponent_name=  "Combined", opponent_idx = 5, name = "win_rate_vs_combined", games = 12000)