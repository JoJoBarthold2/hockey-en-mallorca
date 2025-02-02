import numpy as np
import pylab as plt
import os
import pickle

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def _save_plot(env_name, name):

    plot_dir = f"{env_name}/stats/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{name}.png")
    plt.savefig(plot_path, dpi = 300)
    print(f"Plot saved at {plot_path}")

def plot_returns(returns, env_name, name = "returns_plot", save = True):

    stats_np = np.asarray(returns)
    fig = plt.figure(figsize=(6,3.8))
    plt.plot(stats_np[:,1], label="return")
    plt.plot(running_mean(stats_np[:,1],20), label="smoothed-return")
    plt.legend()
    plt.xlabel("Test Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward evolution during training")

    if save:
        _save_plot(env_name, name)

def plot_losses(losses, env_name, name = "losses_plot", save = True):

    losses_np = np.asarray(losses)
    fig = plt.figure(figsize=(6,3.8))
    plt.plot(losses_np)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Loss evolution during training")

    if save:
        _save_plot(env_name, name)

def save_stats(env_name, stats, losses):

    os.makedirs(f"{env_name}/stats/pkl", exist_ok=True)
    stats_path = os.path.join(f"{env_name}/stats/pkl", "stats.pkl")
    losses_path = os.path.join(f"{env_name}/stats/pkl", "losses.pkl")

    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    with open(losses_path, "wb") as f:
        pickle.dump(losses, f)

def load_stats(env_name):

    stats_path = os.path.join(f"{env_name}/stats/pkl", "stats.pkl")
    losses_path = os.path.join(f"{env_name}/stats/pkl", "losses.pkl")

    with open(stats_path, "rb") as f:
        loaded_stats = pickle.load(f)
    with open(losses_path, "rb") as f:
        loaded_losses = pickle.load(f)

    return loaded_stats, loaded_losses

def save_test_results(env_name, test_rewards, name = "test_results", save_figure = True):

    os.makedirs(f"{env_name}/stats/pkl", exist_ok=True)
    test_path = os.path.join(f"{env_name}/stats/pkl", f"{name}.pkl")

    with open(test_path, "wb") as f:
        pickle.dump(test_rewards, f)

    fig = plt.figure(figsize=(6,3.8))
    plt.bar(range(1, len(test_rewards) + 1), test_rewards)
    plt.xlabel("Test Episode")
    plt.ylabel("Total Reward")
    plt.title("Test Performance")

    if save_figure:
        _save_plot(env_name, name)

def load_test_stats(env_name):
    stats_path = os.path.join(f"{env_name}/stats/pkl", "test_results.pkl")

    with open(stats_path, "rb") as f:
        loaded_stats = pickle.load(f)
    return loaded_stats

def save_betas(env_name, betas, name = "beta"):

        os.makedirs(f"{env_name}/stats/pkl", exist_ok=True)
        beta_path = os.path.join(f"{env_name}/stats/pkl", f"{name}.pkl")

        with open(beta_path, "wb") as f:
            pickle.dump(betas, f)

def load_betas(env_name, name = "beta"):

    beta_path = os.path.join(f"{env_name}/stats/pkl", f"{name}.pkl")

    with open(beta_path, "rb") as f:
        betas = pickle.load(f)
    
    return betas

def plot_beta_evolution(env_name, betas, save_figure = True, name = "beta_evolution"):
    plt.figure(figsize=(10, 5))
    plt.plot(betas, marker='o', linestyle='-', color='b', label='Beta Values')
    plt.xlabel('Iteration')
    plt.ylabel('Beta Value')
    plt.title('Evolution of Beta Values')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    if save_figure:
        _save_plot(env_name, name)

def save_epsilons(env_name, epsilons, name = "epsilon"):

    os.makedirs(f"{env_name}/stats/pkl", exist_ok=True)
    beta_path = os.path.join(f"{env_name}/stats/pkl", f"{name}.pkl")

    with open(beta_path, "wb") as f:
        pickle.dump(epsilons, f)

def load_epsilons(env_name, name = "epsilon"):
    
    beta_path = os.path.join(f"{env_name}/stats/pkl", f"{name}.pkl")

    with open(beta_path, "rb") as f:
        epsilons = pickle.load(f)

        return epsilons

def plot_epsilon_evolution(env_name, epsilons, save_figure = True, name = "epsilon_evolution"):
    
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons, marker='o', linestyle='-', color='b', label='Epsilon Values')
    plt.xlabel('Iteration')
    plt.ylabel('Epsilon Value')
    plt.title('Evolution of Epsilon Values')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    if save_figure:
        _save_plot(env_name, name)

def save_match_history(env_name, match_history, name = "match_history"):

    os.makedirs(f"{env_name}/stats/pkl", exist_ok = True)
    np.savez(f"{env_name}/stats/pkl/{name}", match_history = match_history)

def load_match_history(env_name, name = "match_history"):
    
    data= np.load(f"{env_name}/stats/pkl/{name}.npz")
    return data["match_history"]

import matplotlib.pyplot as plt
import numpy as np

def plot_match_history(env_name, match_history, opponents, name="match_history", save_figure = True):

    match_history = np.array(match_history)

    losses = np.sum(match_history == -1, axis=1)
    draws = np.sum(match_history == 0, axis=1)
    wins = np.sum(match_history == 1, axis=1)

    total_matches = wins + draws + losses
    
    win_rates = np.where(total_matches > 0, wins / total_matches, 0)

    n = len(opponents)
    indices = np.arange(n)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(indices - width, losses, width, label='Losses (-1)', color='red')
    ax.bar(indices, draws, width, label='Draws (0)', color='gray')
    ax.bar(indices + width, wins, width, label='Wins (1)', color='green')
    for i in range(n):
        max_count = max(losses[i], draws[i], wins[i])
        ax.text(indices[i], max_count + 0.5, f"Winrate: {win_rates[i]:.1%}", ha='center')
    ax.set_xlabel("Opponents")
    ax.set_ylabel("Number of Matches")
    ax.set_title("Results by Opponent")
    ax.set_xticks(indices)
    ax.set_xticklabels(opponents)
    ax.legend()
    fig.tight_layout()
    if save_figure:
        _save_plot(env_name, name)
    plt.close()
