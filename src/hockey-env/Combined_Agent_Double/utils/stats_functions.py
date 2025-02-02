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

def plot_match_evolution_by_chunks(env_name, match_history, opponents_names, chunk_size, name="match_evolution", save_figure = True):

    n_opponents = len(opponents_names)
    fig, axes = plt.subplots(n_opponents, 1, figsize=(10, 4 * n_opponents), sharex=False)
    
    if n_opponents == 1:
        axes = [axes]
        
    for i in range(n_opponents):
        results = match_history[i]
        ax = axes[i]
        if len(results) == 0:
            ax.set_title(f"{opponents_names[i]} (no games played)")
            continue
        
        chunks = [results[j:j+chunk_size] for j in range(0, len(results), chunk_size)]
        chunk_indices = np.arange(1, len(chunks) + 1)
        
        win_rates = []
        wins_chunk = []
        draws_chunk = []
        losses_chunk = []
        
        for chunk in chunks:
            wins = chunk.count(1)
            draws = chunk.count(0)
            losses = chunk.count(-1)
            total = wins + draws + losses
            wr = wins / total if total > 0 else 0
            win_rates.append(wr)
            wins_chunk.append(wins)
            draws_chunk.append(draws)
            losses_chunk.append(losses)
        
        ax.plot(chunk_indices, wins_chunk, marker="o", label="Wins", color="green")
        ax.plot(chunk_indices, draws_chunk, marker="o", label="Draws", color="gray")
        ax.plot(chunk_indices, losses_chunk, marker="o", label="Losses", color="red")
        ax.set_ylabel("Number of Matches")
        ax.set_title(f"Evolution for {opponents_names[i]}")
        
        ax2 = ax.twinx()
        ax2.plot(chunk_indices, win_rates, marker="o", linestyle="--", label="Win Rate", color="blue")
        ax2.set_ylabel("Win Rate")
        ax2.set_ylim(0, 1)
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
    
    fig.tight_layout()
    if save_figure:
        _save_plot(env_name, name)