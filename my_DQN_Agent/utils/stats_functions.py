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

def plot_returns(returns, env_name, name = "Returns_plot", save = True):

    stats_np = np.asarray(returns)
    fig = plt.figure(figsize=(6,3.8))
    plt.plot(stats_np[:,1], label="return")
    plt.plot(running_mean(stats_np[:,1],20), label="smoothed-return")
    plt.legend()

    if save:
        _save_plot(env_name, name)

def plot_losses(losses, env_name, name = "Losses_plot", save = True):

    losses_np = np.asarray(losses)
    fig = plt.figure(figsize=(6,3.8))
    plt.plot(losses_np)

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