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
    
def save_winrate(env_name, winrates, name = "winrates"):

    os.makedirs(f"{env_name}/stats/pkl", exist_ok=True)
    winrate_path = os.path.join(f"{env_name}/stats/pkl", f"{name}.pkl")

    with open(winrate_path, "wb") as f:
        pickle.dump(winrates, f)

def load_winrates(env_name, name = "winrates"):
    
    winrate_path = os.path.join(f"{env_name}/stats/pkl", f"{name}.pkl")

    with open(winrate_path, "rb") as f:
        winrates = pickle.load(f)

        return winrates