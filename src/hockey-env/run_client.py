from __future__ import annotations

import argparse
from gymnasium import spaces
import hockey.hockey_env as h_env
from Agents.utils.actions import MORE_ACTIONS
from comprl.client import Agent, launch_client
from Agents.Random.random_agent import RandomAgent
from Agents.Tapas_en_Mallorca.old.Agent import Combined_Agent
from Agents.Prio_n_step.Prio_DQN_Agent import Prio_DQN_Agent
from Agents.Combined_Agent_Double.Dueling_DDQN_Agent import Dueling_DDQN_Agent as Previous_Combined_Agent


SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
SEEDS_TEST = [291 + i for i in range(10)]

seed = SEED_TRAIN_1

USE_MORE_ACTIONS = True

# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["random", "Prio_DQN", "Previous_Combined_DQN", "DDDQN", Combined_Agent],
        default="random",
        help="Which agent to use.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=-1,
        help="Episode number to load the weights from",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    episode = args.episode
    episode = "training_finished" if episode == -1 else "episode_" + str(episode)
    print(f"Episode: {episode}")
    agent: Agent
    if args.agent == "random":
        agent = RandomAgent(seed=seed)
    elif args.agent == "Prio_DQN":
        env_name = "../weights/prio_agent_self_play_17_2_25"
        env = h_env.HockeyEnv()
        h_env.HockeyEnv().seed(seed)

        state_space = env.observation_space

        if USE_MORE_ACTIONS:
            action_space = spaces.Discrete(len(MORE_ACTIONS))
        else:
            action_space = env.discrete_action_space

        agent = Prio_DQN_Agent(
            state_space,
            action_space,
            seed=seed,
            eps=0.01,
            learning_rate=0.0001,
            hidden_sizes=[256, 256],
            n_steps=5,
            use_more_actions=USE_MORE_ACTIONS,
            env=env,
        )
        agent.Q.load(env_name, name=episode)

    elif args.agent == "Previous_Combined_DQN":
        env_name = "../weights/combined_training_6_2_25"
        env = h_env.HockeyEnv()
        h_env.HockeyEnv().seed(seed)

        state_space = env.observation_space

        if USE_MORE_ACTIONS:
            action_space = spaces.Discrete(len(MORE_ACTIONS))
        else:
            action_space = env.discrete_action_space

        agent = Previous_Combined_Agent(
            state_space,
            action_space,
            seed=seed,
            eps=0.01,
            learning_rate=0.0001,
            hidden_sizes=[256, 256],
            n_steps=5,
            use_more_actions=USE_MORE_ACTIONS,
            env=env,
        )
        agent.Q.load(env_name, name=episode)

    elif args.agent == "DDDQN":
        env_name = "../weights/combined_training_6_2_25"
        env = h_env.HockeyEnv()
        h_env.HockeyEnv().seed(seed)

        state_space = env.observation_space

        if USE_MORE_ACTIONS:
            action_space = spaces.Discrete(len(MORE_ACTIONS))
        else:
            action_space = env.discrete_action_space

        agent = Previous_Combined_Agent(
            state_space,
            action_space,
            seed=seed,
            eps=0.01,
            learning_rate=0.0001,
            hidden_sizes=[256, 256],
            n_steps=5,
            use_more_actions=USE_MORE_ACTIONS,
            env=env,
        )
        agent.Q.load(env_name, name=episode)
    elif args.agent == "Combined_agent":
        env_name = "../weights/Dueling_Double_DQN_Prio_n_step_5"
        env = h_env.HockeyEnv()
        h_env.HockeyEnv().seed(seed)

        state_space = env.observation_space

        if USE_MORE_ACTIONS:
            action_space = spaces.Discrete(len(MORE_ACTIONS))
        else:
            action_space = env.discrete_action_space

        agent = Combined_Agent(
            state_space,
            action_space,
            env = env,
            seed = seed,
            use_eps_decay = False,
            use_dueling = True,
            use_double = True,
            use_noisy = True,
            use_prio = True,
            n_step = args.n_step,
            hidden_sizes = [256, 256]
        )
        agent.Q.load(env_name, name=episode)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
