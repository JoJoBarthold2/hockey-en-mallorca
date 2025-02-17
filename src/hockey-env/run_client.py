from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np
from comprl.client import Agent, launch_client
from gymnasium import spaces

from Prio_n_step_Agent.Prio_DQN_Agent import Prio_DQN_Agent
from Prio_n_step_Agent.utils.random_agent import RandomAgent
from Prio_n_step_Agent.utils.actions import MORE_ACTIONS

from Combined_Agent_Double.Dueling_DDQN_Agent import Dueling_DDQN_Agent as Combined_Agent


SEED_TRAIN_1 = 7489
SEED_TRAIN_2 = 1312
SEEDS_TEST = [291 + i for i in range(10)]

seed = SEED_TRAIN_1

USE_MORE_ACTIONS = True


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "Prio_DQN", "Combined_DQN"],
        default="weak",
        help="Which agent to use.",
    )
    parser.add_argument("--episode" , type=int, default= -1, help="Episode number to load the weights from")
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    episode = args.episode
    episode = "training_finished" if episode == -1 else  "episode_" + str(episode)
    print(f"Episode: {episode}")
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "Prio_DQN":
        env_name = "../weights/pure_prio_training_2_2_25"
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

    elif args.agent == "Combined_DQN":
        env_name = "../weights/combined_training_6_2_25"
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
            seed=seed,
            eps=0.01,
            learning_rate=0.0001,
            hidden_sizes=[256, 256],
            n_steps=5,
            use_more_actions=USE_MORE_ACTIONS,
            env=env,
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
