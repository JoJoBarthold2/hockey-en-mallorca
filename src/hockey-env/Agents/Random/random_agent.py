import numpy as np
from comprl.client import Agent

class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def __init__(self, seed):
        np.random.seed(seed)

    def act(self, obs):
        return np.random.uniform(-1,1,4)

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