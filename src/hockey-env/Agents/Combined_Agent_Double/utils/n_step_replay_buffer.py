import numpy as np
from collections import deque

class N_Step_ReplayBuffer:
    """N-step replay buffer. Adapted to be similar to Memory from the lecture can be used in Combination with PER

    transitions have the shape : (ob, a, reward, ob_new, done)"""

    def __init__(self, obs_dim, max_size=100000, n_steps=1, discount=0.95):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size
        self.n_steps = n_steps
        self.discount = discount
        self.n_step_buffer = deque(maxlen=n_steps)

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.n_step_buffer.append(transitions_new)

        if len(self.n_step_buffer) < self.n_steps:
            return transitions_new  # not enough transitions to add n-step transition

        # add n-step transition
        observation, action = self.n_step_buffer[0][
            :2
        ]  # get the first action and observation from the buffer
        reward, next_observation, done = self._get_n_step_info(self.current_idx)

        self.transitions[self.current_idx, :] = (
            np.asarray(  # add the n-step transition to the buffer
                [observation, action, reward, next_observation, done], dtype=object
            )
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size
        return self.n_step_buffer[0]

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds, :]

    def sample_from_idx(self, idx, batch=1):
        assert (
            len(idx) == batch
        ), f"Batch size {batch} does not match the length of the index in sample_from_idx {len(idx)}"

        assert (
            batch <= self.size
        ), "Batch size is larger than the buffer size in sample_from_idx"

        return self.transitions[idx, :]

    def _get_n_step_info(self, idx):
        rew, next_obs, done = self.n_step_buffer[-1][
            -3:
        ]  # take the last transitions out of the n_step_buffer

        # now we go back in the buffer and add the rewards
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + self.discount * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done