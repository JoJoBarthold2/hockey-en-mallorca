import numpy.random as random
import numpy as np
import logging

from Agents.utils.segment_tree import SumSegmentTree, MinSegmentTree

class PrioritizedReplayBuffer:
    """Prioritized Replay buffer. Adapted to be similar to Memory from the lecture
     transitions have the shape : (ob, a, reward, ob_new, done)

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
        self,
        max_size: int,
        batch_size: int = 32,
        alpha: float = 0.6,
    ):
        """Initialization."""
        assert alpha >= 0

        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

        # PER Stuff
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.batch_size = batch_size

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        # SegmentTree
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self, transitions_new
    ):  # corresponding to add_transition in original memory.py
        if self.size == 0:
            # Transitions shape:  (array([-0.56146526, -0.8275003 , -0.9523803 ], dtype=float32), 3, -4.787382871690309, array([-0.6188719 , -0.78549194, -1.4230055 ], dtype=float32), False)

            print("Creating new buffer with size: ", self.max_size)
            print("Transitions shape: ", len(transitions_new))
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, batch=128, beta: float = 0.4):
        """Sample a batch of experiences."""
        self.batch_size = batch
        assert self.size >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        logging.debug(f"Transitions shape: {self.transitions.shape}")
        logging.debug(f"Transitions type: {type(self.transitions)}")
        logging.debug(f"Indices type: {type(indices)}")
        logging.debug(f"indices: {indices}")
        logging.debug(f"Number of indices: {len(indices)}")
        logging.debug(f"Indices shape: {indices.shape}")
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        return self.transitions[indices], indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.size

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, self.size - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return np.array(indices, dtype=int)

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        logging.debug(f"p_sample: {p_sample}")
        logging.debug(f"self.size: {self.size}")
        weight = (p_sample * self.size) ** (-beta)
        weight = weight / max_weight

        return weight

    def get_all_transitions(self):
        return self.transitions[0 : self.size]
