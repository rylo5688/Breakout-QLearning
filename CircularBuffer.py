"""
Replay experience: Helps prevent our network from diverging
    At each QLearning Iteration, we add all relevant information to a finite-size memory instead of updating the model based on the last step
    This helps because in reinforcement learning there is a chance that we will forget previous state if we do not see it for a while. Replay experience
    helps prevent this because we keep it in memory and therefore we can still show those old frames to the network.
    (Disadvantage = This requires a lot of memory)
"""
from random import randint
class CircularBuffer:
    """
    Circular buffer that represents our replay experience for the neural network
    NOTE: This will use a lot of memory
    """

    def __init__(self, size):
        # NOTE: We use size + 1 to help us differentiate between full and empty
        self.data = [None] * (size + 1)
        self.size = size + 1 # This represents the total size of the array
        self.start = 0
        self.end = 0

    def append(self, value):
        self.data[self.end] = value
        self.end = (self.end + 1) % self.size

        # Since we have size + 1, we added one to many items, so we will just trash that element self.start is pointing at
        if self.end == self.start:
            self.start = (self.start + 1) % self.size

    # Gets a random sample of the given size
    def random_sample(self, size):
        sample = []
        n_items = len(self)
        for _ in range(size):
            r_index = randint(0, n_items)
            sample.append(self[r_index])
        return sample

    def __getitem__(self, index):
        # Return the item at the index starting from self.start
        return self.data[(self.start + index) % self.size]

    # This returns the number of items in the CircularBuffer object
    def __len__(self):
        if self.end < self.start:
            return self.end + self.size - self.start
        else:
            return self.end - self.start