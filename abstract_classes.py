from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class RewardFunction(ABC):
    """
    Abstract class for defining a reward function.
    """

    @abstractmethod
    def compute_reward(self, state):
        """
        Compute the reward for a given state transition.

        :param state: The current state.
        :return: The computed reward.
        """
        pass

class RewardAggregator(ABC):
    """
    Abstract class to aggregate rewards from multiple RewardFunction instances and compute a loss.
    """

    def __init__(self, reward_functions, loss_function=nn.MSELoss()):
        """
        Initializes the RewardAggregator with multiple RewardFunction instances and a loss function.

        :param reward_functions: List of RewardFunction instances.
        :param loss_function: A PyTorch loss function, default is Mean Squared Error Loss.
        """
        self.reward_functions = reward_functions
        self.loss_function = loss_function

    @abstractmethod
    def aggregate_rewards(self, states, actions, next_states):
        """
        Aggregate rewards from multiple RewardFunction instances.

        :param states: List of states, each corresponding to a RewardFunction.
        :param actions: List of actions, each corresponding to a RewardFunction.
        :param next_states: List of next states, each corresponding to a RewardFunction.
        :return: Aggregated reward.
        """
        pass

    @abstractmethod
    def compute_loss(self, predicted_rewards, actual_rewards):
        """
        Compute the loss from aggregated rewards.

        :param predicted_rewards: Tensor of predicted rewards.
        :param actual_rewards: Tensor of actual rewards.
        :return: The computed loss.
        """
        pass