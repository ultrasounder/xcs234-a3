import torch
import torch.nn as nn
import torch.distributions as ptd

from abc import ABC, abstractmethod
from utils.network_utils import np2torch


class BasePolicy(ABC):

    def __init__(self, device):
        ABC.__init__(self)
        self.device = device

    @abstractmethod
    def action_distribution(self, observations):
        """
        Defines the conditional probability distribution over actions given an observation
        from the environment

        Args:
            observations (torch.Tensor):  observation of state from the environment
                                        (shape [batch size, dim(observation space)])

        Returns:
            distribution (torch.distributions.Distribution): represents the conditional distributions over
                                                            actions given the observations. Note: a pytorch
                                                            Distribution can have a batch size, and represent
                                                            many distributions.

        Note:
            See https://pytorch.org/docs/stable/distributions.html#distribution for further details
            on distributions in Pytorch. This is an abstract method and must be overridden by subclasses.
            It will return an object representing the policy's conditional
            distribution(s) given the observations. The distribution will have a
            batch shape matching that of observations, to allow for a different
            distribution for each observation in the batch.
        """
        pass

    def act(self, observations):
        """
        Samples actions to be used to act in the environment

        Args:
            observations (torch.Tensor):  observation of states from the environment
                                        (shape [batch size, dim(observation space)])


        Returns:
            sampled_actions (np.array): actions sampled from the distribution over actions resulting from the
                                        learnt policy (shape [batch size, *shape of action])

        TODO:
            Call self.action_distribution to get the distribution over actions,
            then sample from that distribution. You will have to convert the
            actions to a numpy array, via numpy(). See Converting torch Tensor to numpy Array section of the following tutorial
            for further details: https://pytorch.org/tutorials/beginner/former_torchies/tensor_tutorial.html.
            Before converting to numpy, take into consideration the current device of the tensor and whether
            this can be directly converted to a numpy array. Further details can be found here:
            https://pytorch.org/docs/stable/generated/torch.Tensor.cpu.html.
            Put the result in a variable called sampled_actions (which will be returned).
        """
        observations = np2torch(observations, device=self.device)
        ### START CODE HERE ###
        # Get the distribution over actions given the observations
        action_distribution = self.action_distribution(observations)
        # Sample actions from the distribution
        sampled_actions_tensor = action_distribution.sample()
        # Convert the sampled actions to numpy array
        sampled_actions = sampled_actions_tensor.cpu().numpy()
        
        ### END CODE HERE ###
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network, device):
        nn.Module.__init__(self)
        BasePolicy.__init__(self, device)
        self.network = network
        self.device = device

    def action_distribution(self, observations):
        """
        Args:
            observations (torch.Tensor):  observation of states from the environment
                                        (shape [batch size, dim(observation space)])

        Returns:
            distribution (torch.distributions.Categorical): represent the conditional distribution over
                                                            actions given a particular observation

        Notes:
            See https://pytorch.org/docs/stable/distributions.html#categorical for more details on
            categorical distributions in Pytorch
        """
        ### START CODE HERE ###
        # Pass the observations through the network to get the logits
        # make sure the observations are on the correct device
        observations = observations.to(self.device)
        logits = self.network(observations)
        # Move logits to the same device as observations to ensure compatibility with actions
        logits = logits.to(observations.device)
        # Create a categorical distribution using the logits
        distribution = torch.distributions.Categorical(logits=logits)
        ### END CODE HERE ###
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):
    """

    Args:
        network ():
        action_dim (int): the dimension of the action space

    TODO:
        After the basic initialization, you should create a nn.Parameter of
        shape [dim(action space)] and assign it to self.log_std.
        A reasonable initial value for log_std is 0 (corresponding to an
        initial std of 1), but you are welcome to try different values.

        Don't forget to assign the created nn.Parameter to the correct device.

        For more information on nn.Parameter please consult the following
        documentation https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
    """

    def __init__(self, network, action_dim, device):
        nn.Module.__init__(self)
        BasePolicy.__init__(self, device)
        self.network = network
        self.device = device
        ### START CODE HERE ###
        # Create a learnable parameter for log_std with initial value of 0
    # This corresponds to an initial std of 1 (since exp(0) = 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim).to(device))
        ### END CODE HERE ###

    def std(self):
        """
        Returns:
            std (torch.Tensor):  the standard deviation for each dimension of the policy's actions
                                (shape [dim(action space)])

        Hint:
            It can be computed from self.log_std
        """
        ### START CODE HERE ###
        # Convert from log space to standard deviation using exponential function
        std = torch.exp(self.log_std)
        ### END CODE HERE ###
        return std

    def action_distribution(self, observations):
        """
        Args:
            observations (torch.Tensor):  observation of states from the environment
                                        (shape [batch size, dim(observation space)])

        Returns:
            distribution (torch.distributions.Distribution): a pytorch distribution representing
                a diagonal Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()

        Note:
            PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of torch.distributions.MultivariateNormal

            Please consult the following documentation for further details on
            the use of probability distributions in Pytorch:
            https://pytorch.org/docs/stable/distributions.html
        """
        ### START CODE HERE ###
        observations_device = observations.device  # Remember original device
        # Ensure the observations are on the correct device
        observations = observations.to(self.device)
        # Get the mean from the network
        means = self.network(observations)
        # Get the standard deviation from the log_std parameter
        stds = self.std()
        # Move results back to the original observations device
        means = means.to(observations_device)
        stds = stds.to(observations_device)
        # create a batch appropriate scale_tril(lower triangular matrix) for the multivariate normal distribution
        # First, expand the stds to match the batch size of the means
        batch_size = means.size(0)
        batch_stds = stds.expand(means.size(0), -1)
        
        # create a diagonal covariance matrix using the stds
        scale_tril = torch.diag_embed(batch_stds)
        # Create a multivariate normal distribution with the means and scale_tril
        distribution = torch.distributions.MultivariateNormal(loc=means, scale_tril=scale_tril)
        ### END CODE HERE ###
        return distribution
