import torch

class BaseAgent(object):
    """
    Class for the basic agent objects.
    To define your own agent, subclass this class and implement the functions below.
    """

    def __init__(self, 
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 num_checkpoints,
                 save_timesteps,
                 env_valid=None,
                 storage_valid=None):
        """
        env: (gym.Env) environment following the openAI Gym API
        """
        self.env = env
        self.policy = policy
        self.logger = logger
        self.storage = storage
        self.device = device
        self.num_checkpoints = num_checkpoints
        self.save_timesteps = save_timesteps
        if self.save_timesteps is None:
            self.use_save_intervals = True
        else:
            self.use_save_intervals = False
        self.env_valid = env_valid
        self.storage_valid = storage_valid
        self.t = 0

    def predict(self, obs):
        """
        Predict the action with the given input
        """
        pass

    def update_policy(self):
        """
        Train the neural network model
        """
        pass

    def train(self, num_timesteps):
        """
        Train the agent with collecting the trajectories
        """
        pass

    def evaluate(self):
        """
        Evaluate the agent
        """
        pass
