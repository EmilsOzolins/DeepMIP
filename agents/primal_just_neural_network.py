import numpy as np

from agents.primal_nn import extract_current_ip_instance, NetworkPolicy
from data.mip_instance import MIPInstance


class ObservationFunction:  # This allows customizing information received by the solver

    def __init__(self, problem):
        super(ObservationFunction, self).__init__()
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific observations
        self.previous_file_name = None

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        if done:
            return None

        return extract_current_ip_instance(model.as_pyscipopt())


class Policy():
    def __init__(self, problem):
        # called once for each problem benchmark
        self.rng = np.random.RandomState()
        self.problem = problem  # to devise problem-specific policies
        self.network_policy = NetworkPolicy(problem)

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation):
        ip = observation  # type: MIPInstance
        return self.network_policy(action_set, ip)
