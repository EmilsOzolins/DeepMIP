from abc import ABC, abstractmethod


class MIPDataset(ABC):

    @property
    @abstractmethod
    def required_output_bits(self):
        pass

    @abstractmethod
    def create_metrics(self):
        pass

    @abstractmethod
    def evaluate_model_outputs(self, binary_assignment, decimal_assignment, batched_data):
        pass

    @abstractmethod
    def get_metrics(self):
        pass
