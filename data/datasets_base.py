from abc import ABC, abstractmethod
from typing import List

from metrics.general_metrics import Metrics
from utils.data_utils import MIPBatchHolder


class MIPDataset(ABC):

    @property
    @abstractmethod
    def required_output_bits(self):
        pass

    @property
    @abstractmethod
    def test_metrics(self) -> List[Metrics]:
        pass

    @property
    @abstractmethod
    def train_metrics(self) -> List[Metrics]:
        pass

    @abstractmethod
    def decode_model_outputs(self, model_output, batch_holder: MIPBatchHolder):
        pass
