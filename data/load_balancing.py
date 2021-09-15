import glob
import gzip
import pickle
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from data.datasets_base import MIPDataset
from data.mip_instance import MIPInstance
from metrics.general_metrics import Metrics
from metrics.mip_metrics import MIPMetrics
from utils.data import MIPBatchHolder


class LoadBalancingDataset(MIPDataset, Dataset):

    def __init__(self, data_folder="/host-dir/mip_data/load_balancing/train") -> None:
        self._instances = glob.glob(data_folder + "/*.pickle.gz")

    @property
    def required_output_bits(self):
        return 1

    @property
    def test_metrics(self) -> List[Metrics]:
        return [MIPMetrics()]

    @property
    def train_metrics(self) -> List[Metrics]:
        return []

    def decode_model_outputs(self, model_output, batch_holder: MIPBatchHolder):
        output = torch.squeeze(model_output)
        mask = batch_holder.integer_mask
        inv_mask = 1 - mask
        return torch.round(output * mask) + output * inv_mask

    def __getitem__(self, index: int) -> Dict:
        file_name = self._instances[index]

        with gzip.open(file_name, mode="rb") as file:
            mip_instance: MIPInstance = pickle.load(file)

        return {"mip": mip_instance,
                "optimal_solution": torch.as_tensor([float('nan')], dtype=torch.float32)}

    def __len__(self) -> int:
        return len(self._instances)
