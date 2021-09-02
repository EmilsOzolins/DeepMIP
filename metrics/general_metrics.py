from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List


class Metrics(ABC):
    def __init__(self) -> None:
        self._value = defaultdict(float)
        self._count = defaultdict(float)

    @abstractmethod
    def update(self, **kwargs):
        pass

    @property
    def result(self):
        return self._value

    @property
    def numpy_result(self):
        return {k: v.detach().cpu().numpy() for k, v in self._value.items()}


class AverageMetrics(Metrics):

    def __init__(self) -> None:
        super().__init__()

    def update(self, **kwargs) -> None:
        for key, val in kwargs.items():
            self._count[key] += 1
            self._value[key] += (val - self._value[key]) / self._count[key]


class MetricsHandler(Metrics):

    def __init__(self, *metrics) -> None:
        super().__init__()
        self._metrics: List[Metrics] = list(metrics)

    def update(self, **kwargs):
        for m in self._metrics:
            m.update(**kwargs)

    @property
    def result(self):
        res = {}
        for m in self._metrics:
            res.update(m.result)
        return res

    @property
    def numpy_result(self):
        res = {}
        for m in self._metrics:
            res.update(m.numpy_result)
        return res
