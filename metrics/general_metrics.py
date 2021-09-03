from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List


class Metrics(ABC):
    """ Not intended for direct subclassing.
    """
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


class StackableMetrics(Metrics, ABC):
    """
    Use this for case-specific metrics, for example metrics for Sudoku.
    """
    pass


class PrimitiveMetrics(Metrics, ABC):
    """
    Use this for metrics that simply calculate some statistics over data,
    e.g. Average, Median, etc.
    It can be used as building block for StackableMetrics.
    """
    pass


class AverageMetrics(PrimitiveMetrics):

    def __init__(self) -> None:
        super().__init__()

    def update(self, **kwargs) -> None:
        for key, val in kwargs.items():
            self._count[key] += 1
            self._value[key] += (val - self._value[key]) / self._count[key]


class MetricsHandler(StackableMetrics):

    def __init__(self, *metrics) -> None:
        super().__init__()

        for m in metrics:
            if not isinstance(m, StackableMetrics):
                raise RuntimeError("Only StackableMetrics can be handled by MetricsHandler!"
                                   "Please override StackableMetrics or handle metrics yourself.")

        self._metrics: List[StackableMetrics] = list(metrics)

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
