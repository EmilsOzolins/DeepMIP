from collections import defaultdict


class AverageMetric:

    def __init__(self) -> None:
        self._value = defaultdict(float)
        self._count = defaultdict(float)

    def update(self, values: dict) -> None:
        for key, val in values.items():
            self._count[key] += 1
            self._value[key] += (val - self._value[key]) / self._count[key]

    @property
    def result(self):
        return self._value

    @property
    def numpy_result(self):
        return {k: v.detach().cpu().numpy() for k, v in self._value.items()}
