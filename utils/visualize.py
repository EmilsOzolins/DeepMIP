import numpy as np


def format_metrics(mode: str, step: int, metrics: dict, max_line_len=200):
    formatted_metrics = []

    for key, value in metrics.items():
        if isinstance(value, float) or (isinstance(value, np.ndarray) and value.size == 1):
            formatted_metrics.append(f"[{key}={value:.6f}]")
        elif isinstance(value, int):
            formatted_metrics.append(f"[{key}={value}]")
        else:
            raise NotImplementedError(f"Formatting for {type(value)} type not implemented!")

    line_start = f"[step={step}] [{mode}]"
    formatted_output = [line_start]
    line_len = len(line_start)

    for m in formatted_metrics:
        if line_len + len(m) + 1 > max_line_len:
            formatted_output.append("\n")
            formatted_output.append(line_start)
            line_len = len(line_start)

        formatted_output.append(" ")
        formatted_output.append(m)
        line_len += len(m) + 1

    return "".join(formatted_output)
