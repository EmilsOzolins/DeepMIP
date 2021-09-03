import numpy as np


def format_metrics(mode: str, step: int, metrics: dict, max_line_len=120):
    formatted_metrics = []

    for key, value in metrics.items():
        if isinstance(value, float) or (isinstance(value, np.ndarray) and value.size == 1):
            formatted_metrics.append(f"[{key}={value:.4f}]")
        elif isinstance(value, int):
            formatted_metrics.append(f"[{key}={value}]")
        else:
            raise NotImplementedError(f"Formatting for {type(value)} type not implemented!")

    line_start = f"[step={step}] [{mode}]"
    formatted_output = [line_start]

    for m in formatted_metrics:
        if len(formatted_output) + len(m) + 1 > max_line_len:
            formatted_output.append("\n")
            formatted_output.append(line_start)

        formatted_output.append(" ")
        formatted_output.append(m)

    return "".join(formatted_output)
