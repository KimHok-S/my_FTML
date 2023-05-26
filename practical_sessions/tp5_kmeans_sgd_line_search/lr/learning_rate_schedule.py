"""
    Set the schedule of the learning rate during GD or SGD
"""
import math


def learning_rate_schedule(gamma_0, iteration, schedule) -> float:
    """
        Set the schedule of the learning rate during GD or SGD
        :param gamma_0: initial learning rate
        :param iteration: current iteration
        :param schedule: schedule of the learning rate
        :return: learning rate
    """
    if schedule == "constant":
        gamma = gamma_0
    elif schedule == "decreasing 1":
        gamma = gamma_0 / (1 + iteration)
    elif schedule == "decreasing 2":
        gamma = gamma_0 / math.sqrt(1 + iteration)
    else:
        raise ValueError("Unknown schedule")
    return gamma
