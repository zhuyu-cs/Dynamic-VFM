from mmcv.runner import HOOKS, LrUpdaterHook
from math import cos, pi

@HOOKS.register_module()
class CosineRestartLrUpdaterHook(LrUpdaterHook):
    """Cosine annealing with restarts learning rate scheme.

    Args:
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float], optional): Restart weights at each
            restart iteration. Default: [1].
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 periods,
                 by_epoch=True,
                 restart_weights=[1],
                 min_lr=None,
                 min_lr_ratio=None,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.by_epoch = by_epoch
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        super(CosineRestartLrUpdaterHook, self).__init__(**kwargs)

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
        else:
            progress = runner.iter

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        return annealing_cos(base_lr, target_lr[idx], alpha, current_weight)


def get_position_from_periods(iteration, cumulative_periods):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_periods = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 3.

    Args:
        iteration (int): Current iteration.
        cumulative_periods (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_periods):
        if iteration < period:
            return i
    raise ValueError(f'Current iteration {iteration} exceeds '
                     f'cumulative_periods {cumulative_periods}')

def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out
