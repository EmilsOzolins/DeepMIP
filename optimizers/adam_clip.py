from typing import Optional, Callable

import torch
from torch.optim import Adam


class Adam_clip(Adam):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False,
                 clip_multiplier=3.0, clip_epsilon=1e-3):
        self.clip_multiplier = clip_multiplier
        self.clip_epsilon = clip_epsilon
        super(Adam_clip, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    state = self.state[p]

                    if len(state) != 0:
                        grad_sq = state['exp_avg_sq'] # Exponential moving average of squared gradient values
                        step = state['step']
                        beta_2_power = beta2 ** step

                        clipVal = torch.sqrt(torch.sum(grad_sq) / (1.0 - beta_2_power)) * self.clip_multiplier + self.clip_epsilon
                        grad = self.clip_by_norm(grad, clipVal)
                        p.grad = grad

        return super().step(closure)

    def clip_by_norm(self, grad, clip_norm):
        l2sum = torch.sqrt(torch.sum(torch.square(grad)))
        intermediate = grad * clip_norm
        values_clip = intermediate / torch.maximum(l2sum, clip_norm)
        return values_clip


