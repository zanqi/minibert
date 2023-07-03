from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


# https://paperswithcode.com/method/adamw#:~:text=AdamW%20is%20a%20stochastic%20optimization,decay%20from%20the%20gradient%20update.
class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO
                beta1, beta2 = group["betas"]
                state["step"] = 1 if "step" not in state else state["step"] + 1
                step = state["step"]
                m_key = "m"
                v_key = "v"
                state[m_key] = (
                    torch.zeros_like(grad)
                    if m_key not in state
                    else state[m_key]
                )
                state[v_key] = (
                    torch.zeros_like(grad)
                    if v_key not in state
                    else state[v_key]
                )
                m = state[m_key]
                v = state[v_key]
                m.mul_(beta1).add_(grad, alpha=1 - beta1) # m = m * beta1 + grad * (1 - beta1)
                v.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                ) # v = v * beta2 + grad * grad * (1 - beta2)

                # bias correction in a single variable
                alphaT = (
                    alpha * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                    if group["correct_bias"]
                    else alpha
                )

                # update parameters
                p.data.addcdiv_(
                    m, v.sqrt() + group["eps"], value=-alphaT
                ) # p = p - m * alphaT / (v.sqrt() + eps)

                # weight decay
                p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

        return loss
