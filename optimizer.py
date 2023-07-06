import copy
import random
from typing import Callable, Iterable, Tuple
import math
import numpy as np

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

class PCGrad:
    def __init__(self, optimizer, reduction="mean"):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        """
        clear the gradient of the parameters
        """

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        """
        update the parameters with the gradient
        """

        return self._optim.step()

    def pc_backward(self, objectives):
        """
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        """

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == "sum":
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            exit("invalid reduction method")

        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        """
        set the modified gradients to the network
        """

        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        """
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx : idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        """
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad