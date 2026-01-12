"""Math utils functions."""

import torch


def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Use float32-compatible epsilon (1e-7 instead of 1e-15)
        x = x.clamp(-1 + 1e-7, 1 - 1e-7)
        ctx.save_for_backward(x)
        # Keep in original dtype (float32 for MPS compatibility)
        z = x.float()  # Changed from .double() for MPS compatibility
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Keep in original dtype (float32 for MPS compatibility)
        z = x.float()  # Changed from .double() for MPS compatibility
        # Use float32-compatible epsilon (1e-7 instead of 1e-15)
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-7).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Use float32-compatible epsilon (1e-7 instead of 1e-15)
        x = x.clamp(min=1.0 + 1e-7)
        ctx.save_for_backward(x)
        # Keep in original dtype (float32 for MPS compatibility)
        z = x.float()  # Changed from .double() for MPS compatibility
        # Use float32-compatible epsilon (1e-7 instead of 1e-15)
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-7).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5
