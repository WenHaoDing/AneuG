import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.distributions.uniform import Uniform
from models.ghd_reconstruct import GHD_Reconstruct


def wgan_gradient_penalty(D, ghd_real, ghd_fake, 
                          ghd_reconstruct: GHD_Reconstruct, mean, std, use_norm):
    """Calculates the gradient penalty loss for WGAN GP"""
    B = ghd_real.shape[0]
    # Random weight term for interpolation between real and fake samples
    alpha = Uniform(0, 1).sample((B, 1)).to(ghd_real.device)
    # Get random interpolation between real and fake samples
    ghd_interpolate = (alpha * ghd_real + ((1 - alpha) * ghd_fake)).requires_grad_(True)
    data_interpolate = ghd_reconstruct.forward(ghd_interpolate, mean, std, use_norm)
    interpolate_validity = D(data_interpolate)
    fake = Variable(torch.Tensor(B, 1).fill_(1.0).cuda(), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=interpolate_validity,
        inputs=data_interpolate.x,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty