import torch

def simple_guidance_fn(out, cfg):
    uncondition, condtion = out.chunk(2, dim=0)
    out = uncondition + cfg * (condtion - uncondition)
    return out

def guidance_fn_with_rescale(out, cfg, rescale_factor=0.7):
    """
    Apply classifier-free guidance (CFG) to the model's raw output, with variance rescaling (rescale_cfg).

    Args:
        out (torch.Tensor): the model's raw output, its unconditional and conditional halves.
        cfg (float): the guidance scale.
        rescale_factor (float): the rescale factor; common values are 0.5 to 0.8, and 0.7 is a good start.

    Returns:
        torch.Tensor: the output after CFG and rescale_cfg.
    """
    uncondition, condition = out.chunk(2, dim=0)

    guided_out = uncondition + cfg * (condition - uncondition)

    std_condition = torch.std(condition, dim=(1,2,3), keepdim=True)
    std_guided = torch.std(guided_out, dim=(1,2,3), keepdim=True)
    
    scale = std_condition / (std_guided + 1e-6)
    print(scale.mean())
    rescaled_out = guided_out * (scale * rescale_factor + 1.0 * (1.0 - rescale_factor))
    return rescaled_out

def c3_guidance_fn(out, cfg):
    # guidance function in DiT/SiT, seems like a bug not a feature?
    uncondition, condtion = out.chunk(2, dim=0)
    out = condtion
    out[:, :3] = uncondition[:, :3] + cfg * (condtion[:, :3] - uncondition[:, :3])
    return out