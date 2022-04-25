import torch
import json
import os
import imageio
import numpy as np
import torch
from func_tool import get_minibatches, sample_pdf


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    background_prior = None,
    latent_code = None,
    ray_dirs_fake = None
):
    cfg_num_coarse = 64
    cfg_lindisp = False
    cfg_perturb = True
    cfg_numfine = 64
    cfg_chunksize = 2048 if mode == 'train' else 65536
    cfg_radiance_field_noise_std = 0.1 if mode == 'train' else 0.
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6].clone() 
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]
    
    t_vals = torch.linspace(
        0.0,
        1.0,
        cfg_num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand([num_rays, cfg_num_coarse])

    # Get intervals between samples.
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
    lower = torch.cat((z_vals[..., :1], mids), dim=-1)
    
    # Stratified samples in those intervals.
    t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
    z_vals = lower + (upper - lower) * t_rand
        
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
    # Uncomment to dump a ply file visualizing camera rays and sampling points
    #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy())
    ray_batch[...,3:6] = ray_dirs_fake[0][...,3:6] 

    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        cfg_chunksize,
        encode_position_fn,
        encode_direction_fn,
        expressions,
        latent_code
    )
    
    # make last RGB values of each ray, the background
    # return rgb_map, disp_map, acc_map, weights, depth_map
    if background_prior is not None:
        radiance_field[:,-1,:3] = background_prior

    (
        rgb_coarse, disp_coarse, acc_coarse, weights, _,
    ) = volume_render_radiance_field(
        radiance_field, z_vals, rd,
        radiance_field_noise_std=cfg_radiance_field_noise_std,
        white_background=False,
        background_prior=background_prior
    )

    rgb_fine, disp_fine, acc_fine = None, None, None
    
    # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid,
        weights[..., 1:-1],
        cfg_numfine,
    )
    z_samples = z_samples.detach()

    z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
    # pts -> (N_rays, N_samples + N_importance, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(
        model_fine,
        pts,
        ray_batch,
        cfg_chunksize,
        encode_position_fn,
        encode_direction_fn,
        expressions,
        latent_code
    )
    # make last RGB values of each ray, the background
    if background_prior is not None:
        radiance_field[:, -1, :3] = background_prior

    # added use of weights
    # return rgb_map, disp_map, acc_map, weights, surface_depth
    rgb_fine, disp_fine, acc_fine, weights, depth_fine = volume_render_radiance_field( 
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=cfg_radiance_field_noise_std,
        white_background=False,
        background_prior=background_prior
    )

    # changed last return val to fine_weights
    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, weights[:,-1] 



def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn, expressions = None, latent_code = None):
  
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    
    if expressions is None:
        preds = [network_fn(batch) for batch in batches]
    elif latent_code is not None:
        preds = [network_fn(batch, expressions, latent_code) for batch in batches]
    else:
        preds = [network_fn(batch, expressions) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )

    del embedded, input_dirs_flat
    return radiance_field


def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    background_prior = None
):
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)


    if background_prior is not None:
        rgb = torch.sigmoid(radiance_field[:, :-1, :3])
        rgb = torch.cat((rgb, radiance_field[:, -1, :3].unsqueeze(1)), dim=1)
    else:
        rgb = torch.sigmoid(radiance_field[..., :3])

    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    sigma_a[:,-1] += 1e-6 
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    surface_depth = None
    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values
    depth_map = depth_map.sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, surface_depth



def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod