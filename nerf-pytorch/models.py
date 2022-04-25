import torch
import json
import os
import imageio
import numpy as np
import torch
from rendering import get_minibatches, predict_and_render_radiance

    
class ConditionalBlendshapePaperNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=10,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True,
        latent_code_dim=32

    ):
        super(ConditionalBlendshapePaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        # positional encoding으로 인한 dim size 계산
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz # 63
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir # 24
        
        # 제시된 expression dim size
        self.dim_expression = include_expression # 76
        self.dim_latent_code = latent_code_dim # 32

        self.use_viewdirs = use_viewdirs
        
        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, 256))
        
        for i in range(1, 6):
            # skip connection in 4th layer
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
                
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
            
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x,  expr=None, latent_code=None):
        # torch.Size([2048, 87]) torch.Size([76]) torch.Size([32])
        # print(x.shape, expr.shape, latent_code.shape)
        
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        # print(xyz.shape, dirs.shape)
        # x = xyz # self.relu(self.layers_xyz[0](xyz))
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1)
            x = initial
            
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
            
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
            
        x = self.relu(x)
        
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
            
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    background_prior=None,
    latent_code = None,
    ray_directions_ablation = None
):
    viewdirs = None
    
    # use_viewdirs: True
    # Provide ray directions as input
    viewdirs = ray_directions
    viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
    viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]] # to return fine depth map
    
    ro = ray_origins.view((-1, 3))
    rd = ray_directions.view((-1, 3))
    ray_directions_ablation = ray_directions
    rd_ablations = ray_directions_ablation.view((-1, 3))
    
    cfg_near, cfg_far = 0.2, 0.8
    near = cfg_near * torch.ones_like(rd[..., :1])
    far = cfg_far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    rays_ablation = torch.cat((ro, rd_ablations, near, far), dim=-1)
    
    viewdirs = None  
    # Provide ray directions as input
    viewdirs = ray_directions_ablation
    viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
    viewdirs = viewdirs.view((-1, 3))
    
    cfg_chunksize = 2048 if mode == 'train' else 65536

    # minibatch size(chunksize) 만큼 ray 가져오기
    batches_ablation = get_minibatches(rays_ablation, chunksize=cfg_chunksize)
    batches = get_minibatches(rays, chunksize=cfg_chunksize)
    assert(batches[0].shape == batches[0].shape)
    background_prior = get_minibatches(background_prior, chunksize=cfg_chunksize) if background_prior is not None else background_prior
    
    # pred: [rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, weights[:,-1]]
    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            expressions = expressions,
            background_prior = background_prior[i] if background_prior is not None else background_prior,
            latent_code = latent_code,
            ray_dirs_fake = batches_ablation
        )
        for i,batch in enumerate(batches)
    ]
    print(pred[0].shape)

    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)

