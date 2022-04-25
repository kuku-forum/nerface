# Dynamic  Neural Radiance  Fields for Monocular  4D  Facial  Avatar  Reconstruction

## 📌 Brief Information

> Dead line: `2022.04.27 수`
>
> Tech stack:  `pytorch 1.7` `python 3.7` `numpy 1.21`
>
> IDE: `vscode`, `jupyter notebook`
>
> Environment: `GPU-titan X(vram: 12GB)`,` Memory-32GB`, `Window`



## 📌 Summary

> Nerf 모델 및 저자가 공개한 Open source에서 핵심 `Module` 추출
>
> `Validation dataset`는 `load `하지 않았으며 `train dataset` 을 통해 학습이 진행되도록 구현함
>
> 핵심 모듈: `Nerf Model`, `Positional Encoding`, `Ray Extraction`, `volume_render_radiance_field`



## 📌 Module

#### ✍ model_fine, model_coarse: 메인 학습 모델

```python
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
```

```python
ConditionalBlendshapePaperNeRFModel(
  (layers_xyz): ModuleList(
    (0): Linear(in_features=171, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=256, bias=True)
    (2): Linear(in_features=256, out_features=256, bias=True)
    (3): Linear(in_features=427, out_features=256, bias=True)
    (4): Linear(in_features=256, out_features=256, bias=True)
    (5): Linear(in_features=256, out_features=256, bias=True)
  )
  (fc_feat): Linear(in_features=256, out_features=256, bias=True)
  (fc_alpha): Linear(in_features=256, out_features=1, bias=True)
  (layers_dir): ModuleList(
    (0): Linear(in_features=280, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=128, bias=True)
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): Linear(in_features=128, out_features=128, bias=True)
  )
  (fc_rgb): Linear(in_features=128, out_features=3, bias=True)
)
```



#### ✍ Positional_encoding:  frequency를 통한 차원 확장

``` python
def positional_encoding(tensor, num_encoding_functions, include_input=True, log_sampling=True):
    
    '''
    Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    
    torch.linspace(start, end, steps, *, out=None, dtype=None, 
                    layout=torch.strided, device=None, requires_grad=False) → Tensor
    '''
    
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    
    if log_sampling:
        # frequency_bands -> tensor([1., 2., 4., 8., 16., 32., 64., 128., 256., 512.])
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        # frequency_bands -> tensor([1.0000,  57.7778, 114.5556, 171.3333, 228.1111, 284.8889, 341.6667, 398.4445, 455.2222, 512.0000])
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)
```



#### ✍ get_ray_bundle: pose, intrinsics를 통한 ray_direction, ray_center 추출

```` python
def get_ray_bundle(height: int, width: int, intrinsics, tform_cam2world: torch.Tensor, center = [0.5,0.5]):
    '''
    Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length CHANGED TO INTRINSICS (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    guy: changed focal length to array of fx fy
    intrinsics = [fx fy cx cy] where cx cy in [0,1] relative to image size
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    '''
    
    ii, jj = meshgrid_xy(
        torch.arange(width, dtype=tform_cam2world.dtype,
                     device=tform_cam2world.device).to(tform_cam2world),
        torch.arange(height, dtype=tform_cam2world.dtype, device=tform_cam2world.device),
    )

    if intrinsics.shape<(4,):
        intrinsics = [intrinsics, intrinsics, 0.5, 0.5]
        
    directions = torch.stack(
        [
            (ii - width * intrinsics[2]) / intrinsics[0],
            -(jj - height * intrinsics[3]) / intrinsics[1],
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions


def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor):
    '''
    Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    '''
    
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)
````



#### ✍ volume_render_radiance_field: radiance field에서 Ray의 sampling된 color, weight를 합하여 반환

```` python
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
````



## 📚What I got



#### 1. Radiance field에 대한 이해

> 해당 논문은 `Nerf`를 기반으로한 `Human Portrait Video Synthesis` 방법론 이다.
>
> 여기서 핵심은 2D image를 Radiance field에 매핑하고 이를 Rendering 하여 novel view, pose, expressions를 가진 2D image를 Reconstruction 한다.
>
> 여기서 미리 제공된 Pose와 Intrinsics를 통해 Radiance Field로 매핑이 가능하다는 것을 확인할 수 있었다.

#### 2. TO DO

> 1. Rendering에 대한 연구가 필요하다. 
>
>    하지만 model weight 파일의 부재로 제공된 수식의 효용성 파악이 어렵다. 
>    따라서 추가 학습을 통해 wieght 파일을 추출하고 디버깅의 필요성이 대두된다.
>
> 2. 저자가 제공하는 sample dataset에서 Pose, Bbox, Intrinsics, Expression Data에 대한 정보가 부족하다.
>    특히 Bbox 영역의 rendering 과정에서 reconstruction이 얼마나 명확하게 되었는지 정확한 확인이 어렵다.
>    이를 추출하기 위해선 두가지 방법론이 조사할 필요가 있는 것으로 판단된다.
>
>    1. 데이터 촬영 시 데이터 생성
>    2. Face2Face 및 colmap 기반 데이터 추출



## Reference

```
@InProceedings{Gafni_2021_CVPR,
    author    = {Gafni, Guy and Thies, Justus and Zollh{\"o}fer, Michael and Nie{\ss}ner, Matthias},
    title     = {Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {8649-8658},
    github    = {https://github.com/gafniguy/4D-Facial-Avatars}
}

@inproceedings{mildenhall2020nerf,
	title	={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
	autho	r={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
	year	={2020},
    booktitle	={ECCV},
    github    = {https://github.com/krrish94/nerf-pytorch}
}
```

