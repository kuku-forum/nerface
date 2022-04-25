import torch
import json
import os
import imageio
import numpy as np
import torch
from tqdm import tqdm

def load_flame_data(basedir, half_res=False, testskip=1, debug=False, expressions=True,load_frontal_faces=False, load_bbox=True, test=False):
    print("starting data loading")
    
    splits = ["train"]
    select_type = 'train'
    # metas: train, val, test json data load
    # json data: camera_angle_x, frames[file_path, bbox, transform_matrix, expression], intrinsics
    metas = {}
    with open(os.path.join(basedir, f"transforms_train.json"), "r") as fp:
        metas[select_type] = json.load(fp)
    
    all_imgs = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    counts = [0]
    
    # meta: train json
    meta = metas[select_type]
    imgs = []
    poses = []
    expressions = []
    bboxs = []
    skip = testskip

    # frame 이미지 순서대로 laod
    for frame in tqdm(meta["frames"][::skip]):
        try:
            # stack img, pose, expressions, bounding-box
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
            expressions.append(np.array(frame["expression"]))
            
            if load_bbox:
                if "bbox" not in frame.keys():
                    bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                else:
                    bboxs.append(np.array(frame["bbox"]))
        except:
            pass
            
    # 데이터 ndarray 변환 및 정규화(normalization)
    imgs = (np.array(imgs) / 255.0).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    expressions = np.array(expressions).astype(np.float32)
    bboxs = np.array(bboxs).astype(np.float32)

    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)
    all_expressions.append(expressions)
    all_bboxs.append(bboxs)
    
    # splits별 frame 개수 추가
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    
    # 프레임별 모든 데이터를 concatenate
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0)
    
    # H/W, camera_angle_x 추출
    H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta["camera_angle_x"])
    # focal = 초점거리 계산
    # focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    
    # 카메라 내부 파라미터
    # fx & fy: focal lengths, cx & cy : principal point.
    # fx는 초점거리(렌즈중심에서 이미지 센서까지의 거리)가 가로 
    # 방향 셀 크기(간격)의 몇 배인지를 나타내고 fy는 초점거리가 
    # 세로 방향 센서 셀 크기(간격)의 몇 배인지를 나타냅니다
    #  cx, cy는 카메라 렌즈의 중심
    intrinsics = np.array(meta["intrinsics"])

    imgs = [torch.from_numpy(imgs[i]) for i in range(imgs.shape[0])]
    imgs = torch.stack(imgs, 0)
    
    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    bboxs[:,0:2] *= H
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")

    return imgs, poses, int(H), int(W), intrinsics, i_split, expressions, bboxs



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


def get_embedding_function(num_encoding_functions, include_input=True, log_sampling=True):
    # Returns a lambda function that internally calls positional_encoding.
    return lambda x: positional_encoding(x, num_encoding_functions, include_input, log_sampling)
    
    


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
        torch.arange(width, dtype=tform_cam2world.dtype, device=tform_cam2world.device).to(tform_cam2world),
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