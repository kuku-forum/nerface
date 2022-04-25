import os
import numpy as np
import torch
from PIL import Image
from models import ConditionalBlendshapePaperNeRFModel, run_one_iter_of_nerf
from preprocessing import load_flame_data, get_embedding_function, get_ray_bundle, meshgrid_xy
from func_tool import img2mse
from tqdm import tqdm
from torchsummary import summary

def main():
    
    # Load dataset
    root = '.\\nerf-pytorch\\nerface_dataset'
    # [N, W, H, C], ([N, 4, 4]), 512, 512, (4,), 14, ([N, 76]), ([N, 4])
    images, poses, H, W, focal, i_split, expressions, bboxs = load_flame_data(basedir= root + '\\person_1_sample', half_res=False, testskip=1 )
    i_train = i_split[0]
    
    
    print("done loading data")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    encode_position_fn = get_embedding_function(num_encoding_functions=10, include_input=True, log_sampling=True)
    encode_direction_fn = get_embedding_function(num_encoding_functions=4, include_input=False, log_sampling=True)

    # Initialize a coarse-resolution model.
    model_coarse = ConditionalBlendshapePaperNeRFModel(
        num_encoding_fn_xyz=10,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=False,
        use_viewdirs=True,
        num_layers=4,
        hidden_size=256,
        include_expression=True
    )
    model_coarse.to(device)
    
    # If a fine-resolution model is specified, initialize it.
    model_fine = ConditionalBlendshapePaperNeRFModel(
        num_encoding_fn_xyz=10,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=False,
        use_viewdirs=True,
        num_layers = 4,
        hidden_size =256,
        include_expression=True
    )
    model_fine.to(device)

    # load GT background and resize
    print("loading GT background to condition on")
    background = Image.open(os.path.join(root + '\\person_1', 'bg', '00050.png'))
    background.thumbnail((H,W))
    background = torch.from_numpy(np.array(background).astype(np.float32)).to(device)
    background = background/255

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    trainable_parameters += list(model_fine.parameters())
    
     # 임의의 latent vector 
    latent_codes = torch.zeros(len(i_train),32, device=device)
    print("initialized latent codes with shape %d X %d" % (latent_codes.shape[0], latent_codes.shape[1]))
    
    trainable_parameters.append(latent_codes)
    latent_codes.requires_grad = True

    optimizer = torch.optim.Adam(
        [{'params':trainable_parameters}, {'params': background, 'lr': 5.0E-4}],
        lr=5.0E-4)

    # Prepare importance sampling maps
    # bounding box를 통해 중요한 영역을 선정
    ray_importance_sampling_maps = []
    p = 0.9
    print("computing boundix boxes probability maps")
    for i in tqdm(i_train):
        bbox = bboxs[i]
        probs = np.zeros((H,W))
        probs.fill(1-p)
        
        # probs[bbox[0]:bbox[1], bbox[2]:bbox[3]] = p
        probs = (1/probs.sum()) * probs
        ray_importance_sampling_maps.append(probs.reshape(-1))

    
    print("Starting loop")
    train_iters = 1000000
    for i in tqdm(range(train_iters)):
        
        model_coarse.train()
        # if model_fine:
        #     model_coarse.train()
        
        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        background_ray_values = None
        
        # 선정된 idx의 img, pos, expression, latent_vector 추출
        # print(i_train[0])
        img_idx = np.random.choice(i_train)
        # print(img_idx)
        img_target = images[img_idx].to(device)
        pose_target = poses[img_idx, :3, :4].to(device)
        expression_target = expressions[img_idx].to(device) # vector
        # print(latent_codes.shape)
        latent_code = latent_codes[img_idx].to(device)
        
        # ray 추출
        # get_ray_bundle
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
        # Cartesian coordinate system?
        coords = torch.stack(meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)), dim=-1)
        coords = coords.reshape((-1, 2))
        
        # Use importance sampling to sample mainly in the bbox with prob p
        # 2048개의 중요 index 추출
        select_inds = np.random.choice(coords.shape[0], size=(2048), replace=False, p=ray_importance_sampling_maps[img_idx])

        select_inds = coords[select_inds]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]

        target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
        background_ray_values = background[select_inds[:, 0], select_inds[:, 1], :]
        
        # color 추출
        rgb_coarse, _, _, rgb_fine, _, _, _ = run_one_iter_of_nerf(
            H, W,
            focal,
            model_coarse,
            model_fine,
            ray_origins,
            ray_directions,
            mode="train",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            expressions = expression_target,
            background_prior=background_ray_values,
            latent_code = latent_code

        )
        target_ray_values = target_s

        # 타겟과 복원한 컬러를 MSE loss 측정
        coarse_loss = torch.nn.functional.mse_loss(rgb_coarse[..., :3], target_ray_values[..., :3])
        fine_loss = torch.nn.functional.mse_loss(rgb_fine[..., :3], target_ray_values[..., :3]) if rgb_fine is not None else 0.0
        
        latent_code_loss = torch.norm(latent_code) * 0.0005
        background_loss = torch.zeros(1, device=device)
        
        loss_total = coarse_loss + fine_loss + latent_code_loss*10 + background_loss         
        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Learning rate updates
        num_decay_steps = 250 * 1000
        lr_new = 5.0E-4 * (0.1 ** (i / num_decay_steps))
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new


        # Validation 현재 구동 안시켜 놓았음
        if (i % 1000 == 0 or i == 999999 and False ):
            model_coarse.eval()

            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                
                loss = 0
                for img_idx in i_val[:2]:
                    img_target = images[img_idx].to(device)

                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(
                        H, W, focal, pose_target
                    )
                    rgb_coarse, _, _, rgb_fine, _, _ , _ = run_one_iter_of_nerf(
                        H, W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        expressions = expression_target,
                        background_prior = background.view(-1,3),
                        latent_code = torch.zeros(32).to(device),
                    )
                    
                    #print("did one val")
                    target_ray_values = img_target
                    coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                    curr_loss, curr_fine_loss = 0.0, 0.0
                    
                    if rgb_fine is not None:
                        curr_fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                        curr_loss = curr_fine_loss
                    else:
                        curr_loss = coarse_loss
                    loss += curr_loss + curr_fine_loss
                    

if __name__ == "__main__":
    main()