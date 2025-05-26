import sys, logging,random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import optix as ox
import cupy as cp
import time
from torchvision.utils import save_image, make_grid
from pathlib import Path

from tqdm import tqdm
from classes import point_cloud,scene

from optix_raycasting import optix_utils as u_ox
from optix_raycasting import render_optix as r_ox

from scripts import test
from utils import general_utils as utilities
from utils.optim_utils import define_optimizer_manager
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_depth(depth: torch.Tensor, near: float=0.2, far: float=None) -> torch.Tensor:
    """copied and adapted from R3DG"""
    # depth = depth[0].detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()
    colormap = matplotlib.colormaps['turbo']
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = max( near if near else depth.min(), 0)
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]

    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return torch.from_numpy(out_depth).float().cuda().permute(2,0,1)

def train(config, quiet=True):
  mempool = cp.get_default_memory_pool()

  learnable_point_cloud=point_cloud.PointCloud(data_type=config.pointcloud.data_type,device=device)
  opt_scene=scene.Scene(config=config,pointcloud=learnable_point_cloud,train_resolution_scales=config.scene.train_resolution_scales,test_resolution_scales=config.scene.test_resolution_scales,
                      )  

  opt_scene.pointcloud.check_contiguous()

  losses = []
  mean_psnr_train = []
  mean_psnr_test = []
  
  supersampling=(config.training.supersampling_x,config.training.supersampling_y)
  buffer_size=supersampling[0]*supersampling[1]*opt_scene.getTrainCameras()[0].image_height*opt_scene.getTrainCameras()[0].image_width*config.training.max_prim_slice
  hit_prim_idx=cp.zeros((buffer_size),dtype=cp.int32)

  if not quiet:
    print("config.scene.source_path",config.scene.source_path)
    print("Number of points:",opt_scene.pointcloud.positions.shape[0])
    print("Number of iterations:",config.training.n_iters)

  config.training.optimization.position.init_lr = float(opt_scene.cameras_extent)*config.training.optimization.position.init_lr
  config.training.optimization.position.final_lr = float(opt_scene.cameras_extent)*config.training.optimization.position.final_lr
  [optim_manag_positions,optim_manag_rgb,optim_manag_spherical_harmonics,optim_manag_densities,optim_manag_scales,optim_manag_quaternions,optim_manag_sph_gauss_features,optim_manag_bandwidth_sharpness,optim_manag_lobe_axis]=define_optimizer_manager(config.training.optimization,opt_scene.pointcloud,[],[],[],[],[],[],[],[],[])
  opt_scene.pointcloud.setup_optimizers({"xyz":optim_manag_positions,"rgb":optim_manag_rgb,"sh":optim_manag_spherical_harmonics,"density":optim_manag_densities,
                                        "scales":optim_manag_scales,"quaternions":optim_manag_quaternions,
                                        "sph_gauss_features":optim_manag_sph_gauss_features,
                                        "bandwidth_sharpness":optim_manag_bandwidth_sharpness,
                                        "lobe_axis":optim_manag_lobe_axis})
  opt_scene.pointcloud.optim_managers.save_lr()

  cfg_train=config.training

  viewpoint_stack = None

  first_iter=0
  if cfg_train.checkpoint>0:
    first_iter=opt_scene.pointcloud.restore_model(config.training.checkpoint,config.training.checkpoint_folder,config.training.optimization)

  start_time = time.time()

  # setup visualization dir, not part of this by default
  Path(config.save.screenshots, "visualize").mkdir(exist_ok=True, parents=True)
  
  for iter in tqdm(range(first_iter,config.training.n_iters)):
    # Pick a random Camera
    if not viewpoint_stack:
        viewpoint_stack = opt_scene.getTrainCameras(config.scene.train_resolution_scales).copy()

    viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))
    
    gt_image = viewpoint_cam.original_image.cuda()
    
    if iter==first_iter:
      ################# OPTIX #################
      ctx=u_ox.create_context(log=log)
      pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                num_payload_values=1,
                                                num_attribute_values=0,
                                                exception_flags=ox.ExceptionFlags.NONE,
                                                pipeline_launch_params_variable_name="params")
      module_fwd = u_ox.create_module(ctx, pipeline_options,stage="forward")
      program_grps_fwd = u_ox.create_program_groups(ctx, module_fwd)
      pipeline_fwd = u_ox.create_pipeline(ctx, program_grps_fwd, pipeline_options)
      module_bwd = u_ox.create_module(ctx, pipeline_options,stage="backward")
      program_grps_bwd = u_ox.create_program_groups(ctx, module_bwd)
      pipeline_bwd = u_ox.create_pipeline(ctx, program_grps_bwd, pipeline_options)
      with torch.no_grad():
        cp_positions,cp_scales,cp_quaternion,cp_densities=utilities.torch2cupy(opt_scene.pointcloud.positions,
                                                                              opt_scene.pointcloud.get_scale(),opt_scene.pointcloud.get_normalized_quaternion(),
                                                                              opt_scene.pointcloud.densities)
      L1,L2,L3=u_ox.quaternion_to_rotation(cp_quaternion)
      bboxes = u_ox.compute_ellipsoids_bbox(cp_positions,cp_scales,L1,L2,L3,cp_densities)
      gas = u_ox.create_acceleration_structure(ctx, bboxes)

    update = False
    
    settings=r_ox.RenderOptixSettings(ctx,update, gas, program_grps_fwd, pipeline_fwd,
                  program_grps_bwd, pipeline_bwd, viewpoint_cam,
                  config.training.max_prim_slice,opt_scene.pointcloud.num_sph_gauss, iteration=iter, jitter=config.training.jitter, rnd_sample=config.training.rnd_sample,supersampling=(config.training.supersampling_x,config.training.supersampling_y),white_background=config.scene.white_background,
                  hit_prim_idx=hit_prim_idx)
    positions,scales,normalized_quaternions,densities,color_features,sph_gauss_features,bandwidth_sharpness,lobe_axis=opt_scene.pointcloud.get_data(normalized_quaternion=not(cfg_train.normalize_quaternion),normalized_lobe_axis=not(cfg_train.normalize_lobe_axis))
    image, ray_bbox, ray_normals, ray_density, ray_depth=r_ox.RenderOptixFunction.apply(positions,scales,normalized_quaternions,densities,color_features,
                                         sph_gauss_features,bandwidth_sharpness,lobe_axis,
                                         settings)
    image_mean=utilities.reduce_supersampling(viewpoint_cam.image_width,viewpoint_cam.image_height,image,supersampling)
    image_mean=image_mean.permute(2,0,1)

    Ll1 = l1_loss(image_mean, gt_image)
    lambda_ssim = 0.20
    train_loss = (1.0 - lambda_ssim) * Ll1 + lambda_ssim * (1.0 - ssim(image_mean, gt_image))
    
    train_loss.backward()

    torch.cuda.synchronize()
    
    # ----------------------------------------------------------------------------------------------
    # VISUALIZE TRAINING PROGRESS
    # ----------------------------------------------------------------------------------------------
    # opt_scene.getTrainCameras(config.scene.train_resolution_scales).copy()
    # for now use selected random camera
    if iter % 10 == 0:
      
      bbox_mean=utilities.reduce_supersampling(viewpoint_cam.image_width,viewpoint_cam.image_height,ray_bbox,supersampling)
      bbox_mean=bbox_mean.permute(2,0,1)
      

      
      visualization_list = [
        # make_grid requires shape BxCxHxW
        image_mean,
        # target images have shape H,W,C
        gt_image,
        visualize_depth( ray_depth.reshape(viewpoint_cam.image_height, viewpoint_cam.image_width) ),
        # (render_pkg["depth_var"] / 0.001).clamp_max(1).reshape(1, viewpoint_cam.height, viewpoint_cam.width).repeat(3,1,1),
        torch.zeros_like(image_mean),
        ray_density.reshape(1, viewpoint_cam.image_height, viewpoint_cam.image_width).repeat(3,1,1),  # opacity
        ray_normals.reshape(viewpoint_cam.image_height, viewpoint_cam.image_width, 3).permute(2,0,1) * 0.5 + 0.5,
        bbox_mean,
      ]
      
      grid = torch.stack(visualization_list, dim=0)
      grid = make_grid(grid, nrow=4)
      scale = grid.shape[-2] / 800
      grid = torch.nn.functional.interpolate(grid[None], (int(grid.shape[-2]/scale), int(grid.shape[-1]/scale)))[0]
      save_image(grid, Path(config.save.screenshots, "visualize", f"{iter:06d}.png"))
    
    # ----------------------------------------------------------------------------------------------
    
    # print("Maximum norm of the gradient of the positions:",torch.max(torch.norm(opt_scene.pointcloud.positions.grad,dim=1)))

    if opt_scene.pointcloud.positions.grad is not None:
      opt_scene.pointcloud.accumulate_gradient(opt_scene.pointcloud.positions.grad)
      opt_scene.pointcloud.accumulate_gradient_gaussians_not_visible(opt_scene.pointcloud.positions.grad)

    opt_scene.pointcloud.optim_managers.step()
    opt_scene.pointcloud.optim_managers.zero_grad()

    opt_scene.pointcloud.optim_managers.save_lr()

    losses.append(train_loss.item())
    
    if iter%1000==0 and iter>0: 
      opt_scene.pointcloud.check_required_grad()
      if not quiet:
        tqdm.write(f"Number points : {len(opt_scene.pointcloud.positions)}")

    if (iter>0) and (iter%500==0):
      if not quiet:
        # opt_scene.pointcloud.save_pt_epoch(iter,folder_tensors=config.save.models)
        opt_scene.pointcloud.save_model(iter,config.save.models)
        print("Number gaussians not visible deleted : ", len(opt_scene.pointcloud.densities)-torch.sum(opt_scene.pointcloud.num_accum_gnv>0).item())
      #if (iter==config.training.n_iters-1):
      # opt_scene.pointcloud.delete_gaussians_not_seen()
      opt_scene.pointcloud.reset_gradient_accum_gaussians_not_visible()
    
    # Densification
    if iter < cfg_train.densify_until_iter and iter>0:
      if iter > cfg_train.densify_from_iter and iter % cfg_train.densification_interval == 0:
        opt_scene.pointcloud.densify_and_prune(cfg_train.densify_grad_threshold, u_ox.SIGMA_THRESHOLD, opt_scene.cameras_extent, quiet=quiet)
        mempool.free_all_blocks()      
    if cfg_train.unlock_color_features:     
      unlock_freq=cfg_train.unlock_freq
      if (iter>0) and (iter%unlock_freq==0)and iter<=(config.training.limit_degree_tot*unlock_freq):
        degree_unlock=iter//unlock_freq
        if degree_unlock<=config.training.limit_degree_sh:
          opt_scene.pointcloud.unlock_spherical_harmonics((degree_unlock+1)**2)
      
      if iter%unlock_freq==0 and iter<=(config.training.limit_degree_tot*unlock_freq):
        degree_unlock=iter//unlock_freq
        if degree_unlock>config.training.limit_degree_sh:
          opt_scene.pointcloud.unlock_spherical_gaussians(2*degree_unlock+1)

      if config.training.limit_degree_sh==-1:
        with torch.no_grad():
          #Ahnilate the effect of rgb colors
          opt_scene.pointcloud.rgb*=0.0

    if (iter%100==0):
      opt_scene.pointcloud.compute_3D_filter(cameras=opt_scene.getTrainCameras())

    #Force gaussian scale to be superior to log(0.006)
    with torch.no_grad():
      opt_scene.pointcloud.densities[opt_scene.pointcloud.densities<0]=0
      if cfg_train.normalize_quaternion:
        opt_scene.pointcloud.normalize_quaternion()
      if opt_scene.pointcloud.num_sph_gauss>0 and cfg_train.normalize_lobe_axis:
        opt_scene.pointcloud.normalize_lobe_axis()
      if not quiet:
        if iter%1000==0:
          print("Min minimum size gaussian: ", torch.min(opt_scene.pointcloud.filter_3D))
          print("Max minimum size gaussian: ", torch.max(opt_scene.pointcloud.filter_3D))
          print("Min gaussian scale: ", torch.min(opt_scene.pointcloud.scales))
          print("Max gaussian scale: ", torch.max(opt_scene.pointcloud.scales))
      mask = opt_scene.pointcloud.scales<opt_scene.pointcloud.filter_3D.repeat(1,3)
      opt_scene.pointcloud.scales[mask] = opt_scene.pointcloud.filter_3D.repeat(1,3)[mask]
      #opt_scene.pointcloud.scales[opt_scene.pointcloud.scales<-6.5]=-6.5
      opt_scene.pointcloud.scales[opt_scene.pointcloud.scales>cfg_train.max_scale]=cfg_train.max_scale

    ############################################################################################################
    ################################ Saving and testing part ##################################################
    ############################################################################################################
    if (not quiet and iter%1000==0) or (iter==config.training.n_iters-1):
      #Save the model
      if not quiet:
        opt_scene.pointcloud.save_model(iter,config.save.models)

      if (iter==config.training.n_iters-1):
        end_time=time.time()
        training_time=end_time-start_time
        # opt_scene.pointcloud.save_model(iter,config.save.models)
      tqdm.write(f"[ITER]: {iter} Number points : {len(opt_scene.pointcloud.positions)}")

      ## Train
      PSNR_list_train_scales=[]
      for scale in config.scene.train_resolution_scales:
        PSNR_list_train,gt_images_list,images_list=test.inference(opt_scene.pointcloud,opt_scene.getTrainCameras([scale]),config.training.max_prim_slice,rnd_sample=config.training.rnd_sample,supersampling=(config.training.supersampling_x,config.training.supersampling_y), white_background=config.scene.white_background)
        PSNR_list_train_scales.append(np.mean(PSNR_list_train))
        tqdm.write(f"[ITER]: {iter} \tPSNR Train scale {1.0/scale} (mean): {np.mean(PSNR_list_train):.3f}")
        if not quiet:
          tqdm.write(f"[ITER]: {iter} \tPSNR Train scale {1.0/scale} (min): {np.min(PSNR_list_train):.3f} \tIndex image {np.argmin(PSNR_list_train)}")
          tqdm.write(f"[ITER]: {iter} \tPSNR Train scale {1.0/scale} (max): {np.max(PSNR_list_train):.3f} \tIndex image {np.argmax(PSNR_list_train)}")
        for i in range(len(images_list)):
          plt.imsave(config.save.screenshots+"/train/"+"iter"+str(iter)+"scale"+str(1.0/scale)+"_"+str(i)+"_pred.png",images_list[i])
      
      ## Test
      if config.scene.eval:
        PSNR_list_test_scales=[]
        SSIM_list_test_scales=[]
        LPIPS_list_test_scales=[]
        for scale in config.scene.test_resolution_scales:
          PSNR_list_test,gt_images_list,images_list=test.inference(opt_scene.pointcloud,opt_scene.getTestCameras([scale]),config.training.max_prim_slice,rnd_sample=config.training.rnd_sample,supersampling=(config.training.supersampling_x,config.training.supersampling_y), white_background=config.scene.white_background)
          PSNR_list_test_scales.append(np.mean(PSNR_list_test))
          ##
          SSIM_test,LPIPS_test=0,0
          for i in range(len(images_list)):
            permute_images=torch.tensor(images_list[i].transpose(2,0,1),dtype=torch.float32,device=device)[None,...]
            permute_gt_images=torch.tensor(gt_images_list[i].transpose(2,0,1),dtype=torch.float32,device=device)[None,...]
            SSIM_test+=ssim(permute_images, permute_gt_images).item()
            LPIPS_test+=lpips(permute_images, permute_gt_images, net_type='vgg').item()
          SSIM_list_test_scales.append(SSIM_test/len(images_list))
          LPIPS_list_test_scales.append(LPIPS_test/len(images_list))
          tqdm.write(f"[ITER]: {iter} \tSSIM Test scale {1.0/scale} (mean): {SSIM_test/len(images_list):.3f}")
          tqdm.write(f"[ITER]: {iter} \tLPIPS Test scale {1.0/scale} (mean): {LPIPS_test/len(images_list):.3f}")
          ##
          tqdm.write(f"[ITER]: {iter} \tPSNR Test scale {1.0/scale} (mean): {np.mean(PSNR_list_test):.3f}")
          if not quiet:
            tqdm.write(f"[ITER]: {iter} \tPSNR Test scale {1.0/scale} (min): {np.min(PSNR_list_test):.3f} \tIndex image {np.argmin(PSNR_list_test)}")
            tqdm.write(f"[ITER]: {iter} \tPSNR Test scale {1.0/scale} (max): {np.max(PSNR_list_test):.3f} \tIndex image {np.argmax(PSNR_list_test)}")
          for i in range(len(images_list)):
            plt.imsave(config.save.screenshots+"/test/"+"iter"+str(iter)+"scale"+str(1.0/scale)+"_"+str(i)+"_pred.png",images_list[i])

      #Save PSNR and SSIM in a file
      with open(config.save.metrics+'/PSNR_SSIM.txt', 'a') as f:
        f.write(config.scene.source_path+"\n")
        if (iter==config.training.n_iters-1):
          f.write("Training time: "+str(training_time)+"\n")
        f.write(" Iteration: "+str(iter)+"\n")
        f.write("Number points: "+ str(len(opt_scene.pointcloud.positions))+"\n")
        ##Train
        for i,scale in enumerate(config.scene.train_resolution_scales):
          f.write("PSNR Train scale "+str(1.0/scale)+" (mean): "+str(PSNR_list_train_scales[i])+"\n")
        mean_psnr_train.append(np.mean(PSNR_list_train_scales))
        ##Test
        if config.scene.eval:
          for i,scale in enumerate(config.scene.test_resolution_scales):
            f.write("PSNR Test scale "+str(1.0/scale)+" (mean): "+str(PSNR_list_test_scales[i])+"\n")
          ## SSIM and LPIPS
          for i,scale in enumerate(config.scene.test_resolution_scales):
            f.write("SSIM Test scale "+str(1.0/scale)+" (mean): "+str(SSIM_list_test_scales[i])+"\n")
            f.write("LPIPS Test scale "+str(1.0/scale)+" (mean): "+str(LPIPS_list_test_scales[i])+"\n")
          ##
          if len(config.scene.train_resolution_scales)>1:
            f.write("PSNR Train multiscale(mean): "+str(np.mean(PSNR_list_train_scales))+"\n")
          if len(config.scene.test_resolution_scales)>1:
            f.write("PSNR Test multiscale(mean): "+str(np.mean(PSNR_list_test_scales))+"\n")
          f.write("\n")
          mean_psnr_test.append(np.mean(PSNR_list_test_scales))
  ############################################################################################################

  config.training.optimization.position.init_lr = config.training.optimization.position.init_lr/float(opt_scene.cameras_extent)
  config.training.optimization.position.final_lr = config.training.optimization.position.final_lr/float(opt_scene.cameras_extent)
  if config.scene.eval:
    return mean_psnr_test[-1],opt_scene
  else:
    return mean_psnr_train[-1],opt_scene