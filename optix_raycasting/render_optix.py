from utils import general_utils as utilities
# from utils import optix_forward_utils as u_ox_fwd
# from utils import optix_backward_utils as u_ox_bwd
from optix_raycasting import optix_utils as u_ox
import torch
import cupy as cp
import numpy as np

from torch.utils.dlpack import to_dlpack

with open('optix_raycasting/cuda_train/forward/vec_math.h') as f:
    code = f.read()
with open('utils/sh_utils.cu') as f:
    code += f.read()
cupy_module = cp.RawModule(code=code)
cuda_backward_sh = cupy_module.get_function('backward_sh')
cuda_forward_sh = cupy_module.get_function('forward_sh')

def compute_cupy_rgb(camera_center,cupy_positions,cupy_color_features,
                          cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,num_sph_gauss,
                          degree_sh):
  with torch.no_grad():
    num_points=len(cupy_positions)
    block_size=128
    num_blocks=(num_points+block_size-1)//block_size
    cupy_camera_center=cp.fromDlpack(to_dlpack(camera_center.contiguous()))
    cp_colors_rgb=cp.zeros((num_points,3),dtype=cp.float32)
    cuda_forward_sh((num_blocks,),(block_size,),
                      (cupy_camera_center,cupy_positions,cupy_color_features, cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,
                      num_sph_gauss,
                      degree_sh,num_points,cp_colors_rgb))
    return cp_colors_rgb
  
def compute_cupy_sh_grad(camera_center,cupy_positions,cupy_color_features,
                          cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,num_sph_gauss,
                          degree_sh,cp_color_features_grad):
  with torch.no_grad():
    num_points=len(cupy_positions)
    dL_dsh=cp.zeros((num_points*3*(degree_sh+1)**2),dtype=cp.float32)
    dL_dpos=cp.zeros((num_points,3),dtype=cp.float32)
    dL_dsph_gauss=cp.zeros((num_points*num_sph_gauss*3),dtype=cp.float32)
    dL_dbandwidth_sharpness=cp.zeros((num_points*num_sph_gauss),dtype=cp.float32)
    dL_dlobe_axis=cp.zeros((num_points*num_sph_gauss*3),dtype=cp.float32)
    block_size=128
    num_blocks=(num_points+block_size-1)//block_size
    cupy_camera_center=cp.fromDlpack(to_dlpack(camera_center.contiguous()))
    cuda_backward_sh((num_blocks,),(block_size,),
                      (cupy_camera_center,cupy_positions,cupy_color_features, cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,
                      num_sph_gauss,
                      degree_sh,num_points,cp_color_features_grad,dL_dsh,dL_dpos,
                      dL_dsph_gauss,dL_dbandwidth_sharpness,dL_dlobe_axis))
    return dL_dsh, dL_dpos,dL_dsph_gauss,dL_dbandwidth_sharpness,dL_dlobe_axis
  
def unnormalized_quaternion_grad(normalized_quaternions_grad,quaternions):
    sum2 = torch.sum(quaternions**2,dim=1)
    invsum32 = 1.0 / (torch.sqrt(sum2) * sum2)

    quaternions_grad = torch.zeros_like(quaternions)
    quaternions_grad[:,0] = ((sum2 - quaternions[:,0] * quaternions[:,0]) * normalized_quaternions_grad[:,0] - quaternions[:,1] * quaternions[:,0] * normalized_quaternions_grad[:,1] - quaternions[:,2] * quaternions[:,0] * normalized_quaternions_grad[:,2]- quaternions[:,3] * quaternions[:,0] * normalized_quaternions_grad[:,3]) * invsum32
    quaternions_grad[:,1] = (-quaternions[:,0] * quaternions[:,1] * normalized_quaternions_grad[:,0] + (sum2 - quaternions[:,1] * quaternions[:,1]) * normalized_quaternions_grad[:,1] - quaternions[:,2] * quaternions[:,1] * normalized_quaternions_grad[:,2]- quaternions[:,3] * quaternions[:,1] * normalized_quaternions_grad[:,3]) * invsum32
    quaternions_grad[:,2] = (-quaternions[:,0] * quaternions[:,2] * normalized_quaternions_grad[:,0] - quaternions[:,1] * quaternions[:,2] * normalized_quaternions_grad[:,1] + (sum2 - quaternions[:,2] * quaternions[:,2]) * normalized_quaternions_grad[:,2]- quaternions[:,3] * quaternions[:,2] * normalized_quaternions_grad[:,3]) * invsum32
    quaternions_grad[:,3] = (-quaternions[:,0] * quaternions[:,3] * normalized_quaternions_grad[:,0] - quaternions[:,1] * quaternions[:,3] * normalized_quaternions_grad[:,1] - quaternions[:,2] * quaternions[:,3] * normalized_quaternions_grad[:,2] + (sum2 - quaternions[:,3] * quaternions[:,3]) * normalized_quaternions_grad[:,3]) * invsum32
    return quaternions_grad


    
class RenderOptixSettings:
    def __init__(self, context,update, gas, program_grps_fwd, pipeline_fwd,
                 program_grps_bwd, pipeline_bwd, viewpoint_cam,
                 max_prim_slice,num_sph_gauss,iteration,jitter,rnd_sample,supersampling,white_background,
                 hit_prim_idx):
        self.context= context
        self.update = update
        self.gas = gas
        self.program_grps_fwd = program_grps_fwd
        self.pipeline_fwd = pipeline_fwd
        self.program_grps_bwd = program_grps_bwd
        self.pipeline_bwd = pipeline_bwd
        self.viewpoint_cam = viewpoint_cam
        self.max_prim_slice = max_prim_slice
        self.num_sph_gauss=num_sph_gauss
        self.iteration=iteration
        self.jitter=jitter
        self.rnd_sample=rnd_sample
        self.supersampling=supersampling
        self.white_background = white_background
        self.hit_prim_idx=hit_prim_idx


class RenderOptixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,positions,scales,normalized_quaternions,densities,color_features,
                sph_gauss_features,bandwidth_sharpness,lobe_axis,settings):
      cp_positions,cp_scales,cp_quaternions,cp_densities,cp_color_features,cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis=utilities.torch2cupy(
                                positions,
                                scales,
                                normalized_quaternions,
                                densities,
                                color_features.reshape(-1),
                                sph_gauss_features.reshape(-1),
                                bandwidth_sharpness.reshape(-1),
                                lobe_axis.reshape(-1))
      L1,L2,L3=u_ox.quaternion_to_rotation(cp_quaternions)
      bboxes = u_ox.compute_ellipsoids_bbox(cp_positions,cp_scales,L1,L2,L3,cp_densities)
      bb_min=bboxes[:,:3].min(axis=0)
      bb_max=bboxes[:,3:].max(axis=0)
      if settings.update:
        u_ox.update_acceleration_structure(settings.gas, bboxes)
      else:
        settings.gas= u_ox.create_acceleration_structure(settings.context, bboxes)
      sbt_fwd = u_ox.create_sbt(settings.program_grps_fwd, cp_positions,cp_scales,cp_quaternions)
      order_sh=int(np.sqrt(color_features.shape[2]).item()-1)

      cp_rgb=compute_cupy_rgb(settings.viewpoint_cam.camera_center,cp_positions,cp_color_features,
                cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,settings.num_sph_gauss,
                order_sh)
      cp_ray_colors, cp_ray_bbox, cp_ray_normals, cp_ray_density, cp_ray_depth = u_ox.launch_pipeline_forward(settings.pipeline_fwd,sbt_fwd, settings.gas,bb_min,bb_max,settings.viewpoint_cam,
                                            cp_densities,cp_rgb,
                                            cp_positions,cp_scales,cp_quaternions,
                                            max_prim_slice=settings.max_prim_slice,iteration=settings.iteration,
                                            jitter=settings.jitter, rnd_sample=settings.rnd_sample, supersampling=settings.supersampling,
                                            white_background=settings.white_background, hit_prim_idx=settings.hit_prim_idx)

      [ray_colors,rgb, ray_bbox, ray_normals, ray_density, ray_depth ]=utilities.cupy2torch(cp_ray_colors,cp_rgb, cp_ray_bbox, cp_ray_normals, cp_ray_density, cp_ray_depth )

      ctx.settings=settings
      ctx.save_for_backward(positions,scales,normalized_quaternions,densities,color_features,
                            sph_gauss_features,bandwidth_sharpness,lobe_axis,rgb,ray_colors.clone())
      ctx.bb_min=bb_min
      ctx.bb_max=bb_max
      ctx.order_sh=order_sh
      return ray_colors, ray_bbox, ray_normals, ray_density, ray_depth


    @staticmethod
    def backward(ctx, dloss_dray_colors, *args):
      if not dloss_dray_colors.is_contiguous():
        dloss_dray_colors=dloss_dray_colors.contiguous()

      settings=ctx.settings
      bb_min=ctx.bb_min
      bb_max=ctx.bb_max
      order_sh=ctx.order_sh
      positions,scales,normalized_quaternions,densities,color_features,sph_gauss_features,bandwidth_sharpness,lobe_axis,rgb,ray_colors=ctx.saved_tensors
      cp_positions,cp_scales,cp_quaternions,cp_densities,cp_color_features,cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,cp_rgb,cp_ray_colors,cp_dloss_dray_colors=utilities.torch2cupy(
                          positions,
                          scales,
                          normalized_quaternions,
                          densities,
                          color_features.reshape(-1),
                          sph_gauss_features.reshape(-1),
                          bandwidth_sharpness.reshape(-1),
                          lobe_axis.reshape(-1),
                          rgb,
                          ray_colors,
                          dloss_dray_colors)
      sbt_bwd = u_ox.create_sbt(settings.program_grps_bwd, cp_positions,cp_scales,cp_quaternions)
      cp_densities_grad,cp_color_features_grad,cp_positions_grad,cp_scales_grad,cp_quaternions_grad= u_ox.launch_pipeline_backward(
                                                                            settings.pipeline_bwd, sbt_bwd, settings.gas,bb_min,bb_max,
                                                                            settings.viewpoint_cam,
                                                                            cp_densities,cp_rgb,
                                                                            cp_positions,cp_scales,cp_quaternions,
                                                                            cp_ray_colors,cp_dloss_dray_colors,
                                                                            max_prim_slice=settings.max_prim_slice,iteration=settings.iteration,
                                                                            jitter=settings.jitter, rnd_sample=settings.rnd_sample,
                                                                            supersampling=settings.supersampling, 
                                                                            hit_prim_idx=settings.hit_prim_idx
                                                                            )

      cp_sh_grad, dL_dpos_dir,dL_dsph_gauss,dL_dbandwidth_sharpness,dL_dlobe_axis=compute_cupy_sh_grad(settings.viewpoint_cam.camera_center,
                                                           cp_positions,cp_color_features,
                                                           cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,settings.num_sph_gauss,
                                                           order_sh,cp_color_features_grad)
      
      cp_positions_grad+=dL_dpos_dir
      cp_sh_grad=cp_sh_grad.reshape(cp_positions.shape[0],3,(order_sh+1)**2)
      densities_grad,color_features_grad,positions_grad, scales_grad, normalized_quaternions_grad,sph_gauss_feat_grad,bandwidth_sharpness_grad,lobe_axis_grad=utilities.cupy2torch(cp_densities_grad,
                                                                                          cp_sh_grad,cp_positions_grad,cp_scales_grad,cp_quaternions_grad,
                                                                                          dL_dsph_gauss,dL_dbandwidth_sharpness,dL_dlobe_axis)
      if settings.num_sph_gauss!=0:
        bandwidth_sharpness_grad=bandwidth_sharpness_grad.reshape(-1,settings.num_sph_gauss)
        lobe_axis_grad=lobe_axis_grad.reshape(-1,settings.num_sph_gauss,3)
        sph_gauss_feat_grad=sph_gauss_feat_grad.reshape(-1,3,settings.num_sph_gauss)
      else:
        bandwidth_sharpness_grad=None
        lobe_axis_grad=None
        sph_gauss_feat_grad=None
        
        # quick and ugly solution to perform experiment
        if settings.iteration == 0:
          color_features_grad = None

      grads=(positions_grad,scales_grad,normalized_quaternions_grad,densities_grad,color_features_grad,sph_gauss_feat_grad,bandwidth_sharpness_grad,lobe_axis_grad,None)
      return grads      

      