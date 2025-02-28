import optix as ox
import cupy as cp
import os
import math
from torch.utils.dlpack import to_dlpack

#Read the SIGMA_THRESHOLD in respective cuda_train/forward/gaussians_aabb.h
header_train_forward= os.path.join(os.path.dirname(__file__), "cuda_train/forward", "gaussians_aabb.h")
header_train_backward= os.path.join(os.path.dirname(__file__), "cuda_train/backward", "gaussians_aabb.h")
header_test= os.path.join(os.path.dirname(__file__), "cuda_test", "gaussians_aabb.h")
header_gui= os.path.join(os.path.dirname(__file__), "cuda_gui", "gaussians_aabb.h")

def get_SIGMA_THRESHOLD(file_name):
    file=open(file_name, "r")
    lines=file.readlines()
    file.close()
    for line in lines:
        if "SIGMA_THRESHOLD" in line:
            SIGMA_THRESHOLD=line.split()[2]
            #Remove f at the end of the number
            SIGMA_THRESHOLD=float(SIGMA_THRESHOLD[:-1])
            break
    return SIGMA_THRESHOLD

sigma_threshold_train_forward=get_SIGMA_THRESHOLD(header_train_forward)
sigma_threshold_train_backward=get_SIGMA_THRESHOLD(header_train_backward)
sigma_threshold_test=get_SIGMA_THRESHOLD(header_test)
sigma_threshold_gui=get_SIGMA_THRESHOLD(header_gui)
if sigma_threshold_train_forward!=sigma_threshold_train_backward or sigma_threshold_train_forward!=sigma_threshold_test or sigma_threshold_train_forward!=sigma_threshold_gui:
    raise ValueError("SIGMA_THRESHOLD is not the same in all headers")
else:
    SIGMA_THRESHOLD=sigma_threshold_train_forward

def quaternion_to_rotation(quaternion):
    # quaternion = [w, x, y, z]
    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]
    L1=cp.zeros((quaternion.shape[0],3),dtype=cp.float32)
    L2=cp.zeros((quaternion.shape[0],3),dtype=cp.float32)
    L3=cp.zeros((quaternion.shape[0],3),dtype=cp.float32)
    L1[:,0],L1[:,1],L1[:,2]=1 - 2 * (y**2 + z**2), 2 * (x*y - w*z), 2 * (x*z + w*y)
    L2[:,0],L2[:,1],L2[:,2]=2 * (x*y + w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*x)
    L3[:,0],L3[:,1],L3[:,2]=2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)
    return L1,L2,L3

def compute_spheres_bbox(centers,scales):
    out = cp.empty((centers.shape[0], 6), dtype='f4')
    out[:, :3] = centers - 3*scales
    out[:, 3:] = centers + 3*scales
    return out

def compute_ellipsoids_bbox(centers,scales,L1,L2,L3,densities):
    delta=cp.log((densities/SIGMA_THRESHOLD)**2)
    delta[delta<0]=0
    delta=cp.sqrt(delta)

    out = cp.empty((centers.shape[0], 6), dtype='f4')
    
    scales_L1 = cp.linalg.norm(delta[:,None]*scales * L1, axis=1)
    scales_L2 = cp.linalg.norm(delta[:,None]*scales * L2, axis=1)
    scales_L3 = cp.linalg.norm(delta[:,None]*scales * L3, axis=1)
    #I want a Nx3 array with [L1,L2,L3] for each ellipsoid
    out[:, :3] = centers - cp.vstack([scales_L1,scales_L2,scales_L3]).T
    out[:, 3:] = centers + cp.vstack([scales_L1,scales_L2,scales_L3]).T
    return out

def create_acceleration_structure(ctx, bboxes):
    build_input = ox.BuildInputCustomPrimitiveArray([bboxes], num_sbt_records=1, flags=[ox.GeometryFlags.NONE])
    gas = ox.AccelerationStructure(ctx, [build_input], compact=True,allow_update=True)
    return gas

def update_acceleration_structure(gas, bboxes):
    build_input = ox.BuildInputCustomPrimitiveArray([bboxes], num_sbt_records=1, flags=[ox.GeometryFlags.NONE])
    gas.update(build_input)

def create_context(log):
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=False, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = False
    return ctx

def create_module(ctx, pipeline_opts,stage):
    script_dir = os.path.dirname(__file__)
    if stage=="forward":
        cuda_src = os.path.join(script_dir, "cuda_train/forward", "gaussians_aabb.cu")
    elif stage=="backward":
        cuda_src = os.path.join(script_dir, "cuda_train/backward", "gaussians_aabb.cu")
    elif stage=="test":
        cuda_src = os.path.join(script_dir, "cuda_test", "gaussians_aabb.cu")
    elif stage=="gui":
        cuda_src = os.path.join(script_dir, "cuda_gui", "gaussians_aabb.cu")
    else:
        raise ValueError("stage must be 'forward', 'backward', 'test' or 'gui'")
    compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.DEFAULT , opt_level=ox.CompileOptimizationLevel.DEFAULT )
    module = ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)
    return module

def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_IS="__intersection__gaussian",
                                              entry_function_AH="__anyhit__ah")
    return raygen_grp, miss_grp, hit_grp

def create_pipeline(ctx, program_grps, pipeline_options,debug_level=ox.CompileDebugLevel.DEFAULT):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1, debug_level=debug_level)

    pipeline = ox.Pipeline(ctx, compile_options=pipeline_options, link_options=link_opts, program_groups=program_grps)
    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 0)  # max_dc_depth
    return pipeline

def create_sbt(program_grps, positions,scales,quaternions):
    raygen_grp, miss_grp, hit_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp)
    hit_sbt = ox.SbtRecord(hit_grp, names=('positions','scales','quaternions'), formats=('u8','u8','u8'))
    hit_sbt['positions'] = positions.data.ptr
    hit_sbt['scales'] = scales.data.ptr
    hit_sbt['quaternions'] = quaternions.data.ptr
    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)
    return sbt

def launch_pipeline_forward(pipeline : ox.Pipeline, sbt, gas,bbox_min,bbox_max,camera,
                    densities,color_features,positions,scales,quaternions,
                    max_prim_slice,iteration,jitter,rnd_sample, supersampling, white_background,
                    hit_prim_idx):
    ray_size=(camera.image_height*camera.image_width*supersampling[0]*supersampling[1],)
    params_tmp = [
        ('u4',  'iteration'),
        ('u4',  'jitter'),
        ('u4',  'rnd_sample'),
        ('u4',  'max_prim_slice'),
        ('3f4', 'bbox_min'),
        ('3f4', 'bbox_max'),
        ('u4',  'image_width'),
        ('u4',  'image_height'),
        ('3f4', 'cam_eye'),
        ('3f4', 'cam_u'),
        ('3f4', 'cam_v'),
        ('3f4', 'cam_w'),
        ('f4',  'cam_tan_half_fovx'),
        ('f4',  'cam_tan_half_fovy'),
        ('u8',  'densities'),
        ('u8',  'color_features'),
        ('u8',  'positions'),
        ('u8',  'scales'),
        ('u8',  'quaternions'),
        ('u8',  'hit_prim_idx'),
        ('u8',  'ray_colors'),
        ('u8',  'handle'),
        ('u8',  'max_gaussians_exceeded'),
        ('u4',  'white_background'),
        # BASELINE
        ('u8',  'ray_bboxes'),
        ('u8',  'ray_normals'),
        ('u8',  'ray_densities'),
        ('u8',  'ray_depths'),
    ]

    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                formats=[p[0] for p in params_tmp])
    params['iteration'] = iteration
    params['jitter'] = jitter
    params['rnd_sample'] = rnd_sample
    params['max_prim_slice'] = max_prim_slice
    params['bbox_min'] = bbox_min.get()
    params['bbox_max'] = bbox_max.get()
    params['image_width'] = camera.image_width*supersampling[0]
    params['image_height'] = camera.image_height*supersampling[1]
    params['cam_eye'] = cp.fromDlpack(to_dlpack(camera.camera_center)).get()
    params['cam_u'] = cp.fromDlpack(to_dlpack(camera.world_view_transform[:3, 0])).get()
    params['cam_v'] = cp.fromDlpack(to_dlpack(camera.world_view_transform[:3, 1])).get()
    params['cam_w'] = cp.fromDlpack(to_dlpack(camera.world_view_transform[:3, 2])).get()

    # params['cam_tan_half_fov'] = cp.array([cp.tan(camera.FoVx / 2), cp.tan(camera.FoVy / 2)]).get()
    params['cam_tan_half_fovx'] = math.tan(camera.FoVx / 2)
    params['cam_tan_half_fovy'] = math.tan(camera.FoVy / 2)

    params['densities'] = densities.data.ptr
    params['color_features'] = color_features.data.ptr
    params['positions'] = positions.data.ptr
    params['scales'] = scales.data.ptr
    params['quaternions'] = quaternions.data.ptr

    params['hit_prim_idx'] = hit_prim_idx.data.ptr

    cp_ray_colors=cp.zeros((ray_size[0],3), dtype=cp.float32)
    params['ray_colors']=cp_ray_colors.data.ptr
    
    # ---------------------------------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------------------------------
    cp_ray_bbox             = cp.zeros( (ray_size[0],3), dtype=cp.float32 )
    params['ray_bboxes']    = cp_ray_bbox.data.ptr
    
    cp_ray_normals          = cp.zeros( (ray_size[0],3), dtype=cp.float32 )
    params['ray_normals']    = cp_ray_normals.data.ptr
    
    cp_ray_density          = cp.zeros( (ray_size[0],1), dtype=cp.float32 )
    params['ray_densities'] = cp_ray_density.data.ptr
    
    cp_ray_depth            = cp.zeros( (ray_size[0],1), dtype=cp.float32 )
    params['ray_depths']    = cp_ray_depth.data.ptr
    
    # ---------------------------------------------------------------------------------    
    
    params['handle'] = gas.handle

    cp_max_gaussians_exceed = cp.zeros((1), dtype=cp.int32)
    params['max_gaussians_exceeded'] = cp_max_gaussians_exceed.data.ptr

    if white_background:
        params['white_background'] = 1
    else:
        params['white_background'] = 0

    stream = cp.cuda.Stream()
    pipeline.launch(sbt, dimensions=ray_size, params=params, stream=stream)
    stream.synchronize()

    
    if (cp_max_gaussians_exceed[0] > 0):
        if (iteration%1000 == 0):
            print(" ############ Number slabs with max gaussians exceeded : ", cp_max_gaussians_exceed[0])

    return cp_ray_colors, cp_ray_bbox, cp_ray_normals, cp_ray_density, cp_ray_depth

##################################################################

def launch_pipeline_backward(pipeline : ox.Pipeline, sbt, gas,bbox_min,bbox_max,camera,
                    densities,color_features,positions,scales,quaternions,
                    ray_colors,dloss_dray_colors,
                    max_prim_slice,iteration,jitter,rnd_sample,supersampling,
                    hit_prim_idx):
    ray_size=(camera.image_height*camera.image_width*supersampling[0]*supersampling[1],)
    params_tmp = [
        ('u4', 'iteration'),
        ('u4', 'jitter'),
        ('u4', 'rnd_sample'),
        ( 'u4', 'max_prim_slice'),
        ( '3f4', 'bbox_min'),
        ( '3f4', 'bbox_max'),
        ('u4', 'image_width'),
        ('u4', 'image_height'),
        ('3f4', 'cam_eye'),
        ('3f4', 'cam_u'),
        ('3f4', 'cam_v'),
        ('3f4', 'cam_w'),
        ('f4', 'cam_tan_half_fovx'),
        ('f4', 'cam_tan_half_fovy'),
        ( 'u8', 'densities'),
        ( 'u8', 'color_features'),
        ( 'u8', 'positions'),
        ( 'u8', 'scales'),
        ( 'u8', 'quaternions'), #quaternions
        ( 'u8', 'hit_prim_idx'),
        ( 'u8', 'ray_colors'),
        ( 'u8', 'dloss_dray_colors'),
        ( 'u8', 'densities_grad'),
        ( 'u8', 'color_features_grad'),
        ( 'u8', 'positions_grad'),
        ( 'u8', 'scales_grad'),
        ( 'u8', 'quaternions_grad'),
        ( 'u8', 'handle')
    ]

    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                formats=[p[0] for p in params_tmp])
    params['iteration'] = iteration
    params['jitter'] = jitter
    params['rnd_sample'] = rnd_sample
    params['max_prim_slice'] = max_prim_slice
    params['bbox_min'] = bbox_min.get()
    params['bbox_max'] = bbox_max.get()
    params['image_width'] = camera.image_width*supersampling[0]
    params['image_height'] = camera.image_height*supersampling[1]
    params['cam_eye'] = cp.fromDlpack(to_dlpack(camera.camera_center)).get()
    params['cam_u'] = cp.fromDlpack(to_dlpack(camera.world_view_transform[:3, 0])).get()
    params['cam_v'] = cp.fromDlpack(to_dlpack(camera.world_view_transform[:3, 1])).get()
    params['cam_w'] = cp.fromDlpack(to_dlpack(camera.world_view_transform[:3, 2])).get()

    # params['cam_tan_half_fov'] = cp.array([cp.tan(camera.FoVx / 2), cp.tan(camera.FoVy / 2)]).get()
    params['cam_tan_half_fovx'] = math.tan(camera.FoVx / 2)
    params['cam_tan_half_fovy'] = math.tan(camera.FoVy / 2)

    params['densities'] = densities.data.ptr
    params['color_features'] = color_features.data.ptr
    params['positions'] = positions.data.ptr
    params['scales'] = scales.data.ptr
    params['quaternions'] = quaternions.data.ptr

    # hit_prim_idx=cp.zeros((ray_size[0]*max_prim_slice), dtype=cp.int32)
    
    params['hit_prim_idx'] = hit_prim_idx.data.ptr

    params['ray_colors']=ray_colors.data.ptr
    params['dloss_dray_colors']=dloss_dray_colors.data.ptr

    densities_grad=cp.zeros((densities.shape[0]), dtype=cp.float32)
    params['densities_grad']=densities_grad.data.ptr
    color_features_grad=cp.zeros((positions.shape[0],3), dtype=cp.float32)
    params['color_features_grad']=color_features_grad.data.ptr
    positions_grad=cp.zeros((positions.shape[0],3), dtype=cp.float32)
    scales_grad=cp.zeros((scales.shape[0],3), dtype=cp.float32)
    params['positions_grad']=positions_grad.data.ptr
    params['scales_grad']=scales_grad.data.ptr
    quaternions_grad=cp.zeros((quaternions.shape[0],4), dtype=cp.float32)
    params['quaternions_grad']=quaternions_grad.data.ptr

    params['handle'] = gas.handle

    stream = cp.cuda.Stream()
    pipeline.launch(sbt, dimensions=ray_size, params=params, stream=stream)
    stream.synchronize()

    return densities_grad,color_features_grad,positions_grad,scales_grad,quaternions_grad
##################################################################

def launch_pipeline_test(pipeline : ox.Pipeline, sbt, gas,bbox_min,bbox_max,camera,
                    densities,color_features,positions,scales,quaternions,
                    max_prim_slice,rnd_sample,supersampling,white_background):
    ray_size=(camera.image_height*camera.image_width*supersampling[0]*supersampling[1],)
    params_tmp = [
        ( 'u4', 'rnd_sample'),
        ( 'u4', 'max_prim_slice'),
        ( '3f4', 'bbox_min'),
        ( '3f4', 'bbox_max'),
        ('u4', 'image_width'),
        ('u4', 'image_height'),
        ('3f4', 'cam_eye'),
        ('3f4', 'cam_u'),
        ('3f4', 'cam_v'),
        ('3f4', 'cam_w'),
        ('f4', 'cam_tan_half_fovx'),
        ('f4', 'cam_tan_half_fovy'),
        ( 'u8', 'densities'),
        ( 'u8', 'color_features'),
        ( 'u8', 'positions'),
        ( 'u8', 'scales'),
        ( 'u8', 'quaternions'),
        ( 'u8', 'ray_colors'),
        ( 'u8','number_of_gaussians_per_ray'),
        ( 'u8','number_of_gaussians_per_slab'),
        
        ( 'u8', 'handle'),
        ( 'u4', 'white_background')
    ]

    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                formats=[p[0] for p in params_tmp])
    params['rnd_sample'] = rnd_sample
    params['max_prim_slice'] = max_prim_slice
    params['bbox_min'] = bbox_min.get()
    params['bbox_max'] = bbox_max.get()
    params['image_width'] = camera.image_width*supersampling[0]
    params['image_height'] = camera.image_height*supersampling[1]
    params['cam_eye'] = cp.fromDlpack(to_dlpack(camera.camera_center)).get()
    params['cam_u'] = cp.fromDlpack(to_dlpack(camera.world_view_transform[:3, 0])).get()
    params['cam_v'] = cp.fromDlpack(to_dlpack(camera.world_view_transform[:3, 1])).get()
    params['cam_w'] = cp.fromDlpack(to_dlpack(camera.world_view_transform[:3, 2])).get()
    
    # params['cam_tan_half_fov'] =  cp.array([cp.tan(camera.FoVx / 2), cp.tan(camera.FoVy / 2)]).get()
    params['cam_tan_half_fovx'] = math.tan(camera.FoVx / 2)
    params['cam_tan_half_fovy'] = math.tan(camera.FoVy / 2)

    params['densities'] = densities.data.ptr
    params['color_features'] = color_features.data.ptr
    params['positions'] = positions.data.ptr
    params['scales'] = scales.data.ptr
    params['quaternions'] = quaternions.data.ptr

    ray_colors=cp.zeros((ray_size[0],3), dtype=cp.float32)
    params['ray_colors']=ray_colors.data.ptr
    
    number_of_gaussians_per_ray=cp.zeros((ray_size[0],), dtype=cp.int32)
    params['number_of_gaussians_per_ray']=number_of_gaussians_per_ray.data.ptr

    params['handle'] = gas.handle

    if white_background:
        params['white_background'] = 1
    else:
        params['white_background'] = 0
    
    stream = cp.cuda.Stream()
    
    pipeline.launch(sbt, dimensions=ray_size, params=params, stream=stream)
    stream.synchronize()
    
    return ray_colors
