import os, sys, enum, logging

import torch
import cupy as cp
import numpy as np
import optix as ox
from omegaconf import OmegaConf
from PIL import Image
# import ctypes
from scene.dataset_readers import readCamerasFromTransforms,readColmapSceneInfo

import argparse
import glfw, imgui
from utils import general_utils as utilities
from optix_raycasting import optix_utils as u_ox

from gui_utils.gui import init_ui, display_stats
from gui_utils.gl_display import GLDisplay
from gui_utils.cuda_output_buffer import CudaOutputBuffer, CudaOutputBufferType, BufferImageFormat
from gui_utils.gui_state import GeometryState, Params,init_main_params

from gui_utils.glfw_callback import mouse_button_callback, cursor_position_callback, window_size_callback, window_iconify_callback,key_callback,scroll_callback
from gui_utils.state_handlers import init_launch_params, update_state, launch_subframe, display_subframe, init_camera_state
from classes import point_cloud,scene

parser=argparse.ArgumentParser(description="Ray Gauss GUI: You can either display the output of a training iteration or an rg_ply file. In the first case you must provide the output folder and the iteration to display. In the second case you must provide the path to the rg_ply file")
parser.add_argument("-output", type=str, help="Path to output folder")
parser.add_argument("-iter", type=int, help="Iteration to display")
parser.add_argument("-ply_path", type=str, help="Path to data folder")
args=parser.parse_args()

if not((args.ply_path is not None) ^ (args.output is not None and args.iter is not None)):# XOR
    raise ValueError("You must provide either the output folder and the iteration to display or the data path")

if args.ply_path is not None:
    args.ply_path=os.path.abspath(args.ply_path)
    if not os.path.exists(args.ply_path):
        raise ValueError("The path to the rg_ply file is incorrect")
    
gui_mode= 0 if args.ply_path is not None else 1

script_dir = os.path.dirname(os.path.abspath(__file__))

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level= logging.INFO)

log = logging.getLogger()

DEBUG=False

if DEBUG:
    exception_flags=ox.ExceptionFlags.DEBUG | ox.ExceptionFlags.TRACE_DEPTH | ox.ExceptionFlags.STACK_OVERFLOW,
    debug_level = ox.CompileDebugLevel.FULL
    opt_level = ox.CompileOptimizationLevel.LEVEL_0
else:
    exception_flags=ox.ExceptionFlags.NONE
    debug_level = ox.CompileDebugLevel.MINIMAL
    opt_level = ox.CompileOptimizationLevel.LEVEL_3





#------------------------------------------------------------------------------
# Scene data
#------------------------------------------------------------------------------
if gui_mode: #Case where we have an output file
    path_config=os.path.join(args.output,"config","config.yml")
    config=OmegaConf.load(path_config)

    #Extract available iterations from config.pointcloud.pointcloud_storage
    path_available_iterations=os.path.join(args.output,"models")
    #List available iterations thanks to file as densities_iter"iteration".pt
    available_iterations=[int(f.split("chkpnt")[1].split(".pt")[0]) for f in os.listdir(path_available_iterations) if "chkpnt" in f]
    #Sort the list
    available_iterations.sort()

    data_type="float32"
    device=torch.device("cuda")
    pointcloud=point_cloud.PointCloud(data_type=data_type,device=device)
    tested_scene=scene.Scene(config=config,pointcloud=pointcloud,train_resolution_scales=config.scene.train_resolution_scales,test_resolution_scales=config.scene.test_resolution_scales,init_pc=False)
    train_cam_infos=tested_scene.getTrainCameras()
    test_cam_infos=tested_scene.getTestCameras()
    #Permute in shape (H,W,C), mirror the image, convert to numpy, convert to uint8
    train_images=[(train_cam_infos[i].original_image.permute(1,2,0).cpu().numpy()[::-1,:,:]*255).astype(np.uint8) for i in range(len(train_cam_infos))]
    test_images=[(test_cam_infos[i].original_image.permute(1,2,0).cpu().numpy()[::-1,:,:]*255).astype(np.uint8) for i in range(len(test_cam_infos))]
    #Load the model
    pointcloud.restore_model(iteration=args.iter,checkpoint_folder=path_available_iterations)
else:
    pointcloud=point_cloud.PointCloud(data_type="float32",device=torch.device("cuda"))
    pointcloud.load_from_rg_ply(args.ply_path)
positions,scales,normalized_quaternions,densities,color_features,sph_gauss_features,bandwidth_sharpness,lobe_axis=pointcloud.get_data()
num_sph_gauss_features=sph_gauss_features.shape[2]

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
cp_bboxes = u_ox.compute_ellipsoids_bbox(cp_positions,cp_scales,L1,L2,L3,cp_densities)

if (cp_bboxes.shape[0] == 0):
    bb_min=cp.array([0,0,0],dtype=cp.float32)
    bb_max=cp.array([0,0,0],dtype=cp.float32)
else:
    bb_min=cp_bboxes[:,:3].min(axis=0)
    bb_max=cp_bboxes[:,3:].max(axis=0)
max_prim_slice = 1024

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    state = GeometryState()
    degree_sh = int(np.sqrt(pointcloud.harmonic_number).item()-1)
    width,height=800,800
    hit_sphere_idx = cp.zeros((max_prim_slice*width*height),dtype=cp.uint32)
    state.depth_buffer = cp.zeros((width*height),dtype=cp.float32)
    init_main_params(state.params,max_prim_slice=max_prim_slice,degree_sh=degree_sh,max_sh_degree=degree_sh,
                     num_sg=num_sph_gauss_features,max_sg_display=num_sph_gauss_features,bbox_min=bb_min.get(),bbox_max=bb_max.get(),
                     densities=cp_densities,color_features=cp_color_features,sph_gauss_features=cp_sph_gauss_features,
                        bandwidth_sharpness=cp_bandwidth_sharpness,lobe_axis=cp_lobe_axis,positions=cp_positions,scales=cp_scales,
                        quaternions=cp_quaternions,width=800,height=800,
                        hit_sphere_idx=hit_sphere_idx,depth_buffer_ptr=state.depth_buffer)

    state.gt_image = cp.zeros((width*height,4),dtype=cp.uint8)
    #Compute barycenter of the positions
    np_positions=cp.asnumpy(cp_positions)
    barycenter=np_positions.mean(axis=0)
    state.barycenter=barycenter
    print("gui_mode:",gui_mode)
    if gui_mode:
        state.path_available_iterations=path_available_iterations
        state.available_iterations = available_iterations
        state.current_iteration = args.iter
        state.min_iteration=available_iterations[0]
        state.max_iteration=available_iterations[-1]
        state.slider_train_iteration=args.iter

        state.min_idx_test_im,state.max_idx_test_im=0,len(test_cam_infos)-1
        state.min_idx_train_im,state.max_idx_train_im=0,len(train_cam_infos)-1

        state.added_before_iter=state.max_iteration
    
        state.train_cam_infos, state.test_cam_infos = train_cam_infos, test_cam_infos
        state.train_images, state.test_images = train_images, test_images
        
        #Camera parammeters initialize to first train camera center
        first_train_cam_info=train_cam_infos[0]
        T=first_train_cam_info.T
        R=first_train_cam_info.R
        eye = -R@T
        #Project the barycenter on R[:,2] line passing by camera.eye, supposing there that R[:,2] is still normalized
        len_look_at=np.dot(state.barycenter-eye, R[:,2])
        look_at = eye+R[:,2]*len_look_at
        up = -R[:,1]
        fov_y = first_train_cam_info.FoVy*180/np.pi

        init_camera_state(state,eye=eye,look_at=look_at,up=up,fov_y=fov_y)
    else:
        ############################################################################################################
        #By default the camera is looking at the barycenter of the positions
        init_camera_state(state,look_at=state.barycenter)
        ############################################################################################################

    state.pointcloud=pointcloud
    state.gui_mode=gui_mode


    state.time = 0.0


    buffer_format = BufferImageFormat.UCHAR4
    output_buffer_type = CudaOutputBufferType.enable_gl_interop()

    # create_context(state)
    state.ctx=u_ox.create_context(log)
    # create_module(state)
    state.pipeline_opts = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                            num_payload_values=2,
                                            num_attribute_values=0,
                                            exception_flags=ox.ExceptionFlags.NONE,
                                            pipeline_launch_params_variable_name="params")
    state.module=u_ox.create_module(state.ctx,state.pipeline_opts,stage="gui")
    # create_program_groups(state)
    state.raygen_grp, state.miss_grp, state.hit_grp=u_ox.create_program_groups(state.ctx,state.module)
    # create_pipeline(state)
    program_grps=[state.raygen_grp, state.miss_grp, state.hit_grp]
    state.pipeline=u_ox.create_pipeline(state.ctx,program_grps=program_grps,pipeline_options=state.pipeline_opts,debug_level=debug_level)

    state.sbt=u_ox.create_sbt(program_grps=program_grps,positions=cp_positions,scales=cp_scales,quaternions=cp_quaternions)

    init_launch_params(state)

    # build_bvh(state,cp_bboxes)
    state.gas=u_ox.create_acceleration_structure(state.ctx,cp_bboxes)
    state.params.trav_handle = state.gas.handle

    window, impl = init_ui("optixRadianceField", state.params.width, state.params.height)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_window_size_callback(window, window_size_callback)
    glfw.set_window_iconify_callback(window, window_iconify_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_window_user_pointer(window, state)

    output_buffer = CudaOutputBuffer(output_buffer_type, buffer_format,
            state.params.width, state.params.height)

    gl_display = GLDisplay(buffer_format)

    state_update_time = 0.0
    render_time = 0.0
    display_time = 0.0

    tstart = glfw.get_time()
    state.fp_controls.dt=tstart
    while not glfw.window_should_close(window):

        t0 = glfw.get_time()
        glfw.poll_events()
        impl.process_inputs()

        state.time = glfw.get_time() - tstart
        update_state(output_buffer, state)
        t1 = glfw.get_time()
        state_update_time += t1 - t0
        t0 = t1

        launch_subframe(output_buffer, state)
        t1 = glfw.get_time()
        render_time += t1 - t0
        t0 = t1

        display_subframe(output_buffer, gl_display, window)
        display_time += t1 - t0

        if display_stats(state,state_update_time, render_time, display_time):
            state_update_time = 0.0
            render_time = 0.0
            display_time = 0.0

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        state.params.subframe_index = state.params.subframe_index+ 1
    impl.shutdown()
    glfw.terminate()