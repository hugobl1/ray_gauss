from optix_raycasting import optix_utils as u_ox
import cupy as cp
import numpy as np
from utils import general_utils as utilities
from PIL import Image
import glfw
from gui_utils.cupy_gui_util import abs_diff_kernel, copy_kernel, copy_normalize_depth_kernel


#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------
def init_launch_params(state):
    state.params.frame_buffer = 0
    state.params.subframe_index = 0

def handle_gaussians_update(state):
    closest_iteration= min(state.available_iterations, key=lambda x:abs(x-state.slider_train_iteration))
    if not state.current_iteration == closest_iteration:
        #Get closest iteration in available_iterations
        #Get the corresponding tensors
        # folder_tensors=config.pointcloud.pointcloud_storage
        # folder_tensors=os.path.join(args.output,"model")
        # name_densities="densities_iter"+str(closest_iteration)
        # name_scales="scales_iter"+str(closest_iteration)
        # name_quaternions="quaternions_iter"+str(closest_iteration)
        # name_positions="positions_iter"+str(closest_iteration)
        # name_spherical_harmonics="spherical_harmonics_iter"+str(closest_iteration)
        # pointcloud.load_from_pt(name_positions="positions",name_spherical_harmonics="spherical_harmonics",name_densities="densities", name_scales="scales", name_quaternions="quaternions",
        #            folder_tensors="saved_tensors")
        # pointcloud.load_from_pt(name_positions=name_positions,name_spherical_harmonics=name_spherical_harmonics,name_densities=name_densities, name_scales=name_scales, name_quaternions=name_quaternions,
        #              folder_tensors=folder_tensors)
        state.pointcloud.restore_model(iteration=closest_iteration,checkpoint_folder=state.path_available_iterations)
        # positions,scales,normalized_quaternions,densities,color_features=pointcloud.get_data()
        positions,scales,normalized_quaternions,densities,color_features,sph_gauss_features,bandwidth_sharpness,lobe_axis=state.pointcloud.get_data()

        if state.show_added_gaussians:
            iter_added_gaussians=(state.pointcloud.grown_points>0)*(state.pointcloud.grown_points<=state.added_before_iter)
            positions=positions[iter_added_gaussians]
            scales=scales[iter_added_gaussians]
            normalized_quaternions=normalized_quaternions[iter_added_gaussians]
            densities=densities[iter_added_gaussians]
            color_features=color_features[iter_added_gaussians]
            sph_gauss_features=sph_gauss_features[iter_added_gaussians]
            bandwidth_sharpness=bandwidth_sharpness[iter_added_gaussians]
            lobe_axis=lobe_axis[iter_added_gaussians]

        state.params.degree_sh = int(np.sqrt(color_features.shape[2]).item()-1)
        state.params.max_sh_degree = state.params.degree_sh
        state.params.num_sg=sph_gauss_features.shape[2]
        state.params.max_sg_display=state.params.num_sg
        # cp_densities,cp_scales,cp_quaternions,cp_positions,cp_color_features=utilities.torch2cupy(
        #             densities,
        #             scales,
        #             normalized_quaternions,
        #             positions,
        #             color_features.reshape(-1))
        
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
        state.current_iteration = closest_iteration
        state.params.bbox_min = bb_min.get()
        state.params.bbox_max = bb_max.get()
        state.params.densities = cp_densities.data.ptr
        state.params.color_features = cp_color_features.data.ptr
        state.params.sph_gauss_features = cp_sph_gauss_features.data.ptr
        state.params.bandwidth_sharpness = cp_bandwidth_sharpness.data.ptr
        state.params.lobe_axis = cp_lobe_axis.data.ptr
        state.params.positions = cp_positions.data.ptr
        state.params.scales = cp_scales.data.ptr
        state.params.quaternions = cp_quaternions.data.ptr


        program_grps=(state.raygen_grp, state.miss_grp, state.hit_grp)
        state.sbt=u_ox.create_sbt(program_grps=program_grps,positions=cp_positions,scales=cp_scales,quaternions=cp_quaternions)
        state.gas=u_ox.create_acceleration_structure(state.ctx,cp_bboxes)
        state.params.trav_handle = state.gas.handle
        # build_bvh(state,cp_bboxes)

def handle_test_train_update(state):
    if (state.update_train_im):
        state.update_train_im = False
        state.slider_used = True

        train_cam_info=state.train_cam_infos[state.idx_train_im]
        camera = state.camera
        T=train_cam_info.T
        R=train_cam_info.R
        camera.eye = -R@T
        #Project the barycenter on R[:,2] line passing by camera.eye, supposing there that R[:,2] is still normalized
        len_look_at=np.dot(state.barycenter-camera.eye, R[:,2])
        camera.look_at = camera.eye+R[:,2]*len_look_at
        camera.up = -R[:,1]
        camera.fov_y = train_cam_info.FoVy*180/np.pi

        ##############################################
        # state.camera_changed = True
        params = state.params
        camera.aspect_ratio = params.width / float(params.height)
        params.eye = camera.eye
        u,v,w = camera.uvw_frame()
        params.u = u
        params.v = v
        params.w = w
        ##############################################

        state.trackball.reinitialize_orientation_from_camera()

        gt_image=state.train_images[state.idx_train_im]
        if not (state.params.width == gt_image.shape[1] and state.params.height == gt_image.shape[0]):
            h=state.params.height/gt_image.shape[0]
            w_bis=gt_image.shape[1]*h
            #Reshape the image to the size of the window
            gt_image=Image.fromarray(gt_image)
            gt_image=gt_image.resize((int(w_bis),state.params.height),Image.LANCZOS)
            #Pad or crop to the right size
            if w_bis>state.params.width:
                gt_image=gt_image.crop((int((w_bis-state.params.width)/2),0,int((w_bis+state.params.width)/2),state.params.height))
            else:
                #Pad the image
                new_gt_image=Image.new(gt_image.mode,(state.params.width,state.params.height))
                new_gt_image.paste(gt_image,(int((state.params.width-w_bis)/2),0))
                gt_image=new_gt_image
            gt_image=np.array(gt_image.convert("RGB"))
            gt_image=gt_image.reshape(-1,3)
        else:
            #Flatten the first two dimensions
            gt_image=gt_image.reshape(-1,3)
        state.gt_image = cp.array(gt_image)

    if (state.update_test_im):
        state.update_test_im = False
        state.slider_used = True

        test_cam_info=state.test_cam_infos[state.idx_test_im]
        camera = state.camera
        T=test_cam_info.T
        R=test_cam_info.R
        camera.eye = -R@T
        #Project the barycenter on R[:,2] line passing by camera.eye, supposing there that R[:,2] is still normalized
        len_look_at=np.dot(state.barycenter-camera.eye, R[:,2])
        camera.look_at = camera.eye+R[:,2]*len_look_at
        camera.up = -R[:,1]
        camera.fov_y = test_cam_info.FoVy*180/np.pi

        ##############################################
        # state.camera_changed = True
        params = state.params
        camera.aspect_ratio = params.width / float(params.height)
        params.eye = camera.eye
        u,v,w = camera.uvw_frame()
        params.u = u
        params.v = v
        params.w = w
        ##############################################

        state.trackball.reinitialize_orientation_from_camera()

        gt_image=state.test_images[state.idx_test_im]
        #Check if the size of the window is the same as the image
        if not (state.params.width == gt_image.shape[1] and state.params.height == gt_image.shape[0]):
            h=state.params.height/gt_image.shape[0]
            w_bis=gt_image.shape[1]*h
            #Reshape the image to the size of the window
            gt_image=Image.fromarray(gt_image)
            gt_image=gt_image.resize((int(w_bis),state.params.height),Image.LANCZOS)
            #Pad or crop to the right size
            if w_bis>state.params.width:
                gt_image=gt_image.crop((int((w_bis-state.params.width)/2),0,int((w_bis+state.params.width)/2),state.params.height))
            else:
                #Pad the image
                new_gt_image=Image.new(gt_image.mode,(state.params.width,state.params.height))
                new_gt_image.paste(gt_image,(int((state.params.width-w_bis)/2),0))
                gt_image=new_gt_image
            gt_image=np.array(gt_image.convert("RGB"))
            gt_image=gt_image.reshape(-1,3)
        else:
            #Flatten the first two dimensions
            gt_image=gt_image.reshape(-1,3)
        state.gt_image = cp.array(gt_image)

def handle_added_gaussians_update(state):
    if state.update_added_gaussians:
        if state.show_added_gaussians:
            positions,scales,normalized_quaternions,densities,color_features,sph_gauss_features,bandwidth_sharpness,lobe_axis=state.pointcloud.get_data()
            iter_added_gaussians=(state.pointcloud.grown_points>0)*(state.pointcloud.grown_points<=state.added_before_iter)
            positions_added_gaussians=positions[iter_added_gaussians]
            scales_added_gaussians=scales[iter_added_gaussians]
            normalized_quaternions_added_gaussians=normalized_quaternions[iter_added_gaussians]
            densities_added_gaussians=densities[iter_added_gaussians]
            color_features_added_gaussians=color_features[iter_added_gaussians]
            state.params.degree_sh = int(np.sqrt(color_features.shape[2]).item()-1)
            state.params.max_sh_degree = state.params.degree_sh
            cp_densities,cp_scales,cp_quaternions,cp_positions,cp_color_features=utilities.torch2cupy(
                        densities_added_gaussians,
                        scales_added_gaussians,
                        normalized_quaternions_added_gaussians,
                        positions_added_gaussians,
                        color_features_added_gaussians.reshape(-1))

            L1,L2,L3=u_ox.quaternion_to_rotation(cp_quaternions)
            cp_bboxes = u_ox.compute_ellipsoids_bbox(cp_positions,cp_scales,L1,L2,L3,cp_densities)

            if (cp_bboxes.shape[0] == 0):
                bb_min=cp.array([0,0,0],dtype=cp.float32)
                bb_max=cp.array([0,0,0],dtype=cp.float32)
            else:
                bb_min=cp_bboxes[:,:3].min(axis=0)
                bb_max=cp_bboxes[:,3:].max(axis=0)

            state.params.bbox_min = bb_min.get()
            state.params.bbox_max = bb_max.get()
            state.params.densities = cp_densities.data.ptr
            state.params.color_features = cp_color_features.data.ptr
            state.params.positions = cp_positions.data.ptr
            state.params.scales = cp_scales.data.ptr
            state.params.quaternions = cp_quaternions.data.ptr

            program_grps=(state.raygen_grp, state.miss_grp, state.hit_grp)
            state.sbt=u_ox.create_sbt(program_grps=program_grps,positions=cp_positions,scales=cp_scales,quaternions=cp_quaternions)

            # build_bvh(state,cp_bboxes)
            state.gas=u_ox.create_acceleration_structure(state.ctx,cp_bboxes)
            state.params.trav_handle = state.gas.handle
        else:
            positions,scales,normalized_quaternions,densities,color_features=state.pointcloud.get_data()
            state.params.degree_sh = int(np.sqrt(color_features.shape[2]).item()-1)
            state.params.max_sh_degree = state.params.degree_sh
            cp_densities,cp_scales,cp_quaternions,cp_positions,cp_color_features=utilities.torch2cupy(
                        densities,
                        scales,
                        normalized_quaternions,
                        positions,
                        color_features.reshape(-1))

            L1,L2,L3=u_ox.quaternion_to_rotation(cp_quaternions)
            cp_bboxes = u_ox.compute_ellipsoids_bbox(cp_positions,cp_scales,L1,L2,L3,cp_densities)

            if (cp_bboxes.shape[0] == 0):
                bb_min=cp.array([0,0,0],dtype=cp.float32)
                bb_max=cp.array([0,0,0],dtype=cp.float32)
            else:
                bb_min=cp_bboxes[:,:3].min(axis=0)
                bb_max=cp_bboxes[:,3:].max(axis=0)

            state.params.bbox_min = bb_min.get()
            state.params.bbox_max = bb_max.get()
            state.params.densities = cp_densities.data.ptr
            state.params.color_features = cp_color_features.data.ptr
            state.params.positions = cp_positions.data.ptr
            state.params.scales = cp_scales.data.ptr
            state.params.quaternions = cp_quaternions.data.ptr

            program_grps=(state.raygen_grp, state.miss_grp, state.hit_grp)
            state.sbt=u_ox.create_sbt(program_grps=program_grps,positions=cp_positions,scales=cp_scales,quaternions=cp_quaternions)

            # build_bvh(state,cp_bboxes)
            state.gas=u_ox.create_acceleration_structure(state.ctx,cp_bboxes)
            state.params.trav_handle = state.gas.handle


def handle_camera_update(state):
    if not state.camera_changed:
        return
    #Remove print_error_image and print_gt_image
    state.slider_used = False
    state.print_error_image = False
    state.print_gt_image = False
    state.log_scale_error=False
    state.camera_changed = False

    camera = state.camera
    params = state.params

    camera.aspect_ratio = params.width / float(params.height)
    params.eye = camera.eye

    u,v,w = camera.uvw_frame()
    params.u = u
    params.v = v
    params.w = w

def handle_resize(output_buffer, state):
    if not state.resize_dirty:
        return
    state.resize_dirty = False

    output_buffer.resize(state.params.width, state.params.height)

def update_state(output_buffer, state):
    if state.camera_mode == 1:
        state.camera_changed=state.fp_controls.update()
    if state.gui_mode == 1:
        handle_gaussians_update(state)
        handle_test_train_update(state)
        handle_added_gaussians_update(state)
    handle_camera_update(state) #Remove the error mode
    handle_resize(output_buffer, state)
    

def launch_subframe(output_buffer, state):
    state.params.frame_buffer = output_buffer.map()
    state.pipeline.launch(state.sbt, dimensions=state.launch_dimensions,
            params=state.params.handle, stream=output_buffer.stream)

    output_buffer.stream.synchronize()
    output_buffer.unmap()
    # if state.is_window_focused:
    if state.print_error_image:
        a=output_buffer.map()
        b=state.gt_image
        # Define the grid and block size
        size=state.params.width*state.params.height
        block_size = 256
        grid_size = (size + block_size - 1) // block_size
        abs_diff_kernel((grid_size,), (block_size,), (a, b, size,state.log_scale_error))
        output_buffer.stream.synchronize()
        output_buffer.unmap()

    if state.print_gt_image:
        pbo=output_buffer.map()
        size=state.params.width*state.params.height
        block_size = 256
        grid_size = (size + block_size - 1) // block_size
        copy_kernel((grid_size,), (block_size,), (pbo, state.gt_image, size))
        output_buffer.stream.synchronize()
        output_buffer.unmap()

    if state.print_depth:
        pbo=output_buffer.map()
        #Interpret state.params.depth_buffer as a cupy array
        #Max_depth is an array of size grid_size
        # size=state.params.width*state.params.height
        # max_depth_array=cp.zeros((size),dtype=cp.float32)
        # current_depth_array=state.params.depth_buffer
        # while size>1:
        #     block_size = 256
        #     grid_size = (size + block_size - 1) // block_size
        #     maximum_kernel((grid_size,), (block_size,), (current_depth_array, size,max_depth_array))
        #     size=grid_size
        #     current_depth_array=max_depth_array.copy()
        # max_depth=max_depth_array[0]
        min_depth=state.depth_buffer[state.depth_buffer!=0].min().get()
        max_depth=state.depth_buffer.max().get()
        size=state.params.width*state.params.height
        block_size = 256
        grid_size = (size + block_size - 1) // block_size
        copy_normalize_depth_kernel((grid_size,), (block_size,), (pbo, state.params.depth_buffer_ptr,min_depth, max_depth, size))
        output_buffer.stream.synchronize()
        output_buffer.unmap()




def display_subframe(output_buffer, gl_display, window):
    (framebuf_res_x, framebuf_res_y) = glfw.get_framebuffer_size(window)
    gl_display.display( output_buffer.width, output_buffer.height,
                        framebuf_res_x, framebuf_res_y,
                        output_buffer.get_pbo() )


def init_camera_state(state,eye=None,look_at=None,up=None,fov_y=None):
    camera = state.camera
    camera.eye = (-2.895067, -0.48323154, 2.7631636)
    if eye is not None:
        camera.eye = eye
    camera.look_at = (0, 0, 0)
    if look_at is not None:
        camera.look_at = look_at
    camera.up = (0, 0, 1)
    if up is not None:
        camera.up = up
    camera.fov_y = 35
    if fov_y is not None:
        camera.fov_y = fov_y
    state.camera_changed = True

    trackball = state.trackball
    trackball.move_speed = 10.0
    trackball.set_reference_frame([1,0,0], [0,1,0],[0,0,1])
    trackball.reinitialize_orientation_from_camera()